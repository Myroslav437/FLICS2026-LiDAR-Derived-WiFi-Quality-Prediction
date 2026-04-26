"""Helpers for the per-day time-synchronisation pipeline.

Public types
------------
DayMeta : per-session_date record (concatenated telemetry across all
          CSV files of that day; one entry per calibration day).

Functions
---------
enumerate_days            Phase 1 — enumerate calibration days, characterise.
build_lidar_dissim_cache  Streaming LiDAR scan-to-scan dissimilarity per day
                          (reuses the obsolete pipeline cache when present;
                          otherwise recomputes from the H5 file).
make_day_signature        Phase 2 — build per-day telemetry rot_aware signal,
                          per-day LiDAR scan-dissimilarity, resample to 50 Hz,
                          mask telemetry gaps, z-score.
xcorr_lags                FFT-based normalised cross-correlation, |tau|<=cap.
parabolic_refine          Sub-sample peak refinement (Eq. (1)).
peak_prominence           Quality-gate prominence calculation.
moving_block_bootstrap    Bootstrap uncertainty for tau_hat.
cochran_q                 Inverse-variance-weighted mean + Cochran's Q.
cis_overlap               95% CI overlap test (Gaussian approximation).

Sign convention is in ``config.py``.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math

import h5py
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DayMeta:
    session_date: str
    n_csv_files: int
    csv_files: list[str]

    # telemetry slice (per-day, sorted)
    tel_start: pd.Timestamp
    tel_end: pd.Timestamp
    tel_n: int
    tel_period_s: float
    tel_rate_hz: float
    tel_gap_seconds_sum: float       # total wall-clock seconds inside gaps > GAP_MASK_S
    tel_gap_count: int

    # lidar slice (per-day, intersected with telemetry window)
    lid_start: pd.Timestamp
    lid_end: pd.Timestamp
    lid_n: int
    lid_period_s: float
    lid_rate_hz: float

    # intersection
    intersect_start: pd.Timestamp
    intersect_end: pd.Timestamp
    intersect_seconds: float

    # motion
    motion_seconds: float
    n_stop_start_events: int

    eligible_for_estimation: bool
    eligibility_reason: str = ""

    def to_jsonable(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
        return d


# ---------------------------------------------------------------------------
# Phase 1 — enumerate days
# ---------------------------------------------------------------------------

def _load_lidar_index() -> pd.DataFrame:
    with h5py.File(C.LIDAR_H5, "r") as f:
        ts = f["local_timestamps"][:]
        sd = f["session_date"][:]
    sd = pd.Series(sd).astype(str).str.replace(r"^b'|'$", "", regex=True)
    df = pd.DataFrame({
        "row": np.arange(len(ts)),
        "ts_unix_cet": ts,
        "session_date": sd.values,
    })
    df["ts"] = pd.to_datetime(df["ts_unix_cet"], unit="s")
    return df


def enumerate_days(verbose: bool = True) -> tuple[list[DayMeta], pd.DataFrame, pd.DataFrame]:
    tel = pd.read_parquet(C.TELEMETRY_PARQUET)
    tel["fh7000_timestamp"] = tel["fh7000_timestamp"].astype("datetime64[ns]")
    tel = tel.sort_values(["session_date", "fh7000_timestamp"]).reset_index(drop=True)
    lid = _load_lidar_index()

    days: list[DayMeta] = []
    for sd, g in tel.groupby("session_date", sort=True):
        g = g.sort_values("fh7000_timestamp").reset_index(drop=True)
        ts = g["fh7000_timestamp"]
        tel_start = ts.iloc[0]
        tel_end = ts.iloc[-1]

        dt_tel = ts.diff().dt.total_seconds().dropna()
        dt_clean = dt_tel[(dt_tel > 0) & (dt_tel < 1.0)]
        tel_period = float(dt_clean.median()) if len(dt_clean) else float(dt_tel.median())
        tel_rate = 1.0 / tel_period if tel_period > 0 else float("nan")

        gap_mask = dt_tel > C.GAP_MASK_S
        gap_count = int(gap_mask.sum())
        gap_sum = float(dt_tel[gap_mask].sum())

        lpart = lid[lid["session_date"] == sd]
        margin = pd.Timedelta(seconds=2)
        ll = lpart[(lpart["ts"] >= tel_start - margin)
                   & (lpart["ts"] <= tel_end + margin)]
        if len(ll) >= 2:
            ldt = np.diff(ll["ts_unix_cet"].values)
            ldt = ldt[(ldt > 0) & (ldt < 1.0)]
            lid_period = float(np.median(ldt)) if ldt.size else float("nan")
            lid_rate = 1.0 / lid_period if lid_period and lid_period > 0 else float("nan")
            lid_start = ll["ts"].iloc[0]
            lid_end = ll["ts"].iloc[-1]
        else:
            lid_period = float("nan"); lid_rate = float("nan")
            lid_start = pd.NaT; lid_end = pd.NaT

        intersect_start = max(tel_start, lid_start) if pd.notna(lid_start) else pd.NaT
        intersect_end = min(tel_end, lid_end) if pd.notna(lid_end) else pd.NaT
        intersect_s = (intersect_end - intersect_start).total_seconds() \
            if pd.notna(intersect_start) and pd.notna(intersect_end) else float("nan")

        # motion: sum of (sample-period) over moving samples; matches obsolete defn
        speed = g["speed_mps"].astype(float).abs().fillna(0.0).values
        moving = speed > C.MOTION_SPEED_THRESH
        motion_seconds = float(moving.sum()) * tel_period
        if len(moving) > 1:
            transitions = np.diff(moving.astype(np.int8))
            n_stops = int((transitions == 1).sum())
        else:
            n_stops = 0

        eligible = (
            pd.notna(intersect_s) and intersect_s >= C.MIN_MOTION_FOR_ESTIMATION_S
            and motion_seconds >= C.MIN_MOTION_FOR_ESTIMATION_S
            and len(ll) > 0
        )
        reasons = []
        if not (pd.notna(intersect_s) and intersect_s >= C.MIN_MOTION_FOR_ESTIMATION_S):
            reasons.append(f"intersection<{C.MIN_MOTION_FOR_ESTIMATION_S:.0f}s")
        if motion_seconds < C.MIN_MOTION_FOR_ESTIMATION_S:
            reasons.append(f"motion<{C.MIN_MOTION_FOR_ESTIMATION_S:.0f}s")
        if len(ll) == 0:
            reasons.append("no_lidar_in_window")

        meta = DayMeta(
            session_date=str(sd),
            n_csv_files=int(g["run_file"].nunique()),
            csv_files=sorted(g["run_file"].unique().tolist()),
            tel_start=tel_start, tel_end=tel_end, tel_n=len(g),
            tel_period_s=tel_period, tel_rate_hz=tel_rate,
            tel_gap_seconds_sum=gap_sum, tel_gap_count=gap_count,
            lid_start=lid_start, lid_end=lid_end, lid_n=int(len(ll)),
            lid_period_s=lid_period, lid_rate_hz=lid_rate,
            intersect_start=intersect_start, intersect_end=intersect_end,
            intersect_seconds=intersect_s,
            motion_seconds=motion_seconds,
            n_stop_start_events=n_stops,
            eligible_for_estimation=bool(eligible),
            eligibility_reason="ok" if eligible else ",".join(reasons),
        )
        days.append(meta)
        if verbose:
            print(f"  {sd}: {meta.n_csv_files} CSV(s)  tel={meta.tel_n:,} ({tel_rate:.1f} Hz)  "
                  f"gaps>{C.GAP_MASK_S:.0f}s: {gap_count} ({gap_sum:.1f}s total)  "
                  f"lidar={meta.lid_n:,} ({lid_rate:.1f} Hz)  "
                  f"intersect={meta.intersect_seconds:.1f}s  motion={meta.motion_seconds:.1f}s  "
                  f"-> {meta.eligibility_reason}")

    return days, tel, lid


# ---------------------------------------------------------------------------
# Phase 2a — LiDAR dissimilarity per day (reuse obsolete cache when present)
# ---------------------------------------------------------------------------

def _scan_dissim_pair(prev: np.ndarray, curr: np.ndarray) -> float:
    a = prev.astype(np.float32); b = curr.astype(np.float32)
    inv = float(C.LIDAR_INVALID_MARKER)
    valid = ((a != inv) & (b != inv) & (a > 0) & (b > 0)
             & (a < C.LIDAR_SENSOR_MAX_MM) & (b < C.LIDAR_SENSOR_MAX_MM))
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(b[valid] - a[valid])))


def build_lidar_dissim_cache(verbose: bool = True) -> dict[str, Path]:
    """Build per-day LiDAR dissimilarity parquet, reusing obsolete cache if available."""
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}

    with h5py.File(C.LIDAR_H5, "r") as f:
        sd_all = pd.Series(f["session_date"][:]).astype(str).str.replace(
            r"^b'|'$", "", regex=True).values

    for sd in pd.unique(sd_all):
        out_path = C.CACHE_DIR / f"lidar_sig_{sd}.parquet"
        if out_path.exists():
            if verbose:
                print(f"  [skip] {out_path.name}")
            out_paths[sd] = out_path
            continue
        # Try obsolete cache first
        legacy = C.LIDAR_DISSIM_CACHE_LEGACY / f"lidar_sig_{sd}.parquet"
        if legacy.exists():
            if verbose:
                print(f"  [reuse] copying obsolete cache: {legacy.name}")
            df = pd.read_parquet(legacy)
            df.to_parquet(out_path, compression="zstd", index=False)
            out_paths[sd] = out_path
            continue

        idx = np.where(sd_all == sd)[0]
        if len(idx) < 2:
            if verbose:
                print(f"  [skip] {sd}: <2 scans")
            continue
        start, stop = int(idx[0]), int(idx[-1] + 1)
        n_scans = stop - start
        if verbose:
            print(f"  [build] session={sd} rows {start}..{stop} ({n_scans:,} scans)")

        d_arr = np.full(n_scans - 1, np.nan, dtype=np.float32)
        ts_mid = np.empty(n_scans - 1, dtype=np.float64)
        with h5py.File(C.LIDAR_H5, "r") as f:
            distances = f["distances"]
            local_ts = f["local_timestamps"]
            chunk = C.LIDAR_LOAD_CHUNK
            prev_scan = None; prev_ts = None
            written = 0
            for s0 in range(start, stop, chunk):
                s1 = min(s0 + chunk, stop)
                block = distances[s0:s1, :]
                tblock = local_ts[s0:s1]
                if prev_scan is not None:
                    d_arr[written] = _scan_dissim_pair(prev_scan, block[0])
                    ts_mid[written] = 0.5 * (prev_ts + tblock[0])
                    written += 1
                k = block.shape[0]
                if k >= 2:
                    a = block[:-1].astype(np.float32)
                    b = block[1:].astype(np.float32)
                    inv = float(C.LIDAR_INVALID_MARKER)
                    valid = ((a != inv) & (b != inv) & (a > 0) & (b > 0)
                             & (a < C.LIDAR_SENSOR_MAX_MM) & (b < C.LIDAR_SENSOR_MAX_MM))
                    diff = np.abs(b - a)
                    diff[~valid] = np.nan
                    cnt = valid.sum(axis=1).astype(np.float32)
                    sm = np.nansum(diff, axis=1)
                    pair_d = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
                    d_arr[written:written + k - 1] = pair_d.astype(np.float32)
                    ts_mid[written:written + k - 1] = 0.5 * (tblock[:-1] + tblock[1:])
                    written += (k - 1)
                prev_scan = block[-1].copy()
                prev_ts = float(tblock[-1])
        out = pd.DataFrame({"ts_unix_cet": ts_mid, "d": d_arr})
        out["ts"] = pd.to_datetime(out["ts_unix_cet"], unit="s")
        out["valid"] = np.isfinite(out["d"])
        out.to_parquet(out_path, compression="zstd", index=False)
        out_paths[sd] = out_path
        if verbose:
            print(f"    -> {out_path.name} ({len(out):,} rows, valid={out['valid'].mean():.3f})")
    return out_paths


# ---------------------------------------------------------------------------
# Phase 2b — per-day signature on a 50 Hz grid, gap-masked
# ---------------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.nanmean(x)); sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(t_grid, np.nan, dtype=np.float64)
    t0 = t[ok]; y0 = y[ok]
    order = np.argsort(t0); t0 = t0[order]; y0 = y0[order]
    if len(t0) > 1:
        same = np.r_[False, np.diff(t0) == 0]
        if same.any():
            t0u, idx = np.unique(t0, return_inverse=True)
            sums = np.zeros_like(t0u, dtype=np.float64)
            cnts = np.zeros_like(t0u, dtype=np.float64)
            np.add.at(sums, idx, y0)
            np.add.at(cnts, idx, 1.0)
            y0 = sums / np.maximum(cnts, 1)
            t0 = t0u
    return np.interp(t_grid, t0, y0, left=np.nan, right=np.nan)


def _telemetry_gap_mask(t_samples: np.ndarray, t_grid: np.ndarray,
                        gap_thresh_s: float) -> np.ndarray:
    """Boolean mask over t_grid: True where the grid sample sits inside a
    telemetry gap > gap_thresh_s (i.e. the nearest two telemetry samples on
    either side are separated by > gap_thresh_s and the grid sample falls
    between them).

    The mask is used to zero out the resampled signal in regions where
    linear interpolation across a real recording gap would inject artifacts.
    """
    t_samples = np.asarray(t_samples, dtype=np.float64)
    if len(t_samples) < 2:
        return np.zeros_like(t_grid, dtype=bool)
    # Find the index of the previous telemetry sample for each grid point.
    # Grid points < first sample or >= last sample are NOT inside a gap
    # between two samples; we mark them as "outside" (already excluded by
    # the grid construction below).
    idx_prev = np.searchsorted(t_samples, t_grid, side="right") - 1
    in_range = (idx_prev >= 0) & (idx_prev < len(t_samples) - 1)
    mask = np.zeros_like(t_grid, dtype=bool)
    if in_range.any():
        i = idx_prev[in_range]
        gap = t_samples[i + 1] - t_samples[i]
        mask[in_range] = gap > gap_thresh_s
    return mask


def _build_rot_aware(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the rot_aware = |speed_mps| + R * |omega| signal for one
    sorted telemetry slice. Returns (t_unix, y).
    """
    s_v = g["speed_mps"].astype(float).abs().values
    dt = g["fh7000_timestamp"].diff().dt.total_seconds().values
    dh = g["heading_rad"].astype(float).diff().values
    dh = ((dh + np.pi) % (2 * np.pi)) - np.pi
    valid = (dt > 0.05) & (dt < 1.0)
    omega = np.where(valid, np.abs(dh / np.where(dt > 0, dt, np.nan)), 0.0)
    omega = np.nan_to_num(omega, nan=0.0)
    if (omega > 0).any():
        cap = float(np.nanpercentile(omega[omega > 0], 99))
        omega = np.minimum(omega, cap)
    y = s_v + C.ROT_R_EFF * omega
    t = (g["fh7000_timestamp"].values.astype("datetime64[ns]")
         .astype("int64") / 1e9)
    return t, y


def make_day_signature(meta: DayMeta, telemetry: pd.DataFrame,
                       lidar_sig: pd.DataFrame) -> dict:
    """Build the per-day resampled, z-scored telemetry/LiDAR signature pair.

    Concatenates all telemetry of the day, sorts by timestamp, builds the
    rot_aware signal, resamples both telemetry and LiDAR onto a 50 Hz grid
    over the intersection window, masks telemetry gaps > GAP_MASK_S, and
    z-scores the result.
    """
    g = telemetry[telemetry["session_date"] == meta.session_date].copy()
    g = g.sort_values("fh7000_timestamp").reset_index(drop=True)

    t_t, s_t_raw = _build_rot_aware(g)

    margin = 0.5
    ll = lidar_sig[(lidar_sig["ts_unix_cet"] >= meta.tel_start.timestamp() - margin)
                   & (lidar_sig["ts_unix_cet"] <= meta.tel_end.timestamp() + margin)]
    t_l = ll["ts_unix_cet"].values
    s_l_raw = ll["d"].values

    t_lo = max(meta.tel_start.timestamp(), float(t_l.min()) if t_l.size
               else meta.tel_start.timestamp())
    t_hi = min(meta.tel_end.timestamp(), float(t_l.max()) if t_l.size
               else meta.tel_end.timestamp())
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi - t_lo < 5.0:
        return {}
    n_grid = int(math.floor((t_hi - t_lo) * C.GRID_HZ))
    t_grid = t_lo + np.arange(n_grid) * C.GRID_DT

    s_t = _resample_to_grid(t_t, s_t_raw, t_grid)
    s_l = _resample_to_grid(t_l, s_l_raw, t_grid)

    # Mask telemetry gaps (linear interp across a recording pause is fake).
    gap_mask = _telemetry_gap_mask(t_t, t_grid, C.GAP_MASK_S)

    ok = np.isfinite(s_t) & np.isfinite(s_l)
    if ok.sum() < int(C.GRID_HZ * C.MIN_MOTION_FOR_ESTIMATION_S):
        return {}

    # Linear-fill small NaN gaps (rare); leave masked-by-gap samples to be
    # zeroed below.
    def fill_in(a):
        a = a.copy(); x = np.arange(len(a)); m = np.isfinite(a)
        if m.sum() < 2:
            return None
        a[~m] = np.interp(x[~m], x[m], a[m])
        return a
    s_t_f = fill_in(s_t); s_l_f = fill_in(s_l)
    if s_t_f is None or s_l_f is None:
        return {}

    z_t = _zscore(s_t_f)
    z_l = _zscore(s_l_f)

    # Apply the telemetry-gap mask: zero the affected grid points so they
    # contribute nothing to the FFT cross-correlation. We mask BOTH signals
    # consistently so the cross-correlation numerator has matching support.
    if gap_mask.any():
        z_t = np.where(gap_mask, 0.0, z_t)
        z_l = np.where(gap_mask, 0.0, z_l)

    return {
        "t_grid": t_grid,
        "s_t_raw": s_t_f,
        "s_l_raw": s_l_f,
        "z_t": z_t,
        "z_l": z_l,
        "gap_mask": gap_mask,
        "n_grid_masked": int(gap_mask.sum()),
        "n_grid_total": int(len(t_grid)),
    }


# ---------------------------------------------------------------------------
# Per-CSV-file enumeration + signature (cross-check, NOT applied)
# ---------------------------------------------------------------------------

@dataclass
class CsvMeta:
    session_date: str
    run_file: str

    tel_start: pd.Timestamp
    tel_end: pd.Timestamp
    tel_n: int
    tel_period_s: float
    tel_rate_hz: float

    lid_start: pd.Timestamp
    lid_end: pd.Timestamp
    lid_n: int
    lid_period_s: float
    lid_rate_hz: float

    intersect_start: pd.Timestamp
    intersect_end: pd.Timestamp
    intersect_seconds: float

    motion_seconds: float
    n_stop_start_events: int

    eligible_for_estimation: bool
    eligibility_reason: str = ""

    def to_jsonable(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
        return d


def enumerate_csvs(verbose: bool = True) -> tuple[list[CsvMeta], pd.DataFrame, pd.DataFrame]:
    """Per-(session_date, run_file) enumeration. Used only for the per-CSV
    cross-check appendix; not on the per-day estimation path.
    """
    tel = pd.read_parquet(C.TELEMETRY_PARQUET)
    tel["fh7000_timestamp"] = tel["fh7000_timestamp"].astype("datetime64[ns]")
    tel = tel.sort_values(["session_date", "run_file",
                           "fh7000_timestamp"]).reset_index(drop=True)
    lid = _load_lidar_index()

    csvs: list[CsvMeta] = []
    for (sd, rf), g in tel.groupby(["session_date", "run_file"], sort=True):
        g = g.sort_values("fh7000_timestamp").reset_index(drop=True)
        ts = g["fh7000_timestamp"]
        tel_start = ts.iloc[0]; tel_end = ts.iloc[-1]
        dt_tel = ts.diff().dt.total_seconds().dropna()
        dt_clean = dt_tel[(dt_tel > 0) & (dt_tel < 1.0)]
        tel_period = float(dt_clean.median()) if len(dt_clean) else float(dt_tel.median())
        tel_rate = 1.0 / tel_period if tel_period > 0 else float("nan")

        lpart = lid[lid["session_date"] == sd]
        margin = pd.Timedelta(seconds=2)
        ll = lpart[(lpart["ts"] >= tel_start - margin)
                   & (lpart["ts"] <= tel_end + margin)]
        if len(ll) >= 2:
            ldt = np.diff(ll["ts_unix_cet"].values)
            ldt = ldt[(ldt > 0) & (ldt < 1.0)]
            lid_period = float(np.median(ldt)) if ldt.size else float("nan")
            lid_rate = 1.0 / lid_period if lid_period and lid_period > 0 else float("nan")
            lid_start = ll["ts"].iloc[0]; lid_end = ll["ts"].iloc[-1]
        else:
            lid_period = float("nan"); lid_rate = float("nan")
            lid_start = pd.NaT; lid_end = pd.NaT

        intersect_start = max(tel_start, lid_start) if pd.notna(lid_start) else pd.NaT
        intersect_end = min(tel_end, lid_end) if pd.notna(lid_end) else pd.NaT
        intersect_s = (intersect_end - intersect_start).total_seconds() \
            if pd.notna(intersect_start) and pd.notna(intersect_end) else float("nan")

        speed = g["speed_mps"].astype(float).abs().fillna(0.0).values
        moving = speed > C.MOTION_SPEED_THRESH
        motion_seconds = float(moving.sum()) * tel_period
        if len(moving) > 1:
            transitions = np.diff(moving.astype(np.int8))
            n_stops = int((transitions == 1).sum())
        else:
            n_stops = 0

        eligible = (
            pd.notna(intersect_s) and intersect_s >= C.MIN_MOTION_FOR_ESTIMATION_S
            and motion_seconds >= C.MIN_MOTION_FOR_ESTIMATION_S
            and len(ll) > 0
        )
        reasons = []
        if not (pd.notna(intersect_s) and intersect_s >= C.MIN_MOTION_FOR_ESTIMATION_S):
            reasons.append(f"intersection<{C.MIN_MOTION_FOR_ESTIMATION_S:.0f}s")
        if motion_seconds < C.MIN_MOTION_FOR_ESTIMATION_S:
            reasons.append(f"motion<{C.MIN_MOTION_FOR_ESTIMATION_S:.0f}s")
        if len(ll) == 0:
            reasons.append("no_lidar_in_window")

        csvs.append(CsvMeta(
            session_date=str(sd), run_file=str(rf),
            tel_start=tel_start, tel_end=tel_end, tel_n=len(g),
            tel_period_s=tel_period, tel_rate_hz=tel_rate,
            lid_start=lid_start, lid_end=lid_end, lid_n=int(len(ll)),
            lid_period_s=lid_period, lid_rate_hz=lid_rate,
            intersect_start=intersect_start, intersect_end=intersect_end,
            intersect_seconds=intersect_s,
            motion_seconds=motion_seconds, n_stop_start_events=n_stops,
            eligible_for_estimation=bool(eligible),
            eligibility_reason="ok" if eligible else ",".join(reasons),
        ))
        if verbose:
            print(f"  {sd}/{rf}: tel={len(g):,} ({tel_rate:.1f} Hz)  "
                  f"lidar={len(ll):,} ({lid_rate:.2f} Hz)  "
                  f"intersect={intersect_s:.1f}s  motion={motion_seconds:.1f}s")
    return csvs, tel, lid


def make_csv_signature(meta: CsvMeta, telemetry: pd.DataFrame,
                       lidar_sig: pd.DataFrame) -> dict:
    """Build the per-CSV resampled, z-scored telemetry/LiDAR signature pair.

    Same canonical methodology as ``make_day_signature`` (rot_aware), but
    over only the rows belonging to one CSV file. Used only for the
    cross-check appendix.
    """
    g = telemetry[(telemetry["session_date"] == meta.session_date)
                  & (telemetry["run_file"] == meta.run_file)].copy()
    g = g.sort_values("fh7000_timestamp").reset_index(drop=True)

    t_t, s_t_raw = _build_rot_aware(g)

    margin = 0.5
    ll = lidar_sig[(lidar_sig["ts_unix_cet"] >= meta.tel_start.timestamp() - margin)
                   & (lidar_sig["ts_unix_cet"] <= meta.tel_end.timestamp() + margin)]
    t_l = ll["ts_unix_cet"].values
    s_l_raw = ll["d"].values

    t_lo = max(meta.tel_start.timestamp(), float(t_l.min()) if t_l.size
               else meta.tel_start.timestamp())
    t_hi = min(meta.tel_end.timestamp(), float(t_l.max()) if t_l.size
               else meta.tel_end.timestamp())
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi - t_lo < 5.0:
        return {}
    n_grid = int(math.floor((t_hi - t_lo) * C.GRID_HZ))
    t_grid = t_lo + np.arange(n_grid) * C.GRID_DT

    s_t = _resample_to_grid(t_t, s_t_raw, t_grid)
    s_l = _resample_to_grid(t_l, s_l_raw, t_grid)

    ok = np.isfinite(s_t) & np.isfinite(s_l)
    if ok.sum() < int(C.GRID_HZ * C.MIN_MOTION_FOR_ESTIMATION_S):
        return {}

    def fill_in(a):
        a = a.copy(); x = np.arange(len(a)); m = np.isfinite(a)
        if m.sum() < 2:
            return None
        a[~m] = np.interp(x[~m], x[m], a[m])
        return a
    s_t_f = fill_in(s_t); s_l_f = fill_in(s_l)
    if s_t_f is None or s_l_f is None:
        return {}

    z_t = _zscore(s_t_f); z_l = _zscore(s_l_f)
    return {
        "t_grid": t_grid,
        "s_t_raw": s_t_f, "s_l_raw": s_l_f,
        "z_t": z_t, "z_l": z_l,
        "n_grid_total": int(len(t_grid)),
    }


# ---------------------------------------------------------------------------
# Phase 3 — cross-correlation, peak refinement, prominence, bootstrap
# ---------------------------------------------------------------------------

def xcorr_lags(z_t: np.ndarray, z_l: np.ndarray,
               max_lag_s: float = C.SEARCH_RANGE_S) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import correlate
    n = min(len(z_t), len(z_l))
    z_t = z_t[:n].astype(np.float64, copy=False)
    z_l = z_l[:n].astype(np.float64, copy=False)
    c = correlate(z_t, z_l, mode="full", method="fft")
    rho_full = c / float(n)
    s_t_std = float(np.std(z_t)); s_l_std = float(np.std(z_l))
    denom = s_t_std * s_l_std if s_t_std and s_l_std else 1.0
    rho_full /= denom
    K = int(round(max_lag_s / C.GRID_DT))
    K = min(K, n - 1)
    center = n - 1
    rho = rho_full[center - K: center + K + 1]
    lags = np.arange(-K, K + 1) * C.GRID_DT
    return lags, rho


def parabolic_refine(lags: np.ndarray, rho: np.ndarray) -> tuple[float, int, float]:
    k = int(np.nanargmax(rho))
    if 0 < k < len(rho) - 1:
        y_m1, y_0, y_p1 = rho[k - 1], rho[k], rho[k + 1]
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        tau_hat = float(lags[k] + delta * C.GRID_DT)
        y_peak = float(y_0 - 0.25 * (y_m1 - y_p1) * delta)
    else:
        tau_hat = float(lags[k]); y_peak = float(rho[k])
    return tau_hat, k, y_peak


def peak_prominence(rho: np.ndarray, k_peak: int,
                    exclusion_lag_s: float = C.SECOND_PEAK_MIN_LAG_S) -> float:
    if rho[k_peak] <= 0:
        return 0.0
    n_excl = int(round(exclusion_lag_s / C.GRID_DT))
    lo = max(0, k_peak - n_excl); hi = min(len(rho), k_peak + n_excl + 1)
    mask = np.ones_like(rho, dtype=bool); mask[lo:hi] = False
    other_max = float("-inf")
    for i in range(1, len(rho) - 1):
        if not mask[i]:
            continue
        if rho[i] >= rho[i - 1] and rho[i] >= rho[i + 1]:
            if rho[i] > other_max:
                other_max = rho[i]
    if not np.isfinite(other_max):
        return 1.0
    other_max = max(0.0, other_max)
    return float((rho[k_peak] - other_max) / rho[k_peak])


def moving_block_bootstrap(z_t: np.ndarray, z_l: np.ndarray,
                           B: int = C.BOOTSTRAP_B,
                           block_len_s: float = C.BLOCK_LEN_S,
                           max_lag_s: float = C.SEARCH_RANGE_S,
                           rng: np.random.Generator | None = None,
                           tau_point: float = 0.0,
                           modal_window_s: float = 2.0) -> dict:
    if rng is None:
        rng = np.random.default_rng(C.RNG_SEED)
    n = min(len(z_t), len(z_l))
    z_t = z_t[:n]; z_l = z_l[:n]
    L = max(2, int(round(block_len_s * C.GRID_HZ)))
    n_blocks = int(math.ceil(n / L))
    taus = np.empty(B, dtype=np.float64)
    max_start = n - L
    if max_start <= 0:
        nan = float("nan")
        return {"taus": np.full(B, np.nan), "sigma": nan,
                "ci_low": nan, "ci_high": nan,
                "modal_sigma": nan, "modal_ci_low": nan,
                "modal_ci_high": nan, "modal_fraction": nan,
                "block_len_s": float(block_len_s), "sigma_rmse": nan}
    for b in range(B):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        zt_b = z_t[idx]; zl_b = z_l[idx]
        if zt_b.std() > 0: zt_b = (zt_b - zt_b.mean()) / zt_b.std()
        if zl_b.std() > 0: zl_b = (zl_b - zl_b.mean()) / zl_b.std()
        lags, rho = xcorr_lags(zt_b, zl_b, max_lag_s=max_lag_s)
        tau_b, _, _ = parabolic_refine(lags, rho)
        taus[b] = tau_b
    sigma = float(np.nanstd(taus, ddof=1))
    lo = float(np.nanpercentile(taus, 2.5))
    hi = float(np.nanpercentile(taus, 97.5))
    valid = np.isfinite(taus)
    sigma_rmse = float(np.sqrt(np.nanmean((taus[valid] - tau_point) ** 2))) \
        if valid.any() else float("nan")

    in_mode = np.abs(taus - tau_point) <= modal_window_s
    if in_mode.sum() >= 2:
        modal_sigma = float(np.nanstd(taus[in_mode], ddof=1))
        modal_lo = float(np.nanpercentile(taus[in_mode], 2.5))
        modal_hi = float(np.nanpercentile(taus[in_mode], 97.5))
    else:
        modal_sigma = float("nan")
        modal_lo = modal_hi = float("nan")
    modal_fraction = float(in_mode.mean())
    return {"taus": taus, "sigma": sigma, "ci_low": lo, "ci_high": hi,
            "sigma_rmse": sigma_rmse,
            "modal_sigma": modal_sigma,
            "modal_ci_low": modal_lo, "modal_ci_high": modal_hi,
            "modal_fraction": modal_fraction,
            "block_len_s": float(block_len_s)}


# ---------------------------------------------------------------------------
# Phase 4 — homogeneity / drift helpers
# ---------------------------------------------------------------------------

def cochran_q(taus: np.ndarray, sigmas: np.ndarray) -> dict:
    from scipy.stats import chi2
    taus = np.asarray(taus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(taus) & np.isfinite(sigmas) & (sigmas > 0)
    n = int(ok.sum())
    if n < 2:
        return {"n": n, "tau_bar": float("nan"), "se_tau_bar": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "Q": float("nan"), "df": 0, "p": float("nan"), "I2": float("nan")}
    w = 1.0 / sigmas[ok] ** 2
    tau_bar = float(np.sum(w * taus[ok]) / np.sum(w))
    se = float(1.0 / math.sqrt(np.sum(w)))
    ci_lo, ci_hi = tau_bar - 1.96 * se, tau_bar + 1.96 * se
    Q = float(np.sum(w * (taus[ok] - tau_bar) ** 2))
    df = n - 1
    p = float(1.0 - chi2.cdf(Q, df))
    I2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
    return {"n": n, "tau_bar": tau_bar, "se_tau_bar": se,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "Q": Q, "df": df, "p": p, "I2": I2}


def cis_overlap(t1: float, s1: float, t2: float, s2: float) -> bool:
    lo1, hi1 = t1 - 1.96 * s1, t1 + 1.96 * s1
    lo2, hi2 = t2 - 1.96 * s2, t2 + 1.96 * s2
    return not (hi1 < lo2 or hi2 < lo1)
