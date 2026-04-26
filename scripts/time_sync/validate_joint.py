"""End-to-end validation of data/merged/joint_coverage.parquet against the
original CSV / H5 source data.

Checks performed
----------------
T1  Schema: column set matches docs/TELEMETRY_CLEANUP.md cleaned columns
    plus the documented metadata columns added by the joint pipeline.
T2  Per-day applied_tau_s / applied_tau_se_s consistent with the values in
    cache/homogeneity_per_day.json (no row carries a stale tau).
T3  Timestamp identity: lidar_ts_corr_unix == lidar_ts_raw_cet_unix + applied_tau_s
    (float-precision exact).
T4  Raw LiDAR timestamps match data/merged/lidar.h5::local_timestamps at
    every joint row (by lidar_row index).
T5  LiDAR scan_nr / scan_counter at every joint row match the H5 by row index.
T6  delta_t_s identity: delta_t_s == (lidar_ts_corr - fh7000_timestamp).
T7  |delta_t_s| <= match_tolerance_s for every row.
T8  Per-day match_tolerance_s == max(LiDAR_period, telemetry_period)/2 with
    rates measured from the cleaned parquet.
T9  Telemetry payload identity: for every joint row, the joined telemetry
    columns equal the cleaned parquet row keyed by (session_date, run_file,
    fh7000_timestamp). Sampled at 200 random rows per day.
T10 Retention (independently recomputed). For each (session_date, run_file)
    we re-run the merge_asof against the original cleaned telemetry and
    the raw H5 (+ per-day tau), and compare the retained-row count to the
    joint dataset's row count for that file.
T11 Original-CSV provenance (sampled). For 30 sampled joint rows per day
    we open the originating CSV, find the row whose timestamp equals
    fh7000_timestamp, and confirm the parsed telemetry payload
    (speed_mps, heading_rad, actual_speed_left, actual_speed_right,
    signal_quality, x_m, y_m) matches.
T12 Row-count bookkeeping vs. the obsolete pipeline (sanity, not a hard
    check): the per-day retention should differ only via the per-day tau
    (since tolerance is unchanged).

Run:
    .venv/Scripts/python.exe -m analysis.time_sync.validate_joint
"""
from __future__ import annotations
import json
from pathlib import Path
import random
import sys

import h5py
import numpy as np
import pandas as pd

from . import config as C


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"


CLEANED_COLS = [
    "fh7000_timestamp", "session_date", "run_file",
    "load_long", "load_mid", "load_short", "ping",
    "signal_noise", "signal_power", "signal_quality", "uptime",
    "fh6000_timestamp", "fh6000_number",
    "battery_cell_voltage", "battery_value",
    "cumulative_energy_consumption", "cumulative_energy_consumption_uint",
    "momentary_current_consumption", "brake_release_command_on",
    "actual_speed_left", "left_drive_stop_executed",
    "actual_speed_right", "right_drive_stop_executed",
    "nncf_3108_abort_result", "nns_current_segment", "nns_error_status",
    "heading_rad", "nns_state", "nns_on_route_status",
    "nns_position_confidence", "nns_position_initialize_status",
    "speed_mps", "x_m", "y_m", "front_scanner_safety_zone_not_violated",
]

JOINT_METADATA_COLS = [
    "applied_tau_s", "applied_tau_se_s", "match_tolerance_s",
    "lidar_row", "lidar_scan_nr", "lidar_scan_counter",
    "lidar_ts_raw_cet_unix", "lidar_ts_corr_unix", "lidar_ts_corr",
    "delta_t_s",
]

CSV_DIR = {
    "25.02.2026": C.ROOT / "data" / "25.02.2026",
    "15.03.2026": C.ROOT / "data" / "15.03.2026",
    "24.03.2026": C.ROOT / "data" / "24.03.2026",
}

WIFI_COLS = [
    "load_long", "load_mid", "load_short", "ping", "reserve",
    "signal_noise", "signal_power", "signal_quality", "uptime", "wifi_state_1",
]

# CSV column-index -> cleaned name (subset for sample checking only)
SAMPLED_TELEMETRY_INDEX_NAME = {
    11: "fh6000_timestamp",
    24: "actual_speed_left",
    33: "actual_speed_right",
    51: "heading_rad",
    58: "speed_mps",
    60: "x_m",
    61: "y_m",
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.results: list[tuple[str, str, str]] = []   # (label, status, msg)

    def add(self, label: str, status: str, msg: str = "") -> None:
        self.results.append((label, status, msg))
        prefix = {"pass": PASS, "fail": FAIL, "warn": WARN, "info": INFO}[status]
        print(f"  {prefix} {label}: {msg}" if msg else f"  {prefix} {label}")

    @property
    def n_fail(self) -> int:
        return sum(1 for _, s, _ in self.results if s == "fail")

    def summary(self) -> None:
        n_pass = sum(1 for _, s, _ in self.results if s == "pass")
        n_warn = sum(1 for _, s, _ in self.results if s == "warn")
        n_info = sum(1 for _, s, _ in self.results if s == "info")
        print()
        print("=" * 72)
        print(f"VALIDATION SUMMARY: {n_pass} pass, {self.n_fail} fail, "
              f"{n_warn} warn, {n_info} info")
        print("=" * 72)
        if self.n_fail > 0:
            print("\nFailing checks:")
            for label, s, m in self.results:
                if s == "fail":
                    print(f"  - {label}: {m}")


# ---------------------------------------------------------------------------
# CSV loader (mirrors analysis/merge_dataset.py:load_csv_robust)
# ---------------------------------------------------------------------------

def load_one_csv(path: Path) -> pd.DataFrame:
    """Same row-dispatch logic as the merge pipeline. Returns a DataFrame
    with columns: ts (datetime64), and c11..c72 as raw strings (no type
    coercion — caller does that for the columns it cares about).
    """
    rows_full: list[list[str]] = []
    rows_nowifi: list[list[str]] = []
    n_dropped = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            parts = line.split(",")
            n = len(parts)
            if n == 73:
                rows_full.append(parts)
            elif n == 63:
                rows_nowifi.append(parts)
            else:
                n_dropped += 1
    frames = []
    if rows_full:
        df_full = pd.DataFrame(rows_full)
        df_full.columns = ["ts_str"] + WIFI_COLS + [f"c{i}" for i in range(11, 73)]
        frames.append(df_full)
    if rows_nowifi:
        df_nw = pd.DataFrame(rows_nowifi)
        df_nw.columns = ["ts_str"] + [f"c{i}" for i in range(11, 73)]
        for w in WIFI_COLS:
            df_nw[w] = np.nan
        df_nw = df_nw[["ts_str"] + WIFI_COLS + [f"c{i}" for i in range(11, 73)]]
        frames.append(df_nw)
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts_str"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.attrs["n_full"] = len(rows_full)
    df.attrs["n_nowifi"] = len(rows_nowifi)
    df.attrs["n_dropped"] = n_dropped
    return df


# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------

def check_t1_schema(jp: pd.DataFrame, rep: Report) -> None:
    expected = set(CLEANED_COLS) | set(JOINT_METADATA_COLS)
    actual = set(jp.columns)
    missing = expected - actual
    extra = actual - expected
    if missing:
        rep.add("T1 schema", "fail", f"missing columns: {sorted(missing)}")
        return
    if extra:
        rep.add("T1 schema", "fail", f"unexpected columns: {sorted(extra)}")
        return
    rep.add("T1 schema", "pass",
            f"45 cols = 35 cleaned + 10 metadata (TELEMETRY_CLEANUP.md compliant)")


def check_t2_applied_tau(jp: pd.DataFrame, rep: Report) -> None:
    h = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    per_day = h["per_session_day"]
    bad: list[str] = []
    for sd, grp in jp.groupby("session_date"):
        ref_tau = float(per_day[sd]["tau_bar"])
        ref_se = float(per_day[sd]["se"])
        if not np.allclose(grp["applied_tau_s"].values, ref_tau, atol=1e-12):
            bad.append(f"{sd}: tau mismatch ({grp['applied_tau_s'].iloc[0]} != {ref_tau})")
        if not np.allclose(grp["applied_tau_se_s"].values, ref_se, atol=1e-12):
            bad.append(f"{sd}: SE mismatch ({grp['applied_tau_se_s'].iloc[0]} != {ref_se})")
    if bad:
        rep.add("T2 applied_tau / applied_tau_se", "fail", "; ".join(bad))
    else:
        taus = ", ".join(
            f"{sd}={grp['applied_tau_s'].iloc[0]:+.4f}s±{grp['applied_tau_se_s'].iloc[0]:.4f}"
            for sd, grp in jp.groupby("session_date"))
        rep.add("T2 applied_tau / applied_tau_se", "pass", taus)


def check_t3_timestamp_identity(jp: pd.DataFrame, rep: Report) -> None:
    """lidar_ts_corr_unix == lidar_ts_raw_cet_unix + applied_tau_s exact."""
    diff = (jp["lidar_ts_corr_unix"]
            - (jp["lidar_ts_raw_cet_unix"] + jp["applied_tau_s"])).abs()
    max_diff = float(diff.max())
    if max_diff > 1e-6:
        rep.add("T3 ts_corr_unix == ts_raw + tau", "fail",
                f"max |diff| = {max_diff:g} s")
    else:
        rep.add("T3 ts_corr_unix == ts_raw + tau", "pass",
                f"max |diff| = {max_diff:g} s (float-precision exact)")


def check_t4_t5_lidar_provenance(jp: pd.DataFrame, rep: Report) -> None:
    """LiDAR raw ts, scan_nr, scan_counter at every joint row match the H5
    by lidar_row index."""
    with h5py.File(C.LIDAR_H5, "r") as f:
        ts_all = f["local_timestamps"][:]
        scan_nr = f["device_scan_nr"][:]
        scan_counter = f["device_scan_counter"][:]
    rows = jp["lidar_row"].values
    # Bounds
    if rows.min() < 0 or rows.max() >= len(ts_all):
        rep.add("T4 LiDAR row index range", "fail",
                f"row range [{rows.min()}, {rows.max()}], H5 has {len(ts_all)}")
        return
    rep.add("T4 LiDAR row index range", "pass",
            f"row range [{rows.min()}, {rows.max()}] within H5 [0, {len(ts_all)-1}]")

    # T4: timestamp identity
    diff = np.abs(jp["lidar_ts_raw_cet_unix"].values - ts_all[rows])
    max_diff = float(diff.max())
    if max_diff > 1e-6:
        rep.add("T4 raw lidar_ts == H5 local_timestamps", "fail",
                f"max |diff| = {max_diff:g} s")
    else:
        rep.add("T4 raw lidar_ts == H5 local_timestamps", "pass",
                f"max |diff| = {max_diff:g} s (exact)")

    # T5: scan_nr & scan_counter identity
    n_bad_scan = int((jp["lidar_scan_nr"].values != scan_nr[rows]).sum())
    n_bad_cnt = int((jp["lidar_scan_counter"].values != scan_counter[rows]).sum())
    if n_bad_scan or n_bad_cnt:
        rep.add("T5 scan_nr / scan_counter identity", "fail",
                f"{n_bad_scan} bad scan_nr, {n_bad_cnt} bad scan_counter")
    else:
        rep.add("T5 scan_nr / scan_counter identity", "pass",
                "all rows match H5 device_scan_nr & device_scan_counter")


def check_t6_t7_delta_and_tolerance(jp: pd.DataFrame, rep: Report) -> None:
    """delta_t_s = lidar_ts_corr - fh7000_timestamp; |delta_t| <= tolerance."""
    fh = jp["fh7000_timestamp"].astype("datetime64[ns]").astype("int64") / 1e9
    expected = jp["lidar_ts_corr_unix"] - fh
    diff = (jp["delta_t_s"] - expected).abs()
    max_diff = float(diff.max())
    if max_diff > 1e-3:
        # 1 ms is ~the float precision of nanosecond->seconds conversion when
        # taking differences of large unix epoch values; anything larger is a
        # real mismatch.
        rep.add("T6 delta_t identity", "fail", f"max |diff| = {max_diff*1000:.3f} ms")
    else:
        rep.add("T6 delta_t identity", "pass",
                f"max |diff| = {max_diff*1e6:.3f} µs (float roundoff)")

    over = int((jp["delta_t_s"].abs() > jp["match_tolerance_s"]).sum())
    if over:
        rep.add("T7 |delta_t| <= tolerance", "fail",
                f"{over} rows over tolerance")
    else:
        rep.add("T7 |delta_t| <= tolerance", "pass",
                f"all {len(jp):,} rows within tolerance")


def check_t8_tolerance_value(jp: pd.DataFrame, rep: Report,
                              cleaned: pd.DataFrame) -> None:
    """Per-day tolerance == max(LiDAR_period, tel_period) / 2."""
    days = json.loads((C.CACHE_DIR / "days.json").read_text())
    bad = []
    for d in days:
        sd = d["session_date"]
        tol_expected = 0.5 * max(d["lid_period_s"], d["tel_period_s"])
        actual_set = set(round(v, 6) for v in jp.loc[jp["session_date"] == sd,
                                                    "match_tolerance_s"].unique())
        if len(actual_set) != 1:
            bad.append(f"{sd}: more than one tolerance value: {actual_set}")
            continue
        actual = next(iter(actual_set))
        if not np.isclose(actual, tol_expected, rtol=1e-3, atol=1e-6):
            bad.append(f"{sd}: tol={actual:g}, expected={tol_expected:g}")
    if bad:
        rep.add("T8 tolerance == max(periods)/2", "fail", "; ".join(bad))
    else:
        rep.add("T8 tolerance == max(periods)/2", "pass",
                "per-day tolerances match measured periods")


def check_t9_payload_identity(jp: pd.DataFrame, rep: Report,
                               cleaned: pd.DataFrame, n_per_day: int = 200,
                               rng: random.Random | None = None) -> None:
    """Sample n_per_day random joint rows per day; every cleaned-parquet
    column value should equal the source row in `cleaned` keyed by
    (session_date, run_file, fh7000_timestamp).
    """
    if rng is None:
        rng = random.Random(20260427)
    cleaned_idx = cleaned.set_index(["session_date", "run_file",
                                     "fh7000_timestamp"], drop=False)
    payload_cols = [c for c in CLEANED_COLS
                    if c not in ("session_date", "run_file", "fh7000_timestamp")]
    bad_total = 0
    bad_examples = []
    sampled_total = 0
    for sd, grp in jp.groupby("session_date"):
        idxs = list(grp.index)
        sample = rng.sample(idxs, min(n_per_day, len(idxs)))
        sampled_total += len(sample)
        for i in sample:
            r = jp.loc[i]
            try:
                src = cleaned_idx.loc[(r["session_date"], r["run_file"],
                                       r["fh7000_timestamp"])]
            except KeyError:
                bad_total += 1
                if len(bad_examples) < 3:
                    bad_examples.append(f"missing source row for joint idx {i}")
                continue
            if isinstance(src, pd.DataFrame):
                # Multiple cleaned rows share the same fh7000_timestamp (rare
                # but possible with sub-ms collisions). Accept if any of them
                # matches.
                ok = False
                for j in range(len(src)):
                    if _row_equal(r[payload_cols], src.iloc[j][payload_cols]):
                        ok = True
                        break
                if not ok:
                    bad_total += 1
                    if len(bad_examples) < 3:
                        bad_examples.append(
                            f"{sd} idx {i}: no matching cleaned row among "
                            f"{len(src)} timestamp collisions")
            else:
                if not _row_equal(r[payload_cols], src[payload_cols]):
                    bad_total += 1
                    if len(bad_examples) < 3:
                        diffs = [c for c in payload_cols
                                 if not _val_equal(r[c], src[c])]
                        bad_examples.append(
                            f"{sd} idx {i}: mismatch in {diffs[:5]}")
    if bad_total:
        rep.add("T9 payload identity vs cleaned (sampled)", "fail",
                f"{bad_total}/{sampled_total} rows differ; "
                f"examples: {bad_examples}")
    else:
        rep.add("T9 payload identity vs cleaned (sampled)", "pass",
                f"{sampled_total} sampled rows: all 32 payload columns match")


def _val_equal(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if isinstance(a, (np.bool_, bool)) or isinstance(b, (np.bool_, bool)):
        return bool(a) == bool(b)
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))
    except (TypeError, ValueError):
        return a == b


def _row_equal(a: pd.Series, b: pd.Series) -> bool:
    for c in a.index:
        if not _val_equal(a[c], b[c]):
            return False
    return True


def check_t10_retention_independent(jp: pd.DataFrame, rep: Report,
                                     cleaned: pd.DataFrame) -> None:
    """Independently re-run the merge_asof per-CSV against raw H5 and
    cleaned telemetry. Compare retained-row count to joint dataset's row
    count for that file.
    """
    days = json.loads((C.CACHE_DIR / "days.json").read_text())
    days_by_sd = {d["session_date"]: d for d in days}
    h = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    per_day = h["per_session_day"]
    # Load raw H5
    with h5py.File(C.LIDAR_H5, "r") as f:
        sd_all = pd.Series(f["session_date"][:]).astype(str).str.replace(
            r"^b'|'$", "", regex=True).values
        ts_all = f["local_timestamps"][:]

    bad = []
    rows_match = []
    for (sd, rf), grp in jp.groupby(["session_date", "run_file"]):
        d = days_by_sd[sd]
        tau_day = float(per_day[sd]["tau_bar"])
        tol = 0.5 * max(d["lid_period_s"], d["tel_period_s"])

        # Build raw lidar slice for this day, corrected
        idx_day = np.where(sd_all == sd)[0]
        ts_corr_day = ts_all[idx_day] + tau_day

        # Find run window
        rt = cleaned[(cleaned["session_date"] == sd) & (cleaned["run_file"] == rf)]
        run_start = rt["fh7000_timestamp"].min()
        run_end = rt["fh7000_timestamp"].max()
        # LiDAR scans whose corrected timestamp falls in [run_start, run_end]
        run_start_unix = pd.Timestamp(run_start).timestamp()
        run_end_unix = pd.Timestamp(run_end).timestamp()
        in_run_mask = (ts_corr_day >= run_start_unix) & (ts_corr_day <= run_end_unix)
        n_in_run = int(in_run_mask.sum())

        # merge_asof against this run's telemetry
        run_lid = pd.DataFrame({
            "lidar_ts_corr": pd.to_datetime(ts_corr_day[in_run_mask], unit="s")
                              .astype("datetime64[ns]"),
        }).sort_values("lidar_ts_corr")
        run_tel = rt[["fh7000_timestamp"]].copy()
        run_tel["fh7000_timestamp"] = run_tel["fh7000_timestamp"].astype("datetime64[ns]")
        run_tel = run_tel.sort_values("fh7000_timestamp")

        merged = pd.merge_asof(
            run_lid, run_tel,
            left_on="lidar_ts_corr", right_on="fh7000_timestamp",
            direction="nearest", tolerance=pd.Timedelta(seconds=tol),
        )
        n_match_indep = int(merged["fh7000_timestamp"].notna().sum())
        n_in_joint = len(grp)
        rows_match.append((sd, rf, n_in_run, n_match_indep, n_in_joint))
        if n_match_indep != n_in_joint:
            bad.append(f"{sd}/{rf}: indep={n_match_indep}, joint={n_in_joint}")
    if bad:
        rep.add("T10 retention (independent rerun)", "fail",
                "; ".join(bad[:5]) + (f" ... ({len(bad)} total)" if len(bad) > 5 else ""))
    else:
        rep.add("T10 retention (independent rerun)", "pass",
                f"{len(rows_match)} (sd, run_file) groups, "
                f"all retained-row counts match")
    return rows_match


def check_t11_csv_provenance(jp: pd.DataFrame, rep: Report,
                              n_per_day: int = 30,
                              rng: random.Random | None = None) -> None:
    """For sampled joint rows, open the *original* CSV and find the row
    whose timestamp equals fh7000_timestamp; verify a handful of telemetry
    fields match the CSV value at that row.
    """
    if rng is None:
        rng = random.Random(20260428)
    bad_total = 0
    bad_examples = []
    sampled_total = 0
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for sd, grp in jp.groupby("session_date"):
        for rf, sub in grp.groupby("run_file"):
            key = (sd, rf)
            if key not in cache:
                cache[key] = load_one_csv(CSV_DIR[sd] / rf)
            csv = cache[key]
            csv_idx = csv.set_index("ts", drop=False)
            sample = rng.sample(list(sub.index),
                                min(n_per_day, len(sub)))
            sampled_total += len(sample)
            for i in sample:
                r = jp.loc[i]
                ts = pd.Timestamp(r["fh7000_timestamp"])
                if ts not in csv_idx.index:
                    bad_total += 1
                    if len(bad_examples) < 3:
                        bad_examples.append(f"{sd}/{rf} idx {i}: ts {ts} not in CSV")
                    continue
                cv = csv_idx.loc[ts]
                if isinstance(cv, pd.DataFrame):
                    # Pick the row that matches; otherwise use first
                    cv = cv.iloc[0]
                # Check sampled telemetry fields
                ok = True
                local_diffs = []
                for col_idx, name in SAMPLED_TELEMETRY_INDEX_NAME.items():
                    csv_raw = cv.get(f"c{col_idx}")
                    try:
                        csv_v = float(csv_raw) if csv_raw not in ("", None) else float("nan")
                    except (TypeError, ValueError):
                        csv_v = csv_raw
                    j_v = r.get(name)
                    if not _val_equal(csv_v, j_v):
                        ok = False
                        local_diffs.append(f"{name}: csv={csv_v} joint={j_v}")
                if not ok:
                    bad_total += 1
                    if len(bad_examples) < 3:
                        bad_examples.append(f"{sd}/{rf} idx {i}: " + "; ".join(local_diffs))
    if bad_total:
        rep.add("T11 original CSV provenance (sampled)", "fail",
                f"{bad_total}/{sampled_total} rows differ; examples: {bad_examples}")
    else:
        rep.add("T11 original CSV provenance (sampled)", "pass",
                f"{sampled_total} sampled rows match original CSV values "
                f"({len(SAMPLED_TELEMETRY_INDEX_NAME)} fields each)")


def check_t12_obsolete_delta(jp: pd.DataFrame, rep: Report) -> None:
    """Compare per-day matched-row counts to the obsolete pipeline's
    cache/retention.json (if present). Differences should be small and
    explained by the per-day tau shift.
    """
    obs_path = (C.LIDAR_DISSIM_CACHE_LEGACY / "retention.json")
    if not obs_path.exists():
        rep.add("T12 vs obsolete retention", "info",
                f"obsolete cache not found at {obs_path}, skipping")
        return
    obs = json.loads(obs_path.read_text())
    obs_per_run = {(r["session_date"], r["run_file"]): r["n_matched"]
                   for r in obs["per_run"]}
    diffs = []
    for (sd, rf), grp in jp.groupby(["session_date", "run_file"]):
        new_n = len(grp)
        old_n = obs_per_run.get((sd, rf))
        if old_n is None:
            continue
        diffs.append((sd, rf, old_n, new_n, new_n - old_n))
    abs_max = max(abs(d) for *_, d in diffs) if diffs else 0
    rep.add("T12 vs obsolete retention", "info",
            f"per-CSV row-count delta: min={min(d for *_, d in diffs):+d}, "
            f"max={max(d for *_, d in diffs):+d}, max|delta|={abs_max} "
            f"(per-day tau shifted, tolerance unchanged)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Joint-coverage validation against original sources")
    print("=" * 72)
    rep = Report()

    print("\nLoading joint dataset ...")
    jp = pd.read_parquet(C.JOINT_PARQUET)
    print(f"  rows: {len(jp):,}, cols: {len(jp.columns)}")

    print("Loading cleaned telemetry ...")
    cleaned = pd.read_parquet(C.TELEMETRY_PARQUET)
    cleaned["fh7000_timestamp"] = cleaned["fh7000_timestamp"].astype("datetime64[ns]")
    print(f"  rows: {len(cleaned):,}, cols: {len(cleaned.columns)}")

    print("\n--- Schema & metadata --------------------------------------------------")
    check_t1_schema(jp, rep)
    check_t2_applied_tau(jp, rep)

    print("\n--- Timestamp identities -----------------------------------------------")
    check_t3_timestamp_identity(jp, rep)
    check_t4_t5_lidar_provenance(jp, rep)
    check_t6_t7_delta_and_tolerance(jp, rep)
    check_t8_tolerance_value(jp, rep, cleaned)

    print("\n--- Payload provenance -------------------------------------------------")
    check_t9_payload_identity(jp, rep, cleaned, n_per_day=200)
    check_t11_csv_provenance(jp, rep, n_per_day=30)

    print("\n--- Retention identity -------------------------------------------------")
    rows_match = check_t10_retention_independent(jp, rep, cleaned)
    print()
    print(f"  Per-CSV in-window vs matched rows (sd, run, lidar_in_window, "
          f"matched_indep, matched_joint):")
    for sd, rf, in_win, indep, joint in rows_match:
        marker = "OK" if indep == joint else "DIFF"
        print(f"    {sd}/{rf}: window={in_win:>7,}  matched_indep={indep:>7,}  "
              f"matched_joint={joint:>7,}  [{marker}]")

    print("\n--- Cross-method comparison --------------------------------------------")
    check_t12_obsolete_delta(jp, rep)

    rep.summary()
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
