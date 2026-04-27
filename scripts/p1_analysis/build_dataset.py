"""Phase 1 dataset construction.

Consolidates the time-synced joint parquet (681,593 rows), the LiDAR HDF5,
and the frozen Phase-0 v3 artifacts into one Parquet file with the schema
specified in `docs/proposal_rev7.md` §5.4 and the Phase 1 dataset brief.

Output: ``data/phase1/dataset.parquet`` plus a SHA-256 sidecar.

The frozen feature extractor at
``scripts/p0_analysis/artifacts/feature_extractor.py`` is the source of truth
for LiDAR scalar aggregates and AP-relative features; this script invokes
``FeatureExtractor.transform()`` and adds the columns the extractor does not
produce (sectoral features, telemetry, position, targets, anomaly columns,
provenance) on top.

Run as a module: ``python -m scripts.p1_analysis.build_dataset``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr (Windows default cp1250 cannot encode ², Δ, π, …).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project root on path so we can import scripts.p0_analysis when invoked
# either as a module or as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_analysis import config as C  # noqa: E402
from scripts.p0_analysis.artifacts.feature_extractor import (  # noqa: E402
    FeatureExtractor,
)

SEED = 20260427

DATASET_DIR = ROOT / "data" / "phase1"
DATASET_PATH = DATASET_DIR / "dataset.parquet"
SHA256_PATH = DATASET_DIR / "dataset.sha256"
CACHE_DIR = ROOT / "scripts" / "p1_analysis" / "cache"
ARTIFACTS_DIR = ROOT / "scripts" / "p0_analysis" / "artifacts"
ANOMALY_MASK_PATH = ARTIFACTS_DIR / "anomaly_mask.parquet"
FOV_PATH = ARTIFACTS_DIR / "lidar_fov.json"

CHUNK_SIZE = 20_000

# Sector boundaries from the brief, in degrees. Inclusive lower bound,
# exclusive upper bound — except sector 7 which is inclusive on both
# (so the upper boundary 111.60 is captured).
SECTOR_BOUNDS_DEG = [
    (-110.00, -78.34),
    (-78.34, -46.69),
    (-46.69, -15.03),
    (-15.03, 16.63),
    (16.63, 48.29),
    (48.29, 79.94),
    (79.94, 111.60),
]
N_SECTORS = len(SECTOR_BOUNDS_DEG)

CLUTTER_MM = 4000.0  # matches feature_extractor.CLUTTER_MM

TELEMETRY_FEATURES = [
    "load_long", "load_mid", "load_short",
    "battery_value", "momentary_current_consumption", "nns_state",
]


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

LIDAR_SCALAR_COLS = ["mean_dist_mm", "dist_p90_mm", "clutter_frac",
                     "openness_frac", "mean_front_mm"]
SECTORAL_COLS = (
    [f"mean_dist_sector_{i}_mm" for i in range(1, N_SECTORS + 1)]
    + [f"clutter_frac_sector_{i}" for i in range(1, N_SECTORS + 1)]
)
AP_RELATIVE_COLS = ["dist_to_AP", "sin_angle_to_AP", "cos_angle_to_AP",
                    "clutter_frac_toward_AP", "is_AP_in_FOV"]

PROVENANCE_COLS = ["joint_idx", "session_date", "run_file",
                   "fh7000_timestamp"]
TELEMETRY_COLS = ["speed_mps", "turn_rate"] + TELEMETRY_FEATURES
POSITION_COLS = ["x_m", "y_m"]
TARGET_COLS = ["signal_power", "signal_quality", "ping"]
ANOMALY_COLS = ["anomaly_flag", "_audit_nns_position_confidence",
                "_audit_anomaly_reason"]

# Feature/target columns that, if NaN, mark a row as a telemetry-gap anomaly.
# All eight rows that have NaN in any one of these have NaN in all of them
# (verified at build time by the validator).
ANOMALY_TELEMETRY_NAN_COLS = [
    "load_long", "load_mid", "load_short",
    "signal_power", "signal_quality", "ping",
]

SCHEMA_ORDER = (
    PROVENANCE_COLS
    + TELEMETRY_COLS
    + POSITION_COLS
    + LIDAR_SCALAR_COLS
    + [f"mean_dist_sector_{i}_mm" for i in range(1, N_SECTORS + 1)]
    + [f"clutter_frac_sector_{i}" for i in range(1, N_SECTORS + 1)]
    + AP_RELATIVE_COLS
    + TARGET_COLS
    + ANOMALY_COLS
)
assert len(SCHEMA_ORDER) == 44, f"expected 44 columns, got {len(SCHEMA_ORDER)}"


def expected_dtypes() -> dict[str, str]:
    """Pandas dtype label per column (matches the brief)."""
    out: dict[str, str] = {}
    out["joint_idx"] = "int64"
    out["session_date"] = "string"
    out["run_file"] = "string"
    out["fh7000_timestamp"] = "datetime64[us]"
    for c in (TELEMETRY_COLS + POSITION_COLS + LIDAR_SCALAR_COLS
              + [f"mean_dist_sector_{i}_mm" for i in range(1, N_SECTORS + 1)]
              + [f"clutter_frac_sector_{i}" for i in range(1, N_SECTORS + 1)]
              + ["dist_to_AP", "sin_angle_to_AP", "cos_angle_to_AP",
                 "clutter_frac_toward_AP"]
              + TARGET_COLS
              + ["_audit_nns_position_confidence"]):
        out[c] = "float64"
    out["is_AP_in_FOV"] = "bool"
    out["anomaly_flag"] = "bool"
    out["_audit_anomaly_reason"] = "string"
    return out


# ---------------------------------------------------------------------------
# Sectoral feature computation
# ---------------------------------------------------------------------------

def _build_sector_membership(mask_invalid: np.ndarray) -> list[np.ndarray]:
    """Per-sector boolean masks of length N_BEAMS.

    A beam belongs to sector `i` iff:
        - it is an active beam (slot < N_ACTIVE_BEAMS), AND
        - it is not on the AGV-body / dead-zone mask, AND
        - its angle (deg) falls in [theta_lo_i, theta_hi_i) for sectors 1..6
          and [theta_lo_7, theta_hi_7] for sector 7 (inclusive upper to
          capture the 111.60° boundary).

    The 1e-6 tolerance on the upper bound of sector 7 captures the boundary
    beam at 111.60° despite floating-point drift in `i * 0.2`.
    """
    angles_deg = np.full(C.N_BEAMS, np.nan, dtype=np.float64)
    angles_deg[:C.N_ACTIVE_BEAMS] = (
        C.ANGLE_MIN_DEG + np.arange(C.N_ACTIVE_BEAMS) * C.ANGLE_STEP_DEG
    )
    valid_slot = (np.arange(C.N_BEAMS) < C.N_ACTIVE_BEAMS) & (~mask_invalid)
    out: list[np.ndarray] = []
    for i, (lo, hi) in enumerate(SECTOR_BOUNDS_DEG):
        if i == N_SECTORS - 1:
            in_sec = (angles_deg >= lo) & (angles_deg <= hi + 1e-6)
        else:
            in_sec = (angles_deg >= lo) & (angles_deg < hi)
        out.append(valid_slot & in_sec)
    return out


def _sectoral_chunk(d: np.ndarray, sector_masks: list[np.ndarray]
                    ) -> dict[str, np.ndarray]:
    """Compute mean_dist_sector_i and clutter_frac_sector_i for one chunk.

    Each output is shape (n,) float32; NaN where the sector has no valid beam.
    """
    s = d.astype(np.float32)
    bad_global = (s == 0) | (s == C.INVALID_FAR)
    is_close = s < CLUTTER_MM
    out: dict[str, np.ndarray] = {}
    for k, sec_mask in enumerate(sector_masks, start=1):
        in_sec = sec_mask[None, :]
        eligible = (~bad_global) & in_sec
        n_elig = eligible.sum(axis=1).astype(np.float32)
        # mean distance over eligible beams in the sector
        s_masked = np.where(eligible, s, np.nan)
        with np.errstate(invalid="ignore"):
            mean_d = np.nanmean(s_masked, axis=1)
        # clutter fraction over eligible beams
        n_close = (eligible & is_close).sum(axis=1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            clutter = np.where(n_elig > 0, n_close / n_elig, np.nan)
        out[f"mean_dist_sector_{k}_mm"] = mean_d
        out[f"clutter_frac_sector_{k}"] = clutter.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# LiDAR HDF5 chunked read
# ---------------------------------------------------------------------------

def _read_lidar_chunk(ds, lidar_rows: np.ndarray) -> np.ndarray:
    """Read distances at lidar_rows in original order (matches extractor logic)."""
    n = len(lidar_rows)
    order = np.argsort(lidar_rows)
    sorted_rows = lidar_rows[order]
    lo, hi = int(sorted_rows[0]), int(sorted_rows[-1]) + 1
    span = hi - lo
    if span <= 8 * n:
        buf = ds[lo:hi]
        d_sorted = buf[sorted_rows - lo]
    else:
        d_sorted = np.empty((n, C.N_BEAMS), dtype=np.uint16)
        for k, r in enumerate(sorted_rows):
            d_sorted[k] = ds[int(r)]
    inv_order = np.argsort(order)
    return d_sorted[inv_order]


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def compute_lidar_and_ap_features(jp: pd.DataFrame, fx: FeatureExtractor,
                                  sector_masks: list[np.ndarray]) -> pd.DataFrame:
    """Single-pass LiDAR + AP-relative + sectoral feature computation.

    Mirrors ``FeatureExtractor.transform`` (same logic, same constants) but
    reads each LiDAR chunk only once and adds the 14 sectoral features that
    the frozen extractor does not produce.
    """
    n = len(jp)
    x = jp["x_m"].to_numpy()
    y = jp["y_m"].to_numpy()
    h = jp["heading_rad"].to_numpy()

    x_ap = np.empty(n, dtype=np.float64)
    y_ap = np.empty(n, dtype=np.float64)
    for sd, rec in fx.ap_coords.items():
        m = (jp["session_date"] == sd).to_numpy()
        x_ap[m] = float(rec["x_AP"])
        y_ap[m] = float(rec["y_AP"])
    EPS = 0.05  # matches feature_extractor.EPS_DIST_M
    dist = np.maximum(np.hypot(x_ap - x, y_ap - y), EPS)
    world_ang = np.arctan2(y_ap - y, x_ap - x)
    angle_ego = (world_ang - h + np.pi) % (2 * np.pi) - np.pi
    sin_ap = np.sin(angle_ego)
    cos_ap = np.cos(angle_ego)
    in_fov = (angle_ego >= fx._fov_min_rad) & (angle_ego <= fx._fov_max_rad)

    # Outputs allocated up front
    out: dict[str, np.ndarray] = {}
    for col in LIDAR_SCALAR_COLS:
        out[col] = np.full(n, np.nan, dtype=np.float32)
    for col in SECTORAL_COLS:
        out[col] = np.full(n, np.nan, dtype=np.float32)
    clutter_toward_ap = np.full(n, np.nan, dtype=np.float32)

    # Frozen extractor's helpers, used as the source of truth
    from scripts.p0_analysis.artifacts.feature_extractor import (
        _lidar_scalars_chunk, _clutter_toward_ap_chunk,
    )

    lidar_rows = jp["lidar_row"].to_numpy()
    t0 = time.time()
    with h5py.File(C.LIDAR_H5, "r") as h5:
        ds = h5["distances"]
        for s_i in range(0, n, CHUNK_SIZE):
            e_i = min(s_i + CHUNK_SIZE, n)
            d = _read_lidar_chunk(ds, lidar_rows[s_i:e_i])

            feats = _lidar_scalars_chunk(d, fx.mask_invalid, fx._front_mask)
            for col in LIDAR_SCALAR_COLS:
                out[col][s_i:e_i] = feats[col]

            sec = _sectoral_chunk(d, sector_masks)
            for col in SECTORAL_COLS:
                out[col][s_i:e_i] = sec[col]

            clutter_toward_ap[s_i:e_i] = _clutter_toward_ap_chunk(
                d, angle_ego[s_i:e_i], in_fov[s_i:e_i],
                fx.mask_invalid, fx._beam_angles_rad,
                fx._fov_min_rad, fx._fov_max_rad, fx._half_width_rad,
            )
            if (s_i // CHUNK_SIZE) % 5 == 0:
                pct = 100.0 * e_i / n
                print(f"    LiDAR pass: {e_i:>7,} / {n:,}  ({pct:5.1f}%)  "
                      f"elapsed {time.time() - t0:5.1f}s", flush=True)
    print(f"    LiDAR pass: done in {time.time() - t0:.1f}s")

    df = pd.DataFrame()
    for col in LIDAR_SCALAR_COLS + SECTORAL_COLS:
        df[col] = out[col].astype(np.float64)
    df["dist_to_AP"] = dist.astype(np.float64)
    df["sin_angle_to_AP"] = sin_ap.astype(np.float64)
    df["cos_angle_to_AP"] = cos_ap.astype(np.float64)
    df["clutter_frac_toward_AP"] = clutter_toward_ap.astype(np.float64)
    df["is_AP_in_FOV"] = in_fov.astype(bool)
    return df


def compute_turn_rate(jp: pd.DataFrame) -> np.ndarray:
    """Per-session smoothed d(heading_rad)/dt with a 0.5 s rolling mean.

    Edge rows (start of each session): forward-fill from the first valid value.
    """
    n = len(jp)
    head = jp["heading_rad"].to_numpy()
    ts = jp["fh7000_timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
    sess = jp["session_date"].to_numpy()
    sort_idx = np.lexsort((ts, sess))  # primary: sess, secondary: ts
    inv = np.argsort(sort_idx)
    head_s = head[sort_idx]
    ts_s = ts[sort_idx]
    sess_s = sess[sort_idx]

    # First, raw d(heading)/dt with wrap-to-pi on numerator
    raw = np.full(n, np.nan, dtype=np.float64)
    same_session = np.zeros(n, dtype=bool)
    same_session[1:] = sess_s[1:] == sess_s[:-1]
    dh = head_s[1:] - head_s[:-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    dt_s = (ts_s[1:] - ts_s[:-1]) * 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(dt_s > 0, dh / dt_s, np.nan)
    # Clip raw rates to ±π rad/s before smoothing: |turn_rate| > π rad/s
    # (>180°/s) is unphysical for this AGV class and would otherwise come
    # from heading-glitch noise (e.g. teleport-style anomalies, sub-ms dt).
    rate = np.clip(rate, -np.pi, np.pi)
    raw[1:] = np.where(same_session[1:], rate, np.nan)

    # 0.5-second rolling mean per session, time-aware.
    smoothed = np.full(n, np.nan, dtype=np.float64)
    for sd in pd.unique(sess_s):
        m = sess_s == sd
        idx = np.where(m)[0]
        ts_m = ts_s[idx] * 1e-9
        r_m = raw[idx]
        df = pd.DataFrame({"t": ts_m, "r": r_m})
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("t")
        sm = np.asarray(df["r"].rolling("500ms", min_periods=1).mean()
                        .to_numpy(), dtype=np.float64).copy()
        # Forward-fill the very first row from the first finite value
        first_finite_idx = int(np.argmax(np.isfinite(sm)))
        if np.isfinite(sm[first_finite_idx]):
            sm[:first_finite_idx + 1] = np.where(
                np.isfinite(sm[:first_finite_idx + 1]),
                sm[:first_finite_idx + 1], sm[first_finite_idx])
        smoothed[idx] = sm
    return smoothed[inv]


def assemble(jp: pd.DataFrame, anomaly: pd.DataFrame, fx: FeatureExtractor,
             sector_masks: list[np.ndarray]) -> pd.DataFrame:
    print("[1/5] Computing LiDAR + AP-relative + sectoral features...",
          flush=True)
    feat = compute_lidar_and_ap_features(jp, fx, sector_masks)

    print("[2/5] Computing turn_rate...", flush=True)
    turn_rate = compute_turn_rate(jp)

    print("[3/5] Building telemetry frame...", flush=True)
    speed = np.abs(jp["speed_mps"].to_numpy(dtype=np.float64))
    nan_speed = ~np.isfinite(speed)
    if nan_speed.any():
        sl = np.abs(jp["actual_speed_left"].to_numpy(dtype=np.float64))
        sr = np.abs(jp["actual_speed_right"].to_numpy(dtype=np.float64))
        speed = np.where(nan_speed, 0.5 * (sl + sr), speed)

    out = pd.DataFrame()

    # provenance
    out["joint_idx"] = np.arange(len(jp), dtype=np.int64)
    out["session_date"] = jp["session_date"].astype("string")
    out["run_file"] = jp["run_file"].astype("string")
    # parquet stores ns; convert to us per spec when we write
    out["fh7000_timestamp"] = pd.to_datetime(jp["fh7000_timestamp"]).astype(
        "datetime64[us]")

    # telemetry
    out["speed_mps"] = speed
    out["turn_rate"] = turn_rate.astype(np.float64)
    for col in TELEMETRY_FEATURES:
        out[col] = jp[col].to_numpy(dtype=np.float64)

    # position
    out["x_m"] = jp["x_m"].to_numpy(dtype=np.float64)
    out["y_m"] = jp["y_m"].to_numpy(dtype=np.float64)

    # LiDAR scalar + sectoral + AP-relative
    for col in LIDAR_SCALAR_COLS + SECTORAL_COLS:
        out[col] = feat[col].astype(np.float64)
    out["dist_to_AP"] = feat["dist_to_AP"].astype(np.float64)
    out["sin_angle_to_AP"] = feat["sin_angle_to_AP"].astype(np.float64)
    out["cos_angle_to_AP"] = feat["cos_angle_to_AP"].astype(np.float64)
    out["clutter_frac_toward_AP"] = feat["clutter_frac_toward_AP"].astype(
        np.float64)
    out["is_AP_in_FOV"] = feat["is_AP_in_FOV"].astype(bool)

    # targets
    for col in TARGET_COLS:
        out[col] = jp[col].to_numpy(dtype=np.float64)

    # anomaly — start from the v3 confidence-based mask, then OR in any row
    # whose required telemetry/target columns are NaN. The new
    # `_audit_anomaly_reason` column distinguishes the two cases for audit;
    # `low_confidence` takes priority on the (currently empty) intersection
    # so that the existing v3 per-session counts remain reproducible.
    a = anomaly.set_index("joint_idx").reindex(np.arange(len(jp)))
    low_conf = a["anomaly_flag"].to_numpy().astype(bool)
    out["_audit_nns_position_confidence"] = jp[
        "nns_position_confidence"].to_numpy(dtype=np.float64)

    telemetry_nan = np.zeros(len(jp), dtype=bool)
    for col in ANOMALY_TELEMETRY_NAN_COLS:
        telemetry_nan |= ~np.isfinite(jp[col].to_numpy(dtype=np.float64))

    flag = low_conf | telemetry_nan
    reason = np.where(low_conf, "low_confidence",
                      np.where(telemetry_nan, "telemetry_nan", "none"))
    out["anomaly_flag"] = flag
    out["_audit_anomaly_reason"] = pd.array(reason, dtype="string")

    # column order
    print("[4/5] Reordering columns and sorting...", flush=True)
    out = out[SCHEMA_ORDER]

    out = out.sort_values(["session_date", "fh7000_timestamp"], kind="stable",
                          ignore_index=True)
    print("[5/5] Done assembling.", flush=True)
    return out


# ---------------------------------------------------------------------------
# Write + hash
# ---------------------------------------------------------------------------

def _arrow_schema() -> pa.Schema:
    fields = []
    for col in SCHEMA_ORDER:
        if col == "joint_idx":
            fields.append(pa.field(col, pa.int64()))
        elif col in {"session_date", "run_file", "_audit_anomaly_reason"}:
            fields.append(pa.field(col, pa.string()))
        elif col == "fh7000_timestamp":
            fields.append(pa.field(col, pa.timestamp("us")))
        elif col in {"is_AP_in_FOV", "anomaly_flag"}:
            fields.append(pa.field(col, pa.bool_()))
        else:
            fields.append(pa.field(col, pa.float64()))
    return pa.schema(fields)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    schema = _arrow_schema()
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path, compression="zstd",
        row_group_size=65536, use_dictionary=True,
        write_statistics=False,  # statistics include non-deterministic stats
    )


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 1 dataset construction")
    print("=" * 70)
    np.random.seed(SEED)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] joint_coverage.parquet...")
    jp = pd.read_parquet(C.JOINT_PARQUET).reset_index(drop=True)
    print(f"       {len(jp):,} rows")

    print("[load] anomaly mask...")
    anomaly = pd.read_parquet(ANOMALY_MASK_PATH)
    print(f"       {len(anomaly):,} rows; "
          f"{anomaly['anomaly_flag'].sum():,} flagged")

    print("[load] feature extractor + AGV-body mask + FOV...")
    fx = FeatureExtractor(artifacts_dir=ARTIFACTS_DIR)
    sector_masks = _build_sector_membership(fx.mask_invalid)
    sec_counts = [int(m.sum()) for m in sector_masks]
    print(f"       sector beam counts: {sec_counts}  total={sum(sec_counts)}")

    t0 = time.time()
    df = assemble(jp, anomaly, fx, sector_masks)
    elapsed = time.time() - t0
    print(f"[assemble] {len(df):,} rows × {len(df.columns)} cols "
          f"(wall {elapsed:.1f}s)")

    print("[write] dataset.parquet...")
    write_parquet(df, DATASET_PATH)
    file_size_mb = DATASET_PATH.stat().st_size / 1e6
    digest = sha256_of(DATASET_PATH)
    SHA256_PATH.write_text(digest + "  dataset.parquet\n")
    print(f"       size {file_size_mb:.1f} MB  sha256 {digest}")
    print(f"       hash sidecar: {SHA256_PATH}")

    summary = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "file_size_bytes": int(DATASET_PATH.stat().st_size),
        "sha256": digest,
        "elapsed_s": float(elapsed),
        "session_counts": {
            sd: int((df["session_date"] == sd).sum())
            for sd in C.SESSIONS
        },
        "anomaly_counts": {
            sd: int(((df["session_date"] == sd) & df["anomaly_flag"]).sum())
            for sd in C.SESSIONS
        },
        "anomaly_reason_counts": {
            sd: {
                r: int(((df["session_date"] == sd)
                        & (df["_audit_anomaly_reason"] == r)).sum())
                for r in ("none", "low_confidence", "telemetry_nan")
            }
            for sd in C.SESSIONS
        },
    }
    (CACHE_DIR / "build_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
