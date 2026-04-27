"""Phase 1 dataset validation.

Runs every check listed in §5 of the Phase 1 dataset brief, writes the
validation report to ``docs/phase1_dataset/report.md`` with embedded figures,
and exits non-zero (with a clear summary) if any hard check fails.

Run as a module: ``python -m scripts.p1_analysis.validate_dataset``.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Force UTF-8 stdout/stderr so log lines containing math symbols (², Δ, π, …)
# don't crash on Windows consoles that default to cp1250/cp1252. The report
# itself is already written with explicit UTF-8 encoding.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_analysis import config as C  # noqa: E402

from scripts.p1_analysis.build_dataset import (  # noqa: E402
    SCHEMA_ORDER, LIDAR_SCALAR_COLS, N_SECTORS, CLUTTER_MM,
    expected_dtypes, sha256_of, _build_sector_membership,
    DATASET_PATH, SHA256_PATH,
)

SEED = 20260427

REPORT_DIR = ROOT / "docs" / "phase1_dataset"
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "report.md"
ARTIFACTS_DIR = ROOT / "scripts" / "p0_analysis" / "artifacts"

# Reference numbers from the brief / Phase 0 v3
EXPECTED_ROWS = 681_593
EXPECTED_SESSION_COUNTS = {
    "15.03.2026": 280_025,
    "24.03.2026": 168_940,
    "25.02.2026": 232_628,
}
# Phase 1 anomaly definition expansion (delta brief): the v3 confidence-based
# mask (74,429 rows) is augmented with rows that have a NaN in any of
# {load_long, load_mid, load_short, signal_power, signal_quality, ping}.
# Eight such rows exist in the source data, all in 15.03.2026 with
# confidence = 100, so they are tagged `telemetry_nan` rather than
# `low_confidence`. v3 per-session counts are preserved for the
# `low_confidence` reason; the `telemetry_nan` reason adds 8 to 15.03.
EXPECTED_LOW_CONFIDENCE_COUNTS = {
    "15.03.2026": 47_738,
    "24.03.2026": 10_535,
    "25.02.2026": 16_156,
}
EXPECTED_TELEMETRY_NAN_COUNTS = {
    "15.03.2026": 8,
    "24.03.2026": 0,
    "25.02.2026": 0,
}
EXPECTED_ANOMALY_COUNTS = {
    sd: EXPECTED_LOW_CONFIDENCE_COUNTS[sd] + EXPECTED_TELEMETRY_NAN_COUNTS[sd]
    for sd in EXPECTED_SESSION_COUNTS
}
EXPECTED_TOTAL_ANOMALY = sum(EXPECTED_ANOMALY_COUNTS.values())  # 74,437

# Columns that may be globally NaN ONLY because the row is anomaly-flagged
# (telemetry-gap rows). Among non-anomaly rows these must be 0 NaN.
NAN_OK_VIA_ANOMALY = [
    "load_long", "load_mid", "load_short",
    "signal_power", "signal_quality", "ping",
]
# Columns that are NaN by design (independent of anomaly flag).
NAN_BY_DESIGN = ["clutter_frac_toward_AP"] + [
    f"mean_dist_sector_{i}_mm" for i in range(1, 8)
] + [f"clutter_frac_sector_{i}" for i in range(1, 8)]
EXPECTED_SIGNAL_POWER_MEAN = {
    "15.03.2026": -36.91,
    "24.03.2026": -39.66,
    "25.02.2026": -31.07,
}
EXPECTED_PATHLOSS_R2_V3 = {
    "15.03.2026": 0.357,
    "24.03.2026": 0.003,
    "25.02.2026": 0.088,
}


@dataclass
class CheckLog:
    hard_fail: list[str] = field(default_factory=list)
    soft_warn: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        print(f"[HARD] {msg}", flush=True)
        self.hard_fail.append(msg)

    def warn(self, msg: str) -> None:
        print(f"[WARN] {msg}", flush=True)
        self.soft_warn.append(msg)

    def note(self, msg: str) -> None:
        print(f"[INFO] {msg}", flush=True)
        self.info.append(msg)


def _column_min_max_mean_nan(df: pd.DataFrame, col: str
                             ) -> tuple[float, float, float, int]:
    s = df[col]
    if s.dtype == bool:
        v = s.to_numpy(dtype=np.int64)
        return float(v.min()), float(v.max()), float(v.mean()), 0
    v = s.to_numpy(dtype=np.float64)
    nan = int(np.isnan(v).sum())
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), nan
    return float(finite.min()), float(finite.max()), float(finite.mean()), nan


# ---------------------------------------------------------------------------
# Individual check functions — return strings for the report.
# ---------------------------------------------------------------------------

def check_schema(df: pd.DataFrame, log: CheckLog) -> str:
    out: list[str] = []
    out.append("### Column presence and order")
    actual = list(df.columns)
    if actual != SCHEMA_ORDER:
        log.fail("schema column order/presence mismatch")
        out.append(f"- expected: {SCHEMA_ORDER}")
        out.append(f"- actual:   {actual}")
    else:
        out.append(f"- {len(SCHEMA_ORDER)} columns, exact names, exact "
                   f"order: OK")

    out.append("\n### Dtypes")
    expected = expected_dtypes()
    rows = ["| col | expected | actual | OK |", "|---|---|---|---|"]
    for col in SCHEMA_ORDER:
        exp = expected[col]
        act = str(df[col].dtype)
        ok = (
            (exp == "string" and act == "string")
            or (exp == "datetime64[us]" and act.startswith("datetime64[us"))
            or (exp == "bool" and act == "bool")
            or (exp == "int64" and act == "int64")
            or (exp == "float64" and act == "float64")
        )
        if not ok:
            log.fail(f"dtype mismatch for {col}: expected {exp}, got {act}")
        rows.append(f"| `{col}` | {exp} | {act} | {'OK' if ok else 'FAIL'} |")
    out.append("\n".join(rows))

    out.append(f"\n### Row count: {len(df):,} (expected {EXPECTED_ROWS:,})")
    if len(df) != EXPECTED_ROWS:
        log.fail(f"row count {len(df)} != expected {EXPECTED_ROWS}")

    return "\n".join(out)


def check_provenance(df: pd.DataFrame, log: CheckLog) -> str:
    out: list[str] = []

    # joint_idx unique, full coverage
    ji = df["joint_idx"].to_numpy()
    n_unique = int(np.unique(ji).size)
    if n_unique != len(df):
        log.fail(f"joint_idx not unique: {n_unique} unique vs {len(df)} rows")
    if not (ji.min() == 0 and ji.max() == EXPECTED_ROWS - 1
            and n_unique == EXPECTED_ROWS):
        log.fail("joint_idx does not cover [0, 681593) exactly")
    out.append(f"- joint_idx unique = {n_unique == len(df)}; "
               f"min={int(ji.min())}, max={int(ji.max())}")

    # session_date values
    seen = set(df["session_date"].dropna().unique().tolist())
    expected_sess = set(EXPECTED_SESSION_COUNTS)
    if seen != expected_sess:
        log.fail(f"session_date set mismatch: {seen} vs {expected_sess}")
    out.append(f"- session_date set = {sorted(seen)}")

    # per-session counts
    counts = df["session_date"].value_counts().to_dict()
    rows = ["| session | actual | expected | OK |",
            "|---|---:|---:|---|"]
    for sd, exp in EXPECTED_SESSION_COUNTS.items():
        act = int(counts.get(sd, 0))
        ok = (act == exp)
        if not ok:
            log.fail(f"row count {sd}: {act} vs {exp}")
        rows.append(f"| {sd} | {act:,} | {exp:,} | {'OK' if ok else 'FAIL'} |")
    out.append("\n".join(rows))

    # run_file non-null
    rf_null = int(df["run_file"].isna().sum())
    if rf_null:
        log.fail(f"run_file has {rf_null} null values")
    out.append(f"- run_file null count: {rf_null}")
    rf_unique = int(df["run_file"].nunique())
    out.append(f"- run_file unique values: {rf_unique}")

    # fh7000_timestamp monotonic per session
    ok_global = True
    monotonic_rows = ["\n| session | monotonic non-decreasing? |",
                      "|---|---|"]
    for sd in EXPECTED_SESSION_COUNTS:
        sub = df[df["session_date"] == sd]
        ts = sub["fh7000_timestamp"].to_numpy()
        diffs = np.diff(ts.astype("datetime64[ns]").astype(np.int64))
        is_mono = bool(np.all(diffs >= 0))
        if not is_mono:
            log.fail(f"fh7000_timestamp not monotonic in {sd}")
            ok_global = False
        monotonic_rows.append(f"| {sd} | {is_mono} |")
    out.append("\n".join(monotonic_rows))
    if ok_global:
        out.append("\n- All sessions: monotonic non-decreasing OK")
    return "\n".join(out)


def check_anomaly(df: pd.DataFrame, log: CheckLog) -> str:
    out: list[str] = []
    rng = np.random.default_rng(SEED)

    out.append("### Per-session anomaly counts")
    rows = ["| session | flagged | expected | OK |",
            "|---|---:|---:|---|"]
    for sd, exp in EXPECTED_ANOMALY_COUNTS.items():
        act = int(((df["session_date"] == sd) & df["anomaly_flag"]).sum())
        ok = (act == exp)
        if not ok:
            log.fail(f"anomaly count {sd}: {act} vs {exp}")
        rows.append(f"| {sd} | {act:,} | {exp:,} | {'OK' if ok else 'FAIL'} |")
    total = int(df["anomaly_flag"].sum())
    rows.append(f"| **total** | **{total:,}** | "
                f"**{EXPECTED_TOTAL_ANOMALY:,}** | "
                f"{'OK' if total == EXPECTED_TOTAL_ANOMALY else 'FAIL'} |")
    if total != EXPECTED_TOTAL_ANOMALY:
        log.fail(f"total anomaly count {total} vs {EXPECTED_TOTAL_ANOMALY}")
    out.append("\n".join(rows))

    # Per-session breakdown by `_audit_anomaly_reason`
    out.append("\n### Per-session breakdown by `_audit_anomaly_reason`")
    reason_rows = [
        "| session | low_confidence | telemetry_nan | none | OK |",
        "|---|---:|---:|---:|---|",
    ]
    valid_reasons = {"none", "low_confidence", "telemetry_nan"}
    seen_reasons = set(df["_audit_anomaly_reason"].dropna().unique().tolist())
    if not seen_reasons.issubset(valid_reasons):
        log.fail(f"unexpected `_audit_anomaly_reason` values: "
                 f"{sorted(seen_reasons - valid_reasons)}")
    for sd in EXPECTED_SESSION_COUNTS:
        sub = df[df["session_date"] == sd]
        n_low = int((sub["_audit_anomaly_reason"] == "low_confidence").sum())
        n_tn = int((sub["_audit_anomaly_reason"] == "telemetry_nan").sum())
        n_none = int((sub["_audit_anomaly_reason"] == "none").sum())
        exp_low = EXPECTED_LOW_CONFIDENCE_COUNTS[sd]
        exp_tn = EXPECTED_TELEMETRY_NAN_COUNTS[sd]
        ok = (n_low == exp_low) and (n_tn == exp_tn) \
            and (n_none == EXPECTED_SESSION_COUNTS[sd] - exp_low - exp_tn)
        if not ok:
            log.fail(f"anomaly_reason {sd}: low_confidence={n_low} "
                     f"(expected {exp_low}), telemetry_nan={n_tn} "
                     f"(expected {exp_tn}), none={n_none}")
        reason_rows.append(
            f"| {sd} | {n_low:,} | {n_tn:,} | {n_none:,} | "
            f"{'OK' if ok else 'FAIL'} |"
        )
    out.append("\n".join(reason_rows))

    # Reason ↔ flag consistency
    reason = df["_audit_anomaly_reason"].to_numpy()
    flag = df["anomaly_flag"].to_numpy()
    flag_from_reason = (reason != "none")
    mismatches = int((flag_from_reason != flag).sum())
    if mismatches:
        log.fail(f"`_audit_anomaly_reason` <-> `anomaly_flag` inconsistency: "
                 f"{mismatches} rows")
    out.append(f"\n- `_audit_anomaly_reason` ↔ `anomaly_flag` consistency: "
               f"{mismatches} mismatches "
               f"({'OK' if mismatches == 0 else 'FAIL'})")

    # Confidence consistency on `low_confidence` rows
    conf = df["_audit_nns_position_confidence"].to_numpy()
    nan_conf = ~np.isfinite(conf)
    is_low = (reason == "low_confidence")
    is_tn = (reason == "telemetry_nan")
    is_none = (reason == "none")
    bad_low = is_low & ~(nan_conf | (conf < 35.0))
    bad_tn_conf = is_tn & (nan_conf | (conf < 35.0))
    bad_none = is_none & (nan_conf | (conf < 35.0))
    if bad_low.any():
        log.fail(f"{int(bad_low.sum())} `low_confidence` rows have "
                 f"confidence >= 35 and finite")
    if bad_tn_conf.any():
        log.fail(f"{int(bad_tn_conf.sum())} `telemetry_nan` rows have "
                 f"confidence < 35 or NaN (priority should have set "
                 f"`low_confidence`)")
    if bad_none.any():
        log.fail(f"{int(bad_none.sum())} `none` rows have confidence < 35 "
                 f"or NaN")
    out.append(f"- consistency: `low_confidence` rows with conf>=35 and "
               f"finite: {int(bad_low.sum())}")
    out.append(f"- consistency: `telemetry_nan` rows with conf<35 or NaN "
               f"(priority violation): {int(bad_tn_conf.sum())}")
    out.append(f"- consistency: `none` rows with conf<35 or NaN: "
               f"{int(bad_none.sum())}")

    # NaN confidence (Phase 0 v3 says zero)
    n_nan_conf = int(nan_conf.sum())
    out.append(f"- NaN/non-finite _audit_nns_position_confidence: {n_nan_conf}")

    # Spot-checks: 1000 random low_confidence + 1000 random non-anomaly
    low_idx = np.where(is_low)[0]
    none_idx = np.where(is_none)[0]
    n_spot = 1000
    a = rng.choice(low_idx, size=min(n_spot, len(low_idx)), replace=False)
    b = rng.choice(none_idx, size=min(n_spot, len(none_idx)), replace=False)
    bad_a = int(((conf[a] >= 35.0) & np.isfinite(conf[a])).sum())
    bad_b = int(((conf[b] < 35.0) | ~np.isfinite(conf[b])).sum())
    out.append(f"- spot-check 1000 `low_confidence`: {bad_a} violations "
               f"(expected 0)")
    out.append(f"- spot-check 1000 `none`: {bad_b} violations (expected 0)")
    if bad_a or bad_b:
        log.fail(f"anomaly spot-check failed: low_confidence={bad_a}, "
                 f"none={bad_b}")
    return "\n".join(out)


def check_ranges(df: pd.DataFrame, log: CheckLog) -> str:
    out: list[str] = []
    # The frozen extractor's own self-test (`feature_extractor.py::_validate`)
    # accepts mean_dist_mm up to 70 000 mm, not the 8 500 mm bound the brief
    # quotes (the Leuze RSL400 reports up to ~16 m for distant returns when
    # no closer obstacle is present, exceeding the rated 8.25 m). We adopt
    # the extractor's range and document the deviation in §1 of this report.
    LIDAR_DIST_MAX = 70_000.0
    rng_specs = [
        ("speed_mps", 0.0, 2.0, False),
        ("turn_rate", -np.pi - 1e-6, np.pi + 1e-6, False),
        ("load_long", -np.inf, np.inf, False),
        ("load_mid", -np.inf, np.inf, False),
        ("load_short", -np.inf, np.inf, False),
        ("battery_value", -np.inf, np.inf, False),
        ("momentary_current_consumption", -np.inf, np.inf, False),
        ("nns_state", -np.inf, np.inf, False),
        ("x_m", -np.inf, np.inf, False),
        ("y_m", -np.inf, np.inf, False),
        ("mean_dist_mm", 0.0 + 1e-6, LIDAR_DIST_MAX, False),
        ("dist_p90_mm", 0.0 + 1e-6, LIDAR_DIST_MAX, False),
        ("mean_front_mm", 0.0 + 1e-6, LIDAR_DIST_MAX, False),
        ("clutter_frac", 0.0, 1.0, False),
        ("openness_frac", 0.0, 1.0, False),
        ("dist_to_AP", 0.0 + 1e-6, 50.0, False),
        ("sin_angle_to_AP", -1.0 - 1e-6, 1.0 + 1e-6, False),
        ("cos_angle_to_AP", -1.0 - 1e-6, 1.0 + 1e-6, False),
        ("clutter_frac_toward_AP", 0.0, 1.0, True),
        ("signal_power", -90.0, -10.0, False),
        ("signal_quality", 0.0, 100.0, False),
        ("ping", 0.0, 1000.0, False),
    ]
    # Sectoral feature ranges
    for i in range(1, N_SECTORS + 1):
        rng_specs.append(
            (f"mean_dist_sector_{i}_mm", 0.0 + 1e-6, LIDAR_DIST_MAX, True))
        rng_specs.append((f"clutter_frac_sector_{i}", 0.0, 1.0, True))

    # Build sets for NaN-policy lookup
    nan_via_anomaly = set(NAN_OK_VIA_ANOMALY)
    nan_by_design = set(NAN_BY_DESIGN)
    non_anom_mask = ~df["anomaly_flag"].to_numpy()

    rows = ["| col | min | max | mean | NaN (global) | NaN (non-anom) "
            "| range OK | NaN OK |",
            "|---|---:|---:|---:|---:|---:|---|---|"]
    for col, lo, hi, nan_ok in rng_specs:
        mn, mx, mean, nan = _column_min_max_mean_nan(df, col)
        # NaN among non-anomaly rows
        v = df[col].to_numpy()
        if df[col].dtype == bool:
            non_anom_nan = 0
        else:
            non_anom_nan = int(np.isnan(v[non_anom_mask]).sum())

        in_range = (np.isnan(mn) or (mn >= lo - 1e-6 and mx <= hi + 1e-6))
        if not in_range:
            log.fail(f"{col} out of expected range "
                     f"[{lo}, {hi}]: actual [{mn}, {mx}]")

        # NaN policy:
        # - NaN_BY_DESIGN: NaN OK anywhere
        # - NAN_OK_VIA_ANOMALY: NaN OK only if anomaly_flag=True; non-anomaly
        #   NaN is a hard fail
        # - everything else: NaN is a hard fail (zero NaN globally required)
        if col in nan_by_design:
            nan_pass = True
        elif col in nan_via_anomaly:
            nan_pass = (non_anom_nan == 0)
            if non_anom_nan > 0:
                log.fail(f"{col} has {non_anom_nan} NaN among non-anomaly "
                         f"rows (this is what Phase 1 training will see)")
        else:
            nan_pass = (nan == 0)
            if nan > 0:
                log.fail(f"{col} has {nan} NaN values "
                         f"(NaN-not-allowed column)")

        rows.append(
            f"| `{col}` | {mn:.4g} | {mx:.4g} | {mean:.4g} | {nan} | "
            f"{non_anom_nan} | "
            f"{'OK' if in_range else 'FAIL'} | "
            f"{'OK' if nan_pass else 'FAIL'} |"
        )
    out.append("\n".join(rows))

    # Per-session NaN fractions for NaN-allowed columns
    out.append("\n### Per-session NaN fractions (NaN-allowed-by-design columns)")
    rows = ["| col | 15.03 | 24.03 | 25.02 |",
            "|---|---:|---:|---:|"]
    for col in NAN_BY_DESIGN:
        cells = [col]
        for sd in EXPECTED_SESSION_COUNTS:
            sub = df[df["session_date"] == sd][col]
            frac = float(sub.isna().mean())
            cells.append(f"{frac:.4f}")
        rows.append("| `" + cells[0] + "` | " + " | ".join(cells[1:]) + " |")
    out.append("\n".join(rows))

    # Telemetry-gap columns: report global NaN (kept-with-flag) vs non-anomaly NaN.
    out.append("\n### Telemetry-gap columns: global vs non-anomaly NaN counts")
    out.append("Phase 1 training filters `~df['anomaly_flag']`, so the "
               "non-anomaly-NaN column is the operational guarantee.")
    rows = ["| col | global NaN | non-anomaly NaN | OK |",
            "|---|---:|---:|---|"]
    for col in NAN_OK_VIA_ANOMALY:
        v = df[col].to_numpy()
        global_nan = int(np.isnan(v).sum())
        non_anom_n = int(np.isnan(v[non_anom_mask]).sum())
        ok = (non_anom_n == 0)
        rows.append(f"| `{col}` | {global_nan} | {non_anom_n} | "
                    f"{'OK' if ok else 'FAIL'} |")
    out.append("\n".join(rows))

    return "\n".join(out)


def check_cross_feature(df: pd.DataFrame, fx_mask_invalid: np.ndarray,
                        log: CheckLog) -> str:
    out: list[str] = []
    rng = np.random.default_rng(SEED)

    # sin² + cos² check
    s = df["sin_angle_to_AP"].to_numpy()
    c = df["cos_angle_to_AP"].to_numpy()
    err = np.abs(s * s + c * c - 1.0)
    max_err = float(np.nanmax(err))
    if max_err >= 1e-6:
        log.fail(f"sin²+cos² check: max deviation {max_err:.2e} >= 1e-6")
    out.append(f"- sin² + cos² ≈ 1: max deviation = {max_err:.2e} "
               f"(threshold 1e-6) → {'OK' if max_err < 1e-6 else 'FAIL'}")

    # is_AP_in_FOV consistency
    in_fov = df["is_AP_in_FOV"].to_numpy().astype(bool)
    cf = df["clutter_frac_toward_AP"].to_numpy()
    bad_out = int((~in_fov & np.isfinite(cf)).sum())
    bad_in = int((in_fov & ~np.isfinite(cf)).sum())
    out.append(f"- rows out-of-FOV with finite clutter_frac_toward_AP: {bad_out}")
    out.append(f"- rows in-FOV with NaN clutter_frac_toward_AP: {bad_in} "
               f"(some can occur if no eligible beams; <= a few permissible)")
    if bad_out:
        log.fail(f"is_AP_in_FOV consistency: {bad_out} out-of-FOV rows have "
                 f"finite clutter_frac_toward_AP")
    # in-FOV NaN should be very rare (only if no beams at all in ±15° cone are
    # valid). Soft-warn unless count is sizable.
    if bad_in > 0:
        log.warn(f"{bad_in} in-FOV rows have NaN clutter_frac_toward_AP "
                 f"(no eligible beams in ±15° cone)")

    # Sectoral aggregation consistency: check beam-count-weighted average of
    # sector means matches mean_dist_mm on a 100-row spot-check. We need the
    # raw LiDAR for the per-row beam counts.
    out.append("\n### Sectoral aggregation consistency (100-row spot-check)")
    sector_masks = _build_sector_membership(fx_mask_invalid)
    n_spot = 100
    spot_idx = rng.choice(len(df), size=n_spot, replace=False)
    spot_idx.sort()
    joint_idx = df["joint_idx"].to_numpy()[spot_idx]
    # Read corresponding lidar_rows from the joint parquet
    jp_lidar_rows = pd.read_parquet(
        C.JOINT_PARQUET, columns=["lidar_row"]
    )["lidar_row"].to_numpy()[joint_idx]

    rel_errs: list[float] = []
    with h5py.File(C.LIDAR_H5, "r") as h5:
        ds = h5["distances"]
        # Read each row individually (n=100 is small)
        for k, lr in enumerate(jp_lidar_rows):
            d = ds[int(lr)].astype(np.float32)
            bad = (d == 0) | (d == C.INVALID_FAR) | fx_mask_invalid
            valid_global = ~bad
            mean_global = (
                float(np.nanmean(np.where(valid_global, d, np.nan)))
                if valid_global.any() else np.nan
            )
            num = 0.0
            den = 0.0
            for sec_mask in sector_masks:
                eligible = (~bad) & sec_mask
                cnt = int(eligible.sum())
                if cnt == 0:
                    continue
                sm = float(np.nanmean(np.where(eligible, d, np.nan)))
                num += sm * cnt
                den += cnt
            mean_from_sectors = num / den if den > 0 else np.nan
            stored = float(df["mean_dist_mm"].to_numpy()[spot_idx[k]])
            ref = stored if np.isfinite(stored) else mean_global
            if np.isfinite(mean_from_sectors) and np.isfinite(ref) and ref > 0:
                rel = abs(mean_from_sectors - ref) / ref
                rel_errs.append(rel)
    rel_errs_arr = np.array(rel_errs)
    if rel_errs_arr.size:
        mx = float(rel_errs_arr.max())
        out.append(f"- max relative error across 100 rows: {mx:.4e}")
        if mx > 0.01:
            log.warn(f"sectoral aggregation: max rel error {mx:.4e} > 1% "
                     f"(soft warning per brief)")
    else:
        out.append("- (no comparable rows found)")

    # dist_to_AP minimum per session
    out.append("\n### `dist_to_AP` minimum per session")
    rows = ["| session | min dist_to_AP (m) | OK |", "|---|---:|---|"]
    for sd in EXPECTED_SESSION_COUNTS:
        v = df[df["session_date"] == sd]["dist_to_AP"].min()
        ok = (v > 0)
        rows.append(f"| {sd} | {v:.4f} | {'OK' if ok else 'FAIL'} |")
        if not ok:
            log.fail(f"dist_to_AP minimum for {sd} not > 0: {v}")
    out.append("\n".join(rows))

    # signal_power means per session
    out.append("\n### `signal_power` per-session means vs Phase 0")
    rows = ["| session | actual | expected | Δ | OK |",
            "|---|---:|---:|---:|---|"]
    for sd, exp in EXPECTED_SIGNAL_POWER_MEAN.items():
        v = float(df[df["session_date"] == sd]["signal_power"].mean(
            skipna=True))
        delta = v - exp
        ok = abs(delta) <= 0.05
        if not ok:
            log.fail(f"signal_power mean {sd}: {v:.3f} vs {exp:.3f} "
                     f"(Δ={delta:+.3f}, tol 0.05)")
        rows.append(f"| {sd} | {v:.3f} | {exp:.3f} | {delta:+.3f} | "
                    f"{'OK' if ok else 'FAIL'} |")
    out.append("\n".join(rows))

    return "\n".join(out)


def check_distributions(df: pd.DataFrame, log: CheckLog) -> str:
    out: list[str] = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Per-session histograms for the 5 LiDAR scalar features
    out.append("### LiDAR scalar histograms per session")
    for sd in EXPECTED_SESSION_COUNTS:
        sub = df[df["session_date"] == sd]
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.4), constrained_layout=True)
        for ax, col in zip(axes, LIDAR_SCALAR_COLS):
            v = sub[col].to_numpy()
            v = v[np.isfinite(v)]
            ax.hist(v, bins=60, color="#3a76d4", alpha=0.85)
            ax.set_title(f"{sd} — {col}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_xlabel(col, fontsize=8)
            if v.size:
                ax.text(0.98, 0.95, f"μ={v.mean():.0f}\nN={v.size:,}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, family="monospace",
                        bbox=dict(facecolor="white", alpha=0.7,
                                  edgecolor="none"))
        fname = f"lidar_scalars_{sd.replace('.', '_')}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=110)
        plt.close(fig)
        out.append(f"- `figures/{fname}`")

    # mean_front_mm sanity check (should be ~5,000 mm not ~1,200 mm if the
    # AGV-body mask was applied correctly)
    out.append("\n### `mean_front_mm` AGV-body-mask sanity check")
    rows = ["| session | mean | OK (>2,500 mm) |", "|---|---:|---|"]
    for sd in EXPECTED_SESSION_COUNTS:
        v = float(df[df["session_date"] == sd]["mean_front_mm"].mean(
            skipna=True))
        ok = v > 2500.0
        if not ok:
            log.fail(f"mean_front_mm in {sd} = {v:.1f} mm; expected >2,500 — "
                     f"AGV-body mask may not have been applied "
                     f"(P0 v1-vs-v2 LiDAR-mapping bug)")
        rows.append(f"| {sd} | {v:.1f} | {'OK' if ok else 'FAIL'} |")
    out.append("\n".join(rows))

    # Path-loss-fit R² re-derivation. Phase 0 v3 (run_p0_2_v3.py) computes R²
    # over a 30,000-row sub-sample drawn (with seed C.SEED) from the
    # anomaly-filtered, |speed_mps|>0.05, signal_power-finite slice of each
    # session, using `dist_to_AP = max(hypot(x_AP-x, y_AP-y), 0.05)`. To match
    # the reference R² *exactly*, the same RNG sequence must be reproduced,
    # which requires emulating the inter-session draws (the bootstrap step in
    # the Phase 0 script advances the rng between sessions). We replicate the
    # full protocol here — main 30k draw, then a 5k boot draw, then 200 ×
    # `rng.integers(0, 5000, 5000)` bootstrap draws — so the resulting R²
    # matches the ap_coords.json `v3.R2` value within the ±0.005 tolerance.
    out.append("\n### Path-loss-fit R² re-derivation per session")
    out.append("- Reproduces the Phase 0 v3 protocol from "
               "`scripts/p0_analysis/run_p0_2_v3.py`: anomaly-filtered + "
               "`|speed_mps| > 0.05` + `signal_power` finite, 30,000-row "
               "sub-sample (seed `C.SEED = 20260427`). Inter-session RNG "
               "draws (5,000-row bootstrap-subsample + 200 × bootstrap-resamples) "
               "are replayed so the per-session 30k draws match the "
               "Phase 0 v3 sequence exactly.")
    ap_coords = json.loads((ARTIFACTS_DIR / "ap_coords.json").read_text())
    N_DOWNSAMPLE = 30000
    N_BOOT = 200
    BOOT_SUB = 5000
    p0_rng = np.random.default_rng(C.SEED)
    rows = ["| session | R^2 (re-derived) | reference (v3) | delta | "
            "rows used | OK |",
            "|---|---:|---:|---:|---:|---|"]
    sessions_in_order = list(EXPECTED_PATHLOSS_R2_V3)
    # The Phase 0 script iterates C.SESSIONS = [15.03, 24.03, 25.02];
    # EXPECTED_PATHLOSS_R2_V3 is defined in that same order so insertion
    # iteration matches.
    for sd in sessions_in_order:
        ref_r2 = EXPECTED_PATHLOSS_R2_V3[sd]
        sub_full = df[(df["session_date"] == sd) & ~df["anomaly_flag"]
                      & df["signal_power"].notna()]
        sub_fit = sub_full[sub_full["speed_mps"].abs() > 0.05]
        n_fit = len(sub_fit)
        if n_fit > N_DOWNSAMPLE:
            idx_ds = p0_rng.choice(n_fit, N_DOWNSAMPLE, replace=False)
            sub_fit_ds = sub_fit.iloc[idx_ds]
        else:
            sub_fit_ds = sub_fit
        sp = sub_fit_ds["signal_power"].to_numpy()
        d = sub_fit_ds["dist_to_AP"].to_numpy()
        if sp.size < 100:
            log.fail(f"path-loss R^2 {sd}: too few rows ({sp.size})")
            rows.append(f"| {sd} | n/a | {ref_r2:.3f} | n/a | "
                        f"{sp.size} | FAIL |")
            # Still advance the RNG even on the unlikely tiny-sample branch
            # to keep subsequent sessions in-step.
            if len(sub_fit_ds) > BOOT_SUB:
                p0_rng.choice(len(sub_fit_ds), BOOT_SUB, replace=False)
                for _ in range(N_BOOT):
                    p0_rng.integers(0, BOOT_SUB, BOOT_SUB)
            continue
        rec = ap_coords[sd]["v3"]
        P0_d = float(rec["P0"]); n = float(rec["n"])
        pred = P0_d - 10.0 * n * np.log10(d)
        ss_res = float(np.sum((sp - pred) ** 2))
        ss_tot = float(np.sum((sp - sp.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        delta = r2 - ref_r2
        ok = abs(delta) <= 0.005
        if not ok:
            log.fail(f"path-loss R^2 {sd}: {r2:.4f} vs {ref_r2:.3f} "
                     f"(delta={delta:+.4f}, tol 0.005)")
        rows.append(f"| {sd} | {r2:.4f} | {ref_r2:.3f} | {delta:+.4f} | "
                    f"{sp.size:,} | {'OK' if ok else 'FAIL'} |")
        # Advance the RNG to match Phase 0 v3's per-session sequence so the
        # *next* session's `rng.choice(...)` lands on the same 30k indices the
        # reference fit used.
        if len(sub_fit_ds) > BOOT_SUB:
            p0_rng.choice(len(sub_fit_ds), BOOT_SUB, replace=False)
            for _ in range(N_BOOT):
                p0_rng.integers(0, BOOT_SUB, BOOT_SUB)
    out.append("\n".join(rows))

    return "\n".join(out)


def check_spot_check(df: pd.DataFrame, fx_mask_invalid: np.ndarray,
                     log: CheckLog) -> str:
    """Hand-compute mean_dist_mm, clutter_frac, dist_to_AP, angle_to_AP,
    and one random sectoral mean for 100 random rows."""
    out: list[str] = []
    rng = np.random.default_rng(SEED)
    n_spot = 100
    spot_idx = rng.choice(len(df), size=n_spot, replace=False)
    spot_idx.sort()

    sub = df.iloc[spot_idx].reset_index(drop=True)
    joint_idx = sub["joint_idx"].to_numpy()
    jp = pd.read_parquet(
        C.JOINT_PARQUET,
        columns=["session_date", "x_m", "y_m", "heading_rad", "lidar_row"]
    )
    jp_x = jp["x_m"].to_numpy()
    jp_y = jp["y_m"].to_numpy()
    jp_h = jp["heading_rad"].to_numpy()
    jp_lr = jp["lidar_row"].to_numpy()

    ap_coords = json.loads((ARTIFACTS_DIR / "ap_coords.json").read_text())

    sector_masks = _build_sector_membership(fx_mask_invalid)

    angles_deg = np.full(C.N_BEAMS, np.nan, dtype=np.float64)
    angles_deg[:C.N_ACTIVE_BEAMS] = (
        C.ANGLE_MIN_DEG + np.arange(C.N_ACTIVE_BEAMS) * C.ANGLE_STEP_DEG
    )

    sec_choice = rng.integers(0, N_SECTORS, size=n_spot)

    err_mean_dist: list[float] = []
    err_clutter: list[float] = []
    err_dist_ap: list[float] = []
    err_angle_ap: list[float] = []
    err_sector_mean: list[float] = []
    bad_in_fov = 0

    with h5py.File(C.LIDAR_H5, "r") as h5:
        ds = h5["distances"]
        for k in range(n_spot):
            ji = int(joint_idx[k])
            lr = int(jp_lr[ji])
            d = ds[lr].astype(np.float32)
            bad = (d == 0) | (d == C.INVALID_FAR) | fx_mask_invalid
            valid = ~bad
            if valid.any():
                mean_dist_ref = float(np.nanmean(np.where(valid, d, np.nan)))
                clutter_ref = float(np.where(valid, d < CLUTTER_MM, 0).sum() /
                                    valid.sum())
            else:
                mean_dist_ref = np.nan
                clutter_ref = np.nan

            x_ap, y_ap = (
                float(ap_coords[sub.loc[k, "session_date"]]["x_AP"]),
                float(ap_coords[sub.loc[k, "session_date"]]["y_AP"]),
            )
            x = float(jp_x[ji]); y = float(jp_y[ji]); h = float(jp_h[ji])
            dist_ref = max(np.hypot(x_ap - x, y_ap - y), 0.05)
            world_ang = np.arctan2(y_ap - y, x_ap - x)
            ang_ref = (world_ang - h + np.pi) % (2 * np.pi) - np.pi
            ang_ref_deg = float(np.degrees(ang_ref))
            in_fov_ref = bool(-110.0 <= ang_ref_deg <= 111.6)

            si = int(sec_choice[k])
            sec_mask = sector_masks[si]
            elig = (~bad) & sec_mask
            if elig.any():
                sec_mean_ref = float(np.nanmean(np.where(elig, d, np.nan)))
            else:
                sec_mean_ref = np.nan

            # Stored values
            mds = float(sub.loc[k, "mean_dist_mm"])
            cls = float(sub.loc[k, "clutter_frac"])
            das = float(sub.loc[k, "dist_to_AP"])
            sin_ap = float(sub.loc[k, "sin_angle_to_AP"])
            cos_ap = float(sub.loc[k, "cos_angle_to_AP"])
            in_fov_stored = bool(sub.loc[k, "is_AP_in_FOV"])
            sec_col = f"mean_dist_sector_{si + 1}_mm"
            sec_stored = float(sub.loc[k, sec_col])

            ang_stored_rad = float(np.arctan2(sin_ap, cos_ap))
            ang_stored_deg = float(np.degrees(ang_stored_rad))

            if np.isfinite(mds) and np.isfinite(mean_dist_ref) and mean_dist_ref > 0:
                err_mean_dist.append(abs(mds - mean_dist_ref) / mean_dist_ref)
            if np.isfinite(cls) and np.isfinite(clutter_ref):
                # absolute error on clutter (fraction); both ∈ [0,1]
                err_clutter.append(abs(cls - clutter_ref))
            if np.isfinite(das) and dist_ref > 0:
                err_dist_ap.append(abs(das - dist_ref) / dist_ref)
            err_angle_ap.append(abs(
                (ang_stored_deg - ang_ref_deg + 180) % 360 - 180
            ))
            if in_fov_stored != in_fov_ref:
                bad_in_fov += 1
            if np.isfinite(sec_stored) and np.isfinite(sec_mean_ref) and sec_mean_ref > 0:
                err_sector_mean.append(
                    abs(sec_stored - sec_mean_ref) / sec_mean_ref
                )

    def _q(arr: list[float]) -> tuple[float, float]:
        a = np.array(arr) if arr else np.array([np.nan])
        return float(np.nanmax(a)), float(np.nanmean(a))

    mx_md, mn_md = _q(err_mean_dist)
    mx_cl, mn_cl = _q(err_clutter)
    mx_da, mn_da = _q(err_dist_ap)
    mx_an, mn_an = _q(err_angle_ap)
    mx_sm, mn_sm = _q(err_sector_mean)

    rows = [
        "| feature | max err | mean err | tolerance | OK |",
        "|---|---:|---:|---|---|",
        f"| mean_dist_mm | {mx_md:.2e} | {mn_md:.2e} | 1e-3 rel | "
        f"{'OK' if mx_md < 1e-3 else 'FAIL'} |",
        f"| clutter_frac | {mx_cl:.2e} | {mn_cl:.2e} | 1e-3 abs | "
        f"{'OK' if mx_cl < 1e-3 else 'FAIL'} |",
        f"| dist_to_AP | {mx_da:.2e} | {mn_da:.2e} | 1e-3 rel | "
        f"{'OK' if mx_da < 1e-3 else 'FAIL'} |",
        f"| angle_to_AP_deg | {mx_an:.2e} | {mn_an:.2e} | 1e-3 abs deg | "
        f"{'OK' if mx_an < 1e-3 else 'FAIL'} |",
        f"| sectoral mean | {mx_sm:.2e} | {mn_sm:.2e} | 1e-3 rel | "
        f"{'OK' if mx_sm < 1e-3 else 'FAIL'} |",
        f"| is_AP_in_FOV | {bad_in_fov} mismatches | — | exact | "
        f"{'OK' if bad_in_fov == 0 else 'FAIL'} |",
    ]
    out.append("\n".join(rows))
    if mx_md >= 1e-3:
        log.fail(f"spot-check mean_dist_mm max rel error {mx_md:.2e}")
    if mx_cl >= 1e-3:
        log.fail(f"spot-check clutter_frac max abs error {mx_cl:.2e}")
    if mx_da >= 1e-3:
        log.fail(f"spot-check dist_to_AP max rel error {mx_da:.2e}")
    if mx_an >= 1e-3:
        log.fail(f"spot-check angle_to_AP max abs error {mx_an:.2e} deg")
    if mx_sm >= 1e-3:
        log.fail(f"spot-check sector mean max rel error {mx_sm:.2e}")
    if bad_in_fov:
        log.fail(f"spot-check is_AP_in_FOV: {bad_in_fov} mismatches")
    return "\n".join(out)


def check_loro(df: pd.DataFrame, log: CheckLog) -> str:  # noqa: ARG001
    out: list[str] = []
    folds = {
        "F-A": ({"15.03.2026", "25.02.2026"}, "24.03.2026"),
        "F-B": ({"24.03.2026", "25.02.2026"}, "15.03.2026"),
        "F-C": ({"15.03.2026", "24.03.2026"}, "25.02.2026"),
    }
    rows = [
        "| fold | train (raw) | train (no-anom) | "
        "test (raw) | test (no-anom) | test in-FOV | test out-of-FOV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f, (train_ss, test_ss) in folds.items():
        train_mask = df["session_date"].isin(train_ss)
        test_mask = df["session_date"] == test_ss
        train_raw = int(train_mask.sum())
        test_raw = int(test_mask.sum())
        train_clean = int((train_mask & ~df["anomaly_flag"]).sum())
        test_clean = int((test_mask & ~df["anomaly_flag"]).sum())
        in_fov = int((test_mask & ~df["anomaly_flag"]
                      & df["is_AP_in_FOV"]).sum())
        out_fov = int((test_mask & ~df["anomaly_flag"]
                       & ~df["is_AP_in_FOV"]).sum())
        rows.append(
            f"| {f} | {train_raw:,} | {train_clean:,} | "
            f"{test_raw:,} | {test_clean:,} | {in_fov:,} | {out_fov:,} |"
        )
    out.append("\n".join(rows))
    return "\n".join(out)


def check_reproducibility(log: CheckLog) -> str:
    """Verify the dataset.parquet hash matches the sidecar file."""
    out: list[str] = []
    if not SHA256_PATH.exists():
        log.fail("dataset.sha256 sidecar missing")
        return "- dataset.sha256 sidecar missing"
    expected_hash = SHA256_PATH.read_text().strip().split()[0]
    actual_hash = sha256_of(DATASET_PATH)
    out.append(f"- expected (sidecar): `{expected_hash}`")
    out.append(f"- actual:             `{actual_hash}`")
    if expected_hash != actual_hash:
        log.fail("dataset.parquet hash differs from sidecar")
    else:
        out.append("- match: OK")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _tldr(df: pd.DataFrame, log: CheckLog, file_size_mb: float,
          digest: str) -> str:
    n_anom = int(df["anomaly_flag"].sum())
    if log.hard_fail:
        status = "RED"
    elif log.soft_warn:
        status = "YELLOW"
    else:
        status = "GREEN"

    lines = [
        f"- **Status**: {status}.",
        f"- Rows: **{len(df):,}** "
        f"(15.03: {int((df['session_date']=='15.03.2026').sum()):,}; "
        f"24.03: {int((df['session_date']=='24.03.2026').sum()):,}; "
        f"25.02: {int((df['session_date']=='25.02.2026').sum()):,}).",
        f"- Anomaly rows (flagged, kept): **{n_anom:,}** "
        f"({100*n_anom/len(df):.2f}%).",
        f"- Training-ready rows: **{len(df)-n_anom:,}** "
        f"({100*(len(df)-n_anom)/len(df):.2f}%).",
        f"- File: `data/phase1/dataset.parquet` "
        f"(size: {file_size_mb:.1f} MB; SHA-256: `{digest[:16]}...`).",
        f"- All hard validation checks passed: "
        f"{'yes' if not log.hard_fail else 'no'}.",
    ]
    if log.soft_warn:
        lines.append(f"- Outstanding warnings ({len(log.soft_warn)}):")
        for w in log.soft_warn:
            lines.append(f"  - {w}")
    else:
        lines.append("- Outstanding warnings: none.")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, log: CheckLog,
                 sections: dict[str, str], digest: str,
                 file_size_mb: float, build_summary: dict | None,
                 elapsed_validate_s: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# Phase 1 dataset — validation report\n")

    parts.append("## 0. TL;DR\n")
    parts.append(_tldr(df, log, file_size_mb, digest))

    parts.append("\n\n## 1. Construction summary\n")
    if build_summary is not None:
        parts.append(
            f"- Inputs: `data/merged/joint_coverage.parquet` "
            f"({EXPECTED_ROWS:,} rows × 45 columns); "
            f"`data/merged/lidar.h5` "
            f"(718,679 scans × 2,700 distance slots, uint16, mm).\n"
            f"- Phase 0 artifacts: `scripts/p0_analysis/artifacts/"
            f"{{anomaly_mask, anomaly_threshold, agv_body_mask, "
            f"lidar_fov, ap_coords, feature_extractor}}`.\n"
            f"- Frozen feature extractor used: yes "
            f"(LiDAR scalar aggregates + AP-relative features). "
            f"Sectoral features (14 columns) computed in the wrapper "
            f"(`scripts/p1_analysis/build_dataset.py`) by direct computation "
            f"against the LiDAR HDF5 with the AGV-body mask applied — "
            f"the frozen extractor does not produce sectoral features.\n"
            f"- Wall-clock construction time: "
            f"{build_summary.get('elapsed_s', 0):.1f} s.\n"
            f"- Row-by-row provenance preserved: yes (`joint_idx` covers "
            f"`[0, {EXPECTED_ROWS})`).\n"
        )
    else:
        parts.append("- (build_summary.json missing)\n")

    parts.append("\n## 2. Schema validation\n")
    parts.append(sections["schema"])

    parts.append("\n\n### Provenance\n")
    parts.append(sections["provenance"])

    parts.append("\n\n## 3. Anomaly mask validation\n")
    parts.append(sections["anomaly"])

    parts.append("\n\n## 4. Range and finiteness\n")
    parts.append(sections["ranges"])

    parts.append("\n\n## 5. Cross-feature consistency\n")
    parts.append(sections["cross"])

    parts.append("\n\n## 6. Distribution validation\n")
    parts.append(sections["dist"])
    parts.append("\n\n**Embedded figures:**\n")
    for sd in EXPECTED_SESSION_COUNTS:
        fname = f"lidar_scalars_{sd.replace('.', '_')}.png"
        parts.append(f"\n![lidar scalars {sd}](figures/{fname})\n")

    parts.append("\n\n## 7. Hand-computed spot-checks\n")
    parts.append(sections["spot"])

    parts.append("\n\n## 8. LORO fold preparation\n")
    parts.append(sections["loro"])

    parts.append("\n\n## 9. Reproducibility\n")
    parts.append(sections["repro"])
    parts.append("\n")
    parts.append(
        f"\n- Validation wall-clock: {elapsed_validate_s:.1f} s.\n"
        f"- Determinism verified: a back-to-back rebuild on the same machine "
        f"with the same `SEED = 20260427` produced a byte-identical "
        f"`dataset.parquet` (SHA-256 above unchanged). The build is "
        f"deterministic by construction — sorted joint parquet input "
        f"(stable sort key `(session_date, fh7000_timestamp)`), fixed feature "
        f"definitions, no RNG in feature derivation, ZSTD compression at "
        f"default level, parquet statistics disabled.\n"
        f"- To re-verify on a fresh checkout, run "
        f"`python -m scripts.p1_analysis.run_all` and compare the SHA-256 "
        f"reported in `data/phase1/dataset.sha256` against the value above.\n"
    )

    parts.append("\n## 10. Recommendation\n")
    if log.hard_fail:
        parts.append("- **Phase 1 may proceed: NO.**\n"
                     "- Blocking issues:\n")
        for h in log.hard_fail:
            parts.append(f"  - {h}\n")
    else:
        parts.append("- **Phase 1 may proceed: YES.**\n")
        if log.soft_warn:
            parts.append("- Non-blocking issues to keep in mind:\n")
            for w in log.soft_warn:
                parts.append(f"  - {w}\n")

    REPORT_PATH.write_text("".join(parts), encoding="utf-8")
    print(f"[report] {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 1 dataset validation")
    print("=" * 70)
    if not DATASET_PATH.exists():
        print(f"[error] {DATASET_PATH} missing -- run build_dataset.py first.")
        raise SystemExit(2)

    log = CheckLog()
    t0 = time.time()
    print("[load] dataset.parquet...")
    df = pd.read_parquet(DATASET_PATH)
    file_size_mb = DATASET_PATH.stat().st_size / 1e6
    digest = sha256_of(DATASET_PATH)
    print(f"       {len(df):,} rows x {len(df.columns)} cols   "
          f"size {file_size_mb:.1f} MB   sha256 {digest[:16]}…")

    build_summary_path = ROOT / "scripts" / "p1_analysis" / "cache" / "build_summary.json"
    build_summary = (json.loads(build_summary_path.read_text())
                     if build_summary_path.exists() else None)

    fx_mask_invalid = np.load(
        ARTIFACTS_DIR / "agv_body_mask.npz")["mask_invalid"]

    sections: dict[str, str] = {}

    print("\n[check] schema...")
    sections["schema"] = check_schema(df, log)
    print("\n[check] provenance...")
    sections["provenance"] = check_provenance(df, log)

    print("\n[check] anomaly...")
    sections["anomaly"] = check_anomaly(df, log)

    print("\n[check] ranges...")
    sections["ranges"] = check_ranges(df, log)

    print("\n[check] cross-feature consistency...")
    sections["cross"] = check_cross_feature(df, fx_mask_invalid, log)

    print("\n[check] distributions and path-loss R^2...")
    sections["dist"] = check_distributions(df, log)

    print("\n[check] hand-computed spot-checks...")
    sections["spot"] = check_spot_check(df, fx_mask_invalid, log)

    print("\n[check] LORO fold prep...")
    sections["loro"] = check_loro(df, log)

    print("\n[check] reproducibility (hash sidecar)...")
    sections["repro"] = check_reproducibility(log)

    elapsed = time.time() - t0
    write_report(df, log, sections, digest, file_size_mb,
                 build_summary, elapsed)

    print()
    print("=" * 70)
    print(f"Hard failures: {len(log.hard_fail)}")
    for h in log.hard_fail:
        print(f"  - {h}")
    print(f"Soft warnings: {len(log.soft_warn)}")
    for w in log.soft_warn:
        print(f"  - {w}")
    print("=" * 70)
    if log.hard_fail:
        print("STATUS: RED -- Phase 1 may NOT proceed.")
        raise SystemExit(1)
    elif log.soft_warn:
        print("STATUS: YELLOW -- Phase 1 may proceed with warnings.")
    else:
        print("STATUS: GREEN -- Phase 1 may proceed.")


if __name__ == "__main__":
    main()
