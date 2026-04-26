"""Phase 5: apply per-day offsets and produce a joint-coverage dataset.

Sign convention (also stated in config.py):
    tau_hat > 0  =>  LiDAR clock lags telemetry.
    Correction   =>  add tau_hat to every LiDAR timestamp.

Correction granularity: ONE offset per ``session_date`` (per-day).
This per-day τ̂ is computed from the day's full continuous signal (all
CSV files concatenated) — see run_offsets.py.

Joint-coverage filter:
    For each LiDAR scan, find the nearest telemetry sample on the
    corrected time axis (``pandas.merge_asof`` with ``direction='nearest'``).
    The pair is retained if ``|delta_t| < tolerance``, with
        tolerance = max(LIDAR_period, TELEMETRY_period) / 2
    using each day's measured rates.

Output: ``data/merged/joint_coverage.parquet``
    One row per matched LiDAR scan, with the matched telemetry row joined
    and ``applied_tau_s`` / ``applied_tau_se_s`` carried for downstream
    uncertainty propagation.
"""
from __future__ import annotations
import json

import h5py
import numpy as np
import pandas as pd

from . import config as C


def _per_day_offsets() -> dict[str, dict]:
    h = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    return h["per_session_day"]


def _global_offset() -> dict:
    h = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    return h["included_only_pool"]


def main():
    print("=" * 72)
    print("Phase 5: applying per-day offsets and computing joint coverage")
    print("=" * 72)
    days = json.loads((C.CACHE_DIR / "days.json").read_text())
    days_by_sd = {d["session_date"]: d for d in days}
    per_day = _per_day_offsets()
    global_off = _global_offset()

    print("\nPer-day offsets (applied):")
    print(f"  {'session_date':14s}  {'tau (s)':>10s}  {'SE (s)':>10s}  "
          f"{'95% CI (s)':>22s}  rho_peak  n_csv")
    for sd, d in sorted(per_day.items()):
        ci = f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
        print(f"  {sd:14s}  {d['tau_bar']:+10.4f}  {d['se']:>10.4f}  "
              f"{ci:>22s}  {d['rho_peak']:>8.3f}  {d['n_csv_files']}")
    print()
    print(f"For comparison only (NOT applied):")
    print(f"  global pool tau_bar = {global_off['tau_bar']:+.4f} s "
          f"(95% CI [{global_off['ci_low']:+.4f}, {global_off['ci_high']:+.4f}], "
          f"Q={global_off['Q']:.3f} p={global_off['p']:.3g} I^2={global_off['I2']:.1f}%)")
    print()

    print("Loading telemetry ...")
    tel = pd.read_parquet(C.TELEMETRY_PARQUET)
    tel = tel.sort_values(["session_date", "run_file",
                           "fh7000_timestamp"]).reset_index(drop=True)
    tel["fh7000_timestamp"] = tel["fh7000_timestamp"].astype("datetime64[ns]")

    with h5py.File(C.LIDAR_H5, "r") as f:
        sd_all = pd.Series(f["session_date"][:]).astype(str).str.replace(
            r"^b'|'$", "", regex=True).values
        ts_all = f["local_timestamps"][:]
        scan_nr = f["device_scan_nr"][:]
        scan_counter = f["device_scan_counter"][:]

    matched_chunks: list[pd.DataFrame] = []
    retention_per_day: dict[str, dict] = {}
    retention_per_run: dict[str, dict] = {}

    for sd in pd.unique(sd_all):
        if sd not in per_day:
            print(f"  [skip] {sd}: no offset available")
            continue
        tau_day = float(per_day[sd]["tau_bar"])
        se_day = float(per_day[sd]["se"])
        idx = np.where(sd_all == sd)[0]
        ts_corr_unix = ts_all[idx] + tau_day
        ts_corr_dt = pd.to_datetime(ts_corr_unix, unit="s")

        lid = pd.DataFrame({
            "session_date": sd,
            "applied_tau_s": tau_day,
            "applied_tau_se_s": se_day,
            "lidar_row": idx.astype(np.int64),
            "lidar_scan_nr": scan_nr[idx],
            "lidar_scan_counter": scan_counter[idx],
            "lidar_ts_raw_cet_unix": ts_all[idx],
            "lidar_ts_corr_unix": ts_corr_unix,
            "lidar_ts_corr": ts_corr_dt,
        }).sort_values("lidar_ts_corr").reset_index(drop=True)
        lid["lidar_ts_corr"] = lid["lidar_ts_corr"].astype("datetime64[ns]")

        # Per-run window assignment within the day (so we can carry run_file
        # downstream, like the obsolete pipeline). Run windows come from the
        # per-CSV start/end timestamps in the cleaned telemetry.
        sd_runs = (tel[tel["session_date"] == sd]
                   .groupby("run_file")["fh7000_timestamp"]
                   .agg(["min", "max"])
                   .reset_index())
        run_starts = sd_runs["min"].values.astype("datetime64[ns]")
        run_ends = sd_runs["max"].values.astype("datetime64[ns]")
        run_files = sd_runs["run_file"].tolist()

        lid_run_idx = np.full(len(lid), -1, dtype=np.int32)
        lid_ts = lid["lidar_ts_corr"].values.astype("datetime64[ns]")
        for k in range(len(run_files)):
            in_run = (lid_ts >= run_starts[k]) & (lid_ts <= run_ends[k])
            lid_run_idx[in_run] = k
        run_files_arr = np.array(run_files + [None], dtype=object)
        lid["run_file"] = run_files_arr[np.where(lid_run_idx >= 0,
                                                 lid_run_idx,
                                                 len(run_files))]
        before = len(lid)
        lid = lid[lid["run_file"].notna()].reset_index(drop=True)
        after = len(lid)
        print(f"\n[{sd}] {after:,}/{before:,} LiDAR scans inside CSV-run windows  "
              f"(applied tau = {tau_day:+.4f} s)")

        # ---- Per-run merge_asof using per-day tolerance ----
        meta_day = days_by_sd[sd]
        tel_period = float(meta_day["tel_period_s"])
        lid_period = float(meta_day["lid_period_s"])
        tol_s = max(lid_period, tel_period) / 2.0
        tol = pd.Timedelta(seconds=tol_s)

        day_n_scans = 0; day_n_matched = 0
        for rf in lid["run_file"].unique():
            run_lid = lid[lid["run_file"] == rf].copy()
            run_lid["match_tolerance_s"] = tol_s

            run_tel = tel[(tel["session_date"] == sd) & (tel["run_file"] == rf)].copy()
            run_tel = run_tel.drop(columns=["session_date", "run_file"])
            run_tel = run_tel.sort_values("fh7000_timestamp")
            run_lid = run_lid.sort_values("lidar_ts_corr")

            merged = pd.merge_asof(
                run_lid, run_tel,
                left_on="lidar_ts_corr", right_on="fh7000_timestamp",
                direction="nearest", tolerance=tol,
            )
            n_scans = len(run_lid)
            n_matched = int(merged["fh7000_timestamp"].notna().sum())
            retention = n_matched / n_scans if n_scans else 0.0
            matched = merged[merged["fh7000_timestamp"].notna()].copy()
            matched["delta_t_s"] = (
                matched["lidar_ts_corr"].astype("datetime64[ns]")
                - matched["fh7000_timestamp"]
            ).dt.total_seconds()

            d = matched["delta_t_s"]
            stats = {
                "mean_ms": float(d.mean() * 1000.0),
                "std_ms": float(d.std() * 1000.0),
                "abs_mean_ms": float(d.abs().mean() * 1000.0),
                "max_abs_ms": float(d.abs().max() * 1000.0),
                "p95_abs_ms": float(d.abs().quantile(0.95) * 1000.0),
            }
            retention_per_run[f"{sd}|{rf}"] = {
                "session_date": sd, "run_file": rf,
                "applied_tau_s": tau_day, "applied_tau_se_s": se_day,
                "tolerance_s": tol_s, "tel_period_s": tel_period,
                "lid_period_s": lid_period,
                "n_scans": n_scans, "n_matched": n_matched,
                "retention": retention, "delta_t_stats": stats,
            }
            day_n_scans += n_scans; day_n_matched += n_matched
            print(f"  {rf}: match={n_matched:,}/{n_scans:,} "
                  f"({retention:.1%}, tol={tol_s*1000:.1f} ms);  "
                  f"|dt|: mean={stats['abs_mean_ms']:.1f} ms, "
                  f"p95={stats['p95_abs_ms']:.1f} ms, max={stats['max_abs_ms']:.1f} ms")
            matched_chunks.append(matched)
        retention_per_day[sd] = {
            "session_date": sd,
            "applied_tau_s": tau_day, "applied_tau_se_s": se_day,
            "tolerance_s": tol_s, "tel_period_s": tel_period,
            "lid_period_s": lid_period,
            "n_scans": day_n_scans, "n_matched": day_n_matched,
            "retention": (day_n_matched / day_n_scans) if day_n_scans else 0.0,
        }

    if not matched_chunks:
        raise RuntimeError("No matched chunks produced. Cache empty?")
    out_df = pd.concat(matched_chunks, ignore_index=True)
    front = ["session_date", "run_file",
             "applied_tau_s", "applied_tau_se_s", "match_tolerance_s",
             "lidar_row", "lidar_scan_nr", "lidar_scan_counter",
             "lidar_ts_raw_cet_unix", "lidar_ts_corr_unix",
             "lidar_ts_corr", "fh7000_timestamp", "delta_t_s"]
    rest = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + rest]
    print(f"\nJoint coverage: {len(out_df):,} matched LiDAR-telemetry pairs")

    # ---- Post-hoc validation ----
    print("\n" + "=" * 72)
    print("Post-hoc validation of the regenerated joint dataset")
    print("=" * 72)
    issues: list[str] = []

    n_null_tel = out_df["fh7000_timestamp"].isna().sum()
    print(f"  V1 telemetry timestamp null check : {n_null_tel} nulls "
          + ("OK" if n_null_tel == 0 else "FAIL"))
    if n_null_tel:
        issues.append(f"V1: {n_null_tel} unmatched rows")

    over_tol = (out_df["delta_t_s"].abs() > out_df["match_tolerance_s"]).sum()
    print(f"  V2 |delta_t_s| <= match_tolerance  : {over_tol} violations "
          + ("OK" if over_tol == 0 else "FAIL"))
    if over_tol:
        issues.append(f"V2: {over_tol} rows over tolerance")

    bad_tau_rows = 0
    for sd_, grp in out_df.groupby("session_date"):
        ref = per_day[sd_]["tau_bar"]
        if not np.allclose(grp["applied_tau_s"].values, ref, atol=1e-9):
            bad_tau_rows += len(grp)
    print(f"  V3 applied_tau_s consistency       : {bad_tau_rows} mismatches "
          + ("OK" if bad_tau_rows == 0 else "FAIL"))
    if bad_tau_rows:
        issues.append(f"V3: {bad_tau_rows} rows mismatch reference tau")

    drift = (out_df["lidar_ts_corr_unix"]
             - (out_df["lidar_ts_raw_cet_unix"] + out_df["applied_tau_s"])).abs().max()
    print(f"  V4 ts_corr = ts_raw + tau identity : max |drift|={drift:.3g} s "
          + ("OK" if drift < 1e-6 else "FAIL"))
    if drift >= 1e-6:
        issues.append(f"V4: identity violated by {drift:g} s")

    print("  V5 mean delta_t per session_date:")
    for sd_, grp in out_df.groupby("session_date"):
        m = grp["delta_t_s"].mean() * 1000
        s = grp["delta_t_s"].std() * 1000
        print(f"      {sd_}: mean = {m:+.3f} ms, std = {s:.3f} ms")

    overall_n = sum(v["n_scans"] for v in retention_per_day.values())
    overall_m = sum(v["n_matched"] for v in retention_per_day.values())
    overall_ret = overall_m / max(1, overall_n)
    print(f"  V6 overall retention               : {overall_ret*100:.2f}% "
          f"({overall_m:,}/{overall_n:,})")

    print(f"\nWriting {C.JOINT_PARQUET} ...")
    out_df.to_parquet(C.JOINT_PARQUET, compression="zstd", index=False)
    print(f"  size on disk: {C.JOINT_PARQUET.stat().st_size / 1e6:.1f} MB")

    ret_path = C.CACHE_DIR / "retention_per_day.json"
    ret_path.write_text(json.dumps({
        "per_day": list(retention_per_day.values()),
        "per_run": list(retention_per_run.values()),
        "overall": {"n_scans": overall_n, "n_matched": overall_m,
                    "retention": overall_ret},
        "validation_issues": issues,
    }, indent=2))
    print(f"  wrote {ret_path}")
    if issues:
        print("\n[WARNING] Validation issues detected:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n[OK] All post-hoc validations passed.")
    print("\nPhase 5 DONE")


if __name__ == "__main__":
    main()
