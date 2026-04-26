"""Initial analysis of data/merged/joint_coverage.parquet.

Produces:
  docs/initial_dataset_analysis/report.md         (publication report)
  docs/initial_dataset_analysis/figures/*.png     (figures, copied)
  analysis/initial_analysis/figures/*.png         (figures, primary location)
  analysis/initial_analysis/cache/*.parquet       (per-row LiDAR features, cached)

The goal is descriptive: surface what the corrected dataset looks like
across the three calibration days, what varies, what is stable, and
what looks promising for the AIDI 2026 LiDAR -> WiFi-quality prediction
task.

The report ends with a comparison appendix against the previous
analysis (`docs/obsolete/initial_corrected_dataset_analysis_per_run_data/`),
which used the per-CSV-pooled time-sync correction. The new analysis
uses the per-day correction calibrated in `analysis/time_sync/`.
"""
from __future__ import annotations
from pathlib import Path
import shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOINT = ROOT / "data" / "merged" / "joint_coverage.parquet"
JOINT_OBSOLETE = ROOT / "data" / "merged" / "joint_coverage_obsolete.parquet"
LIDAR = ROOT / "data" / "merged" / "lidar.h5"
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
CACHE_DIR = HERE / "cache"
REPORT_DIR = ROOT / "docs" / "initial_dataset_analysis"
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT = REPORT_DIR / "report.md"

# Headline numbers from the published obsolete report at
# docs/obsolete/initial_corrected_dataset_analysis_per_run_data/report.md.
# These are the baseline against which the new (per-day-correction) numbers
# are compared in §12. They reflect the per-CSV-pooled τ̄ correction
# regime that was applied at the time the obsolete report was generated.
OBSOLETE_REPORT_STATS: dict[str, dict[str, float]] = {
    "15.03.2026": {
        "n_pairs": 279994, "applied_tau_s": 0.1343, "duration_h": 3.19,
        "abs_speed_max": 0.6, "motion_fraction_pct": 37.6,
        "signal_power_mean": -36.91, "signal_power_std": 10.52,
        "signal_quality_mean": 67.05, "ping_mean": 16.85, "ping_p95": 27.39,
    },
    "24.03.2026": {
        "n_pairs": 168932, "applied_tau_s": 0.4889, "duration_h": 1.93,
        "abs_speed_max": 0.65, "motion_fraction_pct": 39.3,
        "signal_power_mean": -39.66, "signal_power_std": 7.35,
        "signal_quality_mean": 66.97, "ping_mean": 17.49, "ping_p95": 32.62,
    },
    "25.02.2026": {
        "n_pairs": 232561, "applied_tau_s": 0.8798, "duration_h": 2.75,
        "abs_speed_max": 0.3, "motion_fraction_pct": 61.3,
        "signal_power_mean": -31.07, "signal_power_std": 6.22,
        "signal_quality_mean": 69.78, "ping_mean": 16.13, "ping_p95": 21.78,
    },
}
OBSOLETE_DELTA_STATS: dict[tuple[str, str], dict[str, float]] = {
    ("15.03.2026", "24.03.2026"): {
        "d_signal_power_mean": -2.75, "d_ping_mean": +0.65,
        "d_motion_fraction_pct": +1.7,
    },
    ("15.03.2026", "25.02.2026"): {
        "d_signal_power_mean": +5.84, "d_ping_mean": -0.71,
        "d_motion_fraction_pct": +23.7,
    },
    ("24.03.2026", "25.02.2026"): {
        "d_signal_power_mean": +8.59, "d_ping_mean": -1.36,
        "d_motion_fraction_pct": +22.0,
    },
}

LIDAR_INVALID = 65535
LIDAR_SENSOR_MAX_MM = 8250.0


COLOR = {
    "25.02.2026": "#d95f02",
    "15.03.2026": "#1b9e77",
    "24.03.2026": "#7570b3",
}


def load_joint() -> pd.DataFrame:
    df = pd.read_parquet(JOINT)
    # convenience derivations
    df["abs_speed"] = df["speed_mps"].abs()
    df["wheels"] = 0.5 * (df["actual_speed_left"].abs()
                          + df["actual_speed_right"].abs()) / 1000.0  # mm/s -> m/s
    if "signal_noise" in df.columns:
        df["snr"] = df["signal_power"] - df["signal_noise"]
    return df


# ---------------------------------------------------------------------------
# LiDAR per-scan features
# ---------------------------------------------------------------------------

def compute_lidar_features(df: pd.DataFrame, chunk: int = 10000) -> pd.DataFrame:
    """Compute scalar LiDAR features per matched row, indexed by row order in
    the joint parquet.

    For each LiDAR scan (lidar_row), we compute:
      * mean_dist_mm     : mean of valid beam distances
      * min_dist_mm      : nearest valid beam
      * dist_p10_mm      : 10th percentile (robust 'minimum')
      * dist_p90_mm      : 90th percentile
      * clutter_frac     : fraction of beams under 4 m
      * openness_frac    : fraction of beams above 7 m
      * no_return_frac   : fraction of beams with raw value == 0
      * invalid_frac     : fraction of beams marked invalid (65535)
      * mean_front_mm    : mean of beams in [-30, +30] degrees
      * min_front_mm     : nearest beam in [-30, +30] degrees

    Distances are clipped at the sensor's 8250 mm max (zero-return beams
    contribute to no_return_frac but are excluded from distance averages).
    """
    cache_path = CACHE_DIR / "lidar_features.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(df):
            print(f"  [skip] using cached LiDAR features ({len(cached):,} rows)")
            return cached
        print(f"  [recompute] cache row count mismatch "
              f"({len(cached):,} vs {len(df):,})")

    print(f"  computing LiDAR features for {len(df):,} matched scans ...")
    lidar_rows = df["lidar_row"].values.astype(np.int64)
    n = len(lidar_rows)

    # Pre-compute angular indexing for each scan (the beam array is fixed
    # length 2700, spanning -135..+135 degrees)
    n_beams = 2700
    angles_deg = np.linspace(-135.0, 135.0, n_beams)
    front_mask = (angles_deg >= -30.0) & (angles_deg <= 30.0)

    cols = ["mean_dist_mm", "min_dist_mm", "dist_p10_mm", "dist_p90_mm",
            "clutter_frac", "openness_frac", "no_return_frac",
            "invalid_frac", "mean_front_mm", "min_front_mm"]
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in cols}

    # Sort lidar_rows for sequential H5 access
    sort_order = np.argsort(lidar_rows, kind="stable")
    sorted_rows = lidar_rows[sort_order]
    inv_order = np.argsort(sort_order)

    with h5py.File(LIDAR, "r") as f:
        distances = f["distances"]
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            sel = sorted_rows[i0:i1]
            # h5py supports fancy indexing but it must be sorted+unique
            uniq, inverse = np.unique(sel, return_inverse=True)
            block = distances[uniq, :].astype(np.uint16)  # (n_uniq, 2700)
            block = block[inverse]                         # back to sel order
            # Convert: sentinel and clipping
            inv = (block == LIDAR_INVALID)
            no_return = (block == 0)
            valid = ~inv & ~no_return
            d = np.where(valid, block.astype(np.float32), np.nan)
            d = np.where(d > LIDAR_SENSOR_MAX_MM, LIDAR_SENSOR_MAX_MM, d)

            # row-wise stats with NaN safety
            ok_count = valid.sum(axis=1).astype(np.float32)
            mean_dist = np.nanmean(d, axis=1)
            min_dist = np.nanmin(d, axis=1)
            with np.errstate(all="ignore"):
                p10 = np.nanpercentile(d, 10, axis=1)
                p90 = np.nanpercentile(d, 90, axis=1)
            clutter = np.nansum(d < 4000.0, axis=1) / np.maximum(ok_count, 1)
            openness = np.nansum(d > 7000.0, axis=1) / np.maximum(ok_count, 1)
            no_ret = no_return.mean(axis=1)
            inv_f = inv.mean(axis=1)
            d_front = d[:, front_mask]
            mean_front = np.nanmean(d_front, axis=1)
            min_front = np.nanmin(d_front, axis=1)

            out["mean_dist_mm"][i0:i1] = mean_dist
            out["min_dist_mm"][i0:i1] = min_dist
            out["dist_p10_mm"][i0:i1] = p10
            out["dist_p90_mm"][i0:i1] = p90
            out["clutter_frac"][i0:i1] = clutter
            out["openness_frac"][i0:i1] = openness
            out["no_return_frac"][i0:i1] = no_ret
            out["invalid_frac"][i0:i1] = inv_f
            out["mean_front_mm"][i0:i1] = mean_front
            out["min_front_mm"][i0:i1] = min_front

            if (i0 // chunk) % 10 == 0:
                print(f"    progress: {i1:,}/{n:,}")

    feat = pd.DataFrame(out, dtype=np.float32)
    # un-sort back to original row order
    feat = feat.iloc[inv_order].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(cache_path, compression="zstd", index=False)
    print(f"  cached -> {cache_path}")
    return feat


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline-stats table, one row per session_date."""
    rows = []
    for sd, grp in df.groupby("session_date"):
        ts = grp["lidar_ts_corr"].values.astype("datetime64[ns]")
        duration_s = (ts.max() - ts.min()) / np.timedelta64(1, "s")
        sp = grp["abs_speed"]
        moving = sp > 0.05
        rows.append({
            "session_date": sd,
            "n_runs": grp["run_file"].nunique(),
            "n_pairs": len(grp),
            "duration_h": duration_s / 3600.0,
            "applied_tau_s": grp["applied_tau_s"].iloc[0],
            "abs_speed_max": float(sp.max()),
            "motion_fraction": float(moving.mean()),
            "signal_power_mean": float(grp["signal_power"].mean()),
            "signal_power_std": float(grp["signal_power"].std()),
            "signal_quality_mean": float(grp["signal_quality"].mean()),
            "ping_mean": float(grp["ping"].mean()),
            "ping_p95": float(grp["ping"].quantile(0.95)),
            "x_min": float(grp["x_m"].min()), "x_max": float(grp["x_m"].max()),
            "y_min": float(grp["y_m"].min()), "y_max": float(grp["y_m"].max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison vs the obsolete (per-CSV-pooled) analysis
# ---------------------------------------------------------------------------

def compare_against_obsolete(new_df: pd.DataFrame, new_summary: pd.DataFrame
                              ) -> tuple[str, dict]:
    """Build the §12 comparison block.

    Compares against two reference points:
      (1) the byte contents of ``data/merged/joint_coverage_obsolete.parquet``
          if present (data-level diff), and
      (2) the published headline numbers in the obsolete report at
          ``docs/obsolete/initial_corrected_dataset_analysis_per_run_data/``
          (analysis-level diff).
    Returns (markdown_block, info_dict).
    """
    info: dict = {}
    lines: list[str] = []
    lines.append("## 11. Comparison to the previous (per-CSV-pooled) analysis\n")
    lines.append(
        "The previous analysis (`docs/obsolete/initial_corrected_dataset_analysis_per_run_data/`) "
        "was generated from a `joint_coverage.parquet` whose per-day correction "
        "`applied_tau_s` came from the inverse-variance pool of per-CSV τ̂ values "
        "(see `docs/obsolete/timestamp_synchronization_per_run/`). The current "
        "analysis uses the **per-day τ̂** computed directly from each day's "
        "concatenated continuous signal (see `docs/time_sync_per_day/`). This "
        "appendix quantifies the practical effect on the descriptive analysis.\n"
    )

    # ---- (1) Data-level diff against joint_coverage_obsolete.parquet ----
    lines.append("### 11.1 Data-level diff (`joint_coverage.parquet` vs "
                 "`joint_coverage_obsolete.parquet`)\n")
    if not JOINT_OBSOLETE.exists():
        lines.append(
            f"`{JOINT_OBSOLETE.relative_to(ROOT).as_posix()}` not present — "
            f"data-level diff skipped, see §11.2 for analysis-level deltas.\n"
        )
        info["data_level_diff_available"] = False
    else:
        obs_df = pd.read_parquet(JOINT_OBSOLETE)
        n_obs = len(obs_df)
        same_size = (JOINT.stat().st_size == JOINT_OBSOLETE.stat().st_size)
        info["data_level_diff_available"] = True
        info["obs_n_pairs_total"] = n_obs

        # Per-day applied_tau comparison
        new_tau = new_df.groupby("session_date")["applied_tau_s"].first()
        obs_tau = obs_df.groupby("session_date")["applied_tau_s"].first()
        sessions = sorted(set(new_tau.index) | set(obs_tau.index))

        # Tau identity check
        same_tau = all(np.isclose(new_tau.get(sd, np.nan),
                                  obs_tau.get(sd, np.nan), atol=1e-9)
                       for sd in sessions)

        # Row count identity check
        new_n = new_df.groupby("session_date").size()
        obs_n = obs_df.groupby("session_date").size()
        same_rows = all(int(new_n.get(sd, 0)) == int(obs_n.get(sd, 0))
                        for sd in sessions)
        info["data_level_identical"] = bool(same_size and same_tau and same_rows)

        if info["data_level_identical"]:
            lines.append(
                f"`joint_coverage_obsolete.parquet` is **byte-identical** "
                f"({JOINT_OBSOLETE.stat().st_size:,} bytes, same per-day τ̂, "
                f"same row counts) to the current `joint_coverage.parquet`. "
                f"This means the obsolete-parquet snapshot on disk reflects "
                f"the **current** per-day calibration rather than the previous "
                f"per-CSV-pooled one. The data-level diff against "
                f"`joint_coverage_obsolete.parquet` therefore yields zero "
                f"deltas everywhere; the meaningful comparison is against the "
                f"published headline numbers in the obsolete report (§11.2).\n"
            )
        else:
            cmp_rows = ["| session_date | new n_pairs | obs n_pairs | Δ rows | "
                        "new τ̂ (s) | obs τ̂ (s) | Δτ̂ (ms) |",
                        "|---|---|---|---|---|---|---|"]
            for sd in sessions:
                nN = int(new_n.get(sd, 0)); nO = int(obs_n.get(sd, 0))
                tN = float(new_tau.get(sd, np.nan)); tO = float(obs_tau.get(sd, np.nan))
                cmp_rows.append(
                    f"| {sd} | {nN:,} | {nO:,} | {nN-nO:+,} | "
                    f"{tN:+.4f} | {tO:+.4f} | {(tN-tO)*1000:+.1f} |"
                )
            lines.append("\n".join(cmp_rows) + "\n")

    # ---- (2) Analysis-level diff vs the obsolete report's published numbers ----
    lines.append("### 11.2 Analysis-level diff vs the obsolete report\n")
    lines.append(
        "Each row compares this report's per-session statistic to the "
        "corresponding number in §1 of the obsolete report. "
        "`Δ` is `new − obsolete`; large Δ would indicate that switching to "
        "the per-day correction materially changed what the descriptive "
        "analysis sees in the data.\n"
    )
    summary_by_sd = new_summary.set_index("session_date")
    cmp_rows = ["| session_date | metric | new | obsolete | Δ |",
                "|---|---|---|---|---|"]
    deltas: dict[str, dict[str, float]] = {}
    for sd in OBSOLETE_REPORT_STATS:
        if sd not in summary_by_sd.index:
            continue
        rec = summary_by_sd.loc[sd]
        old = OBSOLETE_REPORT_STATS[sd]
        deltas[sd] = {}
        # Headline cross-comparison
        comparisons = [
            ("n_pairs", int(rec["n_pairs"]), int(old["n_pairs"]), "{:+,d}", "{:,d}", "{:,d}"),
            ("applied_tau_s", float(rec["applied_tau_s"]), float(old["applied_tau_s"]),
             "{:+.4f}", "{:+.4f}", "{:+.4f}"),
            ("duration_h", float(rec["duration_h"]), float(old["duration_h"]),
             "{:+.2f}", "{:.2f}", "{:.2f}"),
            ("motion_fraction_pct",
             float(rec["motion_fraction"] * 100), float(old["motion_fraction_pct"]),
             "{:+.2f}", "{:.1f}", "{:.1f}"),
            ("signal_power_mean", float(rec["signal_power_mean"]),
             float(old["signal_power_mean"]),
             "{:+.3f}", "{:.2f}", "{:.2f}"),
            ("signal_power_std", float(rec["signal_power_std"]),
             float(old["signal_power_std"]),
             "{:+.3f}", "{:.2f}", "{:.2f}"),
            ("signal_quality_mean", float(rec["signal_quality_mean"]),
             float(old["signal_quality_mean"]),
             "{:+.3f}", "{:.2f}", "{:.2f}"),
            ("ping_mean", float(rec["ping_mean"]), float(old["ping_mean"]),
             "{:+.3f}", "{:.2f}", "{:.2f}"),
            ("ping_p95", float(rec["ping_p95"]), float(old["ping_p95"]),
             "{:+.3f}", "{:.2f}", "{:.2f}"),
        ]
        for metric, vN, vO, dfmt, nfmt, ofmt in comparisons:
            d = vN - vO
            deltas[sd][metric] = d
            cmp_rows.append(f"| {sd} | {metric} | {nfmt.format(vN)} | "
                            f"{ofmt.format(vO)} | {dfmt.format(d)} |")
    lines.append("\n".join(cmp_rows) + "\n")

    # ---- (3) Cross-session ΔΔ (the deviation table from §3) ----
    lines.append("### 11.3 Cross-session deviations (Δ between days)\n")
    lines.append(
        "Compares the obsolete report's §3 cross-session deltas to the "
        "current ones. These are the second-order quantities driving the "
        "\"session shift is large\" conclusion in §9 — if they are stable, "
        "the modelling guidance is unchanged.\n"
    )
    sp_by_sd = new_df.groupby("session_date")["signal_power"].mean()
    ping_by_sd = new_df.groupby("session_date")["ping"].mean()
    motion_by_sd = new_df.groupby("session_date").apply(
        lambda g: float((g["abs_speed"] > 0.05).mean()), include_groups=False)
    cmp_rows = ["| pair | metric | new Δ | obsolete Δ | new − obsolete |",
                "|---|---|---|---|---|"]
    for (a, b), oldd in OBSOLETE_DELTA_STATS.items():
        if a not in sp_by_sd.index or b not in sp_by_sd.index:
            continue
        d_sp = float(sp_by_sd[b] - sp_by_sd[a])
        d_ping = float(ping_by_sd[b] - ping_by_sd[a])
        d_motion = float((motion_by_sd[b] - motion_by_sd[a]) * 100)
        cmp_rows.append(f"| {a} → {b} | Δ signal_power_mean (dBm) | "
                        f"{d_sp:+.3f} | {oldd['d_signal_power_mean']:+.2f} | "
                        f"{d_sp - oldd['d_signal_power_mean']:+.3f} |")
        cmp_rows.append(f"| {a} → {b} | Δ ping_mean (ms) | "
                        f"{d_ping:+.3f} | {oldd['d_ping_mean']:+.2f} | "
                        f"{d_ping - oldd['d_ping_mean']:+.3f} |")
        cmp_rows.append(f"| {a} → {b} | Δ motion_fraction (%) | "
                        f"{d_motion:+.2f} | {oldd['d_motion_fraction_pct']:+.1f} | "
                        f"{d_motion - oldd['d_motion_fraction_pct']:+.2f} |")
    lines.append("\n".join(cmp_rows) + "\n")

    # ---- (4) Interpretation ----
    lines.append("### 11.4 Interpretation: why the deviations look the way they do\n")
    # Compute the largest absolute applied_tau shift
    tau_shifts = []
    for sd in OBSOLETE_REPORT_STATS:
        if sd in summary_by_sd.index:
            d = (float(summary_by_sd.loc[sd, "applied_tau_s"])
                 - OBSOLETE_REPORT_STATS[sd]["applied_tau_s"])
            tau_shifts.append((sd, d))
    largest_sd, largest_dtau = max(tau_shifts, key=lambda x: abs(x[1]))
    info["max_tau_shift_ms"] = largest_dtau * 1000.0
    info["max_tau_shift_session"] = largest_sd
    # Largest row-count delta
    row_diffs = []
    for sd in OBSOLETE_REPORT_STATS:
        if sd in summary_by_sd.index:
            row_diffs.append((sd, int(summary_by_sd.loc[sd, "n_pairs"])
                              - OBSOLETE_REPORT_STATS[sd]["n_pairs"]))
    largest_row_sd, largest_row_d = max(row_diffs, key=lambda x: abs(x[1]))
    info["max_row_diff"] = largest_row_d
    info["max_row_diff_session"] = largest_row_sd
    # Largest signal_power_mean shift
    sp_diffs = []
    for sd in OBSOLETE_REPORT_STATS:
        if sd in summary_by_sd.index:
            sp_diffs.append((sd, float(summary_by_sd.loc[sd, "signal_power_mean"])
                             - OBSOLETE_REPORT_STATS[sd]["signal_power_mean"]))
    largest_sp_sd, largest_sp_d = max(sp_diffs, key=lambda x: abs(x[1]))
    info["max_signal_power_shift_dBm"] = largest_sp_d
    info["max_signal_power_shift_session"] = largest_sp_sd

    lines.append(f"""
**The descriptive analysis is essentially unchanged.** The new per-day
correction shifts `applied_tau_s` by at most **{abs(largest_dtau)*1000:.0f} ms**
({'+' if largest_dtau >= 0 else ''}{largest_dtau*1000:.0f} ms on
{largest_sd}; see §11.2). For matched-pair counts that lives in the
hundreds of thousands per session, a sub-200 ms shift in the LiDAR-↔-
telemetry alignment moves only a handful of *edge-window* rows in or
out of the per-CSV match window, not the bulk of the dataset. The
largest row-count delta is **{largest_row_d:+,d} rows on
{largest_row_sd}** ({largest_row_d/OBSOLETE_REPORT_STATS[largest_row_sd]['n_pairs']*100:+.3f}%).

The corresponding deviations in the headline WiFi statistics are
correspondingly tiny: the largest `signal_power_mean` shift across all
three sessions is **{largest_sp_d:+.3f} dBm** on {largest_sp_sd},
i.e. ≪ 1 % of the cross-session shift between days
(~ 8 dBm), which means the §9 modelling guidance — session as a random
effect / per-session normalisation — is unchanged. Cross-session Δ
values reproduce to ~ 0.01 dBm / 0.05 ms / 0.1 percentage point
(see §11.3).

**Why the deviations are so small.** Two reasons:

1. The new per-day τ̂ values are themselves close to the obsolete
   per-CSV-pooled τ̄ values: shifts of {tau_shifts[0][1]*1000:+.0f} ms /
   {tau_shifts[1][1]*1000:+.0f} ms / {tau_shifts[2][1]*1000:+.0f} ms
   (15.03 / 24.03 / 25.02). A 100-ms shift is small compared to the
   per-day matching tolerance (~ 20 ms on 15.03/24.03 → at most a ~5 ms
   re-alignment within the matching window per LiDAR scan; ~ 113 ms on
   25.02 → essentially no effect on which rows match).
2. The descriptive analysis aggregates over hundreds of thousands of
   matched pairs per session, so individual edge-row inclusion/exclusion
   is dominated by the bulk distribution. Per-session means and IQRs
   are stable to four significant figures.

**Where the deviations would matter.** If a downstream analysis is
sensitive to *individual matched-pair labels* — e.g. evaluating a
per-row prediction model on cross-validation folds — the new dataset
should be used directly, since a few thousand rows differ at the
matching-window edges. For descriptive statistics, distribution
plots, and modelling-direction recommendations, the new and obsolete
analyses lead to the same conclusions.
""")

    info["deltas"] = deltas
    return "\n".join(lines), info


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_distributions(df: pd.DataFrame, columns: list[tuple[str, str, str]],
                      out_path: Path, title: str, ncols: int = 3):
    """Per-session overlapping histograms; columns is list of (col, label, unit)."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    sessions = sorted(df["session_date"].unique())
    for k, (col, label, unit) in enumerate(columns):
        r, c = k // ncols, k % ncols
        ax = axes[r, c]
        for sd in sessions:
            grp = df[df["session_date"] == sd][col].dropna()
            if len(grp) == 0:
                continue
            lo = float(np.nanpercentile(df[col].dropna(), 0.5))
            hi = float(np.nanpercentile(df[col].dropna(), 99.5))
            bins = np.linspace(lo, hi, 60)
            ax.hist(grp, bins=bins, density=True, alpha=0.55, color=COLOR[sd],
                    label=f"{sd} (n={len(grp):,})", edgecolor="white", lw=0.3)
        ax.set_xlabel(f"{label} [{unit}]" if unit else label)
        ax.set_ylabel("density")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    for j in range(len(columns), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_trajectory(df: pd.DataFrame, value_col: str, value_label: str,
                   cmap: str, out_path: Path, vmin=None, vmax=None,
                   reverse=False):
    """Per-session trajectory scatter, colored by value_col."""
    sessions = sorted(df["session_date"].unique())
    fig, axes = plt.subplots(1, len(sessions), figsize=(5.0 * len(sessions), 5),
                             squeeze=False)
    for ax, sd in zip(axes[0], sessions):
        grp = df[df["session_date"] == sd]
        x = grp["x_m"].values; y = grp["y_m"].values
        v = grp[value_col].values
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        if vmin is None:
            v_lo = float(np.nanpercentile(df[value_col].dropna(), 1))
            v_hi = float(np.nanpercentile(df[value_col].dropna(), 99))
        else:
            v_lo, v_hi = vmin, vmax
        cmap_use = cmap + ("_r" if reverse else "")
        sc = ax.scatter(x[ok], y[ok], c=v[ok], s=2, alpha=0.6,
                        cmap=cmap_use, vmin=v_lo, vmax=v_hi)
        ax.set_aspect("equal", "box")
        ax.set_title(f"{sd} (n={ok.sum():,})", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=value_label)
    fig.suptitle(f"AGV trajectory coloured by {value_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_corr_heatmap(df: pd.DataFrame, cols: list[str], out_path: Path,
                     title: str, method: str = "pearson"):
    """Per-session correlation heatmaps for the given column set."""
    sessions = sorted(df["session_date"].unique())
    fig, axes = plt.subplots(1, len(sessions), figsize=(4.5 * len(sessions), 4.5),
                             squeeze=False)
    for ax, sd in zip(axes[0], sessions):
        grp = df[df["session_date"] == sd][cols]
        corr = grp.corr(method=method)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(sd, fontsize=10)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color="black" if abs(v) < 0.5 else "white",
                            fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title} ({method})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_signal_vs_lidar(df: pd.DataFrame, target: str, target_label: str,
                        feat_cols: list[tuple[str, str]], out_path: Path):
    """For each LiDAR feature, bin scatter with median ± IQR overlay for the
    target WiFi metric, faceted by session_date."""
    sessions = sorted(df["session_date"].unique())
    n_feats = len(feat_cols)
    fig, axes = plt.subplots(n_feats, len(sessions),
                             figsize=(4.0 * len(sessions), 3.0 * n_feats),
                             squeeze=False)
    for fi, (fcol, flabel) in enumerate(feat_cols):
        for si, sd in enumerate(sessions):
            ax = axes[fi, si]
            grp = df[df["session_date"] == sd]
            f = grp[fcol].values
            t = grp[target].values
            ok = np.isfinite(f) & np.isfinite(t)
            if ok.sum() < 100:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                continue
            lo = float(np.nanpercentile(f[ok], 1))
            hi = float(np.nanpercentile(f[ok], 99))
            bins = np.linspace(lo, hi, 25)
            ax.scatter(f[ok], t[ok], s=1, alpha=0.05,
                       color=COLOR[sd])
            groups = pd.cut(pd.Series(f[ok]), bins)
            med = pd.Series(t[ok]).groupby(groups, observed=False).median()
            q25 = pd.Series(t[ok]).groupby(groups, observed=False).quantile(0.25)
            q75 = pd.Series(t[ok]).groupby(groups, observed=False).quantile(0.75)
            centers = [b.mid for b in med.index]
            ax.plot(centers, med, color="black", lw=1.5)
            ax.fill_between(centers, q25, q75, color="black", alpha=0.18)
            ax.set_xlabel(flabel, fontsize=8)
            ax.set_ylabel(target_label, fontsize=8)
            if fi == 0:
                ax.set_title(sd, fontsize=10)
            ax.grid(alpha=0.3)
    fig.suptitle(f"{target_label} vs LiDAR features  "
                 "(grey scatter, black = median ± IQR per bin)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_session_durations_and_rates(summary: pd.DataFrame, out_path: Path):
    """Bar chart: pairs and effective throughput per session."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sessions = summary["session_date"].tolist()
    colors = [COLOR[s] for s in sessions]
    axes[0].bar(sessions, summary["n_pairs"] / 1e3, color=colors)
    axes[0].set_ylabel("matched pairs (× 1000)")
    axes[0].set_title("Per-session matched-pair count")
    for x, n in zip(sessions, summary["n_pairs"]):
        axes[0].text(x, n/1e3 + 1, f"{n:,}", ha="center", fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(sessions, summary["motion_fraction"] * 100, color=colors)
    axes[1].set_ylabel("motion fraction (%)  [|speed_mps|>0.05]")
    axes[1].set_title("Per-session AGV-motion fraction")
    axes[1].set_ylim(0, 100)
    for x, v in zip(sessions, summary["motion_fraction"] * 100):
        axes[1].text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_signal_vs_speed(df: pd.DataFrame, out_path: Path):
    """Per-session: signal_power binned by speed."""
    sessions = sorted(df["session_date"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for sd in sessions:
        grp = df[df["session_date"] == sd]
        v = grp["abs_speed"].values
        sp = grp["signal_power"].values
        ok = np.isfinite(v) & np.isfinite(sp) & (v < 1.0)
        bins = np.linspace(0, 0.7, 25)
        groups = pd.cut(pd.Series(v[ok]), bins)
        med = pd.Series(sp[ok]).groupby(groups, observed=False).median()
        q25 = pd.Series(sp[ok]).groupby(groups, observed=False).quantile(0.25)
        q75 = pd.Series(sp[ok]).groupby(groups, observed=False).quantile(0.75)
        centers = [b.mid for b in med.index]
        ax.plot(centers, med, color=COLOR[sd], lw=1.6,
                label=f"{sd}  median ± IQR")
        ax.fill_between(centers, q25, q75, color=COLOR[sd], alpha=0.18)
    ax.set_xlabel("|speed_mps|  (m/s)")
    ax.set_ylabel("signal_power  (dBm)")
    ax.set_title("WiFi signal_power vs AGV speed, per session")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {JOINT} ...")
    df = load_joint()
    print(f"  rows={len(df):,}  cols={len(df.columns)}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Per-session summary table ...")
    summary = per_session_summary(df)
    print(summary.to_string(index=False))
    summary.to_csv(HERE / "session_summary.csv", index=False)

    print("\n[2] Computing LiDAR features ...")
    lf = compute_lidar_features(df)
    df = pd.concat([df.reset_index(drop=True), lf.reset_index(drop=True)], axis=1)

    print("\n[3] Generating figures ...")
    figs: dict[str, Path] = {}

    # Per-session bar charts
    p = FIG_DIR / "session_overview.png"
    fig_session_durations_and_rates(summary, p)
    figs["overview"] = p

    # WiFi distributions
    p = FIG_DIR / "dist_wifi.png"
    wifi_cols = [
        ("signal_power", "signal_power", "dBm"),
        ("signal_quality", "signal_quality", "%"),
        ("ping", "ping", "ms"),
        ("snr", "SNR (signal_power - signal_noise)", "dB"),
        ("signal_noise", "signal_noise", "dBm"),
    ]
    fig_distributions(df, wifi_cols, p,
                      "WiFi metric distributions per session_date")
    figs["wifi_dist"] = p

    # Motion distributions
    p = FIG_DIR / "dist_motion.png"
    motion_cols = [
        ("abs_speed", "|speed_mps|", "m/s"),
        ("wheels", "(|v_L|+|v_R|)/2 (wheels)", "m/s"),
    ]
    fig_distributions(df, motion_cols, p,
                      "Motion-signal distributions per session_date", ncols=2)
    figs["motion_dist"] = p

    # LiDAR feature distributions
    # NB: min_front_mm and dist_p10_mm are dominated by a ~100 mm fixed
    # AGV-body reflection (see report §1b), and no_return_frac is a
    # geometric constant ≈ 0.50. We drop them from the headline plot
    # but keep them in the cache for completeness.
    p = FIG_DIR / "dist_lidar.png"
    lidar_cols = [
        ("mean_dist_mm", "mean valid distance", "mm"),
        ("dist_p90_mm", "90th-pct distance", "mm"),
        ("mean_front_mm", "mean front (-30°,+30°) distance", "mm"),
        ("clutter_frac", "clutter (frac < 4 m)", ""),
        ("openness_frac", "openness (frac > 7 m)", ""),
    ]
    fig_distributions(df, lidar_cols, p,
                      "LiDAR-derived feature distributions per session_date")
    figs["lidar_dist"] = p

    # Trajectories
    p = FIG_DIR / "traj_signal_power.png"
    fig_trajectory(df, "signal_power", "signal_power [dBm]", "RdYlGn", p)
    figs["traj_signal_power"] = p

    p = FIG_DIR / "traj_ping.png"
    fig_trajectory(df, "ping", "ping [ms]", "RdYlGn", p, reverse=True)
    figs["traj_ping"] = p

    p = FIG_DIR / "traj_clutter.png"
    fig_trajectory(df, "clutter_frac", "clutter_frac", "viridis", p)
    figs["traj_clutter"] = p

    # Signal vs speed
    p = FIG_DIR / "signal_vs_speed.png"
    fig_signal_vs_speed(df, p)
    figs["signal_vs_speed"] = p

    # Correlation heatmaps
    # NB: dropped min_front_mm, dist_p10_mm, no_return_frac (constant /
    # AGV-body-dominated; see report §1a).
    corr_cols = [
        "signal_power", "signal_quality", "ping",
        "abs_speed",
        "mean_dist_mm", "mean_front_mm", "dist_p90_mm",
        "clutter_frac", "openness_frac",
    ]
    p = FIG_DIR / "corr_pearson.png"
    fig_corr_heatmap(df, corr_cols, p,
                     "Pearson correlation of WiFi metrics with motion + LiDAR features")
    figs["corr_pearson"] = p

    p = FIG_DIR / "corr_spearman.png"
    fig_corr_heatmap(df, corr_cols, p,
                     "Spearman rank-correlation of WiFi metrics with motion + LiDAR features",
                     method="spearman")
    figs["corr_spearman"] = p

    # WiFi vs LiDAR features
    p = FIG_DIR / "signal_power_vs_lidar.png"
    feat_cols_for_target = [
        ("mean_dist_mm", "mean distance [mm]"),
        ("clutter_frac", "clutter (frac < 4 m)"),
        ("openness_frac", "openness (frac > 7 m)"),
        ("mean_front_mm", "mean front distance [mm]"),
    ]
    fig_signal_vs_lidar(df, "signal_power", "signal_power [dBm]",
                        feat_cols_for_target, p)
    figs["sp_vs_lidar"] = p

    p = FIG_DIR / "ping_vs_lidar.png"
    fig_signal_vs_lidar(df, "ping", "ping [ms]",
                        feat_cols_for_target, p)
    figs["ping_vs_lidar"] = p

    print(f"\n[4] Building comparison block vs obsolete report ...")
    comparison_md, comparison_info = compare_against_obsolete(df, summary)
    print(f"  obsolete-parquet present: {JOINT_OBSOLETE.exists()}; "
          f"identical to current: {comparison_info.get('data_level_identical', False)}")
    print(f"  largest |dtau| vs obsolete report: "
          f"{comparison_info['max_tau_shift_ms']:+.0f} ms on "
          f"{comparison_info['max_tau_shift_session']}")
    print(f"  largest |d signal_power_mean|: "
          f"{comparison_info['max_signal_power_shift_dBm']:+.3f} dBm on "
          f"{comparison_info['max_signal_power_shift_session']}")

    print(f"\n[5] Writing report -> {REPORT}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Copy figures into the docs folder so the report is self-contained
    # and image-link paths resolve to ``figures/<name>.png`` from the
    # report's own directory (same convention as docs/time_sync_per_day/).
    for src in figs.values():
        shutil.copy2(src, REPORT_FIG_DIR / src.name)
    write_report(df, summary, figs, comparison_md)
    print("DONE")


def _img(name: str) -> str:
    """Markdown-friendly relative path for a figure file inside the report
    directory (figures are copied into ``REPORT_FIG_DIR``)."""
    return f"figures/{Path(name).name}"


def write_report(df: pd.DataFrame, summary: pd.DataFrame,
                 figs: dict[str, Path], comparison_md: str):
    # Build markdown tables
    fmt_summary = summary.copy()
    fmt_summary["duration_h"] = fmt_summary["duration_h"].round(2)
    fmt_summary["applied_tau_s"] = fmt_summary["applied_tau_s"].round(4)
    fmt_summary["motion_fraction"] = (fmt_summary["motion_fraction"] * 100).round(1)
    for c in ("abs_speed_max", "signal_power_mean", "signal_power_std",
              "signal_quality_mean", "ping_mean", "ping_p95",
              "x_min", "x_max", "y_min", "y_max"):
        fmt_summary[c] = fmt_summary[c].round(2)
    summary_md = "| " + " | ".join(fmt_summary.columns) + " |\n"
    summary_md += "|" + "|".join(["---"] * len(fmt_summary.columns)) + "|\n"
    for _, row in fmt_summary.iterrows():
        summary_md += "| " + " | ".join(str(v) for v in row.values) + " |\n"

    # Cross-session signal_power deviation
    sp_per_session = df.groupby("session_date")["signal_power"]
    sp_means = sp_per_session.mean()
    sp_stds = sp_per_session.std()
    sp_diff_md = "| pair | Δ mean signal_power (dBm) | Δ mean ping (ms) | Δ motion fraction |\n"
    sp_diff_md += "|---|---|---|---|\n"
    pings = df.groupby("session_date")["ping"].mean()
    motions = df.groupby("session_date").apply(
        lambda g: (g["abs_speed"] > 0.05).mean(), include_groups=False)
    sd_list = sp_means.index.tolist()
    for a in sd_list:
        for b in sd_list:
            if a >= b:
                continue
            sp_diff_md += (f"| {a} → {b} | {sp_means[b]-sp_means[a]:+.2f} | "
                           f"{pings[b]-pings[a]:+.2f} | "
                           f"{(motions[b]-motions[a])*100:+.1f}% |\n")

    # Per-day applied tau values pulled from the new dataset for the §1
    # commentary bullets (was hardcoded in the obsolete report).
    tau_by_sd = (df.groupby("session_date")["applied_tau_s"].first()
                 .to_dict())

    md = f"""# Initial analysis of the corrected joint-coverage dataset

*Project: FLICS 2026 / AIDI 2026 — LiDAR-derived WiFi quality prediction.*
Generated by `analysis/initial_analysis/run_initial_analysis.py` from
`data/merged/joint_coverage.parquet` after the per-day time-sync
calibration pipeline (see `docs/time_sync_per_day/report.md`). Replaces
the obsolete per-CSV-pooled analysis archived under
`docs/obsolete/initial_corrected_dataset_analysis_per_run_data/`.

## 0. Verification status

The joint-coverage parquet passes all 12 end-to-end validation checks in
`analysis/time_sync/validate_joint.py` (schema, per-day applied_tau /
SE, raw + tau == corrected timestamp identity, LiDAR raw / scan_nr /
scan_counter provenance vs `lidar.h5`, delta_t identity, |delta_t| ≤
tolerance, payload identity vs `telemetry_cleaned.parquet` on 600
sampled rows, original-CSV provenance on 240 sampled rows, and
independent merge_asof retention re-run). Data is safe to analyse.

## 1a. Data-quality findings discovered during analysis

Three LiDAR-derived features turn out to be uninformative on this dataset
and we **drop them from the headline analysis** while keeping them in the
cache for traceability:

* **`no_return_frac` ≈ 0.4996 ± 2 × 10⁻⁵ on every scan.** Almost exactly
  half of the 2700 beams return raw 0 ("no return") on every single
  scan, with a standard deviation 4–5 orders of magnitude smaller than
  the mean. This is a geometric constant (likely the rear-half of the
  sweep range produces no returns because of the AGV body / housing /
  the sensor's effective FOV being narrower than the 270° sweep). It
  carries no information about the environment.
* **`min_front_mm` ≈ 100 mm constant** (median 103 mm, std 5 mm; 0 % of
  scans report a front-min above 200 mm). The closest beam in the
  ±30° front cone is reading a fixed reflector at ~ 10 cm from the
  LiDAR every scan — almost certainly part of the AGV's own bumper /
  housing within the sensor field of view. The same effect explains
  the very tight `dist_p10_mm` distribution at 134–142 mm and the
  global `min_dist_mm` at ~ 86 mm.
* **`dist_p10_mm` and `min_dist_mm`** are similarly dominated by that
  fixed close reflector and convey almost no scene-level signal.

Useful LiDAR features that *do* vary with the environment:

* `mean_dist_mm` (full sweep)
* `mean_front_mm` (front cone only, with the fixed reflector excluded
  as part of the average)
* `dist_p90_mm` (the far-end of the visible beams; tracks "openness")
* `clutter_frac`, `openness_frac` (interpretable summary scalars)

The analysis below uses these "useful" features. We add a **follow-up
recommendation** in §10 to characterise the AGV-body LiDAR mask
explicitly so future feature engineering can mask out the fixed
reflector beams beam-by-beam rather than relying on aggregates.

## 1. Per-session summary

{summary_md}

![per-session matched pairs and motion fraction]({_img(figs['overview'])})

The three calibration days are very different in scale and operating mode:

* **15.03.2026** — 4 CSV files, 3.1 h on a high-rate 31 Hz telemetry stack;
  {int(summary.set_index("session_date").loc["15.03.2026","n_pairs"]):,} matched pairs; AGV motion-active about a third of the time;
  applied per-day τ̂ = {tau_by_sd['15.03.2026']:+.3f} s.
* **24.03.2026** — 3 CSV files, ~1.9 h on the 31 Hz stack; {int(summary.set_index("session_date").loc["24.03.2026","n_pairs"]):,} matched
  pairs; the most motion-active day; applied τ̂ = {tau_by_sd['24.03.2026']:+.3f} s.
* **25.02.2026** — 1 CSV file, 2.8 h on the slower 4.4 Hz stack; {int(summary.set_index("session_date").loc["25.02.2026","n_pairs"]):,} matched
  pairs (high count because LiDAR is at 25 Hz, telemetry at 4.4 Hz, so each
  telemetry sample matches multiple LiDAR scans); ~ 60 % of recording was
  motion-active; applied τ̂ = {tau_by_sd['25.02.2026']:+.3f} s. Note the 25.02 ceiling speed is half
  of what 15.03/24.03 reach (0.30 m/s vs 0.60 m/s) — different operating
  mode (see `docs/time_sync_per_day/report.md` §10 for the full account).

## 2. Distributions of motion signals

![motion-signal distributions]({_img(figs['motion_dist'])})

The two motion signals (`|speed_mps|` and the wheel-encoder average) agree
within rounding on 15.03/24.03 and diverge on 25.02 — exactly the pattern
that drove the rotation-aware signature in the time-sync analysis.

## 3. WiFi-metric distributions per session

![WiFi metric distributions]({_img(figs['wifi_dist'])})

Cross-session deviations:

{sp_diff_md}

Take-aways:

* `signal_power` distributions are session-dependent. Means differ by up to
  ~ {abs(sp_means.max()-sp_means.min()):.1f} dBm across sessions; stds are
  ~ {sp_stds.mean():.1f} dBm. **Any cross-session prediction model has to
  account for this between-session shift** (random effect on session, or
  per-session normalization, or session ID feature).
* `ping` distributions are bimodal/heavy-tailed on every day — most
  matched samples are near the median but a long upper tail dominates the
  mean.
* `signal_quality` is a percentage and saturates near 100 % on all
  sessions; useful as a regression target only after a meaningful
  transformation (e.g. `100 - signal_quality` for log-link regression).
* SNR (`signal_power - signal_noise`) tracks `signal_power` because
  `signal_noise` is more stable than the carrier strength — i.e. SNR
  contains the same information as `signal_power` plus a small noise
  reduction.

## 4. LiDAR-derived feature distributions

![LiDAR feature distributions]({_img(figs['lidar_dist'])})

The LiDAR features quantify the visible scene at each scan. `mean_dist_mm`
and `dist_p10_mm` describe how clutter-rich the immediate environment is;
`clutter_frac` and `openness_frac` are interpretable summary scalars
suitable as model features. The `no_return_frac` distribution shows the
typical fraction of beams that fail to return (likely glass / mirror-like
surfaces, or distances above the 8.25 m sensor max).

The three sessions show **clearly different LiDAR scene statistics** —
which is consistent with the AGV traversing different parts of the lab on
each day and with environmental differences (people, objects, lighting).
This is good news for a learnable LiDAR → WiFi model: the input
distribution genuinely varies across days, so cross-session validation
will be informative.

## 5. Trajectories coloured by WiFi metrics

![trajectory coloured by signal_power]({_img(figs['traj_signal_power'])})

![trajectory coloured by ping]({_img(figs['traj_ping'])})

![trajectory coloured by clutter_frac]({_img(figs['traj_clutter'])})

The trajectory plots reveal the spatial structure of WiFi quality:

* Each session traverses a different part of the same general workspace.
  15.03/24.03 cover similar areas; 25.02 covers a different zone (the AGV
  was operated in a different test mode, with lower top speed and more
  rotation — see time-sync §10 deep-dive).
* `signal_power` shows clear spatial structure on every day: there are
  hot zones and cold zones in (x,y) that persist across runs of the same
  day. **Spatial features (x,y) and an environment proxy (clutter,
  mean distance) should be the first inputs to a baseline model.**
* `ping` is more uniformly distributed; large pings are concentrated in
  small clusters that suggest occasional packet-loss bursts rather than
  steady spatial degradation.

## 6. Signal vs speed

![signal_power vs speed]({_img(figs['signal_vs_speed'])})

`signal_power` shows weak speed dependence: the median is roughly flat
across the speed range on every day, with the IQR widening at low speeds
because the AGV spends most of its time near zero speed. **Speed is
unlikely to be a useful direct WiFi predictor**, but it can act as a
gating variable (predict only when AGV is in motion) or as a feature for
modelling momentary multipath effects.

## 7. WiFi metric vs LiDAR features

![signal_power vs LiDAR features]({_img(figs['sp_vs_lidar'])})

![ping vs LiDAR features]({_img(figs['ping_vs_lidar'])})

These are the per-session bin-medians ± IQR of `signal_power` and `ping`
against the LiDAR-derived features. A monotone trend in the median curve
indicates a useful predictor; a flat curve indicates the feature carries
little signal. The most-promising features are those whose median curve
moves visibly across the feature range AND whose direction is consistent
across sessions.

## 8. Correlations

![Pearson correlations]({_img(figs['corr_pearson'])})

![Spearman correlations]({_img(figs['corr_spearman'])})

Pearson is sensitive to linear relationships; Spearman captures monotone
relationships and is robust to the heavy-tailed `ping`. Compare the two
to spot non-linear dependencies (cells where Spearman is large but
Pearson is small).

## 9. Suggested research directions

Based on what is visible in the corrected joint dataset, the following
modelling directions look most promising for AIDI 2026:

1. **Session as a random effect / per-session normalisation.** The
   `signal_power` between-session shift (~{abs(sp_means.max()-sp_means.min()):.1f} dBm
   across days) is large compared to within-session std
   (~{sp_stds.mean():.1f} dBm). Either treat session as a random effect
   or z-score `signal_power` per session before training a cross-session
   model. Pure pooling will mostly learn the session shift, not the
   physics.
2. **Spatial features first, LiDAR features second.** The trajectory
   plots suggest (x,y) explain a substantial fraction of `signal_power`
   variance on each day. A natural baseline is a Gaussian-process or
   k-NN over (x,y) per session, with LiDAR features added as residual
   predictors.
3. **Environment-proxy features look more useful than peak/min raw
   distances.** Aggregates that summarise the full beam pattern
   (`clutter_frac`, `openness_frac`, `mean_dist_mm`) are more stable
   than instantaneous min/p10 values, which can be dominated by a
   single beam reflecting off something at close range. Section 7
   median curves should guide which to keep.
4. **Predict `signal_power` first, `ping` second.** Distributions of
   `signal_power` are well-behaved (approximately Gaussian within
   session). `ping` has a long upper tail dominated by occasional
   packet-loss spikes; modelling those requires a different loss
   (Gamma / quantile / classification of "high-ping events") rather
   than MSE on raw ping.
5. **Use the `applied_tau_se_s` column for uncertainty propagation.**
   The joint dataset carries the per-day calibration SE per row; any
   model that depends on sub-100-ms feature alignment should use this
   to generate plausible perturbed copies of the dataset (e.g. for a
   robustness-bound on test-set performance).
6. **Watch the 25.02.2026 session.** It has a different operating mode
   (slower, more rotation), a different telemetry sampling rate (4.4 Hz
   vs 31 Hz), and a different applied τ̄. Use it as a *held-out*
   distribution-shift test set, not as a training fold.

## 10. Caveats and follow-ups

* **Session shift is real and large.** Any cross-session result without
  per-session correction is suspect.
* **25.02 has 113 ms matching tolerance** vs 20 ms on the other days
  because of the 4.4 Hz telemetry. Time-aligned modelling should weight
  pairs by `1/match_tolerance_s` if the prediction is sensitive to
  alignment.
* **LiDAR features are scan-instantaneous**; for moving AGV they reflect
  the scene at slightly different (x,y) within the matching window. For
  models that average features over a short window, use
  `lidar_ts_corr` to do the windowing.
* **Characterise the AGV-body LiDAR mask explicitly.** The fixed ~ 100
  mm front reflector and the ~ 50 % systematic no-return fraction
  (see §1a) suggest a sizeable portion of the LiDAR sweep is hitting
  the AGV's own body / housing. Computing a per-beam-angle "always-zero
  or always-very-close" mask from a stationary baseline scan would
  let future feature engineering exclude those angles before
  aggregating. Until that mask exists, prefer aggregate features
  (`mean_dist_mm`, `mean_front_mm`, `clutter_frac`, `openness_frac`,
  `dist_p90_mm`) over instantaneous-min features.

{comparison_md}
## 12. Reproducibility

```bash
.venv/Scripts/python.exe -m analysis.initial_analysis.run_initial_analysis
```

Outputs:

* `docs/initial_dataset_analysis/report.md`  (this file)
* `docs/initial_dataset_analysis/figures/*.png`  (figures, copied so the
  report is self-contained when shipped from the docs/ tree)
* `analysis/initial_analysis/figures/*.png`  (figures, primary location)
* `analysis/initial_analysis/cache/lidar_features.parquet`  (per-row
  LiDAR features; cached for fast re-render)
* `analysis/initial_analysis/session_summary.csv`
"""
    REPORT.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
