"""Render figures, tables, and the Project B results report from cached artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, data_io, run_modeling  # noqa: E402

FOLDS = list(config.FOLD_NAMES)
MAIN_VARIANTS = list(config.MAIN_VARIANTS)
DISAMBIG_VARIANTS = ["W4", "W2", "W4pp"]  # rendered as W4 / W4' / W4''
DISAMBIG_LABELS = dict(config.DISAMBIG_LABELS)
GROUP_NAMES = list(config.FEATURE_GROUPS.keys())
STRATA = ["overall", "in_fov", "out_of_fov"]


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _fmt_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"


def _fmt_signed_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _bootstrap_diff_rmse(
    y1_true: np.ndarray,
    y1_pred: np.ndarray,
    y2_true: np.ndarray,
    y2_pred: np.ndarray,
    *,
    B: int = config.BOOTSTRAP_B,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Δ = RMSE(y1) - RMSE(y2) with paired-bootstrap CI. Both series must be aligned (same row order)."""
    rng = np.random.default_rng(seed)
    n = len(y1_true)
    sq1 = (y1_true - y1_pred) ** 2
    sq2 = (y2_true - y2_pred) ** 2
    base = float(np.sqrt(np.mean(sq1)) - np.sqrt(np.mean(sq2)))
    if n == 0:
        return base, float("nan"), float("nan")
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = float(np.sqrt(np.mean(sq1[idx])) - np.sqrt(np.mean(sq2[idx])))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return base, lo, hi


def _load_preds(fold: str, variant: str, hp: str = "locked") -> pd.DataFrame:
    if hp == "locked":
        path = config.CACHE_DIR / f"predictions_{fold}_{variant}.parquet"
    else:
        path = config.CACHE_DIR / f"predictions_{fold}_{variant}_{hp}.parquet"
    return pd.read_parquet(path)


# ----------------------------------------------------------------------
# Spatial regions figure
# ----------------------------------------------------------------------


def render_regions_map(non_anom_with_region: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for r in range(1, config.N_REGIONS + 1):
        sub = non_anom_with_region[non_anom_with_region["region_id"] == r]
        ax.scatter(
            sub["x_m"], sub["y_m"], s=2, alpha=0.4, color=cmap(r - 1), label=f"region {r} (n={len(sub):,})"
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    ax.set_title("Project B spatial regions on 15.03.2026 trajectory")
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    p = config.FIGURES_DIR / "regions_map.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def render_regions_summary_table(non_anom_with_region: pd.DataFrame) -> Path:
    rows = []
    for r in range(1, config.N_REGIONS + 1):
        sub = non_anom_with_region[non_anom_with_region["region_id"] == r]
        n = len(sub)
        if n == 0:
            rows.append({"region": r, "n_rows": 0})
            continue
        n_in_fov = int(sub["is_AP_in_FOV"].sum())
        n_out = n - n_in_fov
        unique_cells = int(sub.groupby([np.round(sub["x_m"], 1), np.round(sub["y_m"], 1)]).ngroups)
        rows.append({
            "region": r,
            "n_rows": n,
            "n_unique_cells_0p1m": unique_cells,
            "frac_in_fov": n_in_fov / n if n > 0 else float("nan"),
            "n_in_fov": n_in_fov,
            "n_out_of_fov": n_out,
            "dist_to_AP_min": float(sub["dist_to_AP"].min()),
            "dist_to_AP_max": float(sub["dist_to_AP"].max()),
            "x_min": float(sub["x_m"].min()),
            "x_max": float(sub["x_m"].max()),
            "y_min": float(sub["y_m"].min()),
            "y_max": float(sub["y_m"].max()),
        })
    df = pd.DataFrame(rows)

    md = ["# Project B — per-region statistics on 15.03.2026", ""]
    md.append("| Region | n_rows | unique 0.1 m cells | frac in-FOV | n_in_FOV | n_out_of_FOV | dist_to_AP range (m) | (x_m, y_m) bbox |")
    md.append("|---:|---:|---:|---:|---:|---:|---|---|")
    for _, r in df.iterrows():
        if int(r["n_rows"]) == 0:
            md.append(f"| {int(r['region'])} | 0 | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {int(r['region'])} | {int(r['n_rows']):,} | {int(r['n_unique_cells_0p1m']):,} | "
            f"{r['frac_in_fov']:.3f} | {int(r['n_in_fov']):,} | {int(r['n_out_of_fov']):,} | "
            f"[{r['dist_to_AP_min']:.2f}, {r['dist_to_AP_max']:.2f}] | "
            f"x [{r['x_min']:.2f}, {r['x_max']:.2f}], y [{r['y_min']:.2f}, {r['y_max']:.2f}] |"
        )
    p = config.TABLES_DIR / "regions_summary.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Headline ablation table
# ----------------------------------------------------------------------


def render_wlro_ablation_table(wlro: pd.DataFrame) -> Path:
    rows = []
    for variant in MAIN_VARIANTS:
        for stratum in STRATA:
            row: dict[str, object] = {"variant": variant, "stratum": stratum}
            for fold in FOLDS:
                hit = wlro[(wlro["fold"] == fold) & (wlro["variant"] == variant) & (wlro["stratum"] == stratum)]
                if hit.empty:
                    row[fold] = "—"
                else:
                    r = hit.iloc[0]
                    row[fold] = f"{r['rmse']:.2f} {_fmt_ci(r['rmse_ci_lo'], r['rmse_ci_hi'])}"
            rows.append(row)
    df = pd.DataFrame(rows)
    md = [
        "# WLRO ablation — RMSE (dB) per fold × variant × stratum",
        "",
        "Bootstrap 95% CI on RMSE in brackets (B = 1000). Within-session leave-region-out, 15.03 only.",
        "",
        "| Variant | Stratum | " + " | ".join(FOLDS) + " |",
        "|---|---|" + "|".join(["---"] * len(FOLDS)) + "|",
    ]
    for _, r in df.iterrows():
        cells = [str(r["variant"]), str(r["stratum"])] + [str(r[f]) for f in FOLDS]
        md.append("| " + " | ".join(cells) + " |")
    p = config.TABLES_DIR / "wlro_ablation_rmse.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Δ_LiDAR / Δ_AP-relative / Δ_position
# ----------------------------------------------------------------------


def render_delta_tables() -> tuple[Path, Path, Path, dict]:
    summary: dict = {"delta_lidar": [], "delta_ap": [], "delta_position": []}
    rows_lidar, rows_ap, rows_pos = [], [], []

    for fold in FOLDS:
        # Δ_LiDAR_within = RMSE(W2) - RMSE(W4), paired bootstrap on test rows.
        for stratum in STRATA:
            preds_w2 = _load_preds(fold, "W2")
            preds_w4 = _load_preds(fold, "W4")
            merged = preds_w2[["joint_idx", "is_AP_in_FOV", "signal_power_true", "signal_power_pred"]].rename(
                columns={"signal_power_pred": "pred_w2"}
            ).merge(
                preds_w4[["joint_idx", "signal_power_pred"]].rename(columns={"signal_power_pred": "pred_w4"}),
                on="joint_idx", how="inner",
            )
            if stratum == "in_fov":
                m = merged[merged["is_AP_in_FOV"]]
            elif stratum == "out_of_fov":
                m = merged[~merged["is_AP_in_FOV"]]
            else:
                m = merged
            yt = m["signal_power_true"].to_numpy(dtype=np.float64)
            base, lo, hi = _bootstrap_diff_rmse(
                yt, m["pred_w2"].to_numpy(dtype=np.float64),
                yt, m["pred_w4"].to_numpy(dtype=np.float64),
            )
            rows_lidar.append({
                "fold": fold, "stratum": stratum, "n": int(len(m)),
                "delta_lidar": base, "ci_lo": lo, "ci_hi": hi,
            })

        # Δ_AP-relative_within = RMSE(W1) - RMSE(W2), paired bootstrap.
        for stratum in STRATA:
            preds_w1 = _load_preds(fold, "W1")
            preds_w2 = _load_preds(fold, "W2")
            merged = preds_w1[["joint_idx", "is_AP_in_FOV", "signal_power_true", "signal_power_pred"]].rename(
                columns={"signal_power_pred": "pred_w1"}
            ).merge(
                preds_w2[["joint_idx", "signal_power_pred"]].rename(columns={"signal_power_pred": "pred_w2"}),
                on="joint_idx", how="inner",
            )
            if stratum == "in_fov":
                m = merged[merged["is_AP_in_FOV"]]
            elif stratum == "out_of_fov":
                m = merged[~merged["is_AP_in_FOV"]]
            else:
                m = merged
            yt = m["signal_power_true"].to_numpy(dtype=np.float64)
            base, lo, hi = _bootstrap_diff_rmse(
                yt, m["pred_w1"].to_numpy(dtype=np.float64),
                yt, m["pred_w2"].to_numpy(dtype=np.float64),
            )
            rows_ap.append({
                "fold": fold, "stratum": stratum, "n": int(len(m)),
                "delta_ap": base, "ci_lo": lo, "ci_hi": hi,
            })

        # Δ_position = RMSE(per-region mean baseline) - RMSE(W0), overall stratum.
        preds_w0 = _load_preds(fold, "W0")
        # Per-region mean baseline = mean signal_power on the training pool for this fold.
        # Reconstruct training mean from non_anom rows excluding test region:
        #   - train pool = non_anom rows with region_id != test_region (val+train, before split).
        # We don't have access to non_anom here without reloading, so compute:
        non_anom = data_io.load_session_non_anomaly()
        region_df = pd.read_parquet(config.ARTIFACTS_DIR / "spatial_regions.parquet")
        with_region = non_anom.merge(region_df, on="joint_idx", how="left", validate="1:1")
        test_region_id = config.FOLD_TO_REGION[fold]
        train_mean = float(
            with_region.loc[with_region["region_id"] != test_region_id, config.TARGET].mean()
        )
        y_true = preds_w0["signal_power_true"].to_numpy(dtype=np.float64)
        y_pred_baseline = np.full_like(y_true, train_mean)
        base, lo, hi = _bootstrap_diff_rmse(
            y_true, y_pred_baseline,
            y_true, preds_w0["signal_power_pred"].to_numpy(dtype=np.float64),
        )
        rows_pos.append({
            "fold": fold, "n": int(len(y_true)),
            "rmse_baseline": float(np.sqrt(np.mean((y_true - train_mean) ** 2))),
            "rmse_w0": float(np.sqrt(np.mean((y_true - preds_w0["signal_power_pred"].to_numpy(dtype=np.float64)) ** 2))),
            "delta_position": base, "ci_lo": lo, "ci_hi": hi,
        })

    df_lidar = pd.DataFrame(rows_lidar)
    df_ap = pd.DataFrame(rows_ap)
    df_pos = pd.DataFrame(rows_pos)

    p1 = config.TABLES_DIR / "delta_lidar_within.md"
    md = [
        "# Δ_LiDAR_within — RMSE(W2) − RMSE(W4) (dB)",
        "",
        "Positive => LiDAR features improve over (position + telemetry + AP-relative).",
        "",
        "| Fold | Stratum | n | Δ_LiDAR | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in df_lidar.iterrows():
        md.append(
            f"| {r['fold']} | {r['stratum']} | {int(r['n'])} | {r['delta_lidar']:+.3f} | "
            f"{_fmt_signed_ci(r['ci_lo'], r['ci_hi'])} |"
        )
    p1.write_text("\n".join(md) + "\n", encoding="utf-8")

    p2 = config.TABLES_DIR / "delta_ap_relative_within.md"
    md = [
        "# Δ_AP-relative_within — RMSE(W1) − RMSE(W2) (dB)",
        "",
        "Positive => AP-relative features improve on (position + telemetry).",
        "",
        "| Fold | Stratum | n | Δ_AP-relative | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in df_ap.iterrows():
        md.append(
            f"| {r['fold']} | {r['stratum']} | {int(r['n'])} | {r['delta_ap']:+.3f} | "
            f"{_fmt_signed_ci(r['ci_lo'], r['ci_hi'])} |"
        )
    p2.write_text("\n".join(md) + "\n", encoding="utf-8")

    p3 = config.TABLES_DIR / "delta_position_within.md"
    md = [
        "# Δ_position — RMSE(per-region mean baseline) − RMSE(W0) (dB)",
        "",
        "Positive => position alone (W0 = x_m, y_m) explains structure beyond the train-region mean. Overall stratum.",
        "",
        "| Fold | n | RMSE(mean baseline) | RMSE(W0) | Δ_position | 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in df_pos.iterrows():
        md.append(
            f"| {r['fold']} | {int(r['n'])} | {r['rmse_baseline']:.3f} | {r['rmse_w0']:.3f} | "
            f"{r['delta_position']:+.3f} | {_fmt_signed_ci(r['ci_lo'], r['ci_hi'])} |"
        )
    p3.write_text("\n".join(md) + "\n", encoding="utf-8")

    summary["delta_lidar"] = df_lidar.to_dict(orient="records")
    summary["delta_ap"] = df_ap.to_dict(orient="records")
    summary["delta_position"] = df_pos.to_dict(orient="records")
    return p1, p2, p3, summary


# ----------------------------------------------------------------------
# Disambiguation table + verdict
# ----------------------------------------------------------------------


def render_disambig_table(disambig: pd.DataFrame) -> tuple[Path, str, dict]:
    md = [
        "# Disambiguation — overall RMSE (dB) for W4 / W4' / W4''",
        "",
        "W4 = full 34-feature model. W4' = position + telemetry + AP-relative (≡ W2; LiDAR removed). "
        "W4'' = position + telemetry + LiDAR (AP-relative removed).",
        "",
        "| Variant | " + " | ".join(FOLDS) + " |",
        "|---|" + "|".join(["---"] * len(FOLDS)) + "|",
    ]
    overall = disambig[disambig["stratum"] == "overall"]
    for variant in DISAMBIG_VARIANTS:
        cells = [DISAMBIG_LABELS[variant]]
        for fold in FOLDS:
            hit = overall[(overall["fold"] == fold) & (overall["variant"] == variant)]
            if hit.empty:
                cells.append("—")
            else:
                r = hit.iloc[0]
                cells.append(f"{r['rmse']:.2f} {_fmt_ci(r['rmse_ci_lo'], r['rmse_ci_hi'])}")
        md.append("| " + " | ".join(cells) + " |")

    threshold = 0.5
    fold_judgments: dict[str, str] = {}
    for fold in FOLDS:
        w4 = overall[(overall["variant"] == "W4") & (overall["fold"] == fold)]["rmse"]
        w4p = overall[(overall["variant"] == "W2") & (overall["fold"] == fold)]["rmse"]
        w4pp = overall[(overall["variant"] == "W4pp") & (overall["fold"] == fold)]["rmse"]
        if w4.empty or w4p.empty or w4pp.empty:
            fold_judgments[fold] = "incomplete"
            continue
        w4v, w4pv, w4ppv = float(w4.iloc[0]), float(w4p.iloc[0]), float(w4pp.iloc[0])
        complementary = (w4v < w4pv - threshold) and (w4v < w4ppv - threshold)
        lidar_removable = abs(w4v - w4pv) < threshold
        ap_removable = abs(w4v - w4ppv) < threshold
        if complementary:
            fold_judgments[fold] = "complementary"
        elif lidar_removable and not complementary:
            fold_judgments[fold] = "LiDAR removable"
        elif ap_removable and not complementary:
            fold_judgments[fold] = "AP-relative removable"
        else:
            fold_judgments[fold] = "mixed/unclear"

    counts: dict[str, int] = {}
    for v in fold_judgments.values():
        counts[v] = counts.get(v, 0) + 1
    if counts.get("complementary", 0) >= 3:
        verdict = "complementary"
    elif counts.get("LiDAR removable", 0) >= 3:
        verdict = "LiDAR removable"
    elif counts.get("AP-relative removable", 0) >= 3:
        verdict = "AP-relative removable"
    else:
        verdict = "mixed/unclear"

    md.append("")
    md.append(f"Per-fold judgments (threshold = {threshold} dB):")
    for fold, j in fold_judgments.items():
        md.append(f"- {fold}: **{j}**")
    md.append("")
    md.append(f"**Overall verdict (≥3 of 5 folds): {verdict}**")

    p = config.TABLES_DIR / "disambig_summary.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")

    return p, verdict, {"per_fold": fold_judgments, "counts": counts, "threshold_dB": threshold}


# ----------------------------------------------------------------------
# Buffer-zone sensitivity
# ----------------------------------------------------------------------


def render_buffer_sensitivity(robustness: pd.DataFrame, wlro: pd.DataFrame) -> tuple[Path, dict]:
    buffer_df = robustness[robustness["check"] == "buffer"].copy()
    md = [
        f"# Buffer-zone sensitivity — {config.BUFFER_FOLD} with vs without 1 m exclusion",
        "",
        f"Drop training rows within {config.BUFFER_DISTANCE_M} m of any held-out region row, "
        f"then refit. Compare RMSE deltas to the no-buffer {config.BUFFER_FOLD} fits.",
        "",
        "| Variant | Stratum | RMSE(no buffer) | RMSE(buffer) | Δ_RMSE (buffer - no) |",
        "|---|---|---:|---:|---:|",
    ]
    rows = []
    for variant in MAIN_VARIANTS:
        for stratum in STRATA:
            no_buf = wlro[
                (wlro["fold"] == config.BUFFER_FOLD)
                & (wlro["variant"] == variant)
                & (wlro["stratum"] == stratum)
            ]
            with_buf = buffer_df[
                (buffer_df["variant"] == variant) & (buffer_df["stratum"] == stratum)
            ]
            if no_buf.empty or with_buf.empty:
                md.append(f"| {variant} | {stratum} | — | — | — |")
                continue
            r0 = float(no_buf.iloc[0]["rmse"])
            r1 = float(with_buf.iloc[0]["rmse"])
            md.append(f"| {variant} | {stratum} | {r0:.3f} | {r1:.3f} | {(r1 - r0):+.3f} |")
            rows.append({"variant": variant, "stratum": stratum, "rmse_no_buffer": r0, "rmse_buffer": r1, "delta": r1 - r0})

    delta_df = pd.DataFrame(rows)
    overall_only = delta_df[delta_df["stratum"] == "overall"]
    threshold = config.BUFFER_RMSE_DELTA_THRESHOLD_DB
    n_majority = (overall_only["delta"].abs() < threshold).sum()
    n_total = len(overall_only)
    flagged = bool(n_majority < (n_total // 2 + 1))
    md.append("")
    md.append(
        f"**Sanity check (overall stratum)**: {int(n_majority)}/{int(n_total)} variants change by < {threshold} dB "
        f"under the buffer."
    )
    if flagged:
        md.append(
            "**Flagged**: majority of variants move by ≥ 0.5 dB under the buffer — leave-region-out may be "
            "contaminated by spatial autocorrelation; treat the headline result with caution."
        )
    else:
        md.append("**OK**: no-buffer leave-region-out result is robust to the buffer-zone exclusion.")
    p = config.TABLES_DIR / "buffer_sensitivity.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p, {"deltas": rows, "flagged": flagged, "threshold_dB": threshold}


# ----------------------------------------------------------------------
# Hyperparameter sensitivity
# ----------------------------------------------------------------------


def render_hyperparam_sensitivity(robustness: pd.DataFrame, wlro: pd.DataFrame) -> tuple[Path, dict]:
    hp_df = robustness[robustness["check"] == "hyperparams"].copy()
    md = [
        "# Hyperparameter sensitivity — W4 locked vs H1 (max_depth=4)",
        "",
        "H1 was the cross-session diagnostic's best alternative. Refit on each Project B fold "
        "to verify the within-session story is not config-specific.",
        "",
        "| Fold | Stratum | RMSE(locked, depth=6) | RMSE(H1, depth=4) | Δ_RMSE (H1 - locked) |",
        "|---|---|---:|---:|---:|",
    ]
    rows = []
    for fold in FOLDS:
        for stratum in STRATA:
            locked = wlro[(wlro["fold"] == fold) & (wlro["variant"] == "W4") & (wlro["stratum"] == stratum)]
            h1 = hp_df[(hp_df["fold"] == fold) & (hp_df["variant"] == "W4") & (hp_df["stratum"] == stratum)]
            if locked.empty or h1.empty:
                md.append(f"| {fold} | {stratum} | — | — | — |")
                continue
            r0 = float(locked.iloc[0]["rmse"])
            r1 = float(h1.iloc[0]["rmse"])
            md.append(f"| {fold} | {stratum} | {r0:.3f} | {r1:.3f} | {(r1 - r0):+.3f} |")
            rows.append({"fold": fold, "stratum": stratum, "rmse_locked": r0, "rmse_h1": r1, "delta": r1 - r0})

    df = pd.DataFrame(rows)
    threshold = config.HYPERPARAM_RMSE_DELTA_THRESHOLD_DB
    overall_only = df[df["stratum"] == "overall"]
    flagged_folds = overall_only[overall_only["delta"].abs() > threshold]["fold"].tolist()
    md.append("")
    if flagged_folds:
        md.append(
            f"**Flagged (overall stratum)**: |Δ_RMSE| > {threshold} dB on folds {flagged_folds}. "
            "Report locked and H1 side by side for these folds."
        )
    else:
        md.append(
            f"**OK**: |Δ_RMSE| < {threshold} dB on all folds (overall stratum). The locked headline is "
            "consistent with H1."
        )
    p = config.TABLES_DIR / "hyperparam_sensitivity.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p, {"deltas": rows, "flagged_folds": flagged_folds, "threshold_dB": threshold}


# ----------------------------------------------------------------------
# Headline ablation chart
# ----------------------------------------------------------------------


def render_ablation_chart(wlro: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, stratum in zip(axes, STRATA):
        x_pos = np.arange(len(MAIN_VARIANTS))
        width = 0.15
        for j, fold in enumerate(FOLDS):
            vals = []
            for variant in MAIN_VARIANTS:
                hit = wlro[(wlro["fold"] == fold) & (wlro["variant"] == variant) & (wlro["stratum"] == stratum)]
                vals.append(float(hit.iloc[0]["rmse"]) if not hit.empty else np.nan)
            ax.bar(x_pos + (j - 2) * width, vals, width=width, label=fold)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(MAIN_VARIANTS, rotation=0)
        ax.set_title(f"stratum: {stratum}")
        ax.set_ylabel("RMSE (dB)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Project B: WLRO ablation RMSE per fold × variant × stratum")
    fig.tight_layout()
    p = config.FIGURES_DIR / "wlro_ablation_chart.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------
# XAI on the SHAP fold
# ----------------------------------------------------------------------


def _shap_path(fold: str) -> Path:
    return config.RESULTS_DIR / f"shap_{fold}.parquet"


def _shap_metadata_path(fold: str) -> Path:
    return config.RESULTS_DIR / f"shap_{fold}_metadata.json"


def render_xai_group_importance(fold_name: str) -> Path:
    df = pd.read_parquet(_shap_path(fold_name))
    rows = []
    for group, members in config.FEATURE_GROUPS.items():
        per_feat = []
        for m in members:
            col = f"shap_{m}"
            if col in df.columns:
                per_feat.append(float(np.mean(np.abs(df[col].to_numpy()))))
        rows.append({"group": group, "sum_mean_abs": float(np.sum(per_feat)), "n_features": len(members)})
    table = pd.DataFrame(rows).sort_values("sum_mean_abs", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table["group"], table["sum_mean_abs"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Σ mean|SHAP| in group (dB)")
    ax.set_title(f"XAI: feature-group importance — {fold_name} (W4)")
    fig.tight_layout()
    p = config.FIGURES_DIR / f"xai_group_importance_{fold_name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)

    md = [
        f"# XAI feature-group importance — {fold_name} (W4)",
        "",
        "Σ mean|SHAP| in group (dB), summed over the group's features.",
        "",
        "| Group | Σ mean\\|SHAP\\| (dB) | # features |",
        "|---|---:|---:|",
    ]
    for _, r in table.iterrows():
        md.append(f"| {r['group']} | {r['sum_mean_abs']:.3f} | {int(r['n_features'])} |")
    table_path = config.TABLES_DIR / f"xai_group_importance_{fold_name}.md"
    table_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p


def render_xai_sign_table(fold_name: str) -> Path:
    """One-fold sign-of-effect table for W4 features (excluding binary is_AP_in_FOV)."""
    from scipy.stats import spearmanr

    df = pd.read_parquet(_shap_path(fold_name))
    cont_features = [
        f for f in config.W4
        if f != "is_AP_in_FOV"
    ]
    cap = 30_000
    if len(df) > cap:
        sub = df.sample(n=cap, random_state=config.SEED)
    else:
        sub = df
    rows = []
    for feat in cont_features:
        if feat not in sub.columns or f"shap_{feat}" not in sub.columns:
            rows.append({"feature": feat, "rho": float("nan"), "sign": "—"})
            continue
        x = sub[feat].to_numpy(dtype=np.float64)
        y = sub[f"shap_{feat}"].to_numpy(dtype=np.float64)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 30:
            rows.append({"feature": feat, "rho": float("nan"), "sign": "—"})
            continue
        rho, _ = spearmanr(x[mask], y[mask])
        if rho is None or np.isnan(rho):
            rows.append({"feature": feat, "rho": float("nan"), "sign": "—"})
            continue
        if abs(rho) < 0.05:
            sign = "0"
        elif rho > 0:
            sign = "+"
        else:
            sign = "-"
        rows.append({"feature": feat, "rho": float(rho), "sign": sign})

    md = [
        f"# XAI sign-of-effect — {fold_name} (W4)",
        "",
        "Sign of Spearman ρ(feature, SHAP). `0` if |ρ| < 0.05.",
        "",
        "| Feature | ρ | Sign |",
        "|---|---:|:---:|",
    ]
    for r in rows:
        rho_str = f"{r['rho']:+.2f}" if not np.isnan(r["rho"]) else "—"
        md.append(f"| `{r['feature']}` | {rho_str} | {r['sign']} |")
    p = config.TABLES_DIR / f"xai_sign_{fold_name}.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p


def render_xai_spatial(fold_name: str) -> Path:
    """Spatial dominance map on the held-out region."""
    df = pd.read_parquet(_shap_path(fold_name))
    cmap = {
        "Position": "#9467bd",
        "Telemetry": "#cccccc",
        "AP-relative": "#d62728",
        "LiDAR scalar": "#1f77b4",
        "LiDAR sectoral": "#2ca02c",
    }
    x = df["x_m"].to_numpy(dtype=np.float64)
    y = df["y_m"].to_numpy(dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    df_v = df.iloc[valid.nonzero()[0]].reset_index(drop=True)
    x, y = x[valid], y[valid]

    group_abs = {}
    for group, members in config.FEATURE_GROUPS.items():
        cols = [f"shap_{m}" for m in members if f"shap_{m}" in df_v.columns]
        if not cols:
            group_abs[group] = np.zeros(len(df_v))
        else:
            group_abs[group] = np.mean(np.abs(df_v[cols].to_numpy()), axis=1)

    cell = 0.5
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    cell_groups: dict[tuple[int, int], dict[str, list[float]]] = {}
    for i, k in enumerate(zip(ix, iy)):
        d = cell_groups.setdefault(k, {g: [] for g in cmap})
        for g in cmap:
            d[g].append(group_abs[g][i])
    cx, cy, cc = [], [], []
    for k, gd in cell_groups.items():
        means = {g: float(np.mean(v)) if v else 0.0 for g, v in gd.items()}
        dom = max(means, key=means.get)
        cx.append(k[0] * cell + cell / 2)
        cy.append(k[1] * cell + cell / 2)
        cc.append(cmap[dom])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cx, cy, c=cc, s=14, marker="s")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    ax.set_title(f"XAI spatial dominance — {fold_name} (W4) on held-out region")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=g)
        for g, c in cmap.items()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    p = config.FIGURES_DIR / f"xai_spatial_{fold_name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------
# Verdict + report writer
# ----------------------------------------------------------------------


def _delta_lookup(delta_summary: dict, kind: str) -> dict:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in delta_summary[kind]:
        delta_key = "delta_lidar" if kind == "delta_lidar" else "delta_ap"
        out[(r["fold"], r["stratum"])] = {
            "delta": float(r[delta_key]),
            "lo": float(r["ci_lo"]),
            "hi": float(r["ci_hi"]),
            "n": int(r["n"]),
        }
    return out


def compute_verdict(delta_summary: dict, disambig_per_fold: dict) -> tuple[str, dict]:
    """WITHIN-SESSION-HELPS / WITHIN-SESSION-NULL / MIXED.

    Criterion: in-FOV Δ_LiDAR_within ≥ 1 dB on at least 3 of 5 folds AND Δ direction is
    consistently positive on majority of folds (overall stratum).
    """
    dl = _delta_lookup(delta_summary, "delta_lidar")
    in_fov_clearing = 0
    in_fov_per_fold = {}
    for fold in FOLDS:
        d = dl.get((fold, "in_fov"))
        in_fov_per_fold[fold] = d["delta"] if d else float("nan")
        if d and d["delta"] >= config.DELTA_LIDAR_THRESHOLD_DB:
            in_fov_clearing += 1
    overall_pos = sum(
        1 for fold in FOLDS
        if dl.get((fold, "overall")) and dl[(fold, "overall")]["delta"] > 0
    )
    overall_neg = sum(
        1 for fold in FOLDS
        if dl.get((fold, "overall")) and dl[(fold, "overall")]["delta"] < 0
    )

    helps_n = config.WITHIN_HELPS_FOLDS_REQUIRED  # 3 of 5
    if in_fov_clearing >= helps_n and overall_pos > overall_neg:
        verdict = "WITHIN-SESSION-HELPS"
    elif in_fov_clearing == 0 and overall_neg >= overall_pos:
        verdict = "WITHIN-SESSION-NULL"
    else:
        # in-between: some folds positive, some not — call it MIXED.
        if in_fov_clearing >= helps_n:
            verdict = "WITHIN-SESSION-HELPS"
        elif in_fov_clearing == 0:
            verdict = "WITHIN-SESSION-NULL"
        else:
            verdict = "MIXED"

    diagnostics = {
        "in_fov_clearing": in_fov_clearing,
        "in_fov_per_fold": in_fov_per_fold,
        "overall_positive_folds": overall_pos,
        "overall_negative_folds": overall_neg,
        "threshold_dB": config.DELTA_LIDAR_THRESHOLD_DB,
        "folds_required": helps_n,
        "disambig_per_fold": disambig_per_fold,
    }
    return verdict, diagnostics


def _paper_framing(verdict: str) -> str:
    if verdict == "WITHIN-SESSION-HELPS":
        return (
            "Project A + B paper: cross-session ablation showed LiDAR does not transfer (Project A); "
            "within-session ablation shows it does help (Project B). The headline becomes a contrast — "
            "LiDAR-derived features encode useful within-session structure but learn route-specific "
            "shortcuts that fail to transfer. Frame the paper around this disconnect."
        )
    if verdict == "WITHIN-SESSION-NULL":
        return (
            "Negative-result paper: LiDAR features add no measurable value to AP-relative geometry "
            "even within a single session, on a dataset with strong geometric coverage. The paper's "
            "contribution is methodological — a careful within-session test that controls for "
            "session-shift confounds and finds no signal."
        )
    return (
        "Mixed-regional paper: LiDAR helps in some spatial regions and hurts/no-effect in others. "
        "The paper should map where LiDAR adds value (e.g., obstructed regions) versus where it "
        "behaves as a noisy position proxy. Useful for deployment guidance."
    )


def write_report(
    *,
    region_summary: dict,
    delta_summary: dict,
    disambig_verdict: str,
    disambig_diagnostics: dict,
    fits: pd.DataFrame,
    verdict: str,
    verdict_diagnostics: dict,
    shap_fold: str,
    shap_metadata: dict,
) -> Path:
    sha = data_io.compute_dataset_sha()
    expected_sha = data_io.read_expected_sha()
    sha_match = sha == expected_sha

    dl = _delta_lookup(delta_summary, "delta_lidar")

    def _delta_str(d: dict | None) -> str:
        if d is None:
            return "—"
        return f"{d['delta']:+.2f} dB [{d['lo']:+.2f}, {d['hi']:+.2f}]"

    md: list[str] = []
    md.append("# Project B results report — within-session deep dive on 15.03.2026\n")
    md.append(
        "Project B is the within-session leave-region-out (WLRO) ablation on 15.03 only. "
        "The cross-session counterpart is Project A (`docs/p1_project_a/results_report.md`).\n"
    )

    # ---- Section 0: TL;DR ----
    md.append("## 0. TL;DR\n")
    md.append(
        "**Feature stack: leakage-fixed.** This report uses the leakage-fixed feature stack: "
        f"3 telemetry features ({', '.join('`'+t+'`' for t in config.LEAKAGE_FIXED_TELEMETRY)}) "
        "plus position, AP-relative geometry, and LiDAR. Five features from the original locked "
        "feature stack were removed post-hoc as either router-side (target leakage), "
        "within-session-only (deployment leakage), or constant sentinel (no information). See "
        f"`MIGRATION_LOG.md` for the full audit trail. (`feature_stack_version = \"{config.FEATURE_STACK_VERSION}\"`).\n"
    )
    md.append("- **Δ_LiDAR_within (W2 − W4) per fold (overall / in-FOV / out-of-FOV)**:")
    for fold in FOLDS:
        ov = _delta_str(dl.get((fold, "overall")))
        inf = _delta_str(dl.get((fold, "in_fov")))
        outf = _delta_str(dl.get((fold, "out_of_fov")))
        md.append(f"  - {fold}: overall {ov}; in-FOV {inf}; out-of-FOV {outf}")
    md.append(
        f"- **Folds clearing the {config.DELTA_LIDAR_THRESHOLD_DB} dB in-FOV Δ_LiDAR_within threshold**: "
        f"{verdict_diagnostics['in_fov_clearing']} / {len(FOLDS)} "
        f"(required ≥ {config.WITHIN_HELPS_FOLDS_REQUIRED})."
    )
    md.append(f"- **Disambiguation verdict**: {disambig_verdict}.")
    md.append(f"- **Verdict**: **{verdict}**.")
    md.append(f"- **Recommended paper framing**: {_paper_framing(verdict)}")
    n_fits_expected = 41
    md.append(
        f"- **All {n_fits_expected} fits succeeded**: "
        f"{'yes' if len(fits) == n_fits_expected else f'no — {len(fits)} fits in inventory'}."
    )
    md.append("")

    # ---- Section 1: Inputs ----
    md.append("## 1. Inputs and protocol\n")
    md.append(f"- Dataset: `data/phase1/dataset.parquet` (SHA-256 `{sha}`).")
    md.append(
        f"  - Expected SHA-256 from `dataset.sha256`: `{expected_sha}` "
        f"({'match' if sha_match else 'MISMATCH'})."
    )
    md.append(f"- Filter: `session_date == '{config.SESSION}'` AND `~anomaly_flag` → 232,279 rows.")
    md.append(f"- Region partition: K-means with k={config.N_REGIONS} on (x_m, y_m), method = `{region_summary['method']}`.")
    md.append("- Per-region row counts:")
    for r in range(1, config.N_REGIONS + 1):
        md.append(f"  - region {r}: {int(region_summary['region_sizes'][r]):,}")
    md.append("- Folds: 5 leave-region-out folds (R-1 … R-5).")
    md.append(
        f"- Validation split: random {int(config.VAL_FRACTION * 100)}% of training rows per fold "
        f"(seeded; IID with train across regions)."
    )
    md.append(
        f"- Buffer-zone sanity check: refit {config.BUFFER_FOLD} after dropping training rows within "
        f"{config.BUFFER_DISTANCE_M} m of any held-out-region row."
    )
    md.append(
        "- Hyperparameter robustness: refit W4 under H1 (max_depth=4) on every fold."
    )
    md.append("")

    # ---- Section 2: Spatial regions ----
    md.append("## 2. Spatial regions\n")
    md.append("![Spatial regions](figures/regions_map.png)\n")
    md.append((config.TABLES_DIR / "regions_summary.md").read_text(encoding="utf-8"))
    md.append("")

    # ---- Section 3: Within-session ablation results ----
    md.append("## 3. Within-session ablation results\n")
    md.append("### 3.1 Headline ablation table (5 folds × 3 strata × 6 variants)\n")
    md.append((config.TABLES_DIR / "wlro_ablation_rmse.md").read_text(encoding="utf-8"))
    md.append("\n![WLRO ablation RMSE](figures/wlro_ablation_chart.png)\n")

    md.append("\n### 3.2 Δ_LiDAR_within per fold per stratum\n")
    md.append((config.TABLES_DIR / "delta_lidar_within.md").read_text(encoding="utf-8"))

    md.append("\n### 3.3 Δ_AP-relative_within per fold per stratum\n")
    md.append((config.TABLES_DIR / "delta_ap_relative_within.md").read_text(encoding="utf-8"))

    md.append("\n#### Δ_position (W0 vs per-region mean baseline)\n")
    md.append((config.TABLES_DIR / "delta_position_within.md").read_text(encoding="utf-8"))

    md.append("\n### 3.4 Disambiguation (W4 / W4' / W4'')\n")
    md.append((config.TABLES_DIR / "disambig_summary.md").read_text(encoding="utf-8"))

    md.append("\n### 3.5 Buffer-zone sensitivity (R-1)\n")
    md.append((config.TABLES_DIR / "buffer_sensitivity.md").read_text(encoding="utf-8"))

    md.append("\n### 3.6 Hyperparameter sensitivity (W4 locked vs H1)\n")
    md.append((config.TABLES_DIR / "hyperparam_sensitivity.md").read_text(encoding="utf-8"))

    # ---- Section 4: XAI ----
    md.append(f"\n## 4. XAI ({shap_fold}, W4)\n")
    md.append(f"- Selection: {shap_metadata.get('selection_justification', 'default')}\n")
    n_used = shap_metadata.get("n_rows_used")
    n_total = shap_metadata.get("subsampled_from")
    n_used_str = f"{int(n_used):,}" if isinstance(n_used, (int, float)) else "n/a"
    n_total_str = f"{int(n_total):,}" if isinstance(n_total, (int, float)) else "n/a"
    md.append(f"- SHAP rows used: {n_used_str} (subsampled from {n_total_str})\n")
    md.append("\n### 4.1 Feature-group importance\n")
    md.append(f"![XAI group importance](figures/xai_group_importance_{shap_fold}.png)\n")
    md.append((config.TABLES_DIR / f"xai_group_importance_{shap_fold}.md").read_text(encoding="utf-8"))
    md.append("\n### 4.2 Sign-of-effect (single-fold)\n")
    md.append((config.TABLES_DIR / f"xai_sign_{shap_fold}.md").read_text(encoding="utf-8"))
    md.append("\n### 4.3 Spatial dominance on the held-out region\n")
    md.append(f"![XAI spatial](figures/xai_spatial_{shap_fold}.png)\n")

    # ---- Section 5: Synthesis ----
    md.append("\n## 5. Synthesis vs Project A\n")
    md.append(_synthesis(verdict, dl, disambig_verdict, disambig_diagnostics))

    # ---- Section 6: Recommendation ----
    md.append("\n## 6. Recommendation\n")
    md.append(f"- Final paper framing: **{verdict}**.\n")
    md.append(_paper_framing(verdict) + "\n")
    if verdict == "WITHIN-SESSION-HELPS":
        md.append(
            "- Project A + B structure: §1–4 = cross-session ablation result (Project A). §5 = "
            "within-session ablation on 15.03 (Project B). §6 = XAI on 15.03 R-X. §7 = synthesis: "
            "what within-session learns that does not transfer.\n"
        )
    elif verdict == "WITHIN-SESSION-NULL":
        md.append(
            "- Negative-result paper structure: §1–3 setup and dataset. §4 cross-session result. "
            "§5 within-session result on the cleanest session. §6 disambiguation: AP-relative "
            "geometry is sufficient. §7 implications for LiDAR-based propagation modelling.\n"
        )
    else:
        md.append(
            "- Mixed-regional paper structure: §1–3 setup. §4 ablation by region. §5 spatial XAI: "
            "where does LiDAR add value? §6 deployment guidance.\n"
        )

    # ---- Section 7: Reproducibility ----
    md.append("\n## 7. Reproducibility\n")
    md.append(f"- Seed: `SEED = {config.SEED}` everywhere (XGBoost, NumPy, K-means).")
    md.append("- Run end-to-end: `python -m scripts.p1_project_b.run_all`.")
    md.append("- Stages: `run_modeling` → `run_xai` → `build_results_report`.")
    md.append(f"- Total wall-clock for fits: {float(fits['wallclock_s'].sum()):.1f} s.")
    md.append("")
    md.append("### Model SHA-256 inventory\n")
    md.append("| fold | variant | hyperparams | n_features | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for _, r in fits.iterrows():
        md.append(
            f"| {r['fold']} | {r['variant']} | {r['hyperparams']} | {int(r['n_features'])} | "
            f"{int(r['best_iteration'])} | {int(r['n_train'])} | {int(r['n_val'])} | "
            f"{int(r['n_test'])} | {float(r['wallclock_s']):.1f} | `{r['model_sha256'][:16]}…` |"
        )

    config.REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    return config.REPORT_PATH


def _synthesis(
    verdict: str,
    dl: dict,
    disambig_verdict: str,
    disambig_diagnostics: dict,
) -> str:
    fc_in = "; ".join(f"{f} {dl[(f, 'in_fov')]['delta']:+.2f}" for f in FOLDS if dl.get((f, "in_fov")))
    lines = [
        f"- **Within-session Δ_LiDAR_within (in-FOV) per fold**: {fc_in}.",
        f"- **Within-session disambiguation**: {disambig_verdict} "
        f"(per-fold: {disambig_diagnostics.get('per_fold', {})}).",
        "- **Compared with Project A (cross-session)**: Project A's F-B in-FOV Δ_LiDAR was −0.52 dB "
        "(LiDAR hurts) and the disambiguation was 'LiDAR removable'. Project B isolates whether the "
        "issue was session shift or LiDAR itself.",
    ]
    if verdict == "WITHIN-SESSION-HELPS":
        lines.append(
            "- **Implication**: LiDAR-derived features encode useful structure within a single session "
            "but fail to transfer across sessions — the cross-session shift is the dominant failure mode."
        )
    elif verdict == "WITHIN-SESSION-NULL":
        lines.append(
            "- **Implication**: even within a single session with consistent route, LiDAR features add "
            "no measurable value beyond AP-relative geometry. The Project A pivot interpretation "
            "(LiDAR is a position proxy) holds within-session as well."
        )
    else:
        lines.append(
            "- **Implication**: LiDAR's value is regional. Some held-out regions show consistent gains; "
            "others do not. The paper should map where, not whether."
        )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:
    print("[report] preparing regions and loading metrics")
    non_anom_with_region, region_assn = run_modeling.prepare_regions()
    region_summary = {
        "method": region_assn.method,
        "region_sizes": region_assn.region_sizes,
    }

    wlro = pd.read_parquet(config.RESULTS_DIR / "wlro_metrics.parquet")
    disambig = pd.read_parquet(config.RESULTS_DIR / "disambig_metrics.parquet")
    robustness = pd.read_parquet(config.RESULTS_DIR / "robustness_metrics.parquet")
    fits = pd.read_parquet(config.RESULTS_DIR / "fit_inventory.parquet")

    print("[report] regions figure + table")
    render_regions_map(non_anom_with_region)
    render_regions_summary_table(non_anom_with_region)

    print("[report] ablation table + chart")
    render_wlro_ablation_table(wlro)
    render_ablation_chart(wlro)

    print("[report] delta tables")
    _, _, _, delta_summary = render_delta_tables()

    print("[report] disambiguation")
    _, disambig_verdict, disambig_diag = render_disambig_table(disambig)

    print("[report] sensitivity tables")
    _, buffer_diag = render_buffer_sensitivity(robustness, wlro)
    _, hp_diag = render_hyperparam_sensitivity(robustness, wlro)

    print("[report] XAI")
    # SHAP fold = whatever was actually run. Pick from filesystem; fall back to default.
    shap_files = sorted(config.RESULTS_DIR.glob("shap_R-*.parquet"))
    if shap_files:
        # Use the SHAP file whose corresponding metadata exists; if multiple, prefer the brief's default.
        candidates = [p for p in shap_files if not p.name.endswith("_metadata.json")]
        # Try default first.
        default_path = config.RESULTS_DIR / f"shap_{config.SHAP_FOLD_DEFAULT}.parquet"
        if default_path.exists():
            shap_path = default_path
        else:
            shap_path = candidates[0]
        shap_fold = shap_path.stem.replace("shap_", "")
    else:
        raise FileNotFoundError("no SHAP parquet found; run run_xai first")
    shap_metadata_path = _shap_metadata_path(shap_fold)
    shap_metadata = json.loads(shap_metadata_path.read_text(encoding="utf-8")) if shap_metadata_path.exists() else {}

    render_xai_group_importance(shap_fold)
    render_xai_sign_table(shap_fold)
    render_xai_spatial(shap_fold)

    print("[report] verdict")
    verdict, verdict_diag = compute_verdict(delta_summary, disambig_diag.get("per_fold", {}))
    print(f"[report] verdict = {verdict}")
    print(f"[report]   diagnostics = {verdict_diag}")

    print("[report] writing report")
    p = write_report(
        region_summary=region_summary,
        delta_summary=delta_summary,
        disambig_verdict=disambig_verdict,
        disambig_diagnostics=disambig_diag,
        fits=fits,
        verdict=verdict,
        verdict_diagnostics=verdict_diag,
        shap_fold=shap_fold,
        shap_metadata=shap_metadata,
    )
    print(f"[report] wrote {p}")


if __name__ == "__main__":
    main()
