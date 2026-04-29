"""Project B — R-4 focused diagnostic.

Tests whether R-4's main-run +1.09 dB Δ_LiDAR_within (in-FOV) survives:
  - Diagnostic A (buffer): 6 fits with locked hyperparameters and a 1 m buffer-zone
    exclusion of training rows near the held-out region 4.
  - Diagnostic B (H1): 5 new fits (W0,W1,W2,W3,W4pp) under H1 (max_depth=4); the
    R-4_W4_H1 fit from the main hyperparameter sensitivity check is reused.

Writes a consolidated metrics parquet and a markdown report. Does not touch any
other fold's artifacts.

Naming: H1 fits follow the existing convention `R-4_{variant}_H1.json` so the
already-cached `R-4_W4_H1.json` is naturally reused. (The brief suggested
`R-4_H1_{variant}.json`; we deviate to avoid two parallel naming schemes for the
same artifact type — documented in the diagnostic report.)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from scripts.p1_project_a import metrics as a_metrics

from . import config, folds as folds_mod, run_modeling


@dataclass
class R4FitResult:
    diagnostic: str          # "buffer" or "H1"
    variant: str
    fold_name: str
    hyperparams: str
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    n_train_dropped_by_buffer: int
    buffer_distance_m: float | None
    wallclock_s: float
    model_sha256: str
    model_path: str
    predictions_path: str


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def run_buffer_fits(non_anom_with_region: pd.DataFrame) -> tuple[list[R4FitResult], list[dict]]:
    """6 locked fits on R-4 with 1 m buffer-zone exclusion."""
    fits: list[R4FitResult] = []
    rows: list[dict] = []

    fold = folds_mod.build_fold_with_buffer(
        non_anom_with_region,
        "R-4",
        buffer_distance_m=config.BUFFER_DISTANCE_M,
    )
    print(
        f"[r4-diag] buffer fold {fold.name}: dropped {fold.n_train_dropped_by_buffer} train rows; "
        f"train n={len(fold.train)}, val n={len(fold.val)}, test n={len(fold.test)}"
    )

    for variant in config.MAIN_VARIANTS:
        features = config.VARIANTS[variant]
        print(f"[r4-diag] fit {fold.name} / {variant} ({len(features)} feat, locked)")
        fr = run_modeling._run_one_fit(fold, variant, features, hyperparams_label="locked")
        fits.append(R4FitResult(
            diagnostic="buffer",
            variant=variant,
            fold_name=fr.fold,
            hyperparams=fr.hyperparams,
            best_iteration=fr.best_iteration,
            n_features=fr.n_features,
            n_train=fr.n_train,
            n_val=fr.n_val,
            n_test=fr.n_test,
            n_train_dropped_by_buffer=fr.n_train_dropped_by_buffer,
            buffer_distance_m=fr.buffer_distance_m,
            wallclock_s=fr.wallclock_s,
            model_sha256=fr.model_sha256,
            model_path=fr.model_path,
            predictions_path=fr.predictions_path,
        ))
        print(f"   best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s")

        preds = pd.read_parquet(config.CACHE_DIR / f"predictions_{fold.name}_{variant}.parquet")
        for sm in a_metrics.evaluate_predictions(preds):
            rows.append({
                "diagnostic": "buffer",
                "variant": variant,
                "fold": "R-4",  # all metrics describe behaviour ON region 4
                "config": "locked-buffer",
                "buffer_distance_m": float(config.BUFFER_DISTANCE_M),
                "n_train_dropped_by_buffer": int(fold.n_train_dropped_by_buffer),
                **sm,
            })

    return fits, rows


def run_h1_fits(non_anom_with_region: pd.DataFrame) -> tuple[list[R4FitResult], list[dict]]:
    """5 new H1 fits on R-4 (W0, W1, W2, W3, W4pp). W4 reuses the existing main-run cache."""
    fits: list[R4FitResult] = []
    rows: list[dict] = []

    fold = folds_mod.build_fold(non_anom_with_region, "R-4")
    print(
        f"[r4-diag] H1 fold R-4 (no buffer): train n={len(fold.train)}, val n={len(fold.val)}, "
        f"test n={len(fold.test)}"
    )

    new_variants = ["W0", "W1", "W2", "W3", "W4pp"]
    for variant in new_variants:
        features = config.VARIANTS[variant]
        print(f"[r4-diag] fit R-4 / {variant} ({len(features)} feat, H1: max_depth=4)")
        fr = run_modeling._run_one_fit(fold, variant, features, hyperparams_label="H1")
        fits.append(R4FitResult(
            diagnostic="H1",
            variant=variant,
            fold_name=fr.fold,
            hyperparams=fr.hyperparams,
            best_iteration=fr.best_iteration,
            n_features=fr.n_features,
            n_train=fr.n_train,
            n_val=fr.n_val,
            n_test=fr.n_test,
            n_train_dropped_by_buffer=0,
            buffer_distance_m=None,
            wallclock_s=fr.wallclock_s,
            model_sha256=fr.model_sha256,
            model_path=fr.model_path,
            predictions_path=fr.predictions_path,
        ))
        print(f"   best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s")

    # All 6 cells: 5 newly fitted + 1 reused (W4 from the main run's H1 sensitivity check).
    all_h1_variants = ["W0", "W1", "W2", "W3", "W4", "W4pp"]
    for variant in all_h1_variants:
        preds_path = config.CACHE_DIR / f"predictions_R-4_{variant}_H1.parquet"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"Expected H1 predictions at {preds_path}; "
                f"the existing R-4_W4_H1 cache (or one of the new fits) is missing."
            )
        preds = pd.read_parquet(preds_path)
        for sm in a_metrics.evaluate_predictions(preds):
            rows.append({
                "diagnostic": "H1",
                "variant": variant,
                "fold": "R-4",
                "config": "H1-no-buffer",
                "buffer_distance_m": None,
                "n_train_dropped_by_buffer": 0,
                **sm,
            })

    return fits, rows


def load_baseline_metrics() -> list[dict]:
    """Load the existing locked-no-buffer R-4 metrics from cached predictions."""
    rows: list[dict] = []
    for variant in config.MAIN_VARIANTS:
        preds_path = config.CACHE_DIR / f"predictions_R-4_{variant}.parquet"
        preds = pd.read_parquet(preds_path)
        for sm in a_metrics.evaluate_predictions(preds):
            rows.append({
                "diagnostic": "baseline",
                "variant": variant,
                "fold": "R-4",
                "config": "locked-no-buffer",
                "buffer_distance_m": None,
                "n_train_dropped_by_buffer": 0,
                **sm,
            })
    return rows


# ---------------------------------------------------------------------------
# Diagnostic deltas + verdict
# ---------------------------------------------------------------------------


def _rmse_for(metrics_df: pd.DataFrame, *, config_tag: str, variant: str, stratum: str) -> float:
    sub = metrics_df[
        (metrics_df["config"] == config_tag)
        & (metrics_df["variant"] == variant)
        & (metrics_df["stratum"] == stratum)
    ]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["rmse"])


def compute_delta_lidar(metrics_df: pd.DataFrame, config_tag: str, stratum: str) -> float:
    """Δ_LiDAR_within = RMSE(W2) - RMSE(W4) for a given config and stratum."""
    w2 = _rmse_for(metrics_df, config_tag=config_tag, variant="W2", stratum=stratum)
    w4 = _rmse_for(metrics_df, config_tag=config_tag, variant="W4", stratum=stratum)
    return float(w2 - w4)


def disambig_judgment(metrics_df: pd.DataFrame, config_tag: str, *, threshold_dB: float = 0.5) -> str:
    """W4 vs W2 (W4') vs W4'' on R-4 in-FOV."""
    w4 = _rmse_for(metrics_df, config_tag=config_tag, variant="W4", stratum="in_fov")
    w2 = _rmse_for(metrics_df, config_tag=config_tag, variant="W2", stratum="in_fov")
    w4pp = _rmse_for(metrics_df, config_tag=config_tag, variant="W4pp", stratum="in_fov")
    if any(np.isnan(v) for v in [w4, w2, w4pp]):
        return "incomplete"
    complementary = (w4 < w2 - threshold_dB) and (w4 < w4pp - threshold_dB)
    lidar_removable = abs(w4 - w2) < threshold_dB
    ap_removable = abs(w4 - w4pp) < threshold_dB
    if complementary:
        return "complementary"
    if lidar_removable and not complementary:
        return "LiDAR removable"
    if ap_removable and not complementary:
        return "AP-relative removable"
    return "mixed/unclear"


def compute_verdict(buffer_delta: float, h1_delta: float, *, threshold_dB: float = 0.5) -> tuple[str, str]:
    """Return (verdict_tag, framing_text)."""
    buffer_pass = buffer_delta >= threshold_dB
    h1_pass = h1_delta >= threshold_dB
    if buffer_pass and h1_pass:
        return (
            "R-4 IS A ROBUST EXEMPLAR",
            "Mixed-regional paper framing supported. R-4's LiDAR-helps result holds under both proper "
            "spatial-extrapolation testing (1 m buffer) and a less-overfitting hyperparameter config (H1). "
            "The +1.09 dB main-run value is real, possibly somewhat softer in magnitude after corrections.",
        )
    if buffer_pass ^ h1_pass:
        return (
            "R-4 IS PARTIALLY ROBUST",
            "Negative-with-caveat paper framing. R-4's LiDAR gap holds under one of the two corrections "
            "but not the other. Disambiguation finding still holds; the headline is the cross-session-and-"
            "mostly-within-session negative result, with R-4 mentioned as a region where the gap narrows "
            "but does not robustly survive both checks.",
        )
    return (
        "R-4 IS ILLUSORY",
        "Cleanly negative paper framing. R-4's main-run +1.09 dB was an artifact of "
        "(a) spatial autocorrelation and/or (b) hyperparameter overfitting. The headline is "
        "'LiDAR doesn't help cross-session and doesn't help within-session under proper testing'.",
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


REPORT_PATH = config.DOCS_DIR / "r4_diagnostic.md"
METRICS_PATH = config.RESULTS_DIR / "r4_diagnostic_metrics.parquet"


def _ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"


def _signed_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def render_per_variant_table(metrics_df: pd.DataFrame, baseline_tag: str, alt_tag: str, alt_label: str) -> str:
    """Render: variant × stratum × {RMSE(baseline), RMSE(alt), Δ}."""
    md = [
        f"| Variant | Stratum | RMSE({baseline_tag}) | RMSE({alt_label}) | Δ_RMSE ({alt_label} − {baseline_tag}) |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in config.MAIN_VARIANTS:
        for stratum in ["overall", "in_fov", "out_of_fov"]:
            r0 = _rmse_for(metrics_df, config_tag=baseline_tag, variant=variant, stratum=stratum)
            r1 = _rmse_for(metrics_df, config_tag=alt_tag, variant=variant, stratum=stratum)
            d = r1 - r0
            md.append(
                f"| {variant} | {stratum} | "
                f"{'—' if np.isnan(r0) else f'{r0:.3f}'} | "
                f"{'—' if np.isnan(r1) else f'{r1:.3f}'} | "
                f"{'—' if np.isnan(d) else f'{d:+.3f}'} |"
            )
    return "\n".join(md)


def render_delta_lidar_table(metrics_df: pd.DataFrame, configs: list[tuple[str, str]]) -> str:
    md = [
        "| Stratum | " + " | ".join(label for _tag, label in configs) + " |",
        "|---|" + "|".join(["---:"] * len(configs)) + "|",
    ]
    for stratum in ["overall", "in_fov", "out_of_fov"]:
        cells = [stratum]
        for tag, _label in configs:
            d = compute_delta_lidar(metrics_df, tag, stratum)
            cells.append("—" if np.isnan(d) else f"{d:+.3f}")
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)


def write_report(
    metrics_df: pd.DataFrame,
    fits: list[R4FitResult],
    *,
    main_run_delta_lidar_in_fov: float = 1.09,
) -> None:
    # Headline numbers.
    buffer_delta_in_fov = compute_delta_lidar(metrics_df, "locked-buffer", "in_fov")
    h1_delta_in_fov = compute_delta_lidar(metrics_df, "H1-no-buffer", "in_fov")
    baseline_delta_in_fov = compute_delta_lidar(metrics_df, "locked-no-buffer", "in_fov")

    verdict_tag, framing_text = compute_verdict(buffer_delta_in_fov, h1_delta_in_fov, threshold_dB=0.5)

    # Disambiguation per config.
    dj_locked = disambig_judgment(metrics_df, "locked-no-buffer")
    dj_buffer = disambig_judgment(metrics_df, "locked-buffer")
    dj_h1 = disambig_judgment(metrics_df, "H1-no-buffer")

    md: list[str] = []
    md.append("# Project B — R-4 focused diagnostic\n")
    md.append(
        "Tests whether R-4's main-run +1.09 dB Δ_LiDAR_within (in-FOV) survives two orthogonal "
        "corrections: a 1 m buffer-zone exclusion (spatial-autocorrelation control) and a less-"
        "aggressive hyperparameter config (H1: max_depth=4).\n"
    )

    # ---- TL;DR ----
    md.append("## 0. TL;DR\n")
    md.append(
        f"- **Buffer-zone Δ_LiDAR_within (in-FOV) on R-4**: {buffer_delta_in_fov:+.3f} dB "
        f"(vs main-run no-buffer {baseline_delta_in_fov:+.3f} dB, also reproduced here)."
    )
    md.append(
        f"- **H1 Δ_LiDAR_within (in-FOV) on R-4**: {h1_delta_in_fov:+.3f} dB "
        f"(vs main-run locked {main_run_delta_lidar_in_fov:+.2f} dB)."
    )
    md.append(f"- **Verdict**: **{verdict_tag}**.")
    md.append(f"- **Recommended paper framing**: {framing_text}")
    md.append(f"- **All 12 fits succeeded**: yes ({len(fits)} new fits + 1 reused R-4_W4_H1 from main run).")
    md.append("")

    # ---- Section 1: Context ----
    md.append("## 1. Context\n")
    md.append(
        "Project B's main run found that R-4 is the only fold with Δ_LiDAR_within ≥ 1 dB on "
        "in-FOV (+1.09 dB; all other folds: −9.06, −0.89, −0.14, −0.04). R-4 was therefore "
        "the candidate exemplar for a mixed-regional paper framing. Two methodological caveats "
        "from the main report cast doubt on whether that R-4 result is real:\n"
    )
    md.append(
        "- The R-1 buffer-zone test showed every variant moves by ≥ 0.5 dB under buffer "
        "(W0 alone moved by +12 dB). Leave-region-out is therefore contaminated by short-range "
        "spatial autocorrelation. R-4 was not previously buffer-tested.\n"
    )
    md.append(
        "- The R-1 H1 hyperparameter sensitivity test showed Δ +4.76 dB improvement under "
        "H1 — i.e., the locked depth-6 config catastrophically overfits R-1. R-4's H1 sensitivity "
        "had been measured only for W4 (one variant), not for the full disambiguation ladder.\n"
    )
    md.append(
        "This diagnostic settles whether R-4's +1.09 dB Δ_LiDAR_within is a clean physical finding "
        "or an artifact of either spatial autocorrelation or overfitting.\n"
    )

    # ---- Section 2: Buffer diagnostic ----
    md.append("## 2. Buffer-zone diagnostic (R-4)\n")
    md.append(
        f"Drop training rows within {config.BUFFER_DISTANCE_M:.1f} m of any R-4 row, refit under "
        "locked hyperparameters, evaluate on the unchanged R-4 test set.\n"
    )
    buffer_fit = next((f for f in fits if f.diagnostic == "buffer"), None)
    if buffer_fit is not None:
        md.append(
            f"Post-buffer training-set size: {buffer_fit.n_train:,} rows "
            f"(dropped {buffer_fit.n_train_dropped_by_buffer:,} rows within {config.BUFFER_DISTANCE_M:.1f} m).\n"
        )
    md.append("### 2.1 Per-variant RMSE: locked-no-buffer vs locked-buffer\n")
    md.append(render_per_variant_table(metrics_df, "locked-no-buffer", "locked-buffer", "locked-buffer"))
    md.append("\n### 2.2 Δ_LiDAR_within = RMSE(W2) − RMSE(W4)\n")
    md.append(render_delta_lidar_table(
        metrics_df,
        configs=[("locked-no-buffer", "locked-no-buffer"), ("locked-buffer", "locked-buffer")],
    ))
    md.append("")
    md.append(f"\n**Disambiguation under locked-buffer (R-4 in-FOV, threshold = 0.5 dB)**: {dj_buffer}.")
    md.append("")
    if buffer_delta_in_fov >= 0.5:
        md.append(
            f"**Verdict (Diagnostic A)**: PASS — Δ_LiDAR_within (in-FOV) under buffer = "
            f"{buffer_delta_in_fov:+.3f} dB ≥ +0.5 dB threshold."
        )
    else:
        md.append(
            f"**Verdict (Diagnostic A)**: FAIL — Δ_LiDAR_within (in-FOV) under buffer = "
            f"{buffer_delta_in_fov:+.3f} dB < +0.5 dB threshold."
        )
    md.append("")

    # ---- Section 3: H1 diagnostic ----
    md.append("## 3. H1 hyperparameter diagnostic (R-4)\n")
    md.append(
        "Refit under H1 (max_depth=4; same eta, λ, n_estimators, early_stop) for all 6 variants. "
        "W4 reuses the existing main-run R-4 H1 fit (`R-4_W4_H1.json`); W0, W1, W2, W3, W4'' are new.\n"
    )
    md.append("### 3.1 Per-variant RMSE: locked vs H1\n")
    md.append(render_per_variant_table(metrics_df, "locked-no-buffer", "H1-no-buffer", "H1"))
    md.append("\n### 3.2 Δ_LiDAR_within = RMSE(W2) − RMSE(W4)\n")
    md.append(render_delta_lidar_table(
        metrics_df,
        configs=[("locked-no-buffer", "locked"), ("H1-no-buffer", "H1")],
    ))
    md.append("")
    md.append(f"\n**Disambiguation under H1 (R-4 in-FOV, threshold = 0.5 dB)**: {dj_h1}.")
    md.append("")
    if h1_delta_in_fov >= 0.5:
        md.append(
            f"**Verdict (Diagnostic B)**: PASS — Δ_LiDAR_within (in-FOV) under H1 = "
            f"{h1_delta_in_fov:+.3f} dB ≥ +0.5 dB threshold."
        )
    else:
        md.append(
            f"**Verdict (Diagnostic B)**: FAIL — Δ_LiDAR_within (in-FOV) under H1 = "
            f"{h1_delta_in_fov:+.3f} dB < +0.5 dB threshold."
        )
    md.append("")

    md.append(
        "Note on the threshold choice: the §6 verdict criteria use a 0.5 dB threshold, more "
        "permissive than the original 1.0 dB pivot threshold from Project A / the main Project B "
        "run. This is intentional — the diagnostic asks whether the +1.09 dB result *survives at "
        "all* under stricter conditions, not whether it independently clears 1 dB. A diagnostic "
        "that is too strict to be informative is no diagnostic.\n"
    )

    # ---- Section 4: Combined verdict ----
    md.append("## 4. Combined verdict\n")
    md.append(f"- Diagnostic A (buffer): Δ_LiDAR_within (in-FOV) = {buffer_delta_in_fov:+.3f} dB.")
    md.append(f"- Diagnostic B (H1):     Δ_LiDAR_within (in-FOV) = {h1_delta_in_fov:+.3f} dB.")
    md.append(f"- Disambiguation per config:")
    md.append(f"  - locked-no-buffer (existing main run): **{dj_locked}**.")
    md.append(f"  - locked-buffer:                        **{dj_buffer}**.")
    md.append(f"  - H1-no-buffer:                         **{dj_h1}**.")
    md.append("")
    md.append(f"### Verdict: **{verdict_tag}**\n")
    md.append(framing_text)
    md.append("")

    if verdict_tag == "R-4 IS A ROBUST EXEMPLAR":
        md.append("\n#### Mixed-regional paper outline\n")
        md.append(
            "- §1–3 setup: dataset, AGV, AP, region partition (the same 5 K-means regions used here).\n"
            "- §4 cross-session leave-one-run-out result (Project A): LiDAR does not transfer.\n"
            "- §5 within-session leave-region-out result (Project B): LiDAR helps in R-4 only, "
            "fails in R-1 catastrophically.\n"
            "- §6 R-4 robustness: the buffer-zone and H1 diagnostics in this report.\n"
            "- §7 spatial XAI: where does LiDAR add value vs. where does it behave as a position "
            "proxy? Tie to the per-region distance-to-AP bands.\n"
            "- §8 deployment guidance: LiDAR-derived features are usable in regions matching "
            "R-4's geometric profile and unreliable in others.\n"
        )
    elif verdict_tag == "R-4 IS PARTIALLY ROBUST":
        md.append("\n#### Negative-with-caveat paper outline\n")
        md.append(
            "- Headline negative result remains the cross-session and within-session disambiguation: "
            "AP-relative geometry largely subsumes LiDAR.\n"
            "- R-4 mentioned as a region where Δ_LiDAR survives one of two corrections; this is "
            "consistent with regional structure but does not robustly support a regional-deployment claim.\n"
            "- The diagnostic itself (this report) goes into §6 as a methodological figure.\n"
        )
    else:  # R-4 IS ILLUSORY
        md.append("\n#### Cleanly negative paper outline\n")
        md.append(
            "- §1–3 setup (as above).\n"
            "- §4 cross-session result: LiDAR does not transfer (Project A).\n"
            "- §5 within-session result: LiDAR does not help within-session either (Project B "
            "main run + this R-4 diagnostic).\n"
            "- §6 disambiguation: AP-relative geometry suffices; LiDAR-derived features behave as "
            "a noisy position proxy under proper testing.\n"
            "- §7 implications for LiDAR-based propagation modelling: features that look promising "
            "in univariate analyses (P0.5 Spearman ρ) and even in some cross-session in-FOV slices "
            "(F-A in-FOV +0.30 dB; F-C in-FOV +1.14 dB) do not survive properly-controlled tests.\n"
        )

    # ---- Section 5: Reproducibility ----
    md.append("\n## 5. Reproducibility\n")
    md.append(f"- Seed: `SEED = {config.SEED}` everywhere.")
    md.append("- Run end-to-end: `python -m scripts.p1_project_b.run_r4_diagnostic`.")
    md.append(
        "- Naming convention: H1 fits use the existing codebase pattern `R-4_{variant}_H1.json` "
        "(matching the pre-cached `R-4_W4_H1.json`), not the brief's suggested "
        "`R-4_H1_{variant}.json` — this avoids two parallel naming schemes for the same artifact "
        "type. Buffer fits use `R-4_buffer_{variant}.json` matching the main-run R-1_buffer "
        "convention."
    )
    total_wall = sum(f.wallclock_s for f in fits)
    md.append(f"- Total wall-clock for new fits: {total_wall:.1f} s.")
    md.append("\n### Model SHA-256 inventory (new fits only; W4 H1 reused from main run)\n")
    md.append("| diagnostic | variant | model file | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for f in fits:
        md.append(
            f"| {f.diagnostic} | {f.variant} | `{f.model_path.replace(chr(92), '/').split('/')[-1]}` | "
            f"{f.best_iteration} | {f.n_train} | {f.n_val} | {f.n_test} | "
            f"{f.wallclock_s:.1f} | `{f.model_sha256[:16]}…` |"
        )
    md.append("")

    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[r4-diag] wrote {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    print("[r4-diag] preparing regions on 15.03 non-anomaly rows")
    non_anom_with_region, _ = run_modeling.prepare_regions()

    print("[r4-diag] === Diagnostic A — buffer-zone (6 fits) ===")
    buffer_fits, buffer_rows = run_buffer_fits(non_anom_with_region)

    print("[r4-diag] === Diagnostic B — H1 (5 new fits + 1 reused) ===")
    h1_fits, h1_rows = run_h1_fits(non_anom_with_region)

    print("[r4-diag] loading locked-no-buffer baseline metrics")
    baseline_rows = load_baseline_metrics()

    metrics_df = pd.DataFrame(buffer_rows + h1_rows + baseline_rows)
    metrics_df.to_parquet(METRICS_PATH, index=False)
    print(f"[r4-diag] wrote {METRICS_PATH} ({len(metrics_df)} metric rows)")

    fits = buffer_fits + h1_fits
    fits_df = pd.DataFrame([asdict(f) for f in fits])
    fits_inventory_path = config.RESULTS_DIR / "r4_diagnostic_fit_inventory.parquet"
    fits_df.to_parquet(fits_inventory_path, index=False)
    print(f"[r4-diag] wrote {fits_inventory_path} ({len(fits_df)} fit rows)")

    print("[r4-diag] writing report")
    write_report(metrics_df, fits)

    print(f"[r4-diag] done in {time.time() - t0:.1f}s; {len(fits)} new fits + 1 reused (R-4_W4_H1).")


if __name__ == "__main__":
    main()
