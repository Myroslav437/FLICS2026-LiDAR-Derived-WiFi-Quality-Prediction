"""Phase 1 hyperparameter robustness diagnostic on F-B.

Tests whether the F-B in-FOV Δ_LiDAR ≤ 1 dB pivot trigger from the locked Phase 1 run is
robust to hyperparameter and validation-protocol choice. Refits B1 and B5 on F-B only.

8 fits total:
  - {H1, H2, H3} × {B1, B5}     (6 fits, chronological val split)
  - {H1, H2}    × {B5} randval  (2 fits, random 10%-per-session val)

Locked F-B predictions are loaded from scripts/p1_project_a/cache/ — not refit.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, data_io, folds, metrics, training


DIAG_DIR = config.DIAGNOSTIC_DIR
DIAG_MODELS = DIAG_DIR / "models"
DIAG_CACHE = DIAG_DIR / "cache"
DIAG_METRICS_PATH = DIAG_DIR / "robustness_metrics.parquet"
DIAG_FIT_INVENTORY_PATH = DIAG_DIR / "robustness_fit_inventory.parquet"

for _d in (DIAG_MODELS, DIAG_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


CONFIGS: dict[str, dict] = {
    "H1": {  # less flexible
        "max_depth": 4,
        "eta": 0.05,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
    },
    "H2": {  # slow, regularized
        "max_depth": 6,
        "eta": 0.02,
        "min_child_weight": 10,
        "reg_lambda": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_estimators": 5000,
        "early_stopping_rounds": 200,
    },
    "H3": {  # strong L2
        "max_depth": 8,
        "eta": 0.05,
        "min_child_weight": 5,
        "reg_lambda": 20.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
    },
}


@dataclasses.dataclass
class FitRecord:
    variant: str
    config: str
    val_protocol: str  # "chronological" | "random"
    best_iteration: int
    n_train: int
    n_val: int
    n_test: int
    wallclock_s: float
    model_sha256: str


# ----------------------------------------------------------------------
# F-B fold builder with random per-session val split
# ----------------------------------------------------------------------


def build_fb_random_val(non_anom: pd.DataFrame) -> folds.FoldSplit:
    """F-B with a random 10%-per-session validation split (test set unchanged)."""
    train_sessions, test_session = config.FOLDS["F-B"]
    test_df = non_anom[non_anom["session_date"] == test_session].copy().reset_index(drop=True)

    rng = np.random.default_rng(config.SEED)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for sess in train_sessions:
        sess_df = non_anom[non_anom["session_date"] == sess].copy().reset_index(drop=True)
        n = len(sess_df)
        n_val = max(1, int(round(n * 0.10)))
        idx = rng.choice(n, size=n_val, replace=False)
        is_val = np.zeros(n, dtype=bool)
        is_val[idx] = True
        val_parts.append(sess_df.loc[is_val].copy())
        train_parts.append(sess_df.loc[~is_val].copy())
        train_counts[sess] = int((~is_val).sum())
        val_counts[sess] = int(is_val.sum())

    return folds.FoldSplit(
        name="F-B-randval",
        train=pd.concat(train_parts, axis=0, ignore_index=True),
        val=pd.concat(val_parts, axis=0, ignore_index=True),
        test=test_df,
        train_session_counts=train_counts,
        val_session_counts=val_counts,
        test_session_counts={test_session: len(test_df)},
    )


# ----------------------------------------------------------------------
# Fit + predict with explicit hyperparameter config
# ----------------------------------------------------------------------


def _xgb_params_for(cfg: dict) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": config.SEED,
        "verbosity": 0,
        "max_depth": cfg["max_depth"],
        "eta": cfg["eta"],
        "min_child_weight": cfg["min_child_weight"],
        "reg_lambda": cfg["reg_lambda"],
        "reg_alpha": 0.0,
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
    }


def fit_with_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    cfg: dict,
) -> tuple[xgb.Booster, int]:
    dtrain = training.make_dmatrix(train_df, features)
    dval = training.make_dmatrix(val_df, features)
    booster = xgb.train(
        params=_xgb_params_for(cfg),
        dtrain=dtrain,
        num_boost_round=cfg["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------


def main() -> None:
    print("[robust] loading dataset")
    non_anom = data_io.load_non_anomaly()

    print("[robust] building F-B fold (chronological val)")
    fold_chrono = folds.build_loro_fold(non_anom, "F-B")
    print(
        f"  chrono train n={len(fold_chrono.train)} {fold_chrono.train_session_counts}, "
        f"val n={len(fold_chrono.val)}, test n={len(fold_chrono.test)}"
    )

    print("[robust] building F-B fold (random val)")
    fold_rand = build_fb_random_val(non_anom)
    print(
        f"  random train n={len(fold_rand.train)} {fold_rand.train_session_counts}, "
        f"val n={len(fold_rand.val)}, test n={len(fold_rand.test)}"
    )

    fits: list[FitRecord] = []
    metric_rows: list[dict] = []

    # ---- 6 chronological-val fits: {H1, H2, H3} × {B1, B5} -------------
    for config_name in ["H1", "H2", "H3"]:
        cfg = CONFIGS[config_name]
        for variant in ["B1", "B5"]:
            features = config.VARIANTS[variant]
            t0 = time.time()
            booster, best_iter = fit_with_config(fold_chrono.train, fold_chrono.val, features, cfg)
            y_pred = booster.predict(
                training.make_dmatrix(fold_chrono.test, features, target=None),
                iteration_range=(0, best_iter + 1),
            )
            preds = training.build_predictions_frame(fold_chrono.test, features, y_pred)

            model_path = DIAG_MODELS / f"F-B_{variant}_{config_name}.json"
            booster.save_model(str(model_path))
            preds_path = DIAG_CACHE / f"predictions_F-B_{variant}_{config_name}.parquet"
            preds.to_parquet(preds_path, index=False)

            wall = time.time() - t0
            fits.append(FitRecord(
                variant=variant, config=config_name, val_protocol="chronological",
                best_iteration=best_iter,
                n_train=len(fold_chrono.train), n_val=len(fold_chrono.val), n_test=len(fold_chrono.test),
                wallclock_s=wall,
                model_sha256=training.model_sha256(model_path),
            ))
            for stratum in metrics.evaluate_predictions(preds):
                metric_rows.append({
                    "variant": variant, "config": config_name, "val_protocol": "chronological",
                    "best_iteration": best_iter, "wallclock_s": wall, **stratum,
                })
            print(f"  fit B={variant} {config_name} chrono best_iter={best_iter} wall={wall:.1f}s")

    # ---- 2 random-val fits: {H1, H2} × {B5} ----------------------------
    for config_name in ["H1", "H2"]:
        cfg = CONFIGS[config_name]
        variant = "B5"
        features = config.VARIANTS[variant]
        tag = f"{config_name}_randval"
        t0 = time.time()
        booster, best_iter = fit_with_config(fold_rand.train, fold_rand.val, features, cfg)
        y_pred = booster.predict(
            training.make_dmatrix(fold_rand.test, features, target=None),
            iteration_range=(0, best_iter + 1),
        )
        preds = training.build_predictions_frame(fold_rand.test, features, y_pred)

        model_path = DIAG_MODELS / f"F-B_{variant}_{tag}.json"
        booster.save_model(str(model_path))
        preds_path = DIAG_CACHE / f"predictions_F-B_{variant}_{tag}.parquet"
        preds.to_parquet(preds_path, index=False)

        wall = time.time() - t0
        fits.append(FitRecord(
            variant=variant, config=tag, val_protocol="random",
            best_iteration=best_iter,
            n_train=len(fold_rand.train), n_val=len(fold_rand.val), n_test=len(fold_rand.test),
            wallclock_s=wall,
            model_sha256=training.model_sha256(model_path),
        ))
        for stratum in metrics.evaluate_predictions(preds):
            metric_rows.append({
                "variant": variant, "config": tag, "val_protocol": "random",
                "best_iteration": best_iter, "wallclock_s": wall, **stratum,
            })
        print(f"  fit B={variant} {tag} best_iter={best_iter} wall={wall:.1f}s")

    # ---- Reuse locked F-B predictions ---------------------------------
    for variant in ["B1", "B5"]:
        preds_path = config.CACHE_DIR / f"predictions_F-B_{variant}.parquet"
        preds = pd.read_parquet(preds_path)
        # best_iteration / wallclock for locked are in the inventory
        inv = pd.read_parquet(config.RESULTS_DIR / "fit_inventory.parquet")
        match = inv[(inv["fold"] == "F-B") & (inv["variant"] == variant)]
        best_iter = int(match.iloc[0]["best_iteration"]) if not match.empty else -1
        wall = float(match.iloc[0]["wallclock_s"]) if not match.empty else float("nan")
        for stratum in metrics.evaluate_predictions(preds):
            metric_rows.append({
                "variant": variant, "config": "locked", "val_protocol": "chronological",
                "best_iteration": best_iter, "wallclock_s": wall, **stratum,
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_parquet(DIAG_METRICS_PATH, index=False)
    fits_df = pd.DataFrame([dataclasses.asdict(f) for f in fits])
    fits_df.to_parquet(DIAG_FIT_INVENTORY_PATH, index=False)
    print(f"[robust] {len(fits)} new fits + locked baseline; metrics rows = {len(metrics_df)}")
    print(f"[robust] metrics → {DIAG_METRICS_PATH}")
    print(f"[robust] fit inventory → {DIAG_FIT_INVENTORY_PATH}")
    print("[robust] (the diagnostic narrative is rendered into results_report.md by build_results_report)")


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _fmt_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"


def _delta_lookup(metrics_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Return Δ_LiDAR per (config_label, stratum). For randval configs, uses
    same-config chronological B1 as the comparator (B1 is not refit with random val)."""
    rows: dict[tuple[str, str], dict[str, float]] = {}

    def _rmse(variant: str, cfg: str, stratum: str) -> float | None:
        hit = metrics_df[
            (metrics_df["variant"] == variant)
            & (metrics_df["config"] == cfg)
            & (metrics_df["stratum"] == stratum)
        ]
        if hit.empty:
            return None
        return float(hit.iloc[0]["rmse"])

    for cfg_label in ["locked", "H1", "H2", "H3", "H1_randval", "H2_randval"]:
        # B1 source: locked uses locked, randval uses underlying H1/H2 chrono B1.
        if cfg_label == "locked":
            b1_cfg = "locked"
            b5_cfg = "locked"
        elif cfg_label.endswith("_randval"):
            b1_cfg = cfg_label.split("_")[0]  # H1 or H2
            b5_cfg = cfg_label
        else:
            b1_cfg = cfg_label
            b5_cfg = cfg_label
        for stratum in ["overall", "in_fov", "out_of_fov"]:
            r1 = _rmse("B1", b1_cfg, stratum)
            r5 = _rmse("B5", b5_cfg, stratum)
            if r1 is None or r5 is None:
                continue
            rows[(cfg_label, stratum)] = {
                "delta": r1 - r5,
                "b1_rmse": r1,
                "b5_rmse": r5,
                "b1_source_config": b1_cfg,
                "b5_source_config": b5_cfg,
            }
    return rows


def compute_fb_infov(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Per-config F-B in-FOV Δ_LiDAR (dB)."""
    delta_lookup = _delta_lookup(metrics_df)
    return {
        cfg: delta_lookup.get((cfg, "in_fov"), {}).get("delta", float("nan"))
        for cfg in ["locked", "H1", "H2", "H3", "H1_randval", "H2_randval"]
    }


def compute_verdict(fb_infov: dict[str, float]) -> str:
    """Pivot-robustness verdict from per-config in-FOV Δ_LiDAR."""
    pivot_threshold = 1.0
    recover_protocol_threshold = 0.5

    chrono_cfgs = ["locked", "H1", "H2", "H3"]
    any_chrono_recovers = any(fb_infov.get(c, float("-inf")) > pivot_threshold for c in chrono_cfgs)

    randval_changes_materially = False
    for cfg in ["H1", "H2"]:
        chrono = fb_infov.get(cfg, float("nan"))
        rand = fb_infov.get(f"{cfg}_randval", float("nan"))
        diff = abs(rand - chrono)
        if not np.isnan(diff) and diff > recover_protocol_threshold:
            randval_changes_materially = True

    if any_chrono_recovers:
        return "RECOVERABLE_HYPERPARAMS"
    if randval_changes_materially and any(
        fb_infov.get(f"{c}_randval", float("-inf")) > pivot_threshold for c in ["H1", "H2"]
    ):
        return "RECOVERABLE_VALIDATION_PROTOCOL"
    return "PIVOT_JUSTIFIED"


def build_diagnostic_section(
    metrics_df: pd.DataFrame,
    fits_df: pd.DataFrame,
    *,
    section_number: int = 10,
) -> str:
    """Render the hyperparameter-robustness narrative as a markdown section.

    Returns a string suitable for inclusion as §{section_number} of the consolidated
    Phase 1 / Project A results report.
    """
    delta_lookup = _delta_lookup(metrics_df)
    fb_infov = compute_fb_infov(metrics_df)
    verdict = compute_verdict(fb_infov)

    randval_diffs: dict[str, float] = {}
    for cfg in ["H1", "H2"]:
        chrono = fb_infov.get(cfg, float("nan"))
        rand = fb_infov.get(f"{cfg}_randval", float("nan"))
        randval_diffs[cfg] = abs(rand - chrono)

    pivot_threshold = 1.0
    cfg_order = ["locked", "H1", "H2", "H3", "H1_randval", "H2_randval"]

    md: list[str] = []
    md.append(f"## {section_number}. Hyperparameter robustness diagnostic\n")
    md.append("Tests whether the F-B in-FOV Δ_LiDAR ≤ 1 dB pivot trigger from §2 is robust to "
              "hyperparameter and validation-protocol choice. Refits B1 and B5 on F-B only; all "
              "other Phase 1 / Project A artifacts (models, predictions, SHAP) are untouched.\n")

    md.append(f"### {section_number}.1 TL;DR\n")
    md.append("F-B in-FOV Δ_LiDAR by config (positive = LiDAR helps; >1 dB clears the pivot threshold):")
    for cfg in cfg_order:
        delta = fb_infov.get(cfg, float("nan"))
        md.append(f"- **{cfg}** ({_config_summary(cfg)}): {delta:+.2f} dB")
    md.append(f"\n**Diagnostic verdict: {verdict}**\n")

    md.append(f"### {section_number}.2 Configurations\n")
    md.append("| Tag | max_depth | eta | min_child_weight | reg_lambda | subsample | colsample_bytree | n_estimators | early_stop |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    locked_summary = "6 | 0.05 | 1 | 1.0 | 1.0 | 1.0 | 2000 | 100"
    md.append(f"| locked (frozen) | {locked_summary} |")
    for tag in ["H1", "H2", "H3"]:
        c = CONFIGS[tag]
        md.append(
            f"| {tag} | {c['max_depth']} | {c['eta']} | {c['min_child_weight']} | "
            f"{c['reg_lambda']} | {c['subsample']} | {c['colsample_bytree']} | "
            f"{c['n_estimators']} | {c['early_stopping_rounds']} |"
        )
    md.append("\n`H1_randval` and `H2_randval` use the same hyperparameters as `H1` / `H2` but a "
              "random-10%-per-session validation split (seed=20260427) instead of "
              "chronological-last-10%. B1 is not refit with random val; the Δ_LiDAR for these "
              "uses the B1 fit from the matching chronological config.\n")

    md.append(f"### {section_number}.3 Per-fit RMSE (dB) on F-B\n")
    md.append("Bootstrap 95% CI on RMSE in brackets (B = 1000).\n")
    md.append("| Variant | Config | Val protocol | best_iter | Stratum | n | RMSE | 95% CI |")
    md.append("|---|---|---|---:|---|---:|---:|---|")
    for variant in ["B1", "B5"]:
        for cfg in cfg_order:
            for stratum in ["overall", "in_fov", "out_of_fov"]:
                hit = metrics_df[
                    (metrics_df["variant"] == variant)
                    & (metrics_df["config"] == cfg)
                    & (metrics_df["stratum"] == stratum)
                ]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                md.append(
                    f"| {variant} | {cfg} | {r['val_protocol']} | "
                    f"{int(r['best_iteration'])} | {stratum} | {int(r['n'])} | "
                    f"{float(r['rmse']):.3f} | {_fmt_ci(float(r['rmse_ci_lo']), float(r['rmse_ci_hi']))} |"
                )

    md.append(f"\n### {section_number}.4 Δ_LiDAR by config × stratum\n")
    md.append("Δ_LiDAR = RMSE(B1) − RMSE(B5). Positive => LiDAR features improve over AP-geometry-only.\n")
    md.append("| Config | Stratum | RMSE(B1) | RMSE(B5) | Δ_LiDAR (dB) | clears 1 dB? |")
    md.append("|---|---|---:|---:|---:|:---:|")
    for cfg in cfg_order:
        for stratum in ["overall", "in_fov", "out_of_fov"]:
            d = delta_lookup.get((cfg, stratum))
            if d is None:
                continue
            clears = "✓" if (stratum == "in_fov" and d["delta"] > pivot_threshold) else (
                "—" if stratum != "in_fov" else "✗"
            )
            md.append(
                f"| {cfg} | {stratum} | {d['b1_rmse']:.3f} | {d['b5_rmse']:.3f} | "
                f"{d['delta']:+.3f} | {clears} |"
            )

    md.append(f"\n### {section_number}.5 best_iteration distribution per config\n")
    md.append("If slow-eta H2 reaches a much higher iteration count, the locked config was "
              "stopping prematurely.\n")
    md.append("| Config | B1 best_iter | B5 best_iter |")
    md.append("|---|---:|---:|")
    for cfg in cfg_order:
        b1 = metrics_df[(metrics_df["variant"] == "B1") & (metrics_df["config"] == cfg)
                        & (metrics_df["stratum"] == "overall")]
        b5 = metrics_df[(metrics_df["variant"] == "B5") & (metrics_df["config"] == cfg)
                        & (metrics_df["stratum"] == "overall")]
        b1_str = f"{int(b1.iloc[0]['best_iteration'])}" if not b1.empty else "—"
        b5_str = f"{int(b5.iloc[0]['best_iteration'])}" if not b5.empty else "—"
        md.append(f"| {cfg} | {b1_str} | {b5_str} |")

    md.append(f"\n### {section_number}.6 Verdict criteria\n")
    md.append("- **PIVOT_JUSTIFIED**: F-B in-FOV Δ_LiDAR ≤ 1 dB on all 6 configs (locked + 5 alternatives). "
              "Result is robust; the proposal's pivot trigger fires.")
    md.append("- **RECOVERABLE_HYPERPARAMS**: at least one of {H1, H2, H3} produces F-B in-FOV "
              "Δ_LiDAR > 1 dB. Result is fragile to hyperparameter choice.")
    md.append("- **RECOVERABLE_VALIDATION_PROTOCOL**: a random-val variant differs from its "
              "chronological-val counterpart by >0.5 dB AND clears the 1 dB threshold. "
              "Chronological-per-session early-stopping was the issue.\n")

    md.append(f"### {section_number}.7 Discussion\n")
    md.append("**best_iteration distribution.** " + _iter_discussion(metrics_df) + "\n")
    md.append("**Out-of-FOV under H2 (slowest learning rate).** " + _outfov_h2_discussion(metrics_df) + "\n")
    md.append("**Random-val sensitivity.**")
    md.append(_randval_discussion(fb_infov, randval_diffs))

    md.append(f"\n### {section_number}.8 Recommendation\n")
    if verdict == "PIVOT_JUSTIFIED":
        md.append("**PIVOT TO PROJECT B.** The pivot trigger is robust across all six configurations. "
                  "The original Phase 1 / Project A conclusion stands; LiDAR-derived structure does "
                  "not transfer across sessions on the diagnostic fold under any reasonable "
                  "hyperparameter choice tested. Proceed with the within-session leave-region-out "
                  "path (15.03 only).")
    elif verdict == "RECOVERABLE_HYPERPARAMS":
        best_cfg = max(["H1", "H2", "H3"], key=lambda c: fb_infov.get(c, float("-inf")))
        md.append(
            f"**RE-RUN PHASE 1 / PROJECT A WITH `{best_cfg}`.** The pivot trigger is fragile to "
            f"hyperparameter choice — `{best_cfg}` (F-B in-FOV Δ_LiDAR = {fb_infov[best_cfg]:+.2f} dB) "
            f"clears the 1 dB threshold. Re-run the full LORO ablation with that configuration "
            f"before deciding between Project A (proceed to writing) and Project B "
            f"(single-session pivot).")
    elif verdict == "RECOVERABLE_VALIDATION_PROTOCOL":
        best_cfg = max(["H1_randval", "H2_randval"], key=lambda c: fb_infov.get(c, float("-inf")))
        md.append(
            f"**RE-RUN PHASE 1 / PROJECT A WITH RANDOM-VAL EARLY STOPPING.** The chronological-last-10% "
            f"validation split was triggering early-stopping prematurely. With `{best_cfg}`, F-B "
            f"in-FOV Δ_LiDAR = {fb_infov[best_cfg]:+.2f} dB. Re-run the full LORO ablation with "
            f"random-per-session validation before deciding between Project A and Project B.")

    md.append(
        f"\n### {section_number}.9 Diagnostic artifacts\n"
        "- 8 model files: `scripts/p1_project_a/diagnostic/models/F-B_{variant}_{config}.json`\n"
        "- Predictions: `scripts/p1_project_a/diagnostic/cache/predictions_F-B_{variant}_{config}.parquet`\n"
        "- Metrics parquet: `scripts/p1_project_a/diagnostic/robustness_metrics.parquet`\n"
        "- Fit inventory: `scripts/p1_project_a/diagnostic/robustness_fit_inventory.parquet`\n"
        "- Re-run: `python -m scripts.p1_project_a.run_robustness`\n"
    )

    md.append(f"\n### {section_number}.10 Diagnostic fit inventory\n")
    md.append("| variant | config | val | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for _, r in fits_df.iterrows():
        md.append(
            f"| {r['variant']} | {r['config']} | {r['val_protocol']} | {int(r['best_iteration'])} | "
            f"{int(r['n_train'])} | {int(r['n_val'])} | {int(r['n_test'])} | "
            f"{float(r['wallclock_s']):.1f} | `{r['model_sha256'][:16]}…` |"
        )

    return "\n".join(md) + "\n"


def load_diagnostic_artifacts() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read the cached robustness parquets if present, else None."""
    if not (DIAG_METRICS_PATH.exists() and DIAG_FIT_INVENTORY_PATH.exists()):
        return None
    return pd.read_parquet(DIAG_METRICS_PATH), pd.read_parquet(DIAG_FIT_INVENTORY_PATH)


def _config_summary(cfg: str) -> str:
    if cfg == "locked":
        return "max_depth=6, eta=0.05"
    if cfg.endswith("_randval"):
        base = cfg.split("_")[0]
        c = CONFIGS[base]
        return f"{base} hyperparams + random val (max_depth={c['max_depth']}, eta={c['eta']})"
    c = CONFIGS[cfg]
    return f"max_depth={c['max_depth']}, eta={c['eta']}, λ={c['reg_lambda']}"


def _iter_discussion(metrics_df: pd.DataFrame) -> str:
    cfg_iters = {}
    for cfg in ["locked", "H1", "H2", "H3", "H1_randval", "H2_randval"]:
        b5 = metrics_df[(metrics_df["variant"] == "B5") & (metrics_df["config"] == cfg)
                        & (metrics_df["stratum"] == "overall")]
        if not b5.empty:
            cfg_iters[cfg] = int(b5.iloc[0]["best_iteration"])
    if not cfg_iters:
        return "No best-iteration data available."
    locked = cfg_iters.get("locked", -1)
    h2 = cfg_iters.get("H2", -1)
    if locked > 0 and h2 > 0:
        ratio = h2 / locked
        if ratio > 3:
            return (f"H2 (slow eta) reached B5 best_iter={h2}, vs the locked config's {locked} "
                    f"({ratio:.1f}× more rounds). The slower learning rate did let the model fit longer "
                    f"before validation loss climbed — but check Δ_LiDAR_H2 above to see whether the "
                    f"extra capacity actually changed the F-B test-set verdict.")
        return (f"H2 (slow eta) reached B5 best_iter={h2}, vs the locked config's {locked} — comparable "
                f"depth of training. The locked config was not stopping conspicuously prematurely.")
    return f"B5 best_iter by config: " + ", ".join(f"{c}={n}" for c, n in cfg_iters.items())


def _outfov_h2_discussion(metrics_df: pd.DataFrame) -> str:
    locked = metrics_df[(metrics_df["variant"] == "B5") & (metrics_df["config"] == "locked")
                        & (metrics_df["stratum"] == "out_of_fov")]
    h2 = metrics_df[(metrics_df["variant"] == "B5") & (metrics_df["config"] == "H2")
                    & (metrics_df["stratum"] == "out_of_fov")]
    if locked.empty or h2.empty:
        return "Insufficient data."
    locked_rmse = float(locked.iloc[0]["rmse"])
    h2_rmse = float(h2.iloc[0]["rmse"])
    delta = locked_rmse - h2_rmse
    if delta > 0.3:
        return (f"H2 lowered B5 out-of-FOV RMSE from {locked_rmse:.2f} to {h2_rmse:.2f} dB "
                f"(−{delta:.2f} dB). Slower learning helps the harder stratum.")
    if delta < -0.3:
        return (f"H2 raised B5 out-of-FOV RMSE from {locked_rmse:.2f} to {h2_rmse:.2f} dB "
                f"(+{abs(delta):.2f} dB). Slower learning underfits the harder stratum here.")
    return (f"B5 out-of-FOV RMSE: locked={locked_rmse:.2f} dB, H2={h2_rmse:.2f} dB "
            f"(Δ={delta:+.2f} dB). H2 makes no material difference on the harder stratum.")


def _randval_discussion(fb_infov: dict, randval_diffs: dict) -> str:
    lines = []
    for cfg in ["H1", "H2"]:
        chrono = fb_infov.get(cfg, float("nan"))
        rand = fb_infov.get(f"{cfg}_randval", float("nan"))
        diff = randval_diffs.get(cfg, float("nan"))
        lines.append(
            f"- **{cfg}**: chrono Δ_LiDAR_in_fov = {chrono:+.2f} dB; randval = {rand:+.2f} dB; "
            f"|Δ| = {diff:.2f} dB."
        )
    any_material = any(d > 0.5 for d in randval_diffs.values() if not np.isnan(d))
    if any_material:
        lines.append("\nAt least one randval variant moves Δ_LiDAR by >0.5 dB — chronological "
                     "early-stopping is materially affecting the comparison.")
    else:
        lines.append("\nNeither randval variant moves Δ_LiDAR by >0.5 dB from its chronological "
                     "counterpart — the validation protocol is not the issue.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
