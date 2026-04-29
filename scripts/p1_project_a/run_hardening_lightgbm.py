"""Hardening B — LightGBM cross-check on F-B.

Fits LightGBM with two hyperparameter sets (default and H1-equiv) on F-B for
both B1 and B5 to test whether the negative result is XGBoost-specific.

Test set, validation split, anomaly filter, FOV stratification all match
Project A's protocol exactly. The XGBoost reference numbers (locked + H1) are
reused from Project A's main and diagnostic caches.

4 new LightGBM fits. Originally `scripts/p1_hardening/run_b_lightgbm.py`;
merged into Project A on 2026-04-28.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config as a_config
from . import data_io as a_data_io
from . import folds as a_folds
from . import lightgbm_training as lgb_train
from . import metrics as a_metrics
from .run_hardening_placebo import FitRow
from .run_robustness import DIAG_CACHE as A_DIAG_CACHE


def _evaluate_and_collect(
    fold_name: str,
    variant: str,
    descriptor: str,
    framework: str,
    config_label: str,
    preds: pd.DataFrame,
    metric_rows: list[dict],
) -> None:
    for sm in a_metrics.evaluate_predictions(preds):
        metric_rows.append({
            "experiment": "B",
            "fold": fold_name,
            "variant": variant,
            "descriptor": descriptor,
            "framework": framework,
            "config": config_label,
            **sm,
        })


def main() -> tuple[pd.DataFrame, list[FitRow]]:
    print("[B] LightGBM cross-check on F-B")
    non_anom = a_data_io.load_non_anomaly()
    fold = a_folds.build_loro_fold(non_anom, "F-B")
    print(f"    train n={len(fold.train)}, val n={len(fold.val)}, test n={len(fold.test)}")

    metric_rows: list[dict] = []
    fits: list[FitRow] = []

    # ---- 4 new LightGBM fits.
    for cfg_label in ["default", "H1_equiv"]:
        params = a_config.LGB_LABELS[cfg_label]
        for variant in ["B1", "B5"]:
            features = a_config.VARIANTS[variant]
            descriptor = f"{variant}_lgb_{cfg_label}"
            t0 = time.time()
            booster, best_iter = lgb_train.fit_booster(fold.train, fold.val, features, params)
            y_pred = lgb_train.predict(booster, fold.test, features, best_iter)
            preds = lgb_train.build_predictions_frame(fold.test, features, y_pred)

            model_path = a_config.MODELS_DIR / f"B_{descriptor}.txt"
            lgb_train.save_model(booster, model_path)
            preds_path = a_config.CACHE_DIR / f"predictions_B_{descriptor}.parquet"
            preds.to_parquet(preds_path, index=False)

            wall = float(time.time() - t0)
            sha = lgb_train.model_sha256(model_path)

            fits.append(FitRow(
                experiment="B",
                fold="F-B",
                variant=variant,
                descriptor=descriptor,
                framework="lightgbm",
                config=cfg_label,
                best_iteration=best_iter,
                n_features=len(features),
                n_train=int(len(fold.train)),
                n_val=int(len(fold.val)),
                n_test=int(len(fold.test)),
                wallclock_s=wall,
                model_sha256=sha,
                model_path=str(model_path.relative_to(a_config.PROJECT_ROOT)),
                predictions_path=str(preds_path.relative_to(a_config.PROJECT_ROOT)),
            ))
            _evaluate_and_collect("F-B", variant, descriptor, "lightgbm", cfg_label, preds, metric_rows)
            print(f"  fit {variant} lgb-{cfg_label} best_iter={best_iter} wall={wall:.1f}s sha={sha[:12]}")

    # ---- Reuse XGBoost reference predictions (Project A).
    locked_cache = a_config.CACHE_DIR
    h1_cache: Path = A_DIAG_CACHE
    refs = {
        ("locked", "B1"): locked_cache / "predictions_F-B_B1.parquet",
        ("locked", "B5"): locked_cache / "predictions_F-B_B5.parquet",
        ("H1", "B1"): h1_cache / "predictions_F-B_B1_H1.parquet",
        ("H1", "B5"): h1_cache / "predictions_F-B_B5_H1.parquet",
    }
    for (cfg, variant), p in refs.items():
        preds = pd.read_parquet(p)
        _evaluate_and_collect("F-B", variant, f"{variant}_xgb_{cfg}", "xgboost", cfg, preds, metric_rows)

    df = pd.DataFrame(metric_rows)
    df.to_parquet(a_config.RESULTS_DIR / "hardening_b_metrics.parquet", index=False)
    return df, fits


if __name__ == "__main__":
    main()
