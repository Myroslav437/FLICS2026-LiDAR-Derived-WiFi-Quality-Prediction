"""Hardening A — Cross-session LiDAR placebo on F-B.

Refits B5 on F-B with the 19 LiDAR columns block-shuffled within each training
session. Compares Δ_LiDAR (placebo) to Δ_LiDAR (real) under both locked and H1
hyperparameter configurations.

The B1 and B5 (real LiDAR) predictions on F-B are reused from Project A's main
cache (locked) and Project A's robustness diagnostic cache (H1).

2 new fits. Originally `scripts/p1_hardening/run_a_cross_session_placebo.py`;
merged into Project A on 2026-04-28.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as a_config
from . import data_io as a_data_io
from . import folds as a_folds
from . import metrics as a_metrics
from . import placebo
from . import training as a_training
from .run_robustness import DIAG_CACHE as A_DIAG_CACHE


@dataclass
class FitRow:
    experiment: str
    fold: str
    variant: str
    descriptor: str
    framework: str
    config: str
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    wallclock_s: float
    model_sha256: str
    model_path: str
    predictions_path: str


def _xgb_params_for(label: str) -> dict:
    return dict(a_config.XGB_LABELS[label])


def _fit_xgb(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], xgb_params: dict
) -> tuple[xgb.Booster, int]:
    dtrain = a_training.make_dmatrix(train_df, features)
    dval = a_training.make_dmatrix(val_df, features)
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=a_config.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=a_config.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


def _build_fb_with_placebo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build F-B with the LiDAR placebo applied to (train+val) per session.

    Returns (train_shuf, val_shuf, test_unshuf, integrity_report).
    Integrity report contains per-column marginal-match summaries plus the
    pre/post Spearman ρ between mean_dist_mm and signal_power, computed on the
    combined train+val pool.
    """
    non_anom = a_data_io.load_non_anomaly()
    fold = a_folds.build_loro_fold(non_anom, "F-B")

    # Recombine the per-session train+val pool, apply per-session block shuffle,
    # then re-split chronologically per session into 90/10 (matching the original
    # split since fh7000_timestamp ordering is preserved through value-only
    # shuffling of the LiDAR columns).
    pool = pd.concat([fold.train, fold.val], axis=0, ignore_index=True)
    pool_sorted = pool.sort_values(["session_date", "fh7000_timestamp"], kind="mergesort").reset_index(drop=True)

    # Pre-shuffle Spearman ρ between mean_dist_mm and signal_power.
    rho_before = placebo.spearman_corr(
        pool_sorted["mean_dist_mm"].to_numpy(dtype=np.float64),
        pool_sorted["signal_power"].to_numpy(dtype=np.float64),
    )

    pool_shuf = placebo.block_shuffle_lidar(pool_sorted, group_col="session_date", seed=a_config.SEED)
    rho_after = placebo.spearman_corr(
        pool_shuf["mean_dist_mm"].to_numpy(dtype=np.float64),
        pool_shuf["signal_power"].to_numpy(dtype=np.float64),
    )

    marginals = placebo.marginal_match(pool_sorted, pool_shuf)
    max_marginal_drift = max(v["max_abs_mean_diff"] for v in marginals.values())

    # Re-do the chronological-per-session 90/10 split on the shuffled pool.
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for sess in pool_shuf["session_date"].unique():
        sess_df = pool_shuf[pool_shuf["session_date"] == sess].copy().reset_index(drop=True)
        s = sess_df.sort_values("fh7000_timestamp", kind="mergesort").reset_index(drop=True)
        n = len(s)
        n_val = int(round(n * 0.10))
        n_val = max(1, min(n - 1, n_val))
        train_parts.append(s.iloc[: n - n_val].copy())
        val_parts.append(s.iloc[n - n_val :].copy())
    train_shuf = pd.concat(train_parts, axis=0, ignore_index=True)
    val_shuf = pd.concat(val_parts, axis=0, ignore_index=True)

    integrity = {
        "n_train_pre": int(len(fold.train)),
        "n_val_pre": int(len(fold.val)),
        "n_test_pre": int(len(fold.test)),
        "n_train_post": int(len(train_shuf)),
        "n_val_post": int(len(val_shuf)),
        "n_test_post": int(len(fold.test)),
        "rho_mean_dist_signal_before": rho_before,
        "rho_mean_dist_signal_after": rho_after,
        "max_marginal_mean_drift": max_marginal_drift,
    }
    return train_shuf, val_shuf, fold.test.copy().reset_index(drop=True), integrity


def _evaluate_and_collect(
    fold_name: str,
    variant: str,
    descriptor: str,
    config_label: str,
    preds: pd.DataFrame,
    metric_rows: list[dict],
) -> None:
    for sm in a_metrics.evaluate_predictions(preds):
        metric_rows.append({
            "experiment": "A",
            "fold": fold_name,
            "variant": variant,
            "descriptor": descriptor,
            "framework": "xgboost",
            "config": config_label,
            **sm,
        })


def main() -> tuple[pd.DataFrame, list[FitRow]]:
    print("[A] cross-session LiDAR placebo on F-B")
    train_shuf, val_shuf, test_df, integrity = _build_fb_with_placebo()
    print(f"    integrity: {integrity}")

    metric_rows: list[dict] = []
    fits: list[FitRow] = []

    # ---- Reuse cached B1 and B5 (real LiDAR) predictions on F-B for both configs.
    locked_cache = a_config.CACHE_DIR
    h1_cache: Path = A_DIAG_CACHE

    real_preds_paths = {
        ("locked", "B1"): locked_cache / "predictions_F-B_B1.parquet",
        ("locked", "B5"): locked_cache / "predictions_F-B_B5.parquet",
        ("H1", "B1"): h1_cache / "predictions_F-B_B1_H1.parquet",
        ("H1", "B5"): h1_cache / "predictions_F-B_B5_H1.parquet",
    }
    for (cfg, variant), p in real_preds_paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"expected cached predictions at {p}; run scripts.p1_project_a.run_modeling and "
                f"scripts.p1_project_a.run_robustness first"
            )
        preds = pd.read_parquet(p)
        _evaluate_and_collect("F-B", variant, "real", cfg, preds, metric_rows)

    # ---- 2 new fits: B5 placebo under locked and H1 hyperparameters.
    features = a_config.VARIANTS["B5"]
    for cfg_label in ["locked", "H1"]:
        descriptor = f"B5_placebo_{cfg_label}"
        t0 = time.time()
        booster, best_iter = _fit_xgb(train_shuf, val_shuf, features, _xgb_params_for(cfg_label))
        y_pred = a_training.predict(booster, test_df, features, best_iter)
        preds = a_training.build_predictions_frame(test_df, features, y_pred)

        model_path = a_config.MODELS_DIR / f"A_{descriptor}.json"
        booster.save_model(str(model_path))
        preds_path = a_config.CACHE_DIR / f"predictions_A_{descriptor}.parquet"
        preds.to_parquet(preds_path, index=False)

        wall = float(time.time() - t0)
        sha = a_training.model_sha256(model_path)

        fits.append(FitRow(
            experiment="A",
            fold="F-B",
            variant="B5",
            descriptor=descriptor,
            framework="xgboost",
            config=cfg_label,
            best_iteration=best_iter,
            n_features=len(features),
            n_train=int(len(train_shuf)),
            n_val=int(len(val_shuf)),
            n_test=int(len(test_df)),
            wallclock_s=wall,
            model_sha256=sha,
            model_path=str(model_path.relative_to(a_config.PROJECT_ROOT)),
            predictions_path=str(preds_path.relative_to(a_config.PROJECT_ROOT)),
        ))
        _evaluate_and_collect("F-B", "B5", "placebo", cfg_label, preds, metric_rows)
        print(f"  fit B5 {cfg_label} placebo best_iter={best_iter} wall={wall:.1f}s sha={sha[:12]}")

    df = pd.DataFrame(metric_rows)
    df.to_parquet(a_config.RESULTS_DIR / "hardening_a_metrics.parquet", index=False)

    # Persist integrity report alongside.
    pd.DataFrame([integrity]).to_parquet(a_config.RESULTS_DIR / "hardening_a_integrity.parquet", index=False)
    return df, fits


if __name__ == "__main__":
    main()
