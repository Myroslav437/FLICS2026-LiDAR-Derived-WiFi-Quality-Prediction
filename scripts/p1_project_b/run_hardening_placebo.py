"""Hardening E — Within-session LiDAR placebo on R-4.

Refits W4 on R-4 under locked hyperparameters with the 19 LiDAR columns
block-shuffled in the training set. Tests whether even the locked-no-buffer
R-4 nominal positive (+1.09 dB) was driven by real LiDAR signal or by the
model fitting to LiDAR's marginal structure.

Determinism note. Project B's `_random_train_val_split` seeds its RNG with
`b_b_config.SEED + (hash(fold_name) & 0x7FFFFFFF)`. Because Python randomises
`hash(str)` between invocations (unless `PYTHONHASHSEED=0`), Project B's
cached W2 and W4 predictions on R-4 reflect a particular random split that
cannot be reproduced in a fresh Python run. To get an apples-to-apples
placebo comparison and run-to-run determinism, this experiment overrides the
random val split with a deterministic permutation seeded by `b_b_config.SEED`,
and refits **all three** variants (W2, W4 real, W4 placebo) on the same
deterministic split. Only the LiDAR columns differ between the W4 real and
W4 placebo training sets; the val set, the test set, the row order, and the
non-LiDAR feature values are byte-identical.

3 new fits. Originally `scripts/p1_hardening/run_e_within_session_placebo.py`;
merged into Project B on 2026-04-28. The placebo helper is imported from
Project A (`scripts.p1_project_a.placebo`) since both A's Hardening A and B's
Hardening E use the same generic block-shuffle.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import metrics as a_metrics
from scripts.p1_project_a import placebo
from scripts.p1_project_a import training as a_training
from scripts.p1_project_a.run_hardening_placebo import FitRow

from . import config as b_config
from .run_modeling import prepare_regions


def _fit_xgb(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], xgb_params: dict
) -> tuple[xgb.Booster, int]:
    dtrain = a_training.make_dmatrix(train_df, features)
    dval = a_training.make_dmatrix(val_df, features)
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=b_config.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=b_config.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


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
            "experiment": "E",
            "fold": fold_name,
            "variant": variant,
            "descriptor": descriptor,
            "framework": "xgboost",
            "config": config_label,
            **sm,
        })


def _deterministic_split(non_anom_with_region: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the R-4 train/val/test split with a deterministic 10%-random val seed.

    Mirrors Project B's `build_fold("R-4")` exactly except for the val-split RNG
    seed: this version uses `b_config.SEED` directly (no hash dependence), so the
    split is byte-identical across Python invocations.
    """
    test_region = b_config.FOLD_TO_REGION["R-4"]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    n = len(train_pool)
    n_val = max(1, int(round(n * b_config.VAL_FRACTION)))
    rng = np.random.default_rng(b_config.SEED)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    val = train_pool.iloc[val_idx].reset_index(drop=True)
    train = train_pool.iloc[tr_idx].reset_index(drop=True)
    return train, val, test_df


def _fit_and_record(
    descriptor: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    fits: list[FitRow],
) -> pd.DataFrame:
    t0 = time.time()
    booster, best_iter = _fit_xgb(train_df, val_df, features, b_config.XGB_LOCKED)
    y_pred = a_training.predict(booster, test_df, features, best_iter)
    preds = a_training.build_predictions_frame(test_df, features, y_pred)
    if "region_id" in test_df.columns:
        preds["region_id"] = test_df["region_id"].to_numpy()

    model_path = b_config.MODELS_DIR / f"E_{descriptor}.json"
    booster.save_model(str(model_path))
    preds_path = b_config.CACHE_DIR / f"predictions_E_{descriptor}.parquet"
    preds.to_parquet(preds_path, index=False)

    wall = float(time.time() - t0)
    sha = a_training.model_sha256(model_path)
    variant_for_row = "W4" if descriptor.startswith("R-4_W4") else "W2"
    fits.append(FitRow(
        experiment="E",
        fold="R-4",
        variant=variant_for_row,
        descriptor=descriptor,
        framework="xgboost",
        config="locked",
        best_iteration=best_iter,
        n_features=len(features),
        n_train=int(len(train_df)),
        n_val=int(len(val_df)),
        n_test=int(len(test_df)),
        wallclock_s=wall,
        model_sha256=sha,
        model_path=str(model_path.relative_to(b_config.PROJECT_ROOT)),
        predictions_path=str(preds_path.relative_to(b_config.PROJECT_ROOT)),
    ))
    print(f"  fit {descriptor} best_iter={best_iter} wall={wall:.1f}s sha={sha[:12]}")
    return preds


def main() -> tuple[pd.DataFrame, list[FitRow]]:
    print("[E] within-session LiDAR placebo on R-4 (deterministic split)")
    non_anom_with_region, _ = prepare_regions()
    train_real, val_real, test_df = _deterministic_split(non_anom_with_region)
    print(
        f"    R-4: train n={len(train_real)}, val n={len(val_real)}, test n={len(test_df)} "
        f"(val seed = b_config.SEED, deterministic)"
    )

    # Apply block-shuffle to the (train+val) pool. Single session here (15.03
    # only), so no group_col — one global permutation. Then re-split using the
    # same row-index boundary as the deterministic split above (pool order is
    # preserved through the column-only shuffle).
    pool = pd.concat([train_real, val_real], axis=0, ignore_index=True)
    rho_before = placebo.spearman_corr(
        pool["mean_dist_mm"].to_numpy(dtype=np.float64),
        pool["signal_power"].to_numpy(dtype=np.float64),
    )
    pool_shuf = placebo.block_shuffle_lidar(pool, group_col=None, seed=b_config.SEED)
    rho_after = placebo.spearman_corr(
        pool_shuf["mean_dist_mm"].to_numpy(dtype=np.float64),
        pool_shuf["signal_power"].to_numpy(dtype=np.float64),
    )
    marginals = placebo.marginal_match(pool, pool_shuf)
    max_marginal_drift = max(v["max_abs_mean_diff"] for v in marginals.values())

    n_train = len(train_real)
    train_shuf = pool_shuf.iloc[:n_train].copy().reset_index(drop=True)
    val_shuf = pool_shuf.iloc[n_train:].copy().reset_index(drop=True)

    metric_rows: list[dict] = []
    fits: list[FitRow] = []

    # ---- 3 new fits on the same deterministic split.
    preds_w2 = _fit_and_record(
        "R-4_W2_real_locked", train_real, val_real, test_df, b_config.VARIANTS["W2"], fits,
    )
    _evaluate_and_collect("R-4", "W2", "R-4_W2_real_locked", "locked", preds_w2, metric_rows)

    preds_w4_real = _fit_and_record(
        "R-4_W4_real_locked", train_real, val_real, test_df, b_config.VARIANTS["W4"], fits,
    )
    _evaluate_and_collect("R-4", "W4", "R-4_W4_real_locked", "locked", preds_w4_real, metric_rows)

    preds_w4_placebo = _fit_and_record(
        "R-4_W4_placebo_locked", train_shuf, val_shuf, test_df, b_config.VARIANTS["W4"], fits,
    )
    _evaluate_and_collect("R-4", "W4", "R-4_W4_placebo_locked", "locked", preds_w4_placebo, metric_rows)

    # ---- Also report Project B's cached W2 and W4 numbers as reference, with
    # an explicit "cache" descriptor so the report can cite both the
    # deterministic E result and the (different-split) Project B baseline.
    for variant in ["W2", "W4"]:
        cache_path = b_config.CACHE_DIR / f"predictions_R-4_{variant}.parquet"
        preds_cache = pd.read_parquet(cache_path)
        _evaluate_and_collect("R-4", variant, f"R-4_{variant}_projectB_cache", "locked", preds_cache, metric_rows)

    df = pd.DataFrame(metric_rows)
    df.to_parquet(b_config.RESULTS_DIR / "hardening_e_metrics.parquet", index=False)

    integrity = {
        "n_train_pre": int(len(train_real)),
        "n_val_pre": int(len(val_real)),
        "n_test_pre": int(len(test_df)),
        "n_train_post": int(len(train_shuf)),
        "n_val_post": int(len(val_shuf)),
        "n_test_post": int(len(test_df)),
        "rho_mean_dist_signal_before": rho_before,
        "rho_mean_dist_signal_after": rho_after,
        "max_marginal_mean_drift": max_marginal_drift,
        "split_seed": b_config.SEED,
        "deterministic_split": True,
    }
    pd.DataFrame([integrity]).to_parquet(b_config.RESULTS_DIR / "hardening_e_integrity.parquet", index=False)
    return df, fits


if __name__ == "__main__":
    main()
