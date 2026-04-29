"""Hardening C — R-4 combined H1 + 1m buffer diagnostic.

Refits all 6 within-session variants (W0..W4, W4'') on the R-4 fold under H1
hyperparameters with a 1m buffer applied to the training pool. Closes the
"what if the two corrections cancel?" logical gap between Project B's
buffer-only and H1-only diagnostics on R-4.

Determinism note. Project B's `_random_train_val_split` seeds its RNG via
`config.SEED + (hash(fold_name) & 0x7FFFFFFF)`. Python randomises `hash(str)`
across invocations unless `PYTHONHASHSEED=0`, so Project B's published model
SHAs are not byte-stable across re-runs. To produce a reproducible H1+buffer
diagnostic, this experiment applies the buffer mask via Project B's
`build_buffer_mask` (deterministic) but then overrides the val split with a
fold-internal `np.random.default_rng(b_config.SEED)` permutation. All 6 W
variants share this single split, so within-experiment Δ_LiDAR_within is
exact.

6 new fits. Originally `scripts/p1_hardening/run_c_r4_combined.py`; merged
into Project B on 2026-04-28.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import metrics as a_metrics
from scripts.p1_project_a import training as a_training
from scripts.p1_project_a.run_hardening_placebo import FitRow

from . import config as b_config
from . import regions as b_regions
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
            "experiment": "C",
            "fold": fold_name,
            "variant": variant,
            "descriptor": descriptor,
            "framework": "xgboost",
            "config": config_label,
            **sm,
        })


def _deterministic_buffered_split(
    non_anom_with_region: pd.DataFrame,
    *,
    buffer_distance_m: float = b_config.BUFFER_DISTANCE_M,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """R-4 train/val/test with the 1m buffer applied and a deterministic val seed."""
    test_region = b_config.FOLD_TO_REGION["R-4"]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool_full = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)

    train_coords = train_pool_full[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    holdout_coords = test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    drop_mask = b_regions.build_buffer_mask(train_coords, holdout_coords, buffer_distance_m)
    n_dropped = int(drop_mask.sum())
    train_pool = train_pool_full[~drop_mask].reset_index(drop=True)

    n = len(train_pool)
    n_val = max(1, int(round(n * b_config.VAL_FRACTION)))
    rng = np.random.default_rng(b_config.SEED)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    val = train_pool.iloc[val_idx].reset_index(drop=True)
    train = train_pool.iloc[tr_idx].reset_index(drop=True)
    return train, val, test_df, n_dropped


def main() -> tuple[pd.DataFrame, list[FitRow]]:
    print("[C] R-4 combined H1+1m-buffer diagnostic (deterministic split)")
    non_anom_with_region, _ = prepare_regions()
    train_df, val_df, test_df, n_dropped = _deterministic_buffered_split(non_anom_with_region)
    print(
        f"    R-4 + buffer: dropped {n_dropped} train rows; "
        f"train n={len(train_df)}, val n={len(val_df)}, test n={len(test_df)} "
        f"(val seed = b_config.SEED, deterministic)"
    )

    metric_rows: list[dict] = []
    fits: list[FitRow] = []
    xgb_params = b_config.XGB_H1
    fold_label = "R-4_buffer"

    for variant in b_config.MAIN_VARIANTS:
        features = b_config.VARIANTS[variant]
        descriptor = f"R-4_buffer_H1_{variant}"
        t0 = time.time()
        booster, best_iter = _fit_xgb(train_df, val_df, features, xgb_params)
        y_pred = a_training.predict(booster, test_df, features, best_iter)
        preds = a_training.build_predictions_frame(test_df, features, y_pred)
        if "region_id" in test_df.columns:
            preds["region_id"] = test_df["region_id"].to_numpy()

        model_path = b_config.MODELS_DIR / f"C_{descriptor}.json"
        booster.save_model(str(model_path))
        preds_path = b_config.CACHE_DIR / f"predictions_C_{descriptor}.parquet"
        preds.to_parquet(preds_path, index=False)

        wall = float(time.time() - t0)
        sha = a_training.model_sha256(model_path)

        fits.append(FitRow(
            experiment="C",
            fold=fold_label,
            variant=variant,
            descriptor=descriptor,
            framework="xgboost",
            config="H1",
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
        _evaluate_and_collect("R-4_buffer_H1", variant, descriptor, "H1", preds, metric_rows)
        print(f"  fit {variant} R-4 H1+buffer best_iter={best_iter} wall={wall:.1f}s sha={sha[:12]}")

    df = pd.DataFrame(metric_rows)
    df.to_parquet(b_config.RESULTS_DIR / "hardening_c_metrics.parquet", index=False)
    return df, fits


if __name__ == "__main__":
    main()
