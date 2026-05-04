"""Drives the 64 lean-features fits (32 lean-A + 32 lean-B).

For each variant_set ∈ {lean_A, lean_B}, runs:
  - Project A: 7 fits (F-{A,B,C} × {B5, B5p}, plus F-B B5 H1 = 7)
  - Project B: 15 fits (R-{1..5} × {W2 locked, W4 locked, W4 H1})
  - R-4 robustness: 5 fits (W2/W4 buffer-locked, W4 H1, W2 H1, W4/W2 H1+buffer)
  - Hardening A (placebo on F-B): 2 fits (B5 placebo locked + H1)
  - Hardening B (LightGBM on F-B): 2 fits (B5 LGB default + H1_equiv)
  - Hardening E (placebo on R-4): 1 fit (W4 placebo locked)

Output paths:
  models      → scripts/p1_lean_features/models/{variant_set}_{descriptor}.json|.txt
  predictions → scripts/p1_lean_features/cache/predictions_{variant_set}_{descriptor}.parquet
  metrics     → scripts/p1_lean_features/results/{variant_set}_metrics.parquet
  fit log     → scripts/p1_lean_features/results/fit_inventory.parquet

Reuses Project A and Project B helpers (training, folds, regions, placebo,
buffer mask, LightGBM training) — see CLAUDE.md / brief §1 for the policy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import config as a_config
from scripts.p1_project_a import data_io as a_data_io
from scripts.p1_project_a import folds as a_folds
from scripts.p1_project_a import lightgbm_training as lgb_train
from scripts.p1_project_a import metrics as a_metrics
from scripts.p1_project_a import placebo as a_placebo
from scripts.p1_project_a import training as a_training

from scripts.p1_project_b import config as b_config
from scripts.p1_project_b import folds as b_folds
from scripts.p1_project_b import regions as b_regions
from scripts.p1_project_b.run_modeling import prepare_regions

from . import feature_lists as fl


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class FitRow:
    variant_set: str            # "lean_A" or "lean_B"
    experiment: str             # "project_a" | "project_b" | "r4_robustness" | "hardening_a" | "hardening_b" | "hardening_e"
    fold: str                   # canonical fold name in the metrics table
    variant: str                # "B5", "B5p", "W2", "W4", ...
    descriptor: str             # unique tag, used for filenames
    framework: str              # "xgboost" | "lightgbm"
    config: str                 # "locked" | "H1" | "default" | "H1_equiv" | "locked-buffer" | "H1+buffer" | "placebo_locked" | "placebo_H1"
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    wallclock_s: float
    model_sha256: str
    model_path: str
    predictions_path: str


def _fit_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    xgb_params: dict,
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


def _verify_features_count(features: list[str], expected: int, descriptor: str) -> None:
    if len(features) != expected:
        raise ValueError(
            f"feature-count mismatch for {descriptor!r}: expected {expected}, got {len(features)}: {features}"
        )


def _record_fit(
    *,
    variant_set: str,
    experiment: str,
    fold: str,
    variant: str,
    descriptor: str,
    framework: str,
    config_label: str,
    booster_path,
    best_iter: int,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wallclock_s: float,
) -> FitRow:
    sha = a_training.model_sha256(booster_path)
    rel_model = str(booster_path.relative_to(fl.PROJECT_ROOT))
    preds_path = fl.CACHE_DIR / f"predictions_{variant_set}_{descriptor}.parquet"
    return FitRow(
        variant_set=variant_set,
        experiment=experiment,
        fold=fold,
        variant=variant,
        descriptor=descriptor,
        framework=framework,
        config=config_label,
        best_iteration=int(best_iter),
        n_features=len(features),
        n_train=int(len(train_df)),
        n_val=int(len(val_df)),
        n_test=int(len(test_df)),
        wallclock_s=float(wallclock_s),
        model_sha256=sha,
        model_path=rel_model,
        predictions_path=str(preds_path.relative_to(fl.PROJECT_ROOT)),
    )


def _save_xgb_and_preds(
    booster: xgb.Booster,
    features: list[str],
    test_df: pd.DataFrame,
    *,
    variant_set: str,
    descriptor: str,
    best_iter: int,
    attach_region_id: bool = False,
) -> tuple[pd.DataFrame, Path]:
    model_path = fl.MODELS_DIR / f"{variant_set}_{descriptor}.json"
    booster.save_model(str(model_path))
    y_pred = a_training.predict(booster, test_df, features, best_iter)
    preds = a_training.build_predictions_frame(test_df, features, y_pred)
    if attach_region_id and "region_id" in test_df.columns:
        preds["region_id"] = test_df["region_id"].to_numpy()
    preds_path = fl.CACHE_DIR / f"predictions_{variant_set}_{descriptor}.parquet"
    preds.to_parquet(preds_path, index=False)
    return preds, model_path


def _save_lgb_and_preds(
    booster,
    features: list[str],
    test_df: pd.DataFrame,
    *,
    variant_set: str,
    descriptor: str,
    best_iter: int,
) -> tuple[pd.DataFrame, Path]:
    model_path = fl.MODELS_DIR / f"{variant_set}_{descriptor}.txt"
    lgb_train.save_model(booster, model_path)
    y_pred = lgb_train.predict(booster, test_df, features, best_iter)
    preds = lgb_train.build_predictions_frame(test_df, features, y_pred)
    preds_path = fl.CACHE_DIR / f"predictions_{variant_set}_{descriptor}.parquet"
    preds.to_parquet(preds_path, index=False)
    return preds, model_path


def _evaluate_to_rows(
    preds: pd.DataFrame,
    *,
    variant_set: str,
    experiment: str,
    fold: str,
    variant: str,
    descriptor: str,
    framework: str,
    config_label: str,
) -> list[dict]:
    rows: list[dict] = []
    for sm in a_metrics.evaluate_predictions(preds):
        rows.append({
            "variant_set": variant_set,
            "experiment": experiment,
            "fold": fold,
            "variant": variant,
            "descriptor": descriptor,
            "framework": framework,
            "config": config_label,
            **sm,
        })
    return rows


# ---------------------------------------------------------------------------
# Project A — F-A/F-B/F-C × {B5, B5'} + F-B B5 H1
# ---------------------------------------------------------------------------


def run_project_a(variant_set: str, non_anom: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    """7 fits per variant_set:
      - F-A/F-B/F-C × B5 (locked)
      - F-A/F-B/F-C × B5p (locked)
      - F-B × B5 (H1)
    """
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    feats_map = fl.project_a_variant_features(variant_set)

    # B5 across F-A, F-B, F-C under locked
    for fold_name in ["F-A", "F-B", "F-C"]:
        fold = a_folds.build_loro_fold(non_anom, fold_name)
        # B5 locked
        for variant, config_label, xgb_params in [
            ("B5", "locked", a_config.XGB_LOCKED),
            ("B5p", "locked", a_config.XGB_LOCKED),
        ]:
            features = feats_map[variant]
            descriptor = f"{fold_name}_{variant}_{config_label}"
            print(f"[A/{variant_set}] fit {descriptor} ({len(features)} feat)")
            t0 = time.time()
            booster, best_iter = _fit_xgb(fold.train, fold.val, features, xgb_params)
            preds, model_path = _save_xgb_and_preds(
                booster, features, fold.test,
                variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
            )
            wall = float(time.time() - t0)
            fits.append(_record_fit(
                variant_set=variant_set, experiment="project_a",
                fold=fold_name, variant=variant, descriptor=descriptor,
                framework="xgboost", config_label=config_label,
                booster_path=model_path, best_iter=best_iter,
                features=features, train_df=fold.train, val_df=fold.val, test_df=fold.test,
                wallclock_s=wall,
            ))
            metric_rows.extend(_evaluate_to_rows(
                preds, variant_set=variant_set, experiment="project_a",
                fold=fold_name, variant=variant, descriptor=descriptor,
                framework="xgboost", config_label=config_label,
            ))
            print(f"   best_iter={best_iter} wall={wall:.1f}s")

    # F-B × B5 (H1)
    fold = a_folds.build_loro_fold(non_anom, "F-B")
    variant = "B5"
    features = feats_map[variant]
    descriptor = "F-B_B5_H1"
    print(f"[A/{variant_set}] fit {descriptor} ({len(features)} feat)")
    t0 = time.time()
    booster, best_iter = _fit_xgb(fold.train, fold.val, features, a_config.XGB_H1)
    preds, model_path = _save_xgb_and_preds(
        booster, features, fold.test,
        variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
    )
    wall = float(time.time() - t0)
    fits.append(_record_fit(
        variant_set=variant_set, experiment="project_a",
        fold="F-B", variant="B5", descriptor=descriptor,
        framework="xgboost", config_label="H1",
        booster_path=model_path, best_iter=best_iter,
        features=features, train_df=fold.train, val_df=fold.val, test_df=fold.test,
        wallclock_s=wall,
    ))
    metric_rows.extend(_evaluate_to_rows(
        preds, variant_set=variant_set, experiment="project_a",
        fold="F-B", variant="B5", descriptor=descriptor,
        framework="xgboost", config_label="H1",
    ))
    print(f"   best_iter={best_iter} wall={wall:.1f}s")

    return fits, metric_rows


# ---------------------------------------------------------------------------
# Project B — R-{1..5} × {W2 locked, W4 locked, W4 H1}
# ---------------------------------------------------------------------------


def run_project_b(variant_set: str, non_anom_with_region: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    feats_map = fl.project_b_variant_features(variant_set)

    for fold_name in b_config.FOLD_NAMES:  # R-1..R-5
        fold = b_folds.build_fold(non_anom_with_region, fold_name)

        for variant, config_label, xgb_params in [
            ("W2", "locked", b_config.HYPERPARAMS_LOCKED),
            ("W4", "locked", b_config.HYPERPARAMS_LOCKED),
            ("W4", "H1", b_config.HYPERPARAMS_H1),
        ]:
            features = feats_map[variant]
            descriptor = f"{fold_name}_{variant}_{config_label}"
            print(f"[B/{variant_set}] fit {descriptor} ({len(features)} feat)")
            t0 = time.time()
            booster, best_iter = _fit_xgb(fold.train, fold.val, features, xgb_params)
            preds, model_path = _save_xgb_and_preds(
                booster, features, fold.test,
                variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
                attach_region_id=True,
            )
            wall = float(time.time() - t0)
            fits.append(_record_fit(
                variant_set=variant_set, experiment="project_b",
                fold=fold_name, variant=variant, descriptor=descriptor,
                framework="xgboost", config_label=config_label,
                booster_path=model_path, best_iter=best_iter,
                features=features, train_df=fold.train, val_df=fold.val, test_df=fold.test,
                wallclock_s=wall,
            ))
            metric_rows.extend(_evaluate_to_rows(
                preds, variant_set=variant_set, experiment="project_b",
                fold=fold_name, variant=variant, descriptor=descriptor,
                framework="xgboost", config_label=config_label,
            ))
            print(f"   best_iter={best_iter} wall={wall:.1f}s")

    return fits, metric_rows


# ---------------------------------------------------------------------------
# R-4 robustness — buffer + H1 + H1+buffer
# ---------------------------------------------------------------------------


def _deterministic_buffered_split_r4(non_anom_with_region: pd.DataFrame):
    """Match Hardening C's deterministic buffer split for byte-stable splits."""
    test_region = b_config.FOLD_TO_REGION["R-4"]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool_full = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    train_coords = train_pool_full[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    holdout_coords = test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    drop_mask = b_regions.build_buffer_mask(train_coords, holdout_coords, b_config.BUFFER_DISTANCE_M)
    n_dropped = int(drop_mask.sum())
    train_pool = train_pool_full[~drop_mask].reset_index(drop=True)
    n = len(train_pool)
    n_val = max(1, int(round(n * b_config.VAL_FRACTION)))
    rng = np.random.default_rng(b_config.SEED)
    perm = rng.permutation(n)
    val = train_pool.iloc[perm[:n_val]].reset_index(drop=True)
    train = train_pool.iloc[perm[n_val:]].reset_index(drop=True)
    return train, val, test_df, n_dropped


def run_r4_robustness(variant_set: str, non_anom_with_region: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    """5 fits per variant_set:
      - R-4_W2_buffer_locked
      - R-4_W4_buffer_locked
      - R-4_W2_H1            (W2 H1 not present in original locked Project B; needed for the comparison)
      - R-4_W4_H1_buffer
      - R-4_W2_H1_buffer
    Note: R-4_W4_H1 is already part of run_project_b(); we don't repeat it.
    """
    fits: list[FitRow] = []
    metric_rows: list[dict] = []
    feats_map = fl.project_b_variant_features(variant_set)

    # ----- W2/W4 with buffer (locked hyperparameters, hash-based val seed via build_fold_with_buffer) -----
    fold = b_folds.build_fold_with_buffer(
        non_anom_with_region, "R-4", buffer_distance_m=b_config.BUFFER_DISTANCE_M,
    )
    print(
        f"[R-4 robust/{variant_set}] buffer fold {fold.name}: dropped "
        f"{fold.n_train_dropped_by_buffer} rows; train n={len(fold.train)}, val n={len(fold.val)}"
    )
    for variant in ["W2", "W4"]:
        features = feats_map[variant]
        descriptor = f"R-4_buffer_{variant}_locked"
        print(f"[R-4 robust/{variant_set}] fit {descriptor} ({len(features)} feat)")
        t0 = time.time()
        booster, best_iter = _fit_xgb(fold.train, fold.val, features, b_config.HYPERPARAMS_LOCKED)
        preds, model_path = _save_xgb_and_preds(
            booster, features, fold.test,
            variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
            attach_region_id=True,
        )
        wall = float(time.time() - t0)
        fits.append(_record_fit(
            variant_set=variant_set, experiment="r4_robustness",
            fold="R-4_buffer", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label="locked-buffer",
            booster_path=model_path, best_iter=best_iter,
            features=features, train_df=fold.train, val_df=fold.val, test_df=fold.test,
            wallclock_s=wall,
        ))
        metric_rows.extend(_evaluate_to_rows(
            preds, variant_set=variant_set, experiment="r4_robustness",
            fold="R-4_buffer", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label="locked-buffer",
        ))
        print(f"   best_iter={best_iter} wall={wall:.1f}s")

    # ----- R-4 W2 under H1 (no buffer) — needed because Project B's H1 sweep was W4-only -----
    fold_h1 = b_folds.build_fold(non_anom_with_region, "R-4")
    variant = "W2"
    features = feats_map[variant]
    descriptor = "R-4_W2_H1"
    print(f"[R-4 robust/{variant_set}] fit {descriptor} ({len(features)} feat)")
    t0 = time.time()
    booster, best_iter = _fit_xgb(fold_h1.train, fold_h1.val, features, b_config.HYPERPARAMS_H1)
    preds, model_path = _save_xgb_and_preds(
        booster, features, fold_h1.test,
        variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
        attach_region_id=True,
    )
    wall = float(time.time() - t0)
    fits.append(_record_fit(
        variant_set=variant_set, experiment="r4_robustness",
        fold="R-4", variant=variant, descriptor=descriptor,
        framework="xgboost", config_label="H1",
        booster_path=model_path, best_iter=best_iter,
        features=features, train_df=fold_h1.train, val_df=fold_h1.val, test_df=fold_h1.test,
        wallclock_s=wall,
    ))
    metric_rows.extend(_evaluate_to_rows(
        preds, variant_set=variant_set, experiment="r4_robustness",
        fold="R-4", variant=variant, descriptor=descriptor,
        framework="xgboost", config_label="H1",
    ))
    print(f"   best_iter={best_iter} wall={wall:.1f}s")

    # ----- W2/W4 H1 + buffer — deterministic split (matches Hardening C) -----
    train_df, val_df, test_df, n_dropped = _deterministic_buffered_split_r4(non_anom_with_region)
    print(
        f"[R-4 robust/{variant_set}] H1+buffer (deterministic): dropped {n_dropped} rows; "
        f"train n={len(train_df)}, val n={len(val_df)}"
    )
    for variant in ["W2", "W4"]:
        features = feats_map[variant]
        descriptor = f"R-4_buffer_H1_{variant}"
        print(f"[R-4 robust/{variant_set}] fit {descriptor} ({len(features)} feat)")
        t0 = time.time()
        booster, best_iter = _fit_xgb(train_df, val_df, features, b_config.HYPERPARAMS_H1)
        preds, model_path = _save_xgb_and_preds(
            booster, features, test_df,
            variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
            attach_region_id=True,
        )
        wall = float(time.time() - t0)
        fits.append(_record_fit(
            variant_set=variant_set, experiment="r4_robustness",
            fold="R-4_buffer", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label="H1+buffer",
            booster_path=model_path, best_iter=best_iter,
            features=features, train_df=train_df, val_df=val_df, test_df=test_df,
            wallclock_s=wall,
        ))
        metric_rows.extend(_evaluate_to_rows(
            preds, variant_set=variant_set, experiment="r4_robustness",
            fold="R-4_buffer", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label="H1+buffer",
        ))
        print(f"   best_iter={best_iter} wall={wall:.1f}s")

    return fits, metric_rows


# ---------------------------------------------------------------------------
# Hardening A — F-B placebo, locked + H1
# ---------------------------------------------------------------------------


def _build_fb_placebo(non_anom: pd.DataFrame):
    """Mirror Hardening A's per-session block-shuffle on F-B train+val."""
    fold = a_folds.build_loro_fold(non_anom, "F-B")
    pool = pd.concat([fold.train, fold.val], axis=0, ignore_index=True)
    pool_sorted = pool.sort_values(["session_date", "fh7000_timestamp"], kind="mergesort").reset_index(drop=True)
    pool_shuf = a_placebo.block_shuffle_lidar(pool_sorted, group_col="session_date", seed=a_config.SEED)
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
    return train_shuf, val_shuf, fold.test.copy().reset_index(drop=True)


def run_hardening_a(variant_set: str, non_anom: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    feats_map = fl.project_a_variant_features(variant_set)
    train_shuf, val_shuf, test_df = _build_fb_placebo(non_anom)
    print(
        f"[A.placebo/{variant_set}] F-B placebo train n={len(train_shuf)}, val n={len(val_shuf)}"
    )
    variant = "B5"
    features = feats_map[variant]
    for cfg_label, xgb_params in [("locked", a_config.XGB_LOCKED), ("H1", a_config.XGB_H1)]:
        descriptor = f"F-B_B5_placebo_{cfg_label}"
        print(f"[A.placebo/{variant_set}] fit {descriptor} ({len(features)} feat)")
        t0 = time.time()
        booster, best_iter = _fit_xgb(train_shuf, val_shuf, features, xgb_params)
        preds, model_path = _save_xgb_and_preds(
            booster, features, test_df,
            variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
        )
        wall = float(time.time() - t0)
        fits.append(_record_fit(
            variant_set=variant_set, experiment="hardening_a",
            fold="F-B", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label=f"placebo_{cfg_label}",
            booster_path=model_path, best_iter=best_iter,
            features=features, train_df=train_shuf, val_df=val_shuf, test_df=test_df,
            wallclock_s=wall,
        ))
        metric_rows.extend(_evaluate_to_rows(
            preds, variant_set=variant_set, experiment="hardening_a",
            fold="F-B", variant=variant, descriptor=descriptor,
            framework="xgboost", config_label=f"placebo_{cfg_label}",
        ))
        print(f"   best_iter={best_iter} wall={wall:.1f}s")
    return fits, metric_rows


# ---------------------------------------------------------------------------
# Hardening B — LightGBM on F-B, B5 only (B1 has no telemetry → unchanged)
# ---------------------------------------------------------------------------


def run_hardening_b(variant_set: str, non_anom: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    feats_map = fl.project_a_variant_features(variant_set)
    fold = a_folds.build_loro_fold(non_anom, "F-B")
    variant = "B5"
    features = feats_map[variant]
    for cfg_label, lgb_params in [("default", a_config.LGB_DEFAULT), ("H1_equiv", a_config.LGB_H1_EQUIV)]:
        descriptor = f"F-B_B5_lgb_{cfg_label}"
        print(f"[B.lgb/{variant_set}] fit {descriptor} ({len(features)} feat)")
        t0 = time.time()
        booster, best_iter = lgb_train.fit_booster(fold.train, fold.val, features, lgb_params)
        preds, model_path = _save_lgb_and_preds(
            booster, features, fold.test,
            variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
        )
        wall = float(time.time() - t0)
        fits.append(_record_fit(
            variant_set=variant_set, experiment="hardening_b",
            fold="F-B", variant=variant, descriptor=descriptor,
            framework="lightgbm", config_label=cfg_label,
            booster_path=model_path, best_iter=best_iter,
            features=features, train_df=fold.train, val_df=fold.val, test_df=fold.test,
            wallclock_s=wall,
        ))
        metric_rows.extend(_evaluate_to_rows(
            preds, variant_set=variant_set, experiment="hardening_b",
            fold="F-B", variant=variant, descriptor=descriptor,
            framework="lightgbm", config_label=cfg_label,
        ))
        print(f"   best_iter={best_iter} wall={wall:.1f}s")

    return fits, metric_rows


# ---------------------------------------------------------------------------
# Hardening E — within-session placebo on R-4
# ---------------------------------------------------------------------------


def _deterministic_split_r4(non_anom_with_region: pd.DataFrame):
    """Mirror Hardening E's deterministic R-4 split."""
    test_region = b_config.FOLD_TO_REGION["R-4"]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    n = len(train_pool)
    n_val = max(1, int(round(n * b_config.VAL_FRACTION)))
    rng = np.random.default_rng(b_config.SEED)
    perm = rng.permutation(n)
    val = train_pool.iloc[perm[:n_val]].reset_index(drop=True)
    train = train_pool.iloc[perm[n_val:]].reset_index(drop=True)
    return train, val, test_df


def run_hardening_e(variant_set: str, non_anom_with_region: pd.DataFrame) -> tuple[list[FitRow], list[dict]]:
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    feats_map = fl.project_b_variant_features(variant_set)
    train_real, val_real, test_df = _deterministic_split_r4(non_anom_with_region)
    pool = pd.concat([train_real, val_real], axis=0, ignore_index=True)
    pool_shuf = a_placebo.block_shuffle_lidar(pool, group_col=None, seed=b_config.SEED)
    n_train = len(train_real)
    train_shuf = pool_shuf.iloc[:n_train].copy().reset_index(drop=True)
    val_shuf = pool_shuf.iloc[n_train:].copy().reset_index(drop=True)
    print(
        f"[E.placebo/{variant_set}] R-4 deterministic split: train n={n_train}, val n={len(val_shuf)}"
    )

    variant = "W4"
    features = feats_map[variant]
    descriptor = "R-4_W4_placebo_locked"
    print(f"[E.placebo/{variant_set}] fit {descriptor} ({len(features)} feat)")
    t0 = time.time()
    booster, best_iter = _fit_xgb(train_shuf, val_shuf, features, b_config.HYPERPARAMS_LOCKED)
    preds, model_path = _save_xgb_and_preds(
        booster, features, test_df,
        variant_set=variant_set, descriptor=descriptor, best_iter=best_iter,
        attach_region_id=True,
    )
    wall = float(time.time() - t0)
    fits.append(_record_fit(
        variant_set=variant_set, experiment="hardening_e",
        fold="R-4", variant=variant, descriptor=descriptor,
        framework="xgboost", config_label="placebo_locked",
        booster_path=model_path, best_iter=best_iter,
        features=features, train_df=train_shuf, val_df=val_shuf, test_df=test_df,
        wallclock_s=wall,
    ))
    metric_rows.extend(_evaluate_to_rows(
        preds, variant_set=variant_set, experiment="hardening_e",
        fold="R-4", variant=variant, descriptor=descriptor,
        framework="xgboost", config_label="placebo_locked",
    ))
    print(f"   best_iter={best_iter} wall={wall:.1f}s")

    return fits, metric_rows


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_for_variant_set(variant_set: str, non_anom: pd.DataFrame, non_anom_with_region: pd.DataFrame) -> tuple[list[FitRow], pd.DataFrame]:
    print(f"\n{'='*70}\n[lean] running variant_set = {variant_set}\n{'='*70}")
    all_fits: list[FitRow] = []
    all_rows: list[dict] = []

    a_fits, a_rows = run_project_a(variant_set, non_anom)
    all_fits.extend(a_fits); all_rows.extend(a_rows)

    b_fits, b_rows = run_project_b(variant_set, non_anom_with_region)
    all_fits.extend(b_fits); all_rows.extend(b_rows)

    r4_fits, r4_rows = run_r4_robustness(variant_set, non_anom_with_region)
    all_fits.extend(r4_fits); all_rows.extend(r4_rows)

    ha_fits, ha_rows = run_hardening_a(variant_set, non_anom)
    all_fits.extend(ha_fits); all_rows.extend(ha_rows)

    hb_fits, hb_rows = run_hardening_b(variant_set, non_anom)
    all_fits.extend(hb_fits); all_rows.extend(hb_rows)

    he_fits, he_rows = run_hardening_e(variant_set, non_anom_with_region)
    all_fits.extend(he_fits); all_rows.extend(he_rows)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = fl.RESULTS_DIR / f"{variant_set}_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    fits_df_local = pd.DataFrame([asdict(f) for f in all_fits])
    fits_local_path = fl.RESULTS_DIR / f"{variant_set}_fits.parquet"
    fits_df_local.to_parquet(fits_local_path, index=False)

    print(f"[lean/{variant_set}] {len(all_fits)} fits, {len(all_rows)} metric rows -> {metrics_path}")
    return all_fits, metrics_df


def main() -> None:
    t0 = time.time()
    print("[lean] loading non-anomaly dataset")
    non_anom = a_data_io.load_non_anomaly()
    print("[lean] preparing 15.03 within-session regions")
    non_anom_with_region, _ = prepare_regions()

    all_fits: list[FitRow] = []
    for variant_set in ["lean_a", "lean_b"]:
        metrics_path = fl.RESULTS_DIR / f"{variant_set}_metrics.parquet"
        fits_local_path = fl.RESULTS_DIR / f"{variant_set}_fits.parquet"
        if metrics_path.exists() and fits_local_path.exists():
            print(f"[lean] {variant_set}: metrics + fits already present; skipping fits.")
            inv = pd.read_parquet(fits_local_path)
            for _, r in inv.iterrows():
                all_fits.append(FitRow(**{k: r[k] for k in FitRow.__dataclass_fields__}))
            continue
        fits, _metrics = run_for_variant_set(variant_set, non_anom, non_anom_with_region)
        all_fits.extend(fits)

    fits_df = pd.DataFrame([asdict(f) for f in all_fits])
    fits_df.to_parquet(fl.RESULTS_DIR / "fit_inventory.parquet", index=False)
    print(f"\n[lean] all done; {len(all_fits)} total fits in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
