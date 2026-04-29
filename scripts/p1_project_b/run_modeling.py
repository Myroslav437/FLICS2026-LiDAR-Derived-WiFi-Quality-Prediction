"""Project B modeling driver.

Runs:
  - Main ablation: 5 folds × 6 variants (W0..W4, W4'') = 30 fits.
  - Buffer-zone sensitivity on R-1: 6 variants × 1 fold = 6 fits.
  - Hyperparameter sensitivity (W4 only, all 5 folds, H1 hyperparams) = 5 fits.

Total: 41 fits. Persists models, prediction caches, and consolidated metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import training as a_training
from scripts.p1_project_a import metrics as a_metrics

from . import config, data_io, folds as folds_mod, regions as regions_mod


@dataclass
class FitResult:
    fold: str
    variant: str
    hyperparams: str         # "locked" or "H1"
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    test_region: int
    n_train_dropped_by_buffer: int
    buffer_distance_m: float | None
    wallclock_s: float
    model_sha256: str
    model_path: str
    predictions_path: str


# ---------------------------------------------------------------------------
# Training helpers (parameterised XGBoost; fall back to Project A when locked)
# ---------------------------------------------------------------------------


def _fit_booster(
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
        num_boost_round=config.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


def _model_filename(fold_name: str, variant: str, hyperparams: str) -> str:
    if hyperparams == "locked":
        return f"{fold_name}_{variant}.json"
    return f"{fold_name}_{variant}_{hyperparams}.json"


def _preds_filename(fold_name: str, variant: str, hyperparams: str) -> str:
    if hyperparams == "locked":
        return f"predictions_{fold_name}_{variant}.parquet"
    return f"predictions_{fold_name}_{variant}_{hyperparams}.parquet"


def _run_one_fit(
    fold: folds_mod.WithinFoldSplit,
    variant: str,
    features: list[str],
    *,
    hyperparams_label: str = "locked",
) -> FitResult:
    t0 = time.time()
    xgb_params = config.HYPERPARAM_LABELS[hyperparams_label]
    booster, best_iter = _fit_booster(fold.train, fold.val, features, xgb_params)
    y_pred = a_training.predict(booster, fold.test, features, best_iter)
    preds = a_training.build_predictions_frame(fold.test, features, y_pred)
    # Augment predictions with region_id so we can stratify spatially later.
    if "region_id" in fold.test.columns:
        preds["region_id"] = fold.test["region_id"].to_numpy()

    model_path = config.MODELS_DIR / _model_filename(fold.name, variant, hyperparams_label)
    booster.save_model(str(model_path))
    preds_path = config.CACHE_DIR / _preds_filename(fold.name, variant, hyperparams_label)
    preds.to_parquet(preds_path, index=False)

    return FitResult(
        fold=fold.name,
        variant=variant,
        hyperparams=hyperparams_label,
        best_iteration=best_iter,
        n_features=len(features),
        n_train=int(len(fold.train)),
        n_val=int(len(fold.val)),
        n_test=int(len(fold.test)),
        test_region=int(fold.test_region),
        n_train_dropped_by_buffer=int(fold.n_train_dropped_by_buffer),
        buffer_distance_m=fold.buffer_distance_m,
        wallclock_s=float(time.time() - t0),
        model_sha256=a_training.model_sha256(model_path),
        model_path=str(model_path.relative_to(config.PROJECT_ROOT)),
        predictions_path=str(preds_path.relative_to(config.PROJECT_ROOT)),
    )


# ---------------------------------------------------------------------------
# Region preparation
# ---------------------------------------------------------------------------


def prepare_regions() -> tuple[pd.DataFrame, regions_mod.RegionAssignment]:
    """Build region labels from the 15.03 non-anomaly rows; persist artifact; return joined frame.

    Idempotent: if `spatial_regions.parquet` already exists with the right joint_idx universe and
    K regions, reuse it. Otherwise rebuild.
    """
    non_anom = data_io.load_session_non_anomaly()

    artifact_path = config.ARTIFACTS_DIR / "spatial_regions.parquet"
    rebuild = True
    if artifact_path.exists():
        try:
            existing = pd.read_parquet(artifact_path)
            same_set = set(existing["joint_idx"].tolist()) == set(non_anom["joint_idx"].tolist())
            same_k = set(existing["region_id"].unique().tolist()) == set(range(1, config.N_REGIONS + 1))
            if same_set and same_k:
                rebuild = False
        except Exception:
            rebuild = True

    if rebuild:
        assn = regions_mod.build_regions(non_anom)
        regions_mod.save_regions(assn)
    else:
        # Build a lightweight RegionAssignment-like view from the cached file (without method
        # metadata; downstream consumers only use df + region_sizes for size checks).
        df = pd.read_parquet(artifact_path)
        sizes = {int(r): int((df["region_id"] == r).sum()) for r in range(1, config.N_REGIONS + 1)}
        assn = regions_mod.RegionAssignment(
            df=df, method="cached", centroids=None, pca_axis=None, pca_origin=None, region_sizes=sizes
        )

    joined = regions_mod.attach_region_id(non_anom, assn.df)
    return joined, assn


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_main_ablation(non_anom_with_region: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[FitResult]]:
    """30 main fits: 5 folds × 6 variants. Plus the W4'/W2 disambig pointer is implied (W4' ≡ W2)."""
    fits: list[FitResult] = []
    wlro_rows: list[dict] = []
    disambig_rows: list[dict] = []

    for fold_name in config.FOLD_NAMES:
        print(f"[modeling] building fold {fold_name}")
        fold = folds_mod.build_fold(non_anom_with_region, fold_name)
        print(
            f"  region {fold.test_region}: train n={len(fold.train)}, val n={len(fold.val)}, test n={len(fold.test)}"
        )
        for variant in config.MAIN_VARIANTS:
            features = config.VARIANTS[variant]
            print(f"  fit {fold_name} / {variant} ({len(features)} feat, locked)")
            fr = _run_one_fit(fold, variant, features, hyperparams_label="locked")
            fits.append(fr)
            print(f"    best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s")

            preds = pd.read_parquet(config.CACHE_DIR / _preds_filename(fold_name, variant, "locked"))
            for sm in a_metrics.evaluate_predictions(preds):
                row = {"fold": fold_name, "variant": variant, **sm}
                wlro_rows.append(row)

    wlro_df = pd.DataFrame(wlro_rows)

    # Disambiguation table: W4 (full), W4' (≡ W2 — reuse), W4'' (no AP-rel).
    # We re-emit the matching rows under the disambig labels.
    for fold_name in config.FOLD_NAMES:
        for variant in ["W4", "W2", "W4pp"]:
            sub = wlro_df[(wlro_df["fold"] == fold_name) & (wlro_df["variant"] == variant)]
            for _, r in sub.iterrows():
                disambig_rows.append({
                    "fold": fold_name,
                    "variant": variant,
                    "disambig_label": config.DISAMBIG_LABELS[variant],
                    "stratum": r["stratum"],
                    "n": int(r["n"]),
                    "rmse": float(r["rmse"]),
                    "mae": float(r["mae"]),
                    "r2": float(r["r2"]),
                    "bias": float(r["bias"]),
                    "rmse_ci_lo": float(r["rmse_ci_lo"]),
                    "rmse_ci_hi": float(r["rmse_ci_hi"]),
                })

    disambig_df = pd.DataFrame(disambig_rows)

    wlro_df.to_parquet(config.RESULTS_DIR / "wlro_metrics.parquet", index=False)
    disambig_df.to_parquet(config.RESULTS_DIR / "disambig_metrics.parquet", index=False)
    return wlro_df, disambig_df, fits


def run_buffer_sensitivity(non_anom_with_region: pd.DataFrame) -> tuple[pd.DataFrame, list[FitResult]]:
    """6 fits: R-1 with 1 m buffer, all 6 variants."""
    fits: list[FitResult] = []
    rows: list[dict] = []

    fold = folds_mod.build_fold_with_buffer(
        non_anom_with_region,
        config.BUFFER_FOLD,
        buffer_distance_m=config.BUFFER_DISTANCE_M,
    )
    print(
        f"[modeling] buffer fold {fold.name}: dropped {fold.n_train_dropped_by_buffer} train rows; "
        f"train n={len(fold.train)}, val n={len(fold.val)}, test n={len(fold.test)}"
    )

    for variant in config.MAIN_VARIANTS:
        features = config.VARIANTS[variant]
        print(f"  fit {fold.name} / {variant} ({len(features)} feat, locked)")
        fr = _run_one_fit(fold, variant, features, hyperparams_label="locked")
        fits.append(fr)
        print(f"    best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s")

        preds = pd.read_parquet(config.CACHE_DIR / _preds_filename(fold.name, variant, "locked"))
        for sm in a_metrics.evaluate_predictions(preds):
            rows.append({
                "fold": fold.name,
                "base_fold": config.BUFFER_FOLD,
                "buffer_distance_m": float(config.BUFFER_DISTANCE_M),
                "n_train_dropped_by_buffer": int(fold.n_train_dropped_by_buffer),
                "variant": variant,
                **sm,
            })

    df = pd.DataFrame(rows)
    return df, fits


def run_hyperparam_sensitivity(non_anom_with_region: pd.DataFrame) -> tuple[pd.DataFrame, list[FitResult]]:
    """5 fits: W4 under H1 (max_depth=4) on each fold."""
    fits: list[FitResult] = []
    rows: list[dict] = []

    for fold_name in config.FOLD_NAMES:
        fold = folds_mod.build_fold(non_anom_with_region, fold_name)
        variant = "W4"
        features = config.VARIANTS[variant]
        print(f"  fit {fold_name} / {variant} ({len(features)} feat, H1: max_depth=4)")
        fr = _run_one_fit(fold, variant, features, hyperparams_label="H1")
        fits.append(fr)
        print(f"    best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s")

        preds = pd.read_parquet(config.CACHE_DIR / _preds_filename(fold_name, variant, "H1"))
        for sm in a_metrics.evaluate_predictions(preds):
            rows.append({
                "fold": fold_name,
                "variant": variant,
                "hyperparams": "H1",
                **sm,
            })

    df = pd.DataFrame(rows)
    return df, fits


def main() -> None:
    t0 = time.time()
    print("[modeling] preparing regions on 15.03 non-anomaly rows")
    non_anom_with_region, region_assn = prepare_regions()
    region_summary = json.dumps(
        {
            "method": region_assn.method,
            "region_sizes": region_assn.region_sizes,
        },
        indent=2,
    )
    print(f"[modeling] region summary: {region_summary}")

    print("[modeling] === main ablation (30 fits) ===")
    wlro_df, disambig_df, main_fits = run_main_ablation(non_anom_with_region)

    print("[modeling] === buffer sensitivity (6 fits) ===")
    buffer_df, buffer_fits = run_buffer_sensitivity(non_anom_with_region)

    print("[modeling] === hyperparam sensitivity (5 fits) ===")
    hp_df, hp_fits = run_hyperparam_sensitivity(non_anom_with_region)

    # Consolidate robustness metrics.
    buffer_df = buffer_df.assign(check="buffer")
    hp_df = hp_df.assign(check="hyperparams")
    robustness_df = pd.concat([buffer_df, hp_df], axis=0, ignore_index=True)
    robustness_df.to_parquet(config.RESULTS_DIR / "robustness_metrics.parquet", index=False)

    fits = main_fits + buffer_fits + hp_fits
    fits_df = pd.DataFrame([asdict(f) for f in fits])
    fits_df.to_parquet(config.RESULTS_DIR / "fit_inventory.parquet", index=False)
    print(
        f"[modeling] all fits complete in {time.time() - t0:.1f}s; total fits = {len(fits)} "
        f"(expected 41)"
    )
    print(
        f"[modeling] wlro rows = {len(wlro_df)}, disambig rows = {len(disambig_df)}, "
        f"robustness rows = {len(robustness_df)}"
    )


if __name__ == "__main__":
    main()
