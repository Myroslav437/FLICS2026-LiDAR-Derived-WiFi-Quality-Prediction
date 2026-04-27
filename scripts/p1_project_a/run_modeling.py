"""Phase 1 modeling driver.

Runs 28 fits:
  - Main ablation: 3 folds × 6 variants (B0..B5)
  - Disambiguation: 3 folds × 2 variants (B5', B5'')
  - RQ4: 2 variants × 2 datasets

Persists models, prediction caches, and consolidated metrics parquet files.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config, data_io, folds, metrics, training


@dataclass
class FitResult:
    fold: str
    variant: str
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    wallclock_s: float
    model_sha256: str


def _run_one_loro(
    fold: folds.FoldSplit,
    variant: str,
    features: list[str],
) -> FitResult:
    t0 = time.time()
    booster, best_iter = training.fit_booster(fold.train, fold.val, features)
    y_pred = training.predict(booster, fold.test, features, best_iter)
    preds = training.build_predictions_frame(fold.test, features, y_pred)

    model_path = config.MODELS_DIR / f"{fold.name}_{variant}.json"
    booster.save_model(str(model_path))
    preds_path = config.CACHE_DIR / f"predictions_{fold.name}_{variant}.parquet"
    preds.to_parquet(preds_path, index=False)

    return FitResult(
        fold=fold.name,
        variant=variant,
        best_iteration=best_iter,
        n_features=len(features),
        n_train=len(fold.train),
        n_val=len(fold.val),
        n_test=len(fold.test),
        wallclock_s=time.time() - t0,
        model_sha256=training.model_sha256(model_path),
    )


def run_main_ablation_and_disambig() -> tuple[pd.DataFrame, pd.DataFrame, list[FitResult]]:
    print("[modeling] loading dataset")
    non_anom = data_io.load_non_anomaly()

    main_variants = ["B0", "B1", "B2", "B3", "B4", "B5"]
    disambig_variants = ["B5p", "B5pp"]

    fits: list[FitResult] = []
    loro_rows: list[dict] = []
    disambig_rows: list[dict] = []

    for fold_name in ["F-A", "F-B", "F-C"]:
        print(f"[modeling] building fold {fold_name}")
        fold = folds.build_loro_fold(non_anom, fold_name)
        print(
            f"  train n={len(fold.train)} (per-session {fold.train_session_counts}), "
            f"val n={len(fold.val)}, test n={len(fold.test)}"
        )
        for variant in main_variants + disambig_variants:
            features = config.VARIANTS[variant]
            print(f"  fit {fold_name} / {variant} ({len(features)} features)")
            fr = _run_one_loro(fold, variant, features)
            fits.append(fr)
            print(
                f"    best_iter={fr.best_iteration} wall={fr.wallclock_s:.1f}s"
            )

            preds = pd.read_parquet(config.CACHE_DIR / f"predictions_{fold_name}_{variant}.parquet")
            for stratum_metrics in metrics.evaluate_predictions(preds):
                row = {"fold": fold_name, "variant": variant, **stratum_metrics}
                if variant in main_variants:
                    loro_rows.append(row)
                else:
                    disambig_rows.append(row)
                # Note B5 also belongs in disambig_metrics for the disambig-table comparison; we'll re-emit below.

    # B5 belongs in both: copy main_ablation B5 rows into disambig table for direct comparison.
    main_df = pd.DataFrame(loro_rows)
    disambig_df = pd.DataFrame(disambig_rows)
    if not main_df.empty:
        b5_rows = main_df[main_df["variant"] == "B5"].copy()
        disambig_df = pd.concat([disambig_df, b5_rows], axis=0, ignore_index=True)

    main_df.to_parquet(config.RESULTS_DIR / "loro_metrics.parquet", index=False)
    disambig_df.to_parquet(config.RESULTS_DIR / "disambig_metrics.parquet", index=False)
    return main_df, disambig_df, fits


def run_rq4() -> tuple[pd.DataFrame, list[FitResult]]:
    """RQ4: B5 + per-session intercept (variant a) and B5 + session_id (variant b),
    fit on same-map and full datasets. Reports residual session-effect via SHAP."""
    import xgboost as xgb

    print("[rq4] loading dataset")
    non_anom = data_io.load_non_anomaly()

    rows: list[dict] = []
    fits: list[FitResult] = []
    base_features = list(config.B5)

    for variant in ["rq4a", "rq4b"]:
        for dataset in ["same_map", "full"]:
            t0 = time.time()
            split = folds.build_rq4_split(non_anom, dataset=dataset)
            train_df, extra = folds.add_rq4_features(split.train, variant)
            val_df, _ = folds.add_rq4_features(split.val, variant)
            test_df, _ = folds.add_rq4_features(split.test, variant)
            features = base_features + extra
            print(
                f"[rq4] fit variant={variant} dataset={dataset} "
                f"ntrain={len(train_df)} nval={len(val_df)} ntest={len(test_df)} nfeat={len(features)}"
            )

            booster, best_iter = training.fit_booster(train_df, val_df, features)
            y_pred = training.predict(booster, test_df, features, best_iter)
            preds = training.build_predictions_frame(test_df, features, y_pred)

            tag = f"rq4_{variant}_{dataset}"
            model_path = config.MODELS_DIR / f"{tag}.json"
            booster.save_model(str(model_path))
            preds_path = config.CACHE_DIR / f"predictions_{tag}.parquet"
            preds.to_parquet(preds_path, index=False)

            # SHAP on test rows
            d_test = training.make_dmatrix(test_df, features, target=None)
            shap = booster.predict(d_test, pred_contribs=True, iteration_range=(0, best_iter + 1))
            # shap shape: (n, n_features + 1); last column is bias
            mean_abs = np.mean(np.abs(shap[:, :-1]), axis=0)
            mean_abs_total = float(mean_abs.sum())
            session_idx = [features.index(c) for c in extra]
            mean_abs_session = float(mean_abs[session_idx].sum())
            shap_session_per_feature = {extra[i]: float(mean_abs[session_idx[i]]) for i in range(len(extra))}

            shap_path = config.RESULTS_DIR / f"shap_{tag}.parquet"
            shap_cols = [f"shap_{c}" for c in features] + ["shap_bias"]
            shap_df = pd.DataFrame(shap, columns=shap_cols)
            for c in config.PROVENANCE_COLS + ["signal_power"]:
                shap_df[c] = test_df[c].to_numpy()
            shap_df.to_parquet(shap_path, index=False)

            # Metrics
            y_true = test_df[config.TARGET].to_numpy(dtype=np.float64)
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            bias = float(np.mean(y_true - y_pred))
            fraction = mean_abs_session / mean_abs_total if mean_abs_total > 0 else float("nan")

            rows.append({
                "variant": variant,
                "dataset": dataset,
                "n_test": int(len(test_df)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "bias": bias,
                "mean_abs_shap_session": mean_abs_session,
                "mean_abs_shap_total": mean_abs_total,
                "fraction_session": fraction,
                "shap_session_breakdown": json.dumps(shap_session_per_feature),
                "session_counts": json.dumps(split.session_counts),
                "best_iteration": best_iter,
            })
            fits.append(
                FitResult(
                    fold=tag,
                    variant=variant,
                    best_iteration=best_iter,
                    n_features=len(features),
                    n_train=len(train_df),
                    n_val=len(val_df),
                    n_test=len(test_df),
                    wallclock_s=time.time() - t0,
                    model_sha256=training.model_sha256(model_path),
                )
            )
            print(
                f"  rmse={rmse:.3f} mae={mae:.3f} r2={r2:.3f} "
                f"shap_session={mean_abs_session:.3f}/{mean_abs_total:.3f}={fraction:.3%}"
            )

    df = pd.DataFrame(rows)
    df.to_parquet(config.RESULTS_DIR / "rq4_metrics.parquet", index=False)
    return df, fits


def main() -> None:
    t0 = time.time()
    main_df, disambig_df, main_fits = run_main_ablation_and_disambig()
    rq4_df, rq4_fits = run_rq4()
    fits = main_fits + rq4_fits
    fits_df = pd.DataFrame([asdict(f) for f in fits])
    fits_df.to_parquet(config.RESULTS_DIR / "fit_inventory.parquet", index=False)
    print(f"[modeling] all fits complete in {time.time() - t0:.1f}s; total fits = {len(fits)}")
    print(f"[modeling] loro metrics rows = {len(main_df)}, disambig rows = {len(disambig_df)}, rq4 rows = {len(rq4_df)}")


if __name__ == "__main__":
    main()
