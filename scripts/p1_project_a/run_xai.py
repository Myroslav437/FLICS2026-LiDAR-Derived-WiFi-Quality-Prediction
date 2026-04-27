"""Phase 1 TreeSHAP driver: per-fold SHAP for the B5 model on test rows."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, data_io, folds, training


def _shap_for_fold(fold_name: str, non_anom: pd.DataFrame) -> pd.DataFrame:
    fold = folds.build_loro_fold(non_anom, fold_name)
    test = fold.test.reset_index(drop=True)
    rng = np.random.default_rng(config.SEED + hash(fold_name) % (1 << 32))

    if len(test) > config.SHAP_SUBSAMPLE_LIMIT:
        idx = rng.choice(len(test), size=config.SHAP_SUBSAMPLE_LIMIT, replace=False)
        idx.sort()
        test = test.iloc[idx].reset_index(drop=True)
        subsampled = True
    else:
        subsampled = False

    features = list(config.B5)
    booster = xgb.Booster()
    model_path = config.MODELS_DIR / f"{fold_name}_B5.json"
    booster.load_model(str(model_path))
    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else booster.num_boosted_rounds() - 1

    d_test = training.make_dmatrix(test, features, target=None)
    t0 = time.time()
    shap = booster.predict(d_test, pred_contribs=True, iteration_range=(0, best_iter + 1))
    print(f"[xai] {fold_name} shap shape={shap.shape} wall={time.time() - t0:.1f}s subsampled={subsampled}")

    # Quick sanity check
    y_pred_from_shap = shap.sum(axis=1)
    y_pred_direct = booster.predict(d_test, iteration_range=(0, best_iter + 1))
    diff = float(np.max(np.abs(y_pred_from_shap - y_pred_direct)))
    print(f"  shap-row-sum vs predict max abs diff = {diff:.6f}")

    out = pd.DataFrame()
    for c in config.PROVENANCE_COLS:
        out[c] = test[c].to_numpy()
    out["signal_power_true"] = test[config.TARGET].to_numpy(dtype=np.float64)
    out["signal_power_pred"] = y_pred_direct.astype(np.float64)
    # Raw feature values (needed for XAI-2 sign-consistency, XAI-4 scatter, etc.)
    for feat in features:
        if feat in out.columns:
            continue
        col = test[feat]
        if col.dtype == bool:
            out[feat] = col.astype("int8").to_numpy()
        else:
            out[feat] = col.to_numpy(dtype=np.float64)
    for i, feat in enumerate(features):
        out[f"shap_{feat}"] = shap[:, i].astype(np.float64)
    out["shap_bias"] = shap[:, -1].astype(np.float64)
    return out


def main() -> None:
    non_anom = data_io.load_non_anomaly()
    for fold_name in ["F-A", "F-B", "F-C"]:
        df = _shap_for_fold(fold_name, non_anom)
        out_path = config.RESULTS_DIR / f"shap_{fold_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[xai] wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
