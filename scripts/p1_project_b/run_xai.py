"""Project B TreeSHAP driver: SHAP for the W4 model on the chosen fold (default R-3).

If R-3 W4 RMSE is unusually bad (>1 dB worse than other folds' median W4 RMSE), pick the
fold whose W4 overall RMSE is closest to the median.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import training as a_training

from . import config, data_io, folds as folds_mod, run_modeling


def select_shap_fold(wlro_df: pd.DataFrame) -> tuple[str, str]:
    """Pick the SHAP fold per the brief: default R-3 unless its W4 overall RMSE is >1 dB worse
    than the median W4 overall RMSE.

    Returns (fold_name, justification_text).
    """
    sub = wlro_df[(wlro_df["variant"] == "W4") & (wlro_df["stratum"] == "overall")]
    if sub.empty:
        return config.SHAP_FOLD_DEFAULT, "no W4 overall metrics found; defaulting to R-3"
    median = float(sub["rmse"].median())
    r3 = sub[sub["fold"] == config.SHAP_FOLD_DEFAULT]
    if r3.empty:
        return config.SHAP_FOLD_DEFAULT, "R-3 W4 metric missing; defaulting to R-3"
    r3_rmse = float(r3.iloc[0]["rmse"])
    if r3_rmse - median > 1.0:
        # Pick the fold whose W4 overall RMSE is closest to the median.
        sub2 = sub.copy()
        sub2["dist"] = (sub2["rmse"] - median).abs()
        chosen = sub2.sort_values("dist").iloc[0]["fold"]
        return str(chosen), (
            f"R-3 W4 overall RMSE = {r3_rmse:.2f} dB exceeds median {median:.2f} dB by >1 dB; "
            f"picked {chosen} as closest-to-median fold."
        )
    return config.SHAP_FOLD_DEFAULT, (
        f"R-3 W4 overall RMSE = {r3_rmse:.2f} dB within 1 dB of median {median:.2f} dB; using R-3."
    )


def run_shap_for_fold(fold_name: str, non_anom_with_region: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fold = folds_mod.build_fold(non_anom_with_region, fold_name)
    test = fold.test.reset_index(drop=True)

    rng = np.random.default_rng(config.SEED + (hash(fold_name) & 0x7FFFFFFF))
    if len(test) > config.SHAP_SUBSAMPLE_LIMIT:
        idx = rng.choice(len(test), size=config.SHAP_SUBSAMPLE_LIMIT, replace=False)
        idx.sort()
        test_sub = test.iloc[idx].reset_index(drop=True)
        subsampled = True
    else:
        test_sub = test
        subsampled = False

    features = list(config.W4)
    booster = xgb.Booster()
    model_path = config.MODELS_DIR / f"{fold_name}_W4.json"
    booster.load_model(str(model_path))
    best_iter = (
        int(booster.best_iteration)
        if booster.best_iteration is not None
        else booster.num_boosted_rounds() - 1
    )

    d_test = a_training.make_dmatrix(test_sub, features, target=None)
    t0 = time.time()
    shap = booster.predict(d_test, pred_contribs=True, iteration_range=(0, best_iter + 1))
    print(f"[xai] {fold_name} shap shape={shap.shape} wall={time.time() - t0:.1f}s subsampled={subsampled}")

    y_pred_from_shap = shap.sum(axis=1)
    y_pred_direct = booster.predict(d_test, iteration_range=(0, best_iter + 1))
    diff = float(np.max(np.abs(y_pred_from_shap - y_pred_direct)))
    print(f"  shap-row-sum vs predict max abs diff = {diff:.6f}")

    out = pd.DataFrame()
    for c in config.PROVENANCE_COLS:
        out[c] = test_sub[c].to_numpy()
    if "region_id" in test_sub.columns:
        out["region_id"] = test_sub["region_id"].to_numpy()
    out["signal_power_true"] = test_sub[config.TARGET].to_numpy(dtype=np.float64)
    out["signal_power_pred"] = y_pred_direct.astype(np.float64)
    for feat in features:
        if feat in out.columns:
            continue
        col = test_sub[feat]
        if col.dtype == bool:
            out[feat] = col.astype("int8").to_numpy()
        else:
            out[feat] = col.to_numpy(dtype=np.float64)
    for i, feat in enumerate(features):
        out[f"shap_{feat}"] = shap[:, i].astype(np.float64)
    out["shap_bias"] = shap[:, -1].astype(np.float64)

    metadata = {
        "fold": fold_name,
        "n_rows_used": int(len(test_sub)),
        "subsampled_from": int(len(test)),
        "shap_consistency_max_abs_diff": diff,
        "best_iteration": int(best_iter),
    }
    return out, metadata


def main() -> None:
    print("[xai] preparing regions and loading metrics")
    non_anom_with_region, _ = run_modeling.prepare_regions()
    wlro_df = pd.read_parquet(config.RESULTS_DIR / "wlro_metrics.parquet")

    fold_name, justification = select_shap_fold(wlro_df)
    print(f"[xai] selected fold = {fold_name} ({justification})")

    df, metadata = run_shap_for_fold(fold_name, non_anom_with_region)
    out_path = config.RESULTS_DIR / f"shap_{fold_name}.parquet"
    df.to_parquet(out_path, index=False)

    metadata["selection_justification"] = justification
    (config.RESULTS_DIR / f"shap_{fold_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[xai] wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
