"""XGBoost training helper used by run_modeling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config


def _to_xgb_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    if "is_AP_in_FOV" in out.columns:
        out["is_AP_in_FOV"] = out["is_AP_in_FOV"].astype("int8")
    return out


def make_dmatrix(df: pd.DataFrame, features: list[str], target: str | None = config.TARGET) -> xgb.DMatrix:
    X = _to_xgb_frame(df, features)
    if target is not None and target in df.columns:
        y = df[target].to_numpy(dtype=np.float64)
        return xgb.DMatrix(X, label=y, feature_names=list(features))
    return xgb.DMatrix(X, feature_names=list(features))


def fit_booster(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
) -> tuple[xgb.Booster, int]:
    dtrain = make_dmatrix(train_df, features)
    dval = make_dmatrix(val_df, features)
    booster = xgb.train(
        params=config.XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=config.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


def predict(booster: xgb.Booster, df: pd.DataFrame, features: list[str], best_iter: int) -> np.ndarray:
    d = make_dmatrix(df, features, target=None)
    return booster.predict(d, iteration_range=(0, best_iter + 1))


def build_predictions_frame(
    test_df: pd.DataFrame,
    features: list[str],
    y_pred: np.ndarray,
) -> pd.DataFrame:
    cols = list(config.PROVENANCE_COLS)
    out = test_df[cols].copy().reset_index(drop=True)
    y_true = test_df[config.TARGET].to_numpy(dtype=np.float64)
    out["signal_power_true"] = y_true
    out["signal_power_pred"] = y_pred
    out["residual"] = y_true - y_pred
    return out


def model_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
