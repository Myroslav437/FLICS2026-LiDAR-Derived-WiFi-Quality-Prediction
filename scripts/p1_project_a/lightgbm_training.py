"""LightGBM training wrapper that mirrors scripts.p1_project_a.training's API.

Same input/output convention:
  - feature matrix + target + validation set in
  - trained model + best_iteration + predictions out
  - same prediction-cache schema (joint_idx, session_date, fh7000_timestamp,
    x_m, y_m, is_AP_in_FOV, signal_power_true, signal_power_pred, residual)
  - same model SHA-256 derivation (hash of saved model file)

Used only by Hardening B (LightGBM cross-check on F-B). Originally lived under
`scripts/p1_hardening/lightgbm_training.py`; merged into Project A on
2026-04-28.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as a_config


def _to_lgb_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    if "is_AP_in_FOV" in out.columns:
        out["is_AP_in_FOV"] = out["is_AP_in_FOV"].astype("int8")
    return out


def fit_booster(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    lgb_params: dict,
) -> tuple[lgb.Booster, int]:
    X_tr = _to_lgb_frame(train_df, features)
    y_tr = train_df[a_config.TARGET].to_numpy(dtype=np.float64)
    X_va = _to_lgb_frame(val_df, features)
    y_va = val_df[a_config.TARGET].to_numpy(dtype=np.float64)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=list(features), free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=list(features), free_raw_data=False, reference=dtrain)

    booster = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=a_config.N_ESTIMATORS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=a_config.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else booster.current_iteration()
    return booster, best_iter


def predict(booster: lgb.Booster, df: pd.DataFrame, features: list[str], best_iter: int) -> np.ndarray:
    X = _to_lgb_frame(df, features)
    return booster.predict(X, num_iteration=best_iter)


def build_predictions_frame(
    test_df: pd.DataFrame,
    features: list[str],
    y_pred: np.ndarray,
) -> pd.DataFrame:
    cols = list(a_config.PROVENANCE_COLS)
    out = test_df[cols].copy().reset_index(drop=True)
    y_true = test_df[a_config.TARGET].to_numpy(dtype=np.float64)
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


def save_model(booster: lgb.Booster, path: Path) -> None:
    booster.save_model(str(path))
