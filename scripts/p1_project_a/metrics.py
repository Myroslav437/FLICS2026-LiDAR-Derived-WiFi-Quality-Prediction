"""Metric computation: RMSE/MAE/R²/bias and bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true - y_pred))


def bootstrap_rmse_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    B: int = config.BOOTSTRAP_B,
    seed: int = config.SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rmses = np.empty(B, dtype=np.float64)
    sq_err = (y_true - y_pred) ** 2
    for b in range(B):
        idx = rng.integers(0, n, n)
        rmses[b] = float(np.sqrt(np.mean(sq_err[idx])))
    lo = float(np.percentile(rmses, 2.5))
    hi = float(np.percentile(rmses, 97.5))
    return lo, hi


def stratum_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, with_ci: bool = True) -> dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "rmse": _rmse(y_true, y_pred) if len(y_true) > 0 else float("nan"),
        "mae": _mae(y_true, y_pred) if len(y_true) > 0 else float("nan"),
        "r2": _r2(y_true, y_pred) if len(y_true) > 0 else float("nan"),
        "bias": _bias(y_true, y_pred) if len(y_true) > 0 else float("nan"),
    }
    if with_ci and len(y_true) > 0:
        lo, hi = bootstrap_rmse_ci(y_true, y_pred)
        out["rmse_ci_lo"] = lo
        out["rmse_ci_hi"] = hi
    else:
        out["rmse_ci_lo"] = float("nan")
        out["rmse_ci_hi"] = float("nan")
    return out


def evaluate_predictions(preds: pd.DataFrame) -> list[dict]:
    """Compute overall, in-FOV, out-of-FOV strata. preds has signal_power_true/_pred + is_AP_in_FOV."""
    y = preds["signal_power_true"].to_numpy(dtype=np.float64)
    p = preds["signal_power_pred"].to_numpy(dtype=np.float64)
    in_fov = preds["is_AP_in_FOV"].to_numpy(dtype=bool)

    rows: list[dict] = []
    fsv = config.FEATURE_STACK_VERSION

    rows.append({"stratum": "overall", "feature_stack_version": fsv, **stratum_metrics(y, p)})
    rows.append({"stratum": "in_fov", "feature_stack_version": fsv, **stratum_metrics(y[in_fov], p[in_fov])})
    rows.append({"stratum": "out_of_fov", "feature_stack_version": fsv, **stratum_metrics(y[~in_fov], p[~in_fov])})

    return rows
