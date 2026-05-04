"""Statistical helpers for the leakage investigation.

All functions are pure (input array(s) -> scalar / dict / DataFrame). No I/O.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def acf_at_lags(x: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Sample autocorrelation (Bartlett / biased estimator) at the given lags.

    Uses FFT-based autocovariance for O(n log n) cost on long series.
    Returns r[k] = sum_t (x_t - mu)(x_{t+k} - mu) / sum_t (x_t - mu)^2.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.full(len(lags), np.nan)
    xc = x - x.mean()
    var = float((xc * xc).sum())
    if var == 0.0:
        out = np.zeros(len(lags))
        out[np.array(lags) == 0] = 1.0
        return out
    # FFT length: at least 2n-1, rounded up to next power of two
    fft_n = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(xc, n=fft_n)
    acov_full = np.fft.irfft(fx * np.conj(fx), n=fft_n)[:n]
    out = []
    for lag in lags:
        if lag < 0 or lag >= n:
            out.append(np.nan)
        else:
            out.append(acov_full[lag] / var)
    return np.asarray(out)


def acf_full(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Vector of autocorrelations from lag 0 .. max_lag (inclusive)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xc = x - x.mean()
    var = float((xc * xc).sum())
    if var == 0.0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    fft_n = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(xc, n=fft_n)
    acov_full = np.fft.irfft(fx * np.conj(fx), n=fft_n)[: max_lag + 1]
    return acov_full / var


def first_crossing(acf_curve: np.ndarray, lags: Sequence[int], threshold: float) -> int | None:
    """Smallest lag in `lags` for which |acf_curve[lag]| < threshold.

    `acf_curve` is indexed by absolute lag (acf_curve[lag] is the value at lag).
    """
    for lag in lags:
        if lag >= len(acf_curve):
            continue
        if abs(acf_curve[lag]) < threshold:
            return int(lag)
    return None


# ---------------------------------------------------------------------------
# Partial correlation (Spearman, residualised on ranked controls)
# ---------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    return sstats.rankdata(a, method="average")


def _residual(y_rank: np.ndarray, Z_rank: np.ndarray) -> np.ndarray:
    """OLS residual of y_rank on [intercept, Z_rank columns]."""
    n = y_rank.shape[0]
    if Z_rank.size == 0:
        return y_rank - y_rank.mean()
    X = np.column_stack([np.ones(n), Z_rank])
    beta, *_ = np.linalg.lstsq(X, y_rank, rcond=None)
    return y_rank - X @ beta


def spearman_partial(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray | None
) -> float:
    """Spearman partial correlation: rank then residualise then Pearson."""
    xr = _rank(x)
    yr = _rank(y)
    if controls is None or controls.size == 0:
        zr = np.empty((xr.shape[0], 0))
    else:
        if controls.ndim == 1:
            controls = controls[:, None]
        zr = np.column_stack([_rank(controls[:, j]) for j in range(controls.shape[1])])
    rx = _residual(xr, zr)
    ry = _residual(yr, zr)
    sx = rx.std()
    sy = ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_partial_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray | None,
    n_boot: int = 200,
    seed: int = 20260427,
    block: int = 200,
) -> tuple[float, float, float]:
    """(point estimate, lo95, hi95) using a moving-block bootstrap.

    Block bootstrap is used instead of an i.i.d. resample because the data
    are time-series and have strong serial correlation; an i.i.d. bootstrap
    would underestimate variance.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    point = spearman_partial(x, y, controls)
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    if controls is None:
        controls = np.empty((n, 0))
    elif controls.ndim == 1:
        controls = controls[:, None]
    block = max(1, min(block, n // 4))
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        estimates[b] = spearman_partial(
            x[idx], y[idx], controls[idx] if controls.size else None
        )
    lo = float(np.nanpercentile(estimates, 2.5))
    hi = float(np.nanpercentile(estimates, 97.5))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Other helpers
# ---------------------------------------------------------------------------

def spearman_with_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 200,
    seed: int = 20260427,
    block: int = 200,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    rho = float(sstats.spearmanr(x, y).statistic)
    block = max(1, min(block, n // 4))
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        estimates[b] = sstats.spearmanr(x[idx], y[idx]).statistic
    lo = float(np.nanpercentile(estimates, 2.5))
    hi = float(np.nanpercentile(estimates, 97.5))
    return rho, lo, hi


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = sstats.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def levene_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    res = sstats.levene(a, b, c, center="median")
    return float(res.statistic), float(res.pvalue)
