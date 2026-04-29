"""LiDAR placebo: within-(sub)group block-shuffle of the 19 ego-frame LiDAR columns.

The shuffle is applied **only to the training set**. The test set is kept intact.
One permutation π is generated per group and applied to all 19 LiDAR columns at once
(block shuffle), preserving the joint distribution among LiDAR features but breaking
their alignment with the rest of the row (telemetry, position, AP-relative, target).

`clutter_frac_toward_AP` and `is_AP_in_FOV` are AP-relative features and are NOT
shuffled.

Used by Project A's Hardening A (cross-session placebo on F-B) and Project B's
Hardening E (within-session placebo on R-4). Originally lived under
`scripts/p1_hardening/placebo.py`; merged into Project A on 2026-04-28 because
both Project A's and Project B's hardening runners share this helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _validate_lidar_columns(df: pd.DataFrame) -> None:
    missing = [c for c in config.LIDAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing LiDAR columns: {missing}")


def block_shuffle_lidar(
    train_df: pd.DataFrame,
    *,
    group_col: str | None = None,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Return a copy of train_df with the 19 LiDAR columns block-shuffled.

    If ``group_col`` is provided (e.g., "session_date"), the shuffle is performed
    independently per group. Within each group, a single permutation π is sampled
    and applied to all 19 LiDAR columns simultaneously — i.e., row i's full LiDAR
    block is replaced by row π(i)'s LiDAR block.

    The seed for each group is ``seed + offset`` where offset is the group's stable
    enumerated position (0, 1, ...). This makes the shuffle deterministic across
    re-runs.
    """
    _validate_lidar_columns(train_df)
    out = train_df.copy().reset_index(drop=True)
    lidar = out[config.LIDAR_COLUMNS].to_numpy(copy=True)

    if group_col is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(out))
        shuffled = lidar[perm]
    else:
        if group_col not in out.columns:
            raise ValueError(f"group_col {group_col!r} not in dataframe")
        shuffled = lidar.copy()
        groups = sorted(out[group_col].unique().tolist())
        for offset, g in enumerate(groups):
            mask = (out[group_col].to_numpy() == g)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            rng = np.random.default_rng(seed + offset)
            perm = rng.permutation(idx.size)
            shuffled[idx] = lidar[idx[perm]]

    out.loc[:, config.LIDAR_COLUMNS] = shuffled
    return out


# ---- Integrity-check helpers -------------------------------------------


def marginal_match(
    df_before: pd.DataFrame, df_after: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Per-LiDAR-column mean/std change after shuffle. Should be ~0 for true block shuffle."""
    out: dict[str, dict[str, float]] = {}
    for c in config.LIDAR_COLUMNS:
        a = df_before[c].to_numpy(dtype=np.float64)
        b = df_after[c].to_numpy(dtype=np.float64)
        out[c] = {
            "mean_diff": float(np.mean(a) - np.mean(b)),
            "std_diff": float(np.std(a) - np.std(b)),
            "max_abs_mean_diff": float(abs(np.mean(a) - np.mean(b))),
        }
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via rank-then-Pearson; no SciPy dependency."""
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy().astype(np.float64, copy=True)
    ry = pd.Series(y).rank().to_numpy().astype(np.float64, copy=True)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)
