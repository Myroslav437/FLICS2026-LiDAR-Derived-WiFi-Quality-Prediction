"""Within-session leave-region-out fold construction.

For each fold R-k (k ∈ {1..5}):
  - test  = rows with region_id == k
  - train = rows with region_id != k, minus a random 10% validation split

Buffer-zone variant: drop training rows within `BUFFER_DISTANCE_M` of any
held-out-region row before the train/val split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, regions


@dataclass
class WithinFoldSplit:
    name: str                   # e.g. "R-1" or "R-1_buffer"
    test_region: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_train_dropped_by_buffer: int  # 0 unless buffer applied
    buffer_distance_m: float | None  # None unless buffer applied


def _random_train_val_split(train_pool: pd.DataFrame, fold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 10% validation split with a fold-stable seed."""
    rng = np.random.default_rng(config.SEED + (hash(fold_name) & 0x7FFFFFFF))
    n = len(train_pool)
    n_val = max(1, int(round(n * config.VAL_FRACTION)))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    val = train_pool.iloc[val_idx].reset_index(drop=True)
    train = train_pool.iloc[tr_idx].reset_index(drop=True)
    return train, val


def build_fold(non_anom_with_region: pd.DataFrame, fold_name: str) -> WithinFoldSplit:
    if fold_name not in config.FOLD_TO_REGION:
        raise ValueError(f"unknown fold name {fold_name!r}; expected one of {config.FOLD_NAMES}")
    test_region = config.FOLD_TO_REGION[fold_name]

    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    train, val = _random_train_val_split(train_pool, fold_name)
    return WithinFoldSplit(
        name=fold_name,
        test_region=test_region,
        train=train,
        val=val,
        test=test_df,
        n_train_dropped_by_buffer=0,
        buffer_distance_m=None,
    )


def build_fold_with_buffer(
    non_anom_with_region: pd.DataFrame,
    fold_name: str,
    buffer_distance_m: float = config.BUFFER_DISTANCE_M,
) -> WithinFoldSplit:
    """Same as build_fold but drop training rows within buffer_distance_m of any held-out row."""
    if fold_name not in config.FOLD_TO_REGION:
        raise ValueError(f"unknown fold name {fold_name!r}")
    test_region = config.FOLD_TO_REGION[fold_name]

    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool_full = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)

    train_coords = train_pool_full[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    holdout_coords = test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    drop_mask = regions.build_buffer_mask(train_coords, holdout_coords, buffer_distance_m)
    n_dropped = int(drop_mask.sum())
    train_pool = train_pool_full[~drop_mask].reset_index(drop=True)

    train, val = _random_train_val_split(train_pool, f"{fold_name}_buffer")
    return WithinFoldSplit(
        name=f"{fold_name}_buffer",
        test_region=test_region,
        train=train,
        val=val,
        test=test_df,
        n_train_dropped_by_buffer=n_dropped,
        buffer_distance_m=buffer_distance_m,
    )
