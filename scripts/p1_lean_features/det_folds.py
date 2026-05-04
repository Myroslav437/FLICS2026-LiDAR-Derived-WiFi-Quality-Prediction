"""Deterministic-seed wrapper around scripts.p1_project_b.folds.

The locked `_random_train_val_split` seeds with `config.SEED + (hash(fold_name) & 0x7FFFFFFF)`.
Python's built-in `hash()` of strings is randomised across invocations unless
`PYTHONHASHSEED=0`, so the locked Project B's val rows differ from any later rerun's
val rows for the same fold name. That non-determinism contaminates §3 of the
lean-features comparison report (lean-vs-locked Δ conflates feature-removal effect
with split-noise).

This module provides a parallel `build_fold` / `build_fold_with_buffer` that produces
identical `WithinFoldSplit` semantics to `scripts.p1_project_b.folds`, except that the
fold-stable seed is derived from a SHA-256 digest of `fold_name`, which is reproducible
across Python invocations and machines. The locked Project B artifacts under
`scripts/p1_project_b/` are untouched — this wrapper only affects the deterministic
re-run under `scripts/p1_lean_features/`.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from scripts.p1_project_b import config as b_config
from scripts.p1_project_b import regions as b_regions
from scripts.p1_project_b.folds import WithinFoldSplit


def deterministic_fold_offset(fold_name: str) -> int:
    """32-bit non-negative offset derived deterministically from fold_name.

    Reproducible across Python invocations, machines, and PYTHONHASHSEED settings.
    """
    digest = hashlib.sha256(fold_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _det_random_train_val_split(
    train_pool: pd.DataFrame, fold_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(b_config.SEED + deterministic_fold_offset(fold_name))
    n = len(train_pool)
    n_val = max(1, int(round(n * b_config.VAL_FRACTION)))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    val = train_pool.iloc[val_idx].reset_index(drop=True)
    train = train_pool.iloc[tr_idx].reset_index(drop=True)
    return train, val


def build_fold(non_anom_with_region: pd.DataFrame, fold_name: str) -> WithinFoldSplit:
    if fold_name not in b_config.FOLD_TO_REGION:
        raise ValueError(f"unknown fold name {fold_name!r}; expected one of {b_config.FOLD_NAMES}")
    test_region = b_config.FOLD_TO_REGION[fold_name]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    train, val = _det_random_train_val_split(train_pool, fold_name)
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
    buffer_distance_m: float = b_config.BUFFER_DISTANCE_M,
) -> WithinFoldSplit:
    if fold_name not in b_config.FOLD_TO_REGION:
        raise ValueError(f"unknown fold name {fold_name!r}")
    test_region = b_config.FOLD_TO_REGION[fold_name]
    test_df = non_anom_with_region[non_anom_with_region["region_id"] == test_region].reset_index(drop=True)
    train_pool_full = non_anom_with_region[non_anom_with_region["region_id"] != test_region].reset_index(drop=True)
    train_coords = train_pool_full[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    holdout_coords = test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    drop_mask = b_regions.build_buffer_mask(train_coords, holdout_coords, buffer_distance_m)
    n_dropped = int(drop_mask.sum())
    train_pool = train_pool_full[~drop_mask].reset_index(drop=True)
    train, val = _det_random_train_val_split(train_pool, f"{fold_name}_buffer")
    return WithinFoldSplit(
        name=f"{fold_name}_buffer",
        test_region=test_region,
        train=train,
        val=val,
        test=test_df,
        n_train_dropped_by_buffer=n_dropped,
        buffer_distance_m=buffer_distance_m,
    )
