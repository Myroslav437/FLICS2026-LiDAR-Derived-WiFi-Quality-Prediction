"""LORO fold construction, F-C cadence decimation, and chronological 90/10 val split.

Also provides the RQ4 within-session 70/15/15 chronological split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import config


@dataclass
class FoldSplit:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_session_counts: dict[str, int]
    val_session_counts: dict[str, int]
    test_session_counts: dict[str, int]


def _chronological_train_val(df_session: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort one session's rows chronologically and split into (train, val) by tail fraction."""
    s = df_session.sort_values("fh7000_timestamp", kind="mergesort").reset_index(drop=True)
    n = len(s)
    n_val = int(round(n * val_frac))
    n_val = max(1, min(n - 1, n_val))
    train = s.iloc[: n - n_val].copy()
    val = s.iloc[n - n_val :].copy()
    return train, val


def _decimate_chronologically(df_session: pd.DataFrame, k: int) -> pd.DataFrame:
    s = df_session.sort_values("fh7000_timestamp", kind="mergesort").reset_index(drop=True)
    return s.iloc[::k].reset_index(drop=True)


def build_loro_fold(non_anom: pd.DataFrame, fold_name: str) -> FoldSplit:
    train_sessions, test_session = config.FOLDS[fold_name]
    test_df = non_anom[non_anom["session_date"] == test_session].copy().reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    apply_cadence_match = fold_name == "F-C"

    for sess in train_sessions:
        sess_df = non_anom[non_anom["session_date"] == sess].copy()
        if apply_cadence_match:
            sess_df = _decimate_chronologically(sess_df, config.CADENCE_DECIMATION_FACTOR)
        tr, va = _chronological_train_val(sess_df, val_frac=0.10)
        train_parts.append(tr)
        val_parts.append(va)
        train_counts[sess] = len(tr)
        val_counts[sess] = len(va)

    train = pd.concat(train_parts, axis=0, ignore_index=True)
    val = pd.concat(val_parts, axis=0, ignore_index=True)

    test_counts = {test_session: len(test_df)}

    return FoldSplit(
        name=fold_name,
        train=train,
        val=val,
        test=test_df,
        train_session_counts=train_counts,
        val_session_counts=val_counts,
        test_session_counts=test_counts,
    )


# ---- RQ4 splits ---------------------------------------------------------


@dataclass
class RQ4Split:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    session_counts: dict[str, dict[str, int]]


def _chronological_70_15_15(df_session: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = df_session.sort_values("fh7000_timestamp", kind="mergesort").reset_index(drop=True)
    n = len(s)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train = s.iloc[:n_train].copy()
    val = s.iloc[n_train : n_train + n_val].copy()
    test = s.iloc[n_train + n_val :].copy()
    return train, val, test


def build_rq4_split(non_anom: pd.DataFrame, *, dataset: str) -> RQ4Split:
    """dataset ∈ {"same_map", "full"}."""
    if dataset == "same_map":
        sessions = ["15.03.2026", "24.03.2026"]
    elif dataset == "full":
        sessions = ["15.03.2026", "24.03.2026", "25.02.2026"]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    sub = non_anom[non_anom["session_date"].isin(sessions)].copy()
    train_parts, val_parts, test_parts = [], [], []
    counts: dict[str, dict[str, int]] = {}
    for sess in sessions:
        sess_df = sub[sub["session_date"] == sess].copy()
        tr, va, te = _chronological_70_15_15(sess_df)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        counts[sess] = {"train": len(tr), "val": len(va), "test": len(te)}

    return RQ4Split(
        name=dataset,
        train=pd.concat(train_parts, axis=0, ignore_index=True),
        val=pd.concat(val_parts, axis=0, ignore_index=True),
        test=pd.concat(test_parts, axis=0, ignore_index=True),
        session_counts=counts,
    )


# ---- Feature-matrix helpers --------------------------------------------


def add_rq4_features(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Append RQ4 session-encoding columns to df (in-place safe; returns new frame)."""
    out = df.copy()
    if variant == "rq4a":
        out["is_session_15_03"] = (out["session_date"] == "15.03.2026").astype("int8")
        out["is_session_24_03"] = (out["session_date"] == "24.03.2026").astype("int8")
        extra = ["is_session_15_03", "is_session_24_03"]
    elif variant == "rq4b":
        mapping = {"15.03.2026": 0, "24.03.2026": 1, "25.02.2026": 2}
        out["session_id"] = out["session_date"].map(mapping).astype("int8")
        extra = ["session_id"]
    else:
        raise ValueError(f"unknown rq4 variant {variant!r}")
    return out, extra
