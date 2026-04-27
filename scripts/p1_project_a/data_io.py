"""Dataset loading and SHA verification."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from . import config


def _read_expected_sha() -> str:
    text = config.DATASET_SHA_PATH.read_text().strip()
    return text.split()[0]


def compute_dataset_sha() -> str:
    h = hashlib.sha256()
    with config.DATASET_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(config.DATASET_PATH)
    return df


def load_non_anomaly() -> pd.DataFrame:
    df = load_dataset()
    return df[~df["anomaly_flag"]].reset_index(drop=True)
