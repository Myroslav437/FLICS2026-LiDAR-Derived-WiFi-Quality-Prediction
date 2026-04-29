"""Project B data loading: 15.03 anomaly-filtered slice."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from scripts.p1_project_a import data_io as a_data_io

from . import config


@lru_cache(maxsize=1)
def load_session_non_anomaly() -> pd.DataFrame:
    """Load 15.03 non-anomaly rows. Reset index so positional indexing is clean."""
    df = a_data_io.load_dataset()
    df = df[df["session_date"] == config.SESSION]
    df = df[~df["anomaly_flag"]]
    return df.reset_index(drop=True)


def compute_dataset_sha() -> str:
    return a_data_io.compute_dataset_sha()


def read_expected_sha() -> str:
    return a_data_io._read_expected_sha()
