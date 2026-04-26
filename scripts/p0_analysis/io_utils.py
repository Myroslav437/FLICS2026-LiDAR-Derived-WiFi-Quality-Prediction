"""I/O helpers: load joint parquet, batch-read LiDAR scans by row index."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from . import config as C


def load_joint(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load (a subset of columns from) the joint parquet, preserving row order."""
    df = pd.read_parquet(C.JOINT_PARQUET, columns=list(columns) if columns else None)
    return df.reset_index(drop=True)


def read_lidar_rows(rows: np.ndarray, h5_path: Path = C.LIDAR_H5) -> np.ndarray:
    """Read distances at the given row indices from lidar.h5.

    Reads in sorted-chunk batches to avoid the catastrophic random-access cost
    of fancy indexing. Returns shape (len(rows), 2700) uint16 in original order.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n == 0:
        return np.empty((0, C.N_BEAMS), dtype=np.uint16)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    out = np.empty((n, C.N_BEAMS), dtype=np.uint16)
    with h5py.File(h5_path, "r") as f:
        ds = f["distances"]
        # Walk sorted rows and pull contiguous-ish ranges in chunks.
        i = 0
        while i < n:
            j = min(i + C.H5_CHUNK, n)
            block = sorted_rows[i:j]
            lo, hi = int(block[0]), int(block[-1]) + 1
            # Heuristic: if the span is dense enough, slice the range and
            # gather; otherwise read each row.
            span = hi - lo
            if span <= 4 * (j - i):
                buf = ds[lo:hi]
                out[order[i:j]] = buf[block - lo]
            else:
                # Sparse — read individually but still in sorted order
                for k, r in zip(order[i:j], block):
                    out[k] = ds[int(r)]
            i = j
    return out


def iter_lidar_chunks(h5_path: Path = C.LIDAR_H5, chunk_size: int = C.H5_CHUNK):
    """Yield (start_row, distances_chunk) over the entire h5."""
    with h5py.File(h5_path, "r") as f:
        ds = f["distances"]
        n = ds.shape[0]
        for s in range(0, n, chunk_size):
            yield s, ds[s:s+chunk_size]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path):
    return json.loads(path.read_text())


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
