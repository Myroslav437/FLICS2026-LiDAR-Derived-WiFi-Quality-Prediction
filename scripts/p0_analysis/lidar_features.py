"""Compute and cache the headline LiDAR scalar features for every joint row.

Used by P0.4 (correlation) and feature_extractor.py (Phase 1).
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import read_json, load_joint


CACHE_PATH = C.CACHE_DIR / "lidar_features.parquet"

# Field names exposed
FIELDS = ["mean_dist_mm", "dist_p90_mm", "clutter_frac",
          "openness_frac", "mean_front_mm"]

CHUNK_ROWS = 20000
CLUTTER_MM = 4000.0
OPEN_MM = 7000.0


def _front_beam_mask(theta_min_deg: float, theta_max_deg: float) -> np.ndarray:
    """Beams within the empirical FOV restricted to |theta| < 30 deg.
    Padding beams (>= N_ACTIVE_BEAMS) are excluded — only active beams
    can be in the front cone."""
    out = np.zeros(C.N_BEAMS, dtype=bool)
    angles = C.ANGLE_MIN_DEG + np.arange(C.N_ACTIVE_BEAMS) * C.ANGLE_STEP_DEG
    in_fov = (angles >= theta_min_deg) & (angles <= theta_max_deg)
    in_front = (angles >= -30.0) & (angles <= 30.0)
    out[:C.N_ACTIVE_BEAMS] = in_fov & in_front
    return out


def _compute_features_chunk(d: np.ndarray, mask_invalid: np.ndarray,
                            front_mask: np.ndarray) -> dict:
    """For a (n_rows, 2700) uint16 chunk, return per-row scalar features."""
    s = d.astype(np.float32)
    invalid_per_beam = (s == 0) | (s == C.INVALID_FAR) | mask_invalid[None, :]
    s_valid = np.where(invalid_per_beam, np.nan, s)

    mean_d = np.nanmean(s_valid, axis=1)
    p90 = np.nanpercentile(s_valid, 90, axis=1)
    is_close = np.where(invalid_per_beam, False, s < CLUTTER_MM)
    is_open = np.where(invalid_per_beam, False, s > OPEN_MM)
    valid_count = (~invalid_per_beam).sum(axis=1).astype(np.float32)
    valid_count = np.maximum(valid_count, 1.0)
    clutter_frac = is_close.sum(axis=1) / valid_count
    openness_frac = is_open.sum(axis=1) / valid_count
    # Front cone
    front_invalid = invalid_per_beam | ~front_mask[None, :]
    front_valid = np.where(front_invalid, np.nan, s)
    mean_front = np.nanmean(front_valid, axis=1)

    return dict(
        mean_dist_mm=mean_d, dist_p90_mm=p90,
        clutter_frac=clutter_frac, openness_frac=openness_frac,
        mean_front_mm=mean_front,
    )


def compute_and_cache(force: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force:
        return pd.read_parquet(CACHE_PATH)

    fov = read_json(C.ARTIFACTS_DIR / "lidar_fov.json")
    mask_invalid = np.load(C.ARTIFACTS_DIR / "agv_body_mask.npz")["mask_invalid"]
    front_mask = _front_beam_mask(fov["theta_min_deg"], fov["theta_max_deg"])

    jp = load_joint(["session_date", "lidar_row"])
    n = len(jp)
    out = {f: np.full(n, np.nan, dtype=np.float32) for f in FIELDS}

    # We iterate joint rows in order, gathering lidar_rows in chunks.
    with h5py.File(C.LIDAR_H5, "r") as h5:
        ds = h5["distances"]
        for s in range(0, n, CHUNK_ROWS):
            e = min(s + CHUNK_ROWS, n)
            rows = jp["lidar_row"].iloc[s:e].to_numpy()
            order = np.argsort(rows)
            sorted_rows = rows[order]
            lo, hi = int(sorted_rows[0]), int(sorted_rows[-1]) + 1
            span = hi - lo
            if span <= 8 * (e - s):
                buf = ds[lo:hi]
                d = buf[sorted_rows - lo]
            else:
                d = np.empty((e - s, C.N_BEAMS), dtype=np.uint16)
                for k, r in enumerate(sorted_rows):
                    d[k] = ds[int(r)]
            feats = _compute_features_chunk(d, mask_invalid, front_mask)
            inv_order = np.argsort(order)  # back to original order
            for f in FIELDS:
                out[f][s:e] = feats[f][inv_order]
            if s % (CHUNK_ROWS * 10) == 0:
                print(f"    lidar features: {s}/{n}")

    df = pd.DataFrame(out)
    df["session_date"] = jp["session_date"].to_numpy()
    df.to_parquet(CACHE_PATH, index=False)
    print(f"  -> wrote {CACHE_PATH}")
    return df


if __name__ == "__main__":
    compute_and_cache(force=True)
