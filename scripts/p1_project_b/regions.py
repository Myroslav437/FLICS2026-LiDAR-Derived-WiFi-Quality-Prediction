"""Spatial region definition for the 15.03 within-session split.

Primary: K-means on 50k-row subsample of (x_m, y_m), labels propagated to full
non-anomaly rows by nearest-cluster-centroid lookup.

Fallback: if any region has fewer than REGION_MIN_ROWS rows, partition the
trajectory into N_REGIONS contiguous chunks along the principal trajectory axis
(PCA of (x_m, y_m)).

Buffer-zone helper builds a "within BUFFER_DISTANCE_M of any region-r row" mask
on a candidate (x, y) array using a KDTree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config


def _kmeans_lloyd(
    points: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iter: int = 200,
    n_init: int = 10,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla Lloyd's K-means with k-means++ init and `n_init` restarts.

    Returns (labels_0_indexed, centroids_K_x_D). Picks the run with lowest inertia.
    """
    n, _d = points.shape
    best_inertia = np.inf
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None

    for run_idx in range(n_init):
        # k-means++ initialisation
        run_rng = np.random.default_rng(seed + run_idx)
        idx0 = int(run_rng.integers(0, n))
        centers = [points[idx0]]
        for _ in range(1, k):
            diffs = points[:, None, :] - np.array(centers)[None, :, :]
            min_sq = np.min(np.sum(diffs * diffs, axis=2), axis=1)
            total = float(min_sq.sum())
            if total <= 0:
                next_idx = int(run_rng.integers(0, n))
            else:
                probs = min_sq / total
                next_idx = int(run_rng.choice(n, p=probs))
            centers.append(points[next_idx])
        centroids = np.array(centers, dtype=np.float64)

        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            diffs = points[:, None, :] - centroids[None, :, :]
            sq = np.sum(diffs * diffs, axis=2)
            new_labels = np.argmin(sq, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centroids = np.empty_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = points[mask].mean(axis=0)
                else:
                    new_centroids[j] = points[run_rng.integers(0, n)]
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            if shift < tol:
                break

        diffs = points - centroids[labels]
        inertia = float(np.sum(diffs * diffs))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids


def _pca_first_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis_unit_vector, origin) for the principal component of `points`."""
    origin = points.mean(axis=0)
    centred = points - origin
    # SVD; first right-singular vector is the principal axis.
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    axis = vh[0]
    axis = axis / np.linalg.norm(axis)
    return axis, origin


@dataclass
class RegionAssignment:
    """The full-row region labels and the metadata that produced them."""

    df: pd.DataFrame  # columns: joint_idx, region_id  (full 15.03 non-anomaly rows)
    method: str       # "kmeans" or "pca_trajectory"
    centroids: np.ndarray | None  # K x 2, None for PCA fallback
    pca_axis: np.ndarray | None   # 2-vector, None for K-means
    pca_origin: np.ndarray | None  # 2-vector, None for K-means
    region_sizes: dict[int, int]


def _per_region_sizes(labels: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in range(1, config.N_REGIONS + 1):
        out[r] = int((labels == r).sum())
    return out


def _kmeans_partition(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run K-means on a subsample, then assign every row by nearest centroid.

    Returns: (labels_1_to_K_for_all_rows, centroids_KxD).
    """
    rng = np.random.default_rng(config.SEED)
    n = coords.shape[0]
    sample_n = min(config.REGION_SUBSAMPLE_N, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    idx.sort()
    _labels_sub, centroids = _kmeans_lloyd(
        coords[idx], k=config.N_REGIONS, seed=config.SEED, n_init=10
    )
    # Nearest-centroid assignment for the full 232k rows.
    diffs = coords[:, None, :] - centroids[None, :, :]
    sq = np.sum(diffs * diffs, axis=2)
    raw = np.argmin(sq, axis=1)  # 0-indexed
    # Re-label so region indices are stable across runs:
    # sort regions by centroid x-coordinate ascending (then by y as a tie-break).
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    inv = np.empty_like(order)
    inv[order] = np.arange(config.N_REGIONS)
    labels = inv[raw] + 1  # 1-indexed regions
    centroids_sorted = centroids[order]
    return labels, centroids_sorted


def _pca_trajectory_partition(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project onto principal trajectory axis and split into K equal-count chunks.

    Returns: (labels_1_to_K, axis_unit_vector, origin).
    """
    axis, origin = _pca_first_axis(coords)
    proj = (coords - origin) @ axis  # 1-D projections
    order = np.argsort(proj, kind="mergesort")
    labels = np.empty(len(coords), dtype=np.int64)
    splits = np.array_split(order, config.N_REGIONS)
    for i, idxs in enumerate(splits, start=1):
        labels[idxs] = i
    return labels, axis, origin


def build_regions(non_anom: pd.DataFrame) -> RegionAssignment:
    coords = non_anom[["x_m", "y_m"]].to_numpy(dtype=np.float64)

    # Try K-means first.
    labels, centroids = _kmeans_partition(coords)
    sizes = _per_region_sizes(labels)
    min_size = min(sizes.values())

    method = "kmeans"
    pca_axis: np.ndarray | None = None
    pca_origin: np.ndarray | None = None

    if min_size < config.REGION_MIN_ROWS:
        # Fallback: PCA-trajectory partition.
        labels, pca_axis, pca_origin = _pca_trajectory_partition(coords)
        sizes = _per_region_sizes(labels)
        method = "pca_trajectory"
        centroids_used: np.ndarray | None = None
    else:
        centroids_used = centroids

    df = pd.DataFrame({
        "joint_idx": non_anom["joint_idx"].to_numpy(),
        "region_id": labels.astype(np.int8),
    })
    return RegionAssignment(
        df=df,
        method=method,
        centroids=centroids_used,
        pca_axis=pca_axis,
        pca_origin=pca_origin,
        region_sizes=sizes,
    )


def save_regions(assn: RegionAssignment) -> None:
    out_path = config.ARTIFACTS_DIR / "spatial_regions.parquet"
    assn.df.to_parquet(out_path, index=False)


def load_regions() -> pd.DataFrame:
    return pd.read_parquet(config.ARTIFACTS_DIR / "spatial_regions.parquet")


def attach_region_id(non_anom: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Return non_anom with a region_id column, joined on joint_idx."""
    return non_anom.merge(region_df, on="joint_idx", how="left", validate="1:1")


def build_buffer_mask(
    train_coords: np.ndarray,
    holdout_coords: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Mark training rows within `radius_m` of any held-out region row.

    Returns a boolean array over `train_coords` rows: True = within the buffer
    (i.e., should be DROPPED from training).
    """
    if len(holdout_coords) == 0 or len(train_coords) == 0:
        return np.zeros(len(train_coords), dtype=bool)
    tree = cKDTree(holdout_coords)
    # query_ball_point would return a list; we just need "is there any neighbour within r?"
    # use query with k=1 and compare distances.
    distances, _ = tree.query(train_coords, k=1, distance_upper_bound=radius_m)
    return np.isfinite(distances) & (distances <= radius_m)
