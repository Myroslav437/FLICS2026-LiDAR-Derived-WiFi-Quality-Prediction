"""P0.3 — AGV-body LiDAR mask characterisation.

Picks a stationary window, computes per-beam zero-fraction, median, std;
identifies the dominant valid sector and writes mask_invalid + lidar_fov.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, read_lidar_rows, write_json


N_TARGET_SCANS = 5000


def _select_motion_active_sample(jp: pd.DataFrame, rng: np.random.Generator
                                 ) -> tuple[str, np.ndarray]:
    """Sample motion-active scans (|speed| > 0.1 m/s) across all sessions.

    Walls and obstacles move through the scan as the AGV moves, but beams
    blocked by the AGV body itself stay pinned at a constant near-zero
    distance. This separates body-mask from environmental near-clutter
    much better than a stationary window does.
    """
    moving = jp["speed_mps"].abs() > 0.1
    candidates = np.where(moving.to_numpy())[0]
    if len(candidates) > N_TARGET_SCANS:
        chosen = rng.choice(candidates, size=N_TARGET_SCANS, replace=False)
        chosen.sort()
    else:
        chosen = candidates
    rows = jp["lidar_row"].iloc[chosen].to_numpy()
    sdname = "moving_sample_all_sessions"
    return sdname, rows


def _compute_per_beam_stats(d: np.ndarray) -> dict:
    """For each beam, compute frac_zero / frac_invalid_far, median valid, std valid.
    Only the first N_ACTIVE_BEAMS slots are populated by the sensor; the rest
    are buffer padding and are reported with frac_zero = 1, NaN median/std.
    """
    n_scans = d.shape[0]
    is_zero = (d == C.INVALID_NEAR)
    is_far = (d == C.INVALID_FAR)
    frac_zero = is_zero.sum(axis=0) / n_scans
    frac_far = is_far.sum(axis=0) / n_scans
    valid_mask = ~(is_zero | is_far)
    median = np.full(C.N_BEAMS, np.nan, dtype=np.float64)
    std = np.full(C.N_BEAMS, np.nan, dtype=np.float64)
    for i in range(C.N_ACTIVE_BEAMS):
        v = d[valid_mask[:, i], i]
        if v.size > 0:
            median[i] = float(np.median(v))
            std[i] = float(np.std(v, ddof=0))
    return dict(frac_zero=frac_zero, frac_far=frac_far,
                median=median, std=std)


def _largest_contiguous(valid: np.ndarray) -> tuple[int, int]:
    """Largest contiguous True run in 1-D bool array. Returns (lo, hi) inclusive."""
    if not valid.any():
        return (-1, -1)
    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    sizes = ends - starts + 1
    k = int(np.argmax(sizes))
    return int(starts[k]), int(ends[k])


def _plot_polar(stats: dict, mask_invalid: np.ndarray, mask_zero: np.ndarray,
                mask_body: np.ndarray, valid_lo: int, valid_hi: int, out):
    # Only plot active beams (padding has no angle assignment)
    n_act = C.N_ACTIVE_BEAMS
    angles = np.deg2rad(C.ANGLE_MIN_DEG + np.arange(n_act) * C.ANGLE_STEP_DEG)
    median = stats["median"][:n_act]
    mi = mask_invalid[:n_act]
    mb = mask_body[:n_act]
    mz = mask_zero[:n_act]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    valid_pts = ~mi & np.isfinite(median)
    ax.scatter(angles[valid_pts], median[valid_pts] / 1000.0,
               s=3, c="C0", label="valid beams (median return, m)")
    if mb.any():
        ax.scatter(angles[mb], np.full(mb.sum(), 0.2),
                   s=10, c="C3", label=f"mask_body (n={int(mb.sum())})")
    if mz.any():
        ax.scatter(angles[mz], np.full(mz.sum(), 0.4),
                   s=10, c="C1", label=f"mask_zero (n={int(mz.sum())})")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(8.0)
    ax.set_title(f"P0.3  per-beam median return  |  valid sector "
                 f"[{C.beam_angle_deg(valid_lo):.1f}°, "
                 f"{C.beam_angle_deg(valid_hi):.1f}°] "
                 f"(active beams = {n_act})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("P0.3  AGV-body LiDAR mask characterization")
    print("=" * 72)
    jp = load_joint(["session_date", "fh7000_timestamp", "speed_mps", "lidar_row"])
    rng = np.random.default_rng(C.SEED)
    sdname, rows = _select_motion_active_sample(jp, rng)
    print(f"  motion-active sample: {sdname}  n_scans={len(rows)}")

    d = read_lidar_rows(rows)
    stats = _compute_per_beam_stats(d)

    # Padding mask: the storage allocates 2700 slots but only the first
    # N_ACTIVE_BEAMS are populated by the sensor — the rest are always
    # zero. This is *not* an angular blockage and must be excluded from
    # the AGV-body analysis.
    mask_pad = np.zeros(C.N_BEAMS, dtype=bool)
    mask_pad[C.N_ACTIVE_BEAMS:] = True

    # mask_zero / mask_body apply only to active beams
    mask_zero = (stats["frac_zero"] > 0.95) & ~mask_pad
    valid_for_body = (stats["median"] < 200) & (stats["std"] < 30)
    valid_for_body = valid_for_body & np.isfinite(stats["median"])
    mask_body = valid_for_body & ~mask_pad
    mask_invalid = mask_zero | mask_body | mask_pad

    valid = ~mask_invalid
    valid_lo, valid_hi = _largest_contiguous(valid)
    theta_min = float(C.beam_angle_deg(valid_lo))
    theta_max = float(C.beam_angle_deg(valid_hi))
    # valid_fraction is reported relative to ACTIVE beams (the only meaningful denominator)
    valid_fraction = float(valid.sum() / C.N_ACTIVE_BEAMS)

    # Save artifacts
    np.savez(C.ARTIFACTS_DIR / "agv_body_mask.npz",
             mask_invalid=mask_invalid,
             mask_zero=mask_zero,
             mask_body=mask_body,
             mask_pad=mask_pad)
    write_json(C.ARTIFACTS_DIR / "lidar_fov.json", {
        "theta_min_deg": theta_min,
        "theta_max_deg": theta_max,
        "valid_fraction": valid_fraction,
        "valid_beam_lo": valid_lo,
        "valid_beam_hi": valid_hi,
        "n_valid_beams": int(valid.sum()),
        "n_active_beams": int(C.N_ACTIVE_BEAMS),
        "n_pad_beams": int(mask_pad.sum()),
        "angle_step_deg": float(C.ANGLE_STEP_DEG),
        "source_session": sdname,
        "n_scans_used": int(len(rows)),
    })
    _plot_polar(stats, mask_invalid, mask_zero, mask_body,
                valid_lo, valid_hi, C.FIGURES_DIR / "p0_3_polar.png")

    # Gate B
    width = theta_max - theta_min
    if valid_fraction < 0.30:
        gate = "RED"
    elif width < 120.0:
        gate = "YELLOW"
    else:
        gate = "GREEN"
    summary = {
        "source_session": sdname,
        "n_scans_used": int(len(rows)),
        "theta_min_deg": theta_min,
        "theta_max_deg": theta_max,
        "sector_width_deg": float(width),
        "n_zero_beams": int(mask_zero.sum()),
        "n_body_beams": int(mask_body.sum()),
        "n_invalid_beams": int(mask_invalid.sum()),
        "valid_fraction": valid_fraction,
        "gate_B": gate,
    }
    write_json(C.CACHE_DIR / "p0_3.json", summary)
    print(f"  valid sector: [{theta_min:.1f}°, {theta_max:.1f}°]  width={width:.1f}°")
    print(f"  invalid beams: {int(mask_invalid.sum())}/{C.N_BEAMS} "
          f"({100*(1-valid_fraction):.1f}% invalid)")
    print(f"  Gate B: {gate}")


if __name__ == "__main__":
    main()
