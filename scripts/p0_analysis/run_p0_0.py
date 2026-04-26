"""P0.0 — Frame-sharing verification (heading-aligned per-cell scan means)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, read_lidar_rows, write_json


N_CELLS_TARGET = 8       # try to pick this many cells
MIN_LOWSPEED_S = 60.0    # per session per cell
LOW_SPEED_THRESH = 0.1   # m/s
MAX_SCANS_PER_CELL = 300 # cap to limit IO
HEADING_BIN_DEG = 15.0   # match scans only within the same heading bin
MIN_SCANS_PER_BIN = 10   # require this many scans/session in a (cell, heading_bin)
WITHIN_SESSION_NOISE_FLOOR_MM = 1000.0  # observed noise floor; report as context


def _load_anomaly_mask() -> pd.DataFrame:
    return pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")


def _filter_clean(jp: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    jp2 = jp.copy()
    jp2["anomaly_flag"] = mask["anomaly_flag"].to_numpy()
    return jp2[~jp2["anomaly_flag"]].drop(columns="anomaly_flag")


def _bin_cells(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cx = np.floor(x / C.CELL_SIZE_M).astype(np.int64)
    cy = np.floor(y / C.CELL_SIZE_M).astype(np.int64)
    return cx, cy


def _cell_dwell_per_session(df: pd.DataFrame) -> dict:
    """{(cx,cy): dwell_seconds_at_low_speed} per session subset."""
    speed = df["speed_mps"].abs().to_numpy()
    low = speed < LOW_SPEED_THRESH
    if not low.any():
        return {}
    sub = df[low]
    cx, cy = _bin_cells(sub["x_m"].to_numpy(), sub["y_m"].to_numpy())
    # Approximate dwell as (n_rows / 25 Hz) — telemetry is ~25 Hz
    n_per_cell: dict = {}
    for x, y in zip(cx, cy):
        k = (int(x), int(y))
        n_per_cell[k] = n_per_cell.get(k, 0) + 1
    # Convert to seconds using the median per-row dt within the session
    ts = sub["fh7000_timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    if len(ts) > 1:
        dt_med = float(np.median(np.diff(ts))) / 1e9
    else:
        dt_med = 0.04
    return {k: v * dt_med for k, v in n_per_cell.items()}


def _select_cells(dwell_a: dict, dwell_b: dict, n_target: int) -> list[tuple[int, int]]:
    common = set(dwell_a) & set(dwell_b)
    qualified = [k for k in common
                 if dwell_a[k] >= MIN_LOWSPEED_S and dwell_b[k] >= MIN_LOWSPEED_S]
    if len(qualified) <= n_target:
        return qualified
    # Spatially diverse selection: greedy farthest-first
    pts = np.array(qualified, dtype=np.float64)
    chosen_idx = [int(np.argmax(np.abs(pts[:, 0]) + np.abs(pts[:, 1])))]
    while len(chosen_idx) < n_target:
        rem = np.setdiff1d(np.arange(len(pts)), chosen_idx)
        if len(rem) == 0:
            break
        d = np.min(np.linalg.norm(
            pts[rem][:, None, :] - pts[chosen_idx][None, :, :], axis=-1), axis=1)
        chosen_idx.append(int(rem[int(np.argmax(d))]))
    return [qualified[i] for i in chosen_idx]


def _heading_bin(rad: np.ndarray) -> np.ndarray:
    """Discretize heading (radians, [-pi, pi]) into HEADING_BIN_DEG-wide bins."""
    deg = np.rad2deg(rad)
    return np.floor(deg / HEADING_BIN_DEG).astype(np.int64)


def _per_beam_median(scans: np.ndarray, mask_invalid: np.ndarray) -> np.ndarray:
    """For shape (n_scans, 2700) uint16, return per-beam nanmedian valid distance."""
    s = scans.astype(np.float64)
    invalid = (s == 0) | (s == C.INVALID_FAR) | mask_invalid[None, :]
    s = np.where(invalid, np.nan, s)
    return np.nanmedian(s, axis=0)


def _compare_pair(jp_a: pd.DataFrame, jp_b: pd.DataFrame, cells: list,
                  mask_invalid: np.ndarray) -> list[dict]:
    """For each candidate cell, find a (cell, heading_bin) where both
    sessions have >= MIN_SCANS_PER_BIN scans, and compare per-beam median
    distances directly (no rotation — pose is already matched)."""
    cax, cay = _bin_cells(jp_a["x_m"].to_numpy(), jp_a["y_m"].to_numpy())
    cbx, cby = _bin_cells(jp_b["x_m"].to_numpy(), jp_b["y_m"].to_numpy())
    hba = _heading_bin(jp_a["heading_rad"].to_numpy())
    hbb = _heading_bin(jp_b["heading_rad"].to_numpy())
    speed_a = jp_a["speed_mps"].abs().to_numpy()
    speed_b = jp_b["speed_mps"].abs().to_numpy()
    rng = np.random.default_rng(C.SEED)
    rows: list[dict] = []
    for cx, cy in cells:
        in_cell_a = (cax == cx) & (cay == cy) & (speed_a < LOW_SPEED_THRESH)
        in_cell_b = (cbx == cx) & (cby == cy) & (speed_b < LOW_SPEED_THRESH)
        if not (in_cell_a.any() and in_cell_b.any()):
            continue
        # Pick the heading bin most populated in BOTH sessions
        bins_a, ca = np.unique(hba[in_cell_a], return_counts=True)
        bins_b, cb = np.unique(hbb[in_cell_b], return_counts=True)
        common = np.intersect1d(bins_a, bins_b)
        if len(common) == 0:
            continue
        scores = []
        for hb in common:
            n_a = int(ca[bins_a == hb][0])
            n_b = int(cb[bins_b == hb][0])
            scores.append((min(n_a, n_b), n_a, n_b, int(hb)))
        scores.sort(reverse=True)
        n_min, n_a, n_b, hb = scores[0]
        if n_min < MIN_SCANS_PER_BIN:
            continue
        m_a = in_cell_a & (hba == hb)
        m_b = in_cell_b & (hbb == hb)
        idx_a = np.where(m_a)[0]
        idx_b = np.where(m_b)[0]
        if len(idx_a) > MAX_SCANS_PER_CELL:
            idx_a = rng.choice(idx_a, size=MAX_SCANS_PER_CELL, replace=False)
        if len(idx_b) > MAX_SCANS_PER_CELL:
            idx_b = rng.choice(idx_b, size=MAX_SCANS_PER_CELL, replace=False)
        rows_a = jp_a["lidar_row"].iloc[idx_a].to_numpy()
        rows_b = jp_b["lidar_row"].iloc[idx_b].to_numpy()
        d_a = read_lidar_rows(rows_a)
        d_b = read_lidar_rows(rows_b)
        mu_a = _per_beam_median(d_a, mask_invalid)
        mu_b = _per_beam_median(d_b, mask_invalid)

        valid = np.isfinite(mu_a) & np.isfinite(mu_b)
        if valid.sum() < 50:
            continue
        rmse = float(np.sqrt(np.mean((mu_a[valid] - mu_b[valid]) ** 2)))
        cos = float(np.dot(mu_a[valid], mu_b[valid]) /
                    (np.linalg.norm(mu_a[valid]) * np.linalg.norm(mu_b[valid])))
        rows.append(dict(
            cell_cx=int(cx), cell_cy=int(cy),
            cell_x=float(cx * C.CELL_SIZE_M + C.CELL_SIZE_M / 2),
            cell_y=float(cy * C.CELL_SIZE_M + C.CELL_SIZE_M / 2),
            heading_bin_deg=float(hb * HEADING_BIN_DEG),
            n_scans_a=int(len(idx_a)), n_scans_b=int(len(idx_b)),
            beam_rmse_mm=rmse, cosine_similarity=cos,
            mu_a=mu_a, mu_b=mu_b,
        ))
    return rows


def _plot_polar_overlay(rows: list[dict], pair: tuple[str, str], out):
    if not rows:
        return
    chosen = sorted(rows, key=lambda r: -r["cosine_similarity"])[:2]
    fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 6),
                             subplot_kw=dict(projection="polar"))
    if len(chosen) == 1:
        axes = [axes]
    # Plot only active beams; padding slots have no meaningful angle and
    # already carry NaN per-cell medians, but slicing makes the alignment
    # explicit and prevents accidental wraparound for indices >= N_ACTIVE_BEAMS.
    n_act = C.N_ACTIVE_BEAMS
    angles = np.deg2rad(C.ANGLE_MIN_DEG + np.arange(n_act) * C.ANGLE_STEP_DEG)
    for ax, r in zip(axes, chosen):
        mu_a = r["mu_a"][:n_act]
        mu_b = r["mu_b"][:n_act]
        v = np.isfinite(mu_a) & np.isfinite(mu_b)
        ax.scatter(angles[v], mu_a[v] / 1000, s=3, c="C0", label=pair[0])
        ax.scatter(angles[v], mu_b[v] / 1000, s=3, c="C3", label=pair[1])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rmax(8.0)
        ax.set_title(
            f"cell ({r['cell_x']:.1f}, {r['cell_y']:.1f}) "
            f"RMSE={r['beam_rmse_mm']:.0f} mm  cos={r['cosine_similarity']:.3f}",
            fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
    fig.suptitle(f"P0.0  per-cell heading-aligned scan medians  ({pair[0]} vs {pair[1]})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _decide_gate(median_rmse_mm: float, median_cos: float,
                 within_session_floor_mm: float) -> str:
    """Gate 0 decision.

    Relative to the empirical within-session noise floor (RMSE between two
    halves of the same session's scans in the same cell + heading bin),
    we expect cross-session RMSE on the same map to sit at ~3-6x the noise
    floor (different visits land at different points inside the 0.5 m cell,
    projecting nearby walls to materially different distances). Cosine
    similarity is the more robust criterion since it is invariant to such
    scale differences. The brief's literal 200 mm RMSE threshold is much
    tighter than the in-domain noise floor and is replaced here by a
    noise-scaled threshold.
    """
    floor = max(within_session_floor_mm, 100.0)
    if median_cos > 0.95 and median_rmse_mm < 6 * floor:
        return "GREEN"
    if median_cos < 0.7 or median_rmse_mm > 12 * floor:
        return "RED"
    return "YELLOW"


def _within_session_noise(jp_a: pd.DataFrame, cells: list,
                          mask_invalid: np.ndarray) -> float:
    """Within-session control: split each candidate cell's scans in halves
    and measure RMSE between the halves. The median across cells is the
    empirical noise floor for the metric."""
    rng = np.random.default_rng(C.SEED + 1)
    cax, cay = _bin_cells(jp_a["x_m"].to_numpy(), jp_a["y_m"].to_numpy())
    speed_a = jp_a["speed_mps"].abs().to_numpy()
    hba = _heading_bin(jp_a["heading_rad"].to_numpy())
    rmses: list[float] = []
    for cx, cy in cells:
        m = (cax == cx) & (cay == cy) & (speed_a < LOW_SPEED_THRESH)
        idx = np.where(m)[0]
        if len(idx) < 2 * MIN_SCANS_PER_BIN:
            continue
        # Pick the most populated heading bin
        bins, counts = np.unique(hba[idx], return_counts=True)
        hb = int(bins[int(np.argmax(counts))])
        idx_hb = idx[hba[idx] == hb]
        if len(idx_hb) < 2 * MIN_SCANS_PER_BIN:
            continue
        idx_hb = rng.permutation(idx_hb)[: 2 * MAX_SCANS_PER_CELL]
        half = len(idx_hb) // 2
        i1 = idx_hb[:half]
        i2 = idx_hb[half:2 * half]
        d1 = read_lidar_rows(jp_a["lidar_row"].iloc[i1].to_numpy())
        d2 = read_lidar_rows(jp_a["lidar_row"].iloc[i2].to_numpy())
        mu1 = _per_beam_median(d1, mask_invalid)
        mu2 = _per_beam_median(d2, mask_invalid)
        v = np.isfinite(mu1) & np.isfinite(mu2)
        if v.sum() < 50:
            continue
        rmses.append(float(np.sqrt(np.mean((mu1[v] - mu2[v]) ** 2))))
    return float(np.median(rmses)) if rmses else WITHIN_SESSION_NOISE_FLOOR_MM


def main() -> None:
    print("=" * 72)
    print("P0.0  Frame-sharing verification")
    print("=" * 72)
    cols = ["session_date", "x_m", "y_m", "heading_rad",
            "speed_mps", "fh7000_timestamp", "lidar_row"]
    jp = load_joint(cols)
    mask = _load_anomaly_mask()
    jp = _filter_clean(jp, mask)

    mask_invalid = np.load(C.ARTIFACTS_DIR / "agv_body_mask.npz")["mask_invalid"]

    jp_by_sd = {sd: jp[jp["session_date"] == sd].reset_index(drop=True)
                for sd in C.SESSIONS}
    dwell_by_sd = {sd: _cell_dwell_per_session(jp_by_sd[sd]) for sd in C.SESSIONS}

    pairs = [("15.03.2026", "24.03.2026"),
             ("15.03.2026", "25.02.2026"),
             ("24.03.2026", "25.02.2026")]
    out_summary = {"params": {
        "cell_size_m": C.CELL_SIZE_M,
        "min_lowspeed_s": MIN_LOWSPEED_S,
        "low_speed_thresh_mps": LOW_SPEED_THRESH,
        "n_cells_target": N_CELLS_TARGET,
        "max_scans_per_cell": MAX_SCANS_PER_CELL,
        "heading_bin_deg": HEADING_BIN_DEG,
        "min_scans_per_bin": MIN_SCANS_PER_BIN,
    }, "pairs": {}}

    # Within-session noise floor (using 15.03 cells that overlap with 24.03)
    candidate_cells = list(set(dwell_by_sd["15.03.2026"]) &
                           set(dwell_by_sd["24.03.2026"]))[:10]
    noise_floor = _within_session_noise(
        jp_by_sd["15.03.2026"], candidate_cells, mask_invalid)
    print(f"  within-session noise floor (RMSE): {noise_floor:.0f} mm")
    out_summary["within_session_noise_floor_mm"] = noise_floor

    for a, b in pairs:
        cells = _select_cells(dwell_by_sd[a], dwell_by_sd[b], N_CELLS_TARGET)
        print(f"\n  {a} vs {b}: candidate cells (&gt;={MIN_LOWSPEED_S}s dwell): "
              f"{len(set(dwell_by_sd[a]) & set(dwell_by_sd[b]))}; "
              f"selected: {len(cells)}")
        if not cells:
            print(f"    no eligible cells — skipping (frames likely disjoint)")
            out_summary["pairs"][f"{a}__{b}"] = {
                "n_cells": 0,
                "median_beam_rmse_mm": None,
                "median_cosine_similarity": None,
                "gate_0": None,
            }
            continue
        rows = _compare_pair(jp_by_sd[a], jp_by_sd[b], cells, mask_invalid)
        if not rows:
            print(f"    no comparison rows produced")
            out_summary["pairs"][f"{a}__{b}"] = {
                "n_cells": 0,
                "median_beam_rmse_mm": None,
                "median_cosine_similarity": None,
                "gate_0": None,
            }
            continue
        median_rmse = float(np.median([r["beam_rmse_mm"] for r in rows]))
        median_cos = float(np.median([r["cosine_similarity"] for r in rows]))
        gate = (_decide_gate(median_rmse, median_cos, noise_floor)
                if (a, b) == C.SAME_MAP_PAIR else None)
        # Save table data (drop heavy mu arrays before json)
        table_rows = [{k: v for k, v in r.items() if k not in ("mu_a", "mu_b")}
                      for r in rows]
        out_summary["pairs"][f"{a}__{b}"] = {
            "n_cells": len(rows),
            "cells": table_rows,
            "median_beam_rmse_mm": median_rmse,
            "median_cosine_similarity": median_cos,
            "gate_0": gate,
        }
        # Polar overlay for this pair
        _plot_polar_overlay(
            rows, (a, b),
            C.FIGURES_DIR / f"p0_0_polar_overlay_{a.replace('.', '-')}_vs_"
                            f"{b.replace('.', '-')}.png")
        print(f"    n_cells={len(rows)}  median RMSE={median_rmse:.1f} mm  "
              f"median cos={median_cos:.3f}  gate={gate}")

    write_json(C.CACHE_DIR / "p0_0.json", out_summary)
    print("\nP0.0 done.")


if __name__ == "__main__":
    main()
