"""P0.1 — Spatial overlap of trajectories."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, write_json


def _cell_set(df: pd.DataFrame) -> set:
    cx = np.floor(df["x_m"].to_numpy() / C.CELL_SIZE_M).astype(np.int64)
    cy = np.floor(df["y_m"].to_numpy() / C.CELL_SIZE_M).astype(np.int64)
    return set(zip(cx.tolist(), cy.tolist()))


def main() -> None:
    print("=" * 72)
    print("P0.1  Spatial overlap of trajectories")
    print("=" * 72)
    jp = load_joint(["session_date", "x_m", "y_m"])
    cells = {sd: _cell_set(jp[jp["session_date"] == sd]) for sd in C.SESSIONS}
    counts = {sd: len(cells[sd]) for sd in C.SESSIONS}
    pairs = [("15.03.2026", "24.03.2026"),
             ("15.03.2026", "25.02.2026"),
             ("24.03.2026", "25.02.2026")]
    pair_stats = {}
    for a, b in pairs:
        inter = cells[a] & cells[b]
        union = cells[a] | cells[b]
        pair_stats[f"{a}__{b}"] = {
            "n_cells_a": counts[a],
            "n_cells_b": counts[b],
            "n_intersection": len(inter),
            "n_a_only": counts[a] - len(inter),
            "n_b_only": counts[b] - len(inter),
            "n_union": len(union),
            "iou": (len(inter) / len(union)) if len(union) else 0.0,
            "pct_a_in_b": (len(inter) / counts[a]) if counts[a] else 0.0,
            "pct_b_in_a": (len(inter) / counts[b]) if counts[b] else 0.0,
        }
        print(f"  {a} vs {b}: |A|={counts[a]:,} |B|={counts[b]:,} "
              f"|A∩B|={len(inter):,} IoU={pair_stats[f'{a}__{b}']['iou']:.3f}")

    # Figure: trajectory overlay (2 panels: same-map A, B alone)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for sd, color in zip(("15.03.2026", "24.03.2026"), ("C0", "C3")):
        sub = jp[jp["session_date"] == sd]
        ax.scatter(sub["x_m"], sub["y_m"], s=1, c=color, alpha=0.4, label=sd)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Map A (15.03 + 24.03 frame)")
    ax.legend(markerscale=8)
    ax = axes[1]
    sub = jp[jp["session_date"] == "25.02.2026"]
    ax.scatter(sub["x_m"], sub["y_m"], s=1, c="C2", alpha=0.4, label="25.02.2026")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Map B (25.02, separate frame)")
    ax.legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "p0_1_overlay.png", dpi=120)
    plt.close(fig)

    # Heatmap of same-map overlap density
    cax_a = np.floor(jp[jp["session_date"] == "15.03.2026"]["x_m"]
                     / C.CELL_SIZE_M).astype(np.int64)
    cay_a = np.floor(jp[jp["session_date"] == "15.03.2026"]["y_m"]
                     / C.CELL_SIZE_M).astype(np.int64)
    cax_b = np.floor(jp[jp["session_date"] == "24.03.2026"]["x_m"]
                     / C.CELL_SIZE_M).astype(np.int64)
    cay_b = np.floor(jp[jp["session_date"] == "24.03.2026"]["y_m"]
                     / C.CELL_SIZE_M).astype(np.int64)
    lo_x = min(cax_a.min(), cax_b.min()); hi_x = max(cax_a.max(), cax_b.max())
    lo_y = min(cay_a.min(), cay_b.min()); hi_y = max(cay_a.max(), cay_b.max())
    grid = np.zeros((hi_y - lo_y + 1, hi_x - lo_x + 1), dtype=np.int8)
    for x, y in zip(cax_a, cay_a):
        grid[y - lo_y, x - lo_x] |= 1
    for x, y in zip(cax_b, cay_b):
        grid[y - lo_y, x - lo_x] |= 2
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.colors.ListedColormap(
        ["white", "#7fb3ff", "#ff7f7f", "#7f7f7f"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin="lower",
              extent=[lo_x * C.CELL_SIZE_M, (hi_x + 1) * C.CELL_SIZE_M,
                      lo_y * C.CELL_SIZE_M, (hi_y + 1) * C.CELL_SIZE_M])
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Map A coverage:  blue=15.03 only  red=24.03 only  grey=both")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "p0_1_overlap_heatmap.png", dpi=120)
    plt.close(fig)

    write_json(C.CACHE_DIR / "p0_1.json", {
        "cell_size_m": C.CELL_SIZE_M,
        "per_session_cell_counts": counts,
        "pair_stats": pair_stats,
    })
    print("P0.1 done.")


if __name__ == "__main__":
    main()
