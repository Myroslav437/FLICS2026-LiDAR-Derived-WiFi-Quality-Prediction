"""P0.6 — Cross-session same-cell consistency (15.03 vs 24.03)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, write_json, read_json


MIN_ROWS_PER_SESSION = 30


def main() -> None:
    print("=" * 72)
    print("P0.6  Cross-session same-cell consistency  (15.03 vs 24.03)")
    print("=" * 72)
    p00 = read_json(C.CACHE_DIR / "p0_0.json")
    same_pair_key = "15.03.2026__24.03.2026"
    gate0 = p00["pairs"][same_pair_key]["gate_0"]
    print(f"  P0.0 Gate 0 (same-map pair): {gate0}")
    if gate0 == "RED":
        print("  Gate 0 is RED — skipping P0.6")
        write_json(C.CACHE_DIR / "p0_6.json",
                   {"skipped": True, "reason": "Gate 0 RED"})
        return

    jp = load_joint(["session_date", "x_m", "y_m", "signal_power"])
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")
    jp["anomaly_flag"] = mask["anomaly_flag"].to_numpy()
    jp = jp[(~jp["anomaly_flag"]) & jp["signal_power"].notna()].reset_index(drop=True)

    # Restrict to same-map sessions
    a = jp[jp["session_date"] == "15.03.2026"].copy()
    b = jp[jp["session_date"] == "24.03.2026"].copy()
    for d in (a, b):
        d["cx"] = np.floor(d["x_m"].to_numpy() / C.CELL_SIZE_M).astype(np.int64)
        d["cy"] = np.floor(d["y_m"].to_numpy() / C.CELL_SIZE_M).astype(np.int64)
    means_a = a.groupby(["cx", "cy"])["signal_power"].agg(["mean", "size"])
    means_b = b.groupby(["cx", "cy"])["signal_power"].agg(["mean", "size"])
    common = means_a.join(means_b, lsuffix="_a", rsuffix="_b", how="inner")
    common = common[(common["size_a"] >= MIN_ROWS_PER_SESSION) &
                    (common["size_b"] >= MIN_ROWS_PER_SESSION)]
    common["delta_dB"] = common["mean_a"] - common["mean_b"]
    print(f"  qualifying cells: {len(common):,}")
    if len(common) == 0:
        print("  no qualifying cells — abort")
        write_json(C.CACHE_DIR / "p0_6.json",
                   {"skipped": True, "reason": "no qualifying cells"})
        return

    abs_delta = common["delta_dB"].abs()
    median_abs = float(abs_delta.median())
    q25 = float(abs_delta.quantile(0.25))
    q75 = float(abs_delta.quantile(0.75))
    median = float(common["delta_dB"].median())

    if median_abs < 4.0:
        gate_d = "GREEN"
    elif median_abs < 7.0:
        gate_d = "YELLOW"
    else:
        gate_d = "RED"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(common["delta_dB"], bins=40, color="C0", edgecolor="black")
    ax.axvline(0, color="grey", linewidth=1)
    ax.axvline(median, color="C3", linewidth=2,
               label=f"median Δ = {median:+.2f} dB")
    ax.set_xlabel("Δ = mean_15.03 - mean_24.03  [dB]")
    ax.set_ylabel("# cells")
    ax.set_title(f"P0.6 same-cell signal_power Δ  (n_cells={len(common)})\n"
                 f"median |Δ| = {median_abs:.2f} dB,  IQR = [{q25:.2f}, {q75:.2f}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "p0_6_delta_histogram.png", dpi=120)
    plt.close(fig)

    summary = {
        "n_cells": int(len(common)),
        "median_delta_dB": median,
        "median_abs_delta_dB": median_abs,
        "q25_abs_delta_dB": q25,
        "q75_abs_delta_dB": q75,
        "gate_D": gate_d,
        "min_rows_per_session": MIN_ROWS_PER_SESSION,
    }
    write_json(C.CACHE_DIR / "p0_6.json", summary)
    print(f"  median Δ = {median:+.2f} dB  median |Δ| = {median_abs:.2f} dB  "
          f"Gate D: {gate_d}")
    print("P0.6 done.")


if __name__ == "__main__":
    main()
