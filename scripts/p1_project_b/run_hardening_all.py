"""Hardening orchestrator for Project B: runs Hardening C and E end-to-end.

Consolidates C,E per-experiment metrics into
`scripts/p1_project_b/results/hardening_metrics.parquet` and per-fit inventory
into `scripts/p1_project_b/results/hardening_fit_inventory.parquet`.

Originally split off from `scripts/p1_hardening/run_all.py` on 2026-04-28
when the hardening package was dissolved.

Usage:
    python -m scripts.p1_project_b.run_hardening_all
"""

from __future__ import annotations

import dataclasses
import time

import pandas as pd

from . import config as b_config
from .run_hardening_combined import main as run_c
from .run_hardening_placebo import main as run_e


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("Hardening experiments (Project B) — orchestrator")
    print("=" * 72)

    print("\n--- Hardening C ---")
    c_df, c_fits = run_c()

    print("\n--- Hardening E ---")
    e_df, e_fits = run_e()

    print("\n--- Consolidation ---")
    metrics_df = pd.concat([c_df, e_df], axis=0, ignore_index=True)
    metrics_df.to_parquet(b_config.RESULTS_DIR / "hardening_metrics.parquet", index=False)
    print(f"  hardening_metrics.parquet rows = {len(metrics_df)} (C,E)")

    fits_df = pd.DataFrame([dataclasses.asdict(f) for f in (c_fits + e_fits)])
    fits_df.to_parquet(b_config.RESULTS_DIR / "hardening_fit_inventory.parquet", index=False)
    print(f"  hardening_fit_inventory.parquet rows = {len(fits_df)} (expected: 6+3 = 9 new fits)")

    print(f"\n[run_hardening_all] complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
