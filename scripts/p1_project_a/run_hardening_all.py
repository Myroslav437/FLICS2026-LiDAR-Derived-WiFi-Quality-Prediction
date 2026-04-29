"""Hardening orchestrator for Project A: runs Hardening A and B end-to-end.

Consolidates A,B per-experiment metrics into
`scripts/p1_project_a/results/hardening_metrics.parquet` and per-fit inventory
into `scripts/p1_project_a/results/hardening_fit_inventory.parquet`.

Originally split off from `scripts/p1_hardening/run_all.py` on 2026-04-28
when the hardening package was dissolved.

Usage:
    python -m scripts.p1_project_a.run_hardening_all
"""

from __future__ import annotations

import dataclasses
import time

import pandas as pd

from . import config as a_config
from .run_hardening_lightgbm import main as run_b
from .run_hardening_placebo import FitRow, main as run_a


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("Hardening experiments (Project A) — orchestrator")
    print("=" * 72)

    print("\n--- Hardening A ---")
    a_df, a_fits = run_a()

    print("\n--- Hardening B ---")
    b_df, b_fits = run_b()

    print("\n--- Consolidation ---")
    metrics_df = pd.concat([a_df, b_df], axis=0, ignore_index=True)
    metrics_df.to_parquet(a_config.RESULTS_DIR / "hardening_metrics.parquet", index=False)
    print(f"  hardening_metrics.parquet rows = {len(metrics_df)} (A,B)")

    fits_df = pd.DataFrame([dataclasses.asdict(f) for f in (a_fits + b_fits)])
    fits_df.to_parquet(a_config.RESULTS_DIR / "hardening_fit_inventory.parquet", index=False)
    print(f"  hardening_fit_inventory.parquet rows = {len(fits_df)} (expected: 2+4 = 6 new fits)")

    print(f"\n[run_hardening_all] complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
