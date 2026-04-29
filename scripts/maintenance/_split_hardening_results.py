"""Split hardening_metrics.parquet and fit_inventory.parquet by experiment.

A,B rows -> scripts/p1_project_a/results/hardening_*.parquet
C,E rows -> scripts/p1_project_b/results/hardening_*.parquet
D has no entries in fit_inventory; D rows in hardening_metrics (none expected) go to dataset analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1].parent

SRC_RESULTS = ROOT / "scripts" / "p1_hardening" / "results"
A_RESULTS = ROOT / "scripts" / "p1_project_a" / "results"
B_RESULTS = ROOT / "scripts" / "p1_project_b" / "results"


def split(src_name: str, a_name: str, b_name: str) -> None:
    src = SRC_RESULTS / src_name
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    if "experiment" not in df.columns:
        raise KeyError(f"missing 'experiment' column in {src}; got {list(df.columns)}")
    a_df = df[df["experiment"].isin(["A", "B"])].reset_index(drop=True)
    b_df = df[df["experiment"].isin(["C", "E"])].reset_index(drop=True)
    other_df = df[~df["experiment"].isin(["A", "B", "C", "E"])].reset_index(drop=True)
    if len(other_df) != 0:
        raise ValueError(
            f"unexpected rows in {src} with experiment={set(other_df['experiment'])}; "
            f"only A/B/C/E expected"
        )
    a_dst = A_RESULTS / a_name
    b_dst = B_RESULTS / b_name
    a_df.to_parquet(a_dst, index=False)
    b_df.to_parquet(b_dst, index=False)
    print(f"{src.name}: total={len(df)} -> A/B={len(a_df)} ({a_dst}), C/E={len(b_df)} ({b_dst})")
    src.unlink()
    print(f"  removed {src}")


def main() -> None:
    split("hardening_metrics.parquet", "hardening_metrics.parquet", "hardening_metrics.parquet")
    split("fit_inventory.parquet",     "hardening_fit_inventory.parquet", "hardening_fit_inventory.parquet")


if __name__ == "__main__":
    main()
