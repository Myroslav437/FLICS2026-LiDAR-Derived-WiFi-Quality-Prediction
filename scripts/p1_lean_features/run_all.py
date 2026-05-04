"""Top-level lean-features orchestrator.

  python -m scripts.p1_lean_features.run_all

Steps:
  1. Snapshot SHA-256 of every locked artifact (Project A, Project B, hardening
     metrics) so we can verify post-run that nothing was modified.
  2. Run the 64 lean fits via run_lean_rerun.main().
  3. Build the three-way comparison + report via run_comparison.main().
  4. Re-snapshot SHA-256 of the locked artifacts and write
     scripts/p1_lean_features/locked_artifacts_unchanged.md with the diff (if
     any).

The lean rerun module names its per-fold metrics parquets `{variant_set}_metrics.parquet`
where variant_set ∈ {lean_A, lean_B}; the comparison module reads
`lean_A_metrics.parquet` and `lean_B_metrics.parquet`. We rename so the
on-disk filename matches that convention.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from scripts.p1_project_a import config as a_config
from scripts.p1_project_b import config as b_config

from . import feature_lists as fl
from . import run_comparison
from . import run_lean_rerun


LOCKED_DIRS_AND_FILES: list[Path] = [
    # Project A
    a_config.RESULTS_DIR,
    a_config.MODELS_DIR,
    a_config.CACHE_DIR,
    a_config.DIAGNOSTIC_DIR,
    # Project B
    b_config.RESULTS_DIR,
    b_config.MODELS_DIR,
    b_config.CACHE_DIR,
    b_config.ARTIFACTS_DIR,
]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_dir_hashes(roots: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(fl.PROJECT_ROOT).as_posix()
                out[rel] = _hash_file(p)
    return out


def _diff_snapshots(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    keys_before = set(before.keys()); keys_after = set(after.keys())
    added = sorted(keys_after - keys_before)
    removed = sorted(keys_before - keys_after)
    changed = sorted(k for k in keys_before & keys_after if before[k] != after[k])
    return added, removed, changed


def write_unchanged_report(before: dict[str, str], after: dict[str, str]) -> None:
    added, removed, changed = _diff_snapshots(before, after)
    out = fl.ANALYSIS_DIR / "locked_artifacts_unchanged.md"
    md: list[str] = []
    md.append("# Locked-artifact non-modification verification\n")
    md.append("Snapshots of every file under the locked Project A and Project B directories, "
              "taken before and after `python -m scripts.p1_lean_features.run_all`. The lean "
              "rerun must not alter any locked artifact.\n")
    md.append(f"- Files snapshotted before run: {len(before)}")
    md.append(f"- Files snapshotted after run:  {len(after)}")
    md.append(f"- Files added during run:       {len(added)}")
    md.append(f"- Files removed during run:     {len(removed)}")
    md.append(f"- Files with changed SHA-256:   {len(changed)}")
    md.append("")

    if not added and not removed and not changed:
        md.append("**Result: PASS** — every locked artifact is byte-identical pre/post run.\n")
    else:
        md.append("**Result: FAIL** — see diff below.\n")

    if changed:
        md.append("## Changed files\n")
        for k in changed:
            md.append(f"- `{k}`")
            md.append(f"  - before: `{before[k][:16]}…`")
            md.append(f"  - after:  `{after[k][:16]}…`")
        md.append("")
    if added:
        md.append("## Added files\n")
        for k in added:
            md.append(f"- `{k}`  (new SHA-256: `{after[k][:16]}…`)")
        md.append("")
    if removed:
        md.append("## Removed files\n")
        for k in removed:
            md.append(f"- `{k}`  (was `{before[k][:16]}…`)")
        md.append("")

    md.append("## Snapshot scope\n")
    for d in LOCKED_DIRS_AND_FILES:
        md.append(f"- `{d.relative_to(fl.PROJECT_ROOT).as_posix()}/`")
    md.append("")

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[lean.run_all] verification report -> {out}")


def main() -> None:
    print("[lean.run_all] STEP 1/4 — snapshotting locked artifacts")
    t0 = time.time()
    before = _snapshot_dir_hashes(LOCKED_DIRS_AND_FILES)
    print(f"  snapshotted {len(before)} files in {time.time() - t0:.1f}s")

    print("\n[lean.run_all] STEP 2/4 — running 64 lean fits")
    t0 = time.time()
    run_lean_rerun.main()
    print(f"  fit time: {time.time() - t0:.1f}s")

    print("\n[lean.run_all] STEP 3/4 — building three-way comparison + report")
    t0 = time.time()
    run_comparison.main()
    print(f"  comparison time: {time.time() - t0:.1f}s")

    print("\n[lean.run_all] STEP 4/4 — verifying locked artifacts unchanged")
    t0 = time.time()
    after = _snapshot_dir_hashes(LOCKED_DIRS_AND_FILES)
    print(f"  re-snapshotted {len(after)} files in {time.time() - t0:.1f}s")
    write_unchanged_report(before, after)


if __name__ == "__main__":
    main()
