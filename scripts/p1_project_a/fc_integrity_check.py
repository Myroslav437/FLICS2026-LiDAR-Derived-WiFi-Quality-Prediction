"""Compare current SHA-256 of locked artefacts against the baseline snapshot.

Reports any added/removed/modified files. Designed to run AFTER
``run_fc_diagnostic`` to confirm no locked artefact has been touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .fc_baseline_snapshot import collect


def main(baseline_path: str = "scripts/p1_project_a/fc_diagnostic_baseline.json") -> int:
    baseline = json.loads(Path(baseline_path).read_text())
    baseline_map = {row["path"]: row["sha256"] for row in baseline}

    roots = [
        "scripts/p1_project_a/models",
        "scripts/p1_project_a/cache",
        "scripts/p1_project_a/results",
        "scripts/p1_project_a/diagnostic",
        "scripts/p1_project_b",
    ]
    excludes = ["fc_diagnostic"]
    current = collect(roots, excludes)
    current_map = dict(current)

    baseline_keys = set(baseline_map)
    current_keys = set(current_map)

    added = sorted(current_keys - baseline_keys)
    removed = sorted(baseline_keys - current_keys)
    modified = sorted(p for p in baseline_keys & current_keys if baseline_map[p] != current_map[p])

    print(f"baseline files: {len(baseline_keys)}")
    print(f"current files:  {len(current_keys)}")
    print(f"added:    {len(added)}")
    print(f"removed:  {len(removed)}")
    print(f"modified: {len(modified)}")

    if added:
        print("\nADDED files (unexpected — should be empty for an integrity check):")
        for p in added:
            print(f"  + {p}")
    if removed:
        print("\nREMOVED files:")
        for p in removed:
            print(f"  - {p}")
    if modified:
        print("\nMODIFIED files:")
        for p in modified:
            print(f"  ~ {p}")
            print(f"    baseline: {baseline_map[p]}")
            print(f"    current:  {current_map[p]}")

    if added or removed or modified:
        print("\n[FAIL] locked artefact integrity check FAILED")
        return 1
    print("\n[OK] locked artefact integrity check passed — 0 changes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/p1_project_a/fc_diagnostic_baseline.json"))
