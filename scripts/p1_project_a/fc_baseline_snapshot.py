"""Snapshot SHA-256 of all locked artefacts BEFORE the F-C diagnostic runs.

Reads from:
  - scripts/p1_project_a/{models,cache,results,diagnostic}/
  - scripts/p1_project_b/

EXCLUDES the new fc_diagnostic/ subdirectories (which the diagnostic creates).

Writes the snapshot to scripts/p1_project_a/fc_diagnostic_baseline.json so
the post-run integrity check can compare against it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(roots: list[str], excludes: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for d, _, files in os.walk(rp):
            d_posix = Path(d).as_posix()
            if any(d_posix.startswith(ex) or ("/" + ex + "/") in ("/" + d_posix + "/") for ex in excludes):
                continue
            for fn in files:
                p = Path(d) / fn
                p_posix = p.as_posix()
                if any(ex in p_posix for ex in excludes):
                    continue
                rows.append((p_posix, sha256_of(p)))
    rows.sort()
    return rows


def main(out_path: str = "scripts/p1_project_a/fc_diagnostic_baseline.json") -> None:
    roots = [
        "scripts/p1_project_a/models",
        "scripts/p1_project_a/cache",
        "scripts/p1_project_a/results",
        "scripts/p1_project_a/diagnostic",
        "scripts/p1_project_b",
    ]
    # Exclude the new fc_diagnostic subdirectories created by this run.
    excludes = ["fc_diagnostic"]

    rows = collect(roots, excludes)
    payload = [{"path": p, "sha256": s} for p, s in rows]
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"snapshot: {len(rows)} files -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/p1_project_a/fc_diagnostic_baseline.json")
