# Locked-artifact non-modification verification

Snapshots of every file under the locked Project A and Project B directories, taken before and after `python -m scripts.p1_lean_features.run_all`. The lean rerun must not alter any locked artifact.

- Files snapshotted before run: 238
- Files snapshotted after run:  238
- Files added during run:       0
- Files removed during run:     0
- Files with changed SHA-256:   0

**Result: PASS** — every locked artifact is byte-identical pre/post run.

## Snapshot scope

- `scripts/p1_project_a/results/`
- `scripts/p1_project_a/models/`
- `scripts/p1_project_a/cache/`
- `scripts/p1_project_a/diagnostic/`
- `scripts/p1_project_b/results/`
- `scripts/p1_project_b/models/`
- `scripts/p1_project_b/cache/`
- `scripts/p1_project_b/artifacts/`
