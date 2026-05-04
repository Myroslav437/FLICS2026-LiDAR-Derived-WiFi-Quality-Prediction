# Deterministic-split rerun — audit log

Snapshots `scripts/p1_project_b/` before and after the deterministic rerun (`python -m scripts.p1_lean_features.run_det_rerun`). The locked artifacts are byte-frozen audit-trail files; this rerun must not modify any of them.

## 1. Determinism smoke test

Five sequential calls to `det_folds.deterministic_fold_offset('R-1')` return `2934265069` every time. The wrapper uses `hashlib.sha256` instead of Python's built-in `hash()`, so the offset is reproducible across Python invocations, machines, and PYTHONHASHSEED settings.

## 2. Locked-artifact non-modification

- Files snapshotted before: 149
- Files snapshotted after:  149
- Files added:              0
- Files removed:            0
- Files with changed SHA-256: 0

**Result: PASS** — every file under `scripts/p1_project_b/` is byte-identical before and after the rerun.

## 3. Rerun summary

- Total fits: 30 (15 lockedFull + 15 leanB)
- Verdict: **SHIFTED**
- max |Δ_lean-B − lockedFull| (deterministic): **6.825 dB**
- max |Δ| on headline rows (R-4 W4 in_fov, locked + H1): **3.906 dB**

## 4. Snapshot scope

- `scripts/p1_project_b/` (recursive, excluding `__pycache__`)
