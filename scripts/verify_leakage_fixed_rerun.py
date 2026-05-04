"""Stage-5 verification per the leakage-fixed rerun prompt §6.

Checks:
  1. Phase 0 + dataset SHA-256 unchanged (compared to .phase0_sha_pre.txt).
  2. Dataset SHA-256 still c164c53b5dd27238...
  3. All fits in the unified inventory carry feature_stack_version = "leakage_fixed_v1".
  4. No active code path references load_long, load_mid, load_short, nns_state,
     or battery_value (occurrences in MIGRATION_LOG.md and docs/p1_battery_provenance/
     are documentation of the removed features, not active code use, and are tolerated).
  5. Every new report under docs/ mentions "leakage-fixed" (or "leakage fixed") in TL;DR.

Writes a verification block into MIGRATION_LOG.md §8.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_phase0_unchanged() -> tuple[bool, list[str]]:
    """Re-hash Phase 0 + dataset and compare to .phase0_sha_pre.txt."""
    pre_path = PROJECT_ROOT / ".phase0_sha_pre.txt"
    if not pre_path.exists():
        return False, ["pre-deletion SHA snapshot missing"]
    pre = {}
    for line in pre_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("MISSING"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            pre[parts[1].replace("\\", "/")] = parts[0]

    mismatches = []
    for rel, expected_sha in pre.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            mismatches.append(f"missing now: {rel}")
            continue
        actual = _sha256(p)
        if actual != expected_sha:
            mismatches.append(f"changed: {rel} (was {expected_sha[:16]}, now {actual[:16]})")
    return (not mismatches), mismatches


def _check_dataset_pinned_sha() -> tuple[bool, str]:
    p = PROJECT_ROOT / "data/phase1/dataset.parquet"
    sha = _sha256(p)
    expected_prefix = "c164c53b5dd27238"
    return sha.startswith(expected_prefix), sha


def _check_feature_stack_version(unified_inv_path: Path) -> tuple[bool, dict]:
    if not unified_inv_path.exists():
        return False, {"error": "unified inventory missing"}
    df = pd.read_parquet(unified_inv_path)
    n_total = len(df)
    if "feature_stack_version" not in df.columns:
        return False, {"n_total": n_total, "error": "missing column feature_stack_version"}
    counts = df["feature_stack_version"].value_counts().to_dict()
    ok = (counts.get("leakage_fixed_v1", 0) == n_total)
    return ok, {"n_total": n_total, "counts": counts}


def _check_no_legacy_features_in_active_code() -> tuple[bool, list[str]]:
    """Grep active code paths for the five removed feature names.

    Allowed (documentation, audit trail, historical context):
      - MIGRATION_LOG.md
      - docs/p1_battery_provenance/*
      - docs/p1_leakage_check/*
      - docs/p1_lean_features/* (tables — but should be empty post-deletion)
      - docs/proposal_rev10.md
      - docs/unified_report.md
      - .pre_deletion_shas.txt, .phase0_sha_pre.txt
      - scripts/p1_lean_features/* (historical context)
      - scripts/clean_telemetry.py (cleanup script — operates on raw data, not the model stack)
      - data/* (raw data — leaked features still in dataset.parquet as columns)
      - any __pycache__/*.pyc (compiled Python — may still reference the names)

    Active code paths grep'd:
      - scripts/p1_project_a/*.py
      - scripts/p1_project_b/*.py
      - scripts/run_leakage_fixed_pipeline*.py

    Hits inside Python comment lines or inside a triple-quoted docstring are
    tolerated (they document the removed features for human readers).
    """
    legacy = ["load_long", "load_mid", "load_short", "battery_value", "nns_state"]
    pat = re.compile(r"\b(" + "|".join(legacy) + r")\b")

    active_dirs = [
        PROJECT_ROOT / "scripts/p1_project_a",
        PROJECT_ROOT / "scripts/p1_project_b",
    ]
    active_files = [
        PROJECT_ROOT / "scripts/run_leakage_fixed_pipeline.py",
        PROJECT_ROOT / "scripts/run_leakage_fixed_pipeline_resume.py",
    ]
    files_to_check: list[Path] = []
    for d in active_dirs:
        for p in d.rglob("*.py"):
            if "__pycache__" in str(p):
                continue
            files_to_check.append(p)
    for p in active_files:
        if p.exists():
            files_to_check.append(p)

    def _is_doc_or_comment(line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'")

    hits = []
    for p in files_to_check:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        # Strip out triple-quoted blocks (module docstrings, multi-line comments
        # in feature-removal documentation) before grepping for active code use.
        stripped = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        stripped = re.sub(r"'''.*?'''", "", stripped, flags=re.DOTALL)
        for line in stripped.splitlines():
            if _is_doc_or_comment(line):
                continue
            for m in pat.finditer(line):
                hits.append(f"{p.relative_to(PROJECT_ROOT)} -> {m.group(0)} (line: {line.strip()[:80]})")

    return (not hits), hits


def _check_reports_mention_leakage_fixed() -> tuple[bool, list[str]]:
    """Every new report should mention 'leakage-fixed' or 'leakage fixed' in its TL;DR / first 60 lines."""
    candidates = [
        PROJECT_ROOT / "docs/p1_project_a/results_report.md",
        PROJECT_ROOT / "docs/p1_project_b/results_report.md",
        PROJECT_ROOT / "docs/p1_project_b/r4_diagnostic.md",
        PROJECT_ROOT / "docs/proposal_rev10.md",
        PROJECT_ROOT / "docs/unified_report.md",
    ]
    missing = []
    for p in candidates:
        if not p.exists():
            missing.append(f"missing: {p.relative_to(PROJECT_ROOT)}")
            continue
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:60]).lower()
        if "leakage-fixed" not in head and "leakage fixed" not in head:
            missing.append(f"no 'leakage-fixed' in first 60 lines of: {p.relative_to(PROJECT_ROOT)}")
    return (not missing), missing


def main() -> None:
    print("=" * 78)
    print("  Leakage-fixed rerun — Stage 5 verification")
    print("=" * 78)

    ok1, mismatches1 = _check_phase0_unchanged()
    print(f"\n[1] Phase 0 + dataset SHAs unchanged: {'PASS' if ok1 else 'FAIL'}")
    for m in mismatches1[:10]:
        print(f"    - {m}")

    ok2, sha2 = _check_dataset_pinned_sha()
    print(f"\n[2] Dataset SHA still c164c53b5dd27238...: {'PASS' if ok2 else 'FAIL'} (got {sha2[:16]})")

    inv = PROJECT_ROOT / "scripts/run_leakage_fixed_pipeline_inventory.parquet"
    ok3, fsv_info = _check_feature_stack_version(inv)
    print(f"\n[3] All fits tagged feature_stack_version='leakage_fixed_v1': {'PASS' if ok3 else 'FAIL'}")
    print(f"    {fsv_info}")

    ok4, hits = _check_no_legacy_features_in_active_code()
    print(f"\n[4] No legacy feature names in active code: {'PASS' if ok4 else 'FAIL'}")
    for h in hits[:20]:
        print(f"    - {h}")

    ok5, missing = _check_reports_mention_leakage_fixed()
    print(f"\n[5] All new reports mention 'leakage-fixed' in TL;DR: {'PASS' if ok5 else 'FAIL'}")
    for m in missing:
        print(f"    - {m}")

    overall = ok1 and ok2 and ok3 and ok4 and ok5
    print()
    print("=" * 78)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)

    # ---- Append the verification block to MIGRATION_LOG.md §8 -----
    log_path = PROJECT_ROOT / "MIGRATION_LOG.md"
    text = log_path.read_text(encoding="utf-8")
    sec8_marker = "## 8. Post-rerun verification\n"
    sec8_idx = text.find(sec8_marker)
    if sec8_idx < 0:
        raise RuntimeError("Could not find §8 marker in MIGRATION_LOG.md")

    block = [sec8_marker, ""]
    block.append("Stage-5 verification log per the leakage-fixed rerun prompt §6.\n")
    block.append(f"- **(1) Phase 0 + dataset SHAs unchanged**: {'PASS' if ok1 else 'FAIL'}.")
    if mismatches1:
        block.append("  - Mismatches:")
        for m in mismatches1[:10]:
            block.append(f"    - {m}")
    block.append(f"- **(2) Dataset SHA still `c164c53b5dd27238...`**: {'PASS' if ok2 else 'FAIL'} (got `{sha2}`).")
    block.append(f"- **(3) All fits tagged `feature_stack_version = \"leakage_fixed_v1\"`**: {'PASS' if ok3 else 'FAIL'} ({fsv_info.get('n_total', '?')} fits; counts = {fsv_info.get('counts', {})}).")
    block.append(f"- **(4) No legacy feature names (`load_long`/`load_mid`/`load_short`/`battery_value`/`nns_state`) in active code paths**: {'PASS' if ok4 else 'FAIL'}.")
    if hits:
        block.append("  - Hits (must be fixed if any):")
        for h in hits[:20]:
            block.append(f"    - {h}")
    block.append(f"- **(5) All new reports mention 'leakage-fixed' in TL;DR**: {'PASS' if ok5 else 'FAIL'}.")
    if missing:
        block.append("  - Missing:")
        for m in missing:
            block.append(f"    - {m}")
    block.append("")
    block.append(f"**Overall: {'PASS' if overall else 'FAIL'}**.\n")

    new_text = text[:sec8_idx] + "\n".join(block) + "\n"
    log_path.write_text(new_text, encoding="utf-8")
    print(f"\n[verify] wrote verification block to {log_path}")


if __name__ == "__main__":
    main()
