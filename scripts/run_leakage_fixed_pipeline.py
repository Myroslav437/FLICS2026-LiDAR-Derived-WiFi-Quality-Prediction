"""End-to-end pipeline driver for the leakage-fixed Phase 1 rerun.

Executes, in order:

  1. Project A modeling (LORO ablation + RQ4)
  2. Project A hyperparameter robustness diagnostic on F-B
  3. Project A hardening (Hardening A — placebo; Hardening B — LightGBM)
  4. Project A XAI (TreeSHAP per fold)
  5. Project A results report
  6. Project B modeling (WLRO + buffer + H1 sensitivity)
  7. Project B R-4 four-corrections diagnostic
  8. Project B hardening (Hardening C — H1+buffer; Hardening E — placebo)
  9. Project B XAI (TreeSHAP on R-3 W4)
 10. Project B results report
 11. Final consolidation: write a unified fit-inventory parquet and append the
     post-rerun SHA-256 inventory + verification log to `MIGRATION_LOG.md`.

Usage: `python -m scripts.run_leakage_fixed_pipeline`
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage(label: str) -> None:
    print()
    print("=" * 78)
    print(f"  STAGE: {label}")
    print("=" * 78)


def main() -> None:
    t_start = time.time()
    print(f"[pipeline] leakage-fixed Phase 1 rerun starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Pre-flight: phase-0 + dataset SHA snapshot --------------------
    p0_files = [
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/anomaly_mask.parquet",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/agv_body_mask.npz",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/lidar_fov.json",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/ap_coords.json",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/feature_extractor.py",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/anomaly_threshold.json",
        PROJECT_ROOT / "scripts/p0_analysis/artifacts/anomaly_mask_v3.parquet",
        PROJECT_ROOT / "data/phase1/dataset.parquet",
    ]
    pre_p0_shas = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(p) for p in p0_files if p.exists()}
    print("[pipeline] pre-rerun Phase 0 + dataset SHAs captured (8 files)")

    # ---- Project A ----------------------------------------------------
    from scripts.p1_project_a import run_modeling as a_modeling
    from scripts.p1_project_a import run_robustness as a_robust
    from scripts.p1_project_a import run_hardening_all as a_hardening
    from scripts.p1_project_a import run_xai as a_xai
    from scripts.p1_project_a import build_results_report as a_report

    _stage("Project A — modeling (LORO ablation + RQ4)")
    a_modeling.main()
    _stage("Project A — hyperparameter robustness diagnostic on F-B")
    a_robust.main()
    _stage("Project A — hardening (A: placebo; B: LightGBM)")
    a_hardening.main()
    _stage("Project A — XAI (TreeSHAP per fold)")
    a_xai.main()
    _stage("Project A — results report")
    a_report.main()

    # ---- Project B ----------------------------------------------------
    from scripts.p1_project_b import run_modeling as b_modeling
    from scripts.p1_project_b import run_r4_diagnostic as b_r4
    from scripts.p1_project_b import run_hardening_all as b_hardening
    from scripts.p1_project_b import run_xai as b_xai
    from scripts.p1_project_b import build_results_report as b_report

    _stage("Project B — modeling (WLRO + buffer + H1 sensitivity)")
    b_modeling.main()
    _stage("Project B — R-4 four-corrections diagnostic")
    b_r4.main()
    _stage("Project B — hardening (C: H1+buffer; E: placebo)")
    b_hardening.main()
    _stage("Project B — XAI (TreeSHAP on R-3 W4)")
    b_xai.main()
    _stage("Project B — results report")
    b_report.main()

    # ---- Consolidation -----------------------------------------------
    _stage("Consolidation — unified fit inventory + post-rerun SHA log")
    consolidate()

    # ---- Phase 0 post-rerun verification -----------------------------
    _stage("Verification — Phase 0 + dataset unchanged")
    post_p0_shas = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(p) for p in p0_files if p.exists()}
    mismatches = [k for k in pre_p0_shas if pre_p0_shas[k] != post_p0_shas.get(k)]
    if mismatches:
        raise RuntimeError(f"Phase 0 / dataset changed during rerun: {mismatches}")
    print("[pipeline] Phase 0 + dataset bytes-identical (8/8 SHA-256 match).")

    elapsed = time.time() - t_start
    print(f"\n[pipeline] complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


def consolidate() -> None:
    """Build the unified fit-inventory parquet and append post-rerun hashes
    to MIGRATION_LOG.md.

    Sources:
      - Project A modeling: scripts/p1_project_a/results/fit_inventory.parquet
      - Project A diagnostic: scripts/p1_project_a/diagnostic/robustness_fit_inventory.parquet
      - Project A hardening: scripts/p1_project_a/results/hardening_fit_inventory.parquet
      - Project B modeling: scripts/p1_project_b/results/fit_inventory.parquet
      - Project B R-4 diagnostic: scripts/p1_project_b/results/r4_diagnostic_fit_inventory.parquet
      - Project B hardening: scripts/p1_project_b/results/hardening_fit_inventory.parquet
    """
    from scripts.p1_project_a import config as a_config
    from scripts.p1_project_b import config as b_config

    sources: list[tuple[str, str, Path]] = [
        ("project_a_modeling", "Project A — main ablation + RQ4",
         a_config.RESULTS_DIR / "fit_inventory.parquet"),
        ("project_a_diagnostic", "Project A — F-B hyperparameter robustness diagnostic",
         a_config.DIAGNOSTIC_DIR / "robustness_fit_inventory.parquet"),
        ("project_a_hardening", "Project A — Hardening A (placebo) + B (LightGBM)",
         a_config.RESULTS_DIR / "hardening_fit_inventory.parquet"),
        ("project_b_modeling", "Project B — WLRO + buffer + H1 sensitivity",
         b_config.RESULTS_DIR / "fit_inventory.parquet"),
        ("project_b_r4_diagnostic", "Project B — R-4 four-corrections diagnostic",
         b_config.RESULTS_DIR / "r4_diagnostic_fit_inventory.parquet"),
        ("project_b_hardening", "Project B — Hardening C (H1+buffer) + E (placebo)",
         b_config.RESULTS_DIR / "hardening_fit_inventory.parquet"),
    ]

    frames = []
    counts: dict[str, int] = {}
    for tag, _label, path in sources:
        if not path.exists():
            print(f"[consolidate] WARNING: missing fit inventory: {path}")
            counts[tag] = 0
            continue
        df = pd.read_parquet(path).copy()
        df.insert(0, "source_group", tag)
        df["feature_stack_version"] = a_config.FEATURE_STACK_VERSION
        frames.append(df)
        counts[tag] = len(df)
        print(f"[consolidate] {tag}: {len(df)} fits")

    unified_path = PROJECT_ROOT / "scripts/run_leakage_fixed_pipeline_inventory.parquet"
    if frames:
        unified = pd.concat(frames, axis=0, ignore_index=True)
        unified.to_parquet(unified_path, index=False)
        print(f"[consolidate] unified inventory -> {unified_path} ({len(unified)} rows)")
    total_fits = sum(counts.values())
    print(f"[consolidate] total fits across all sources: {total_fits}")

    # ---- Post-rerun SHA inventory of new artefacts -------------------
    new_artefacts: list[tuple[str, str]] = []
    artefact_roots = [
        PROJECT_ROOT / "scripts/p1_project_a/models",
        PROJECT_ROOT / "scripts/p1_project_a/cache",
        PROJECT_ROOT / "scripts/p1_project_a/results",
        PROJECT_ROOT / "scripts/p1_project_a/diagnostic",
        PROJECT_ROOT / "scripts/p1_project_b/models",
        PROJECT_ROOT / "scripts/p1_project_b/cache",
        PROJECT_ROOT / "scripts/p1_project_b/results",
        PROJECT_ROOT / "scripts/p1_project_b/artifacts",
    ]
    for root in artefact_roots:
        if not root.exists():
            continue
        for sub in sorted(root.rglob("*")):
            if sub.is_file():
                rel = str(sub.relative_to(PROJECT_ROOT)).replace("\\", "/")
                new_artefacts.append((rel, _sha256(sub)))

    # ---- Append the post-rerun + verification sections to MIGRATION_LOG.md
    log_path = PROJECT_ROOT / "MIGRATION_LOG.md"
    text = log_path.read_text(encoding="utf-8")

    # Replace the §7 placeholder with the actual SHAs.
    sec7_marker = "## 7. Post-rerun SHA-256 hashes\n"
    sec7_idx = text.find(sec7_marker)
    sec8_marker = "## 8. Post-rerun verification\n"
    sec8_idx = text.find(sec8_marker)
    if sec7_idx < 0 or sec8_idx < 0:
        raise RuntimeError("Could not find §7/§8 markers in MIGRATION_LOG.md")

    sec7_lines = [sec7_marker, ""]
    sec7_lines.append(
        f"Post-rerun inventory: {len(new_artefacts)} files across the Project A and Project B "
        "models, caches, results, diagnostic, and artifacts directories.\n"
    )
    sec7_lines.append("Per-source-group fit counts:\n")
    for tag, _label, _path in sources:
        sec7_lines.append(f"- `{tag}`: {counts.get(tag, 0)} fits")
    sec7_lines.append("")
    sec7_lines.append("First 32 rows of the post-rerun model SHA-256 inventory:\n")
    sec7_lines.append("```")
    model_files = [(p, h) for p, h in new_artefacts if p.endswith((".json", ".txt")) and ("models/" in p or "/diagnostic/models/" in p)]
    for p, h in model_files[:32]:
        sec7_lines.append(f"{h}  {p}")
    sec7_lines.append("```\n")
    sec7_lines.append(
        f"Total post-rerun model files: {len(model_files)}.\n"
        "Full per-file inventory persisted at `scripts/run_leakage_fixed_pipeline_inventory.parquet`.\n"
    )
    sec7_block = "\n".join(sec7_lines) + "\n"

    new_text = text[:sec7_idx] + sec7_block + text[sec8_idx:]
    log_path.write_text(new_text, encoding="utf-8")
    print(f"[consolidate] MIGRATION_LOG.md §7 updated with post-rerun SHAs ({len(new_artefacts)} files)")


if __name__ == "__main__":
    main()
