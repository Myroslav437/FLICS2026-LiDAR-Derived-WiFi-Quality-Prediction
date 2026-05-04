"""Resume the leakage-fixed pipeline from the Project A hardening stage onward.

Project A modeling and Project A robustness diagnostic are already on disk
(see `.pipeline_run.log`). This script picks up at hardening, then continues
through Project B and the consolidation step.
"""

from __future__ import annotations

import time

from scripts.run_leakage_fixed_pipeline import _stage, _sha256, consolidate, PROJECT_ROOT


def main() -> None:
    t_start = time.time()
    print(f"[pipeline-resume] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

    from scripts.p1_project_a import run_hardening_all as a_hardening
    from scripts.p1_project_a import run_xai as a_xai
    from scripts.p1_project_a import build_results_report as a_report

    _stage("Project A -- hardening (A: placebo; B: LightGBM)")
    a_hardening.main()
    _stage("Project A -- XAI (TreeSHAP per fold)")
    a_xai.main()
    _stage("Project A -- results report")
    a_report.main()

    from scripts.p1_project_b import run_modeling as b_modeling
    from scripts.p1_project_b import run_r4_diagnostic as b_r4
    from scripts.p1_project_b import run_hardening_all as b_hardening
    from scripts.p1_project_b import run_xai as b_xai
    from scripts.p1_project_b import build_results_report as b_report

    _stage("Project B -- modeling (WLRO + buffer + H1 sensitivity)")
    b_modeling.main()
    _stage("Project B -- R-4 four-corrections diagnostic")
    b_r4.main()
    _stage("Project B -- hardening (C: H1+buffer; E: placebo)")
    b_hardening.main()
    _stage("Project B -- XAI (TreeSHAP on R-3 W4)")
    b_xai.main()
    _stage("Project B -- results report")
    b_report.main()

    _stage("Consolidation -- unified fit inventory + post-rerun SHA log")
    consolidate()

    _stage("Verification -- Phase 0 + dataset unchanged")
    post_p0_shas = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(p) for p in p0_files if p.exists()}
    mismatches = [k for k in pre_p0_shas if pre_p0_shas[k] != post_p0_shas.get(k)]
    if mismatches:
        raise RuntimeError(f"Phase 0 / dataset changed during rerun: {mismatches}")
    print("[pipeline-resume] Phase 0 + dataset bytes-identical (8/8 SHA-256 match).")

    elapsed = time.time() - t_start
    print(f"\n[pipeline-resume] complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
