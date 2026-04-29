"""Verify SHA-256 hashes after the hardening merge.

Compares pre-move hashes (from `merge_hardening_pre_hashes.md`) against post-move
hashes at each file's new location.

The byte-stable set — model files (.json / .txt), prediction caches (.parquet),
results parquets, docs figures + tables — MUST remain unchanged. A mismatch in
this set is a critical failure: the move silently corrupted a file.

The expected-edit set — source code (.py) — is updated in §3.3 of the merge
spec to change `from . import config` style imports to the new package's
`config as a_config` / `config as b_config`. A mismatch in this set is expected
and the post-edit hash is the new source of truth; we record both for audit.

Files that were "split" into multiple destinations
(`hardening_metrics.parquet`, `fit_inventory.parquet`) get a SPLIT entry — the
row-level integrity is verified by the dedicated split tool and by the row
counts (75 metric rows = 42 A/B + 33 C/E; 15 fit rows = 6 A/B + 9 C/E).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
PRE = ROOT / "scripts" / "maintenance" / "merge_hardening_pre_hashes.md"
OUT = ROOT / "scripts" / "maintenance" / "merge_hardening_post_hashes.md"


# Mapping from pre-move relative path -> post-move relative path.
MAPPING: dict[str, str] = {
    # Helper modules
    "scripts/p1_hardening/placebo.py": "scripts/p1_project_a/placebo.py",
    "scripts/p1_hardening/lightgbm_training.py": "scripts/p1_project_a/lightgbm_training.py",
    # Runners (renamed)
    "scripts/p1_hardening/run_a_cross_session_placebo.py": "scripts/p1_project_a/run_hardening_placebo.py",
    "scripts/p1_hardening/run_b_lightgbm.py":             "scripts/p1_project_a/run_hardening_lightgbm.py",
    "scripts/p1_hardening/run_c_r4_combined.py":          "scripts/p1_project_b/run_hardening_combined.py",
    "scripts/p1_hardening/run_e_within_session_placebo.py": "scripts/p1_project_b/run_hardening_placebo.py",
    "scripts/p1_hardening/run_d_dataset_noise_floor.py":  "scripts/p1_dataset_analysis/run_noise_floor.py",
    # Models F-B (A, B) -> Project A
    "scripts/p1_hardening/models/A_B5_placebo_locked.json": "scripts/p1_project_a/models/A_B5_placebo_locked.json",
    "scripts/p1_hardening/models/A_B5_placebo_H1.json":     "scripts/p1_project_a/models/A_B5_placebo_H1.json",
    "scripts/p1_hardening/models/B_B1_lgb_default.txt":     "scripts/p1_project_a/models/B_B1_lgb_default.txt",
    "scripts/p1_hardening/models/B_B1_lgb_H1_equiv.txt":    "scripts/p1_project_a/models/B_B1_lgb_H1_equiv.txt",
    "scripts/p1_hardening/models/B_B5_lgb_default.txt":     "scripts/p1_project_a/models/B_B5_lgb_default.txt",
    "scripts/p1_hardening/models/B_B5_lgb_H1_equiv.txt":    "scripts/p1_project_a/models/B_B5_lgb_H1_equiv.txt",
    # Models C buffer -> Project B
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W0.json":   "scripts/p1_project_b/models/C_R-4_buffer_H1_W0.json",
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W1.json":   "scripts/p1_project_b/models/C_R-4_buffer_H1_W1.json",
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W2.json":   "scripts/p1_project_b/models/C_R-4_buffer_H1_W2.json",
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W3.json":   "scripts/p1_project_b/models/C_R-4_buffer_H1_W3.json",
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W4.json":   "scripts/p1_project_b/models/C_R-4_buffer_H1_W4.json",
    "scripts/p1_hardening/models/C_R-4_buffer_H1_W4pp.json": "scripts/p1_project_b/models/C_R-4_buffer_H1_W4pp.json",
    # Models E placebo -> Project B
    "scripts/p1_hardening/models/E_R-4_W2_real_locked.json":     "scripts/p1_project_b/models/E_R-4_W2_real_locked.json",
    "scripts/p1_hardening/models/E_R-4_W4_real_locked.json":     "scripts/p1_project_b/models/E_R-4_W4_real_locked.json",
    "scripts/p1_hardening/models/E_R-4_W4_placebo_locked.json": "scripts/p1_project_b/models/E_R-4_W4_placebo_locked.json",
    # Cache A, B -> Project A
    "scripts/p1_hardening/cache/predictions_A_B5_placebo_H1.parquet":     "scripts/p1_project_a/cache/predictions_A_B5_placebo_H1.parquet",
    "scripts/p1_hardening/cache/predictions_A_B5_placebo_locked.parquet": "scripts/p1_project_a/cache/predictions_A_B5_placebo_locked.parquet",
    "scripts/p1_hardening/cache/predictions_B_B1_lgb_default.parquet":    "scripts/p1_project_a/cache/predictions_B_B1_lgb_default.parquet",
    "scripts/p1_hardening/cache/predictions_B_B1_lgb_H1_equiv.parquet":   "scripts/p1_project_a/cache/predictions_B_B1_lgb_H1_equiv.parquet",
    "scripts/p1_hardening/cache/predictions_B_B5_lgb_default.parquet":    "scripts/p1_project_a/cache/predictions_B_B5_lgb_default.parquet",
    "scripts/p1_hardening/cache/predictions_B_B5_lgb_H1_equiv.parquet":   "scripts/p1_project_a/cache/predictions_B_B5_lgb_H1_equiv.parquet",
    # Cache C, E -> Project B
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W0.parquet":   "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W0.parquet",
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W1.parquet":   "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W1.parquet",
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W2.parquet":   "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W2.parquet",
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W3.parquet":   "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W3.parquet",
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4.parquet":   "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W4.parquet",
    "scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4pp.parquet": "scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W4pp.parquet",
    "scripts/p1_hardening/cache/predictions_E_R-4_W2_real_locked.parquet":     "scripts/p1_project_b/cache/predictions_E_R-4_W2_real_locked.parquet",
    "scripts/p1_hardening/cache/predictions_E_R-4_W4_real_locked.parquet":     "scripts/p1_project_b/cache/predictions_E_R-4_W4_real_locked.parquet",
    "scripts/p1_hardening/cache/predictions_E_R-4_W4_placebo_locked.parquet": "scripts/p1_project_b/cache/predictions_E_R-4_W4_placebo_locked.parquet",
    # Per-experiment results (renamed at destination)
    "scripts/p1_hardening/results/experiment_a_integrity.parquet": "scripts/p1_project_a/results/hardening_a_integrity.parquet",
    "scripts/p1_hardening/results/experiment_a_metrics.parquet":   "scripts/p1_project_a/results/hardening_a_metrics.parquet",
    "scripts/p1_hardening/results/experiment_b_metrics.parquet":   "scripts/p1_project_a/results/hardening_b_metrics.parquet",
    "scripts/p1_hardening/results/experiment_c_metrics.parquet":   "scripts/p1_project_b/results/hardening_c_metrics.parquet",
    "scripts/p1_hardening/results/experiment_e_integrity.parquet": "scripts/p1_project_b/results/hardening_e_integrity.parquet",
    "scripts/p1_hardening/results/experiment_e_metrics.parquet":   "scripts/p1_project_b/results/hardening_e_metrics.parquet",
    # Experiment D dataset-level outputs
    "scripts/p1_hardening/results/experiment_d_per_cell_sigma.parquet":   "scripts/p1_dataset_analysis/results/noise_floor_per_cell_sigma.parquet",
    "scripts/p1_hardening/results/experiment_d_same_cell_deltas.parquet": "scripts/p1_dataset_analysis/results/noise_floor_same_cell_deltas.parquet",
    "scripts/p1_hardening/results/experiment_d_summary.json":             "scripts/p1_dataset_analysis/results/noise_floor_summary.json",
    # Docs
    "docs/p1_hardening/figures/dataset_noise_floor.png": "docs/p1_dataset_analysis/figures/dataset_noise_floor.png",
    "docs/p1_hardening/tables/dataset_noise_floor.md":   "docs/p1_dataset_analysis/tables/dataset_noise_floor.md",
}

# Files split rather than moved 1:1 — these are checked content-equivalent (rows
# preserved across the two halves) elsewhere; here we just emit a SPLIT entry.
SPLIT: set[str] = {
    "scripts/p1_hardening/results/hardening_metrics.parquet",
    "scripts/p1_hardening/results/fit_inventory.parquet",
}

# Files dissolved (deleted) and content reproduced inside other docs/code.
DISSOLVED: set[str] = {
    "scripts/p1_hardening/__init__.py",        # rewritten as deprecation shim
    "scripts/p1_hardening/config.py",          # absorbed into project A/B configs
    "scripts/p1_hardening/build_results_report.py",  # functionality moved to project A/B reports
    "scripts/p1_hardening/run_all.py",         # split into project_a/run_hardening_all.py + project_b/run_hardening_all.py
    "docs/p1_hardening/results_report.md",     # dissolved into 3 target docs
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_pre() -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in PRE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-fA-F]{64})`\s*\|", ln)
        if m:
            out[m.group(1)] = m.group(2).lower()
    return out


def main() -> None:
    pre = parse_pre()
    lines: list[str] = []
    lines.append("# Post-move SHA-256 verification\n")
    lines.append("Compares pre-move hash (from `merge_hardening_pre_hashes.md`) against post-move hash at the destination.\n")
    lines.append("Source files (.py) are EXPECTED to differ in SHA after Stage 1.3 import-path updates.\n")
    lines.append("Model / cache / results / docs files MUST match exactly; mismatches there are critical failures.\n")
    lines.append("| Pre-path | Post-path | Pre SHA | Post SHA | Status |")
    lines.append("|---|---|---|---|---|")
    n_match = 0
    n_byte_stable_mismatch = 0
    n_source_edited = 0
    n_split = 0
    n_dissolved = 0
    n_missing_post = 0
    for src in sorted(pre.keys()):
        pre_sha = pre[src]
        if src in SPLIT:
            lines.append(f"| `{src}` | (split) | `{pre_sha}` | — | SPLIT |")
            n_split += 1
            continue
        if src in DISSOLVED:
            lines.append(f"| `{src}` | (dissolved) | `{pre_sha}` | — | DISSOLVED |")
            n_dissolved += 1
            continue
        dst = MAPPING.get(src)
        if dst is None:
            lines.append(f"| `{src}` | ??? | `{pre_sha}` | — | UNMAPPED |")
            n_missing_post += 1
            continue
        dst_path = ROOT / dst
        if not dst_path.exists():
            lines.append(f"| `{src}` | `{dst}` | `{pre_sha}` | — | MISSING |")
            n_missing_post += 1
            continue
        post_sha = sha256_of(dst_path)
        if post_sha == pre_sha:
            status = "MATCH"
            n_match += 1
        elif src.endswith(".py"):
            status = "EDITED-EXPECTED"
            n_source_edited += 1
        else:
            status = "MISMATCH (byte-stable file changed)"
            n_byte_stable_mismatch += 1
        lines.append(f"| `{src}` | `{dst}` | `{pre_sha}` | `{post_sha}` | {status} |")
    lines.append("")
    lines.append("## Summary\n")
    lines.append(f"- MATCH (byte-stable file unchanged): {n_match}")
    lines.append(f"- EDITED-EXPECTED (.py source updated in Stage 1.3): {n_source_edited}")
    lines.append(f"- MISMATCH (byte-stable file changed — CRITICAL): {n_byte_stable_mismatch}")
    lines.append(f"- SPLIT: {n_split} (expected for hardening_metrics.parquet, fit_inventory.parquet)")
    lines.append(f"- DISSOLVED: {n_dissolved} (expected for __init__.py, config.py, build_results_report.py, run_all.py, docs results_report.md)")
    lines.append(f"- MISSING/UNMAPPED: {n_missing_post}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(
        f"  MATCH={n_match} EDITED-EXPECTED={n_source_edited} "
        f"MISMATCH={n_byte_stable_mismatch} SPLIT={n_split} "
        f"DISSOLVED={n_dissolved} MISSING/UNMAPPED={n_missing_post}"
    )
    if n_byte_stable_mismatch != 0 or n_missing_post != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
