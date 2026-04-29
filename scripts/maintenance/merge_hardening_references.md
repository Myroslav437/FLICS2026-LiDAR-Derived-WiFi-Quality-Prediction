# Cross-references to `p1_hardening` (case-insensitive grep)

Snapshot taken before Stage 1. Each match is classified by treatment:

- **UPDATE**: live code/doc reference; rewrite to the new path during §3.3.
- **HISTORICAL**: locked-by-revision document (proposal Rev9); leave as-is per spec §3.3.
- **DISSOLVED**: file is being dissolved into a target doc; the rewrite below handles the path implicitly.
- **AUDIT**: maintenance log file; never updated, source of truth.

## Code

| File | Line | Treatment | Note |
|---|---:|---|---|
| `scripts/p1_hardening/__init__.py` | 1–13 | rewrite as deprecation shim | per spec §2.4 |
| `scripts/p1_hardening/config.py` | 28 | DISSOLVED — file deleted | helper config no longer needed; A/B configs absorb the role |
| `scripts/p1_hardening/config.py` | 33 | DISSOLVED — file deleted | as above |
| `scripts/p1_hardening/build_results_report.py` | all | DISSOLVED — file deleted | replaced by extension of project A and B `build_results_report.py` |
| `scripts/p1_hardening/run_a_cross_session_placebo.py` | imports | UPDATE in moved file | becomes `scripts/p1_project_a/run_hardening_placebo.py` |
| `scripts/p1_hardening/run_b_lightgbm.py` | imports | UPDATE in moved file | becomes `scripts/p1_project_a/run_hardening_lightgbm.py` |
| `scripts/p1_hardening/run_c_r4_combined.py` | imports | UPDATE in moved file | becomes `scripts/p1_project_b/run_hardening_combined.py` |
| `scripts/p1_hardening/run_d_dataset_noise_floor.py` | 6 | UPDATE in moved file | becomes `scripts/p1_dataset_analysis/run_noise_floor.py` |
| `scripts/p1_hardening/run_e_within_session_placebo.py` | imports | UPDATE in moved file | becomes `scripts/p1_project_b/run_hardening_placebo.py` |
| `scripts/p1_hardening/run_all.py` | 7 | DISSOLVED → split | replaced by `run_hardening_all.py` in projects a and b |

## Documents

| File | Line | Treatment | Note |
|---|---:|---|---|
| `docs/proposal_rev9.md` | 5 | HISTORICAL | proposal Rev9 abstract sentence references the old location for the hardening report; locked. |
| `docs/proposal_rev9.md` | 130 | HISTORICAL | "Suggested figure" path; locked. The figure is moved physically; the proposal text stays as written per spec §3.3 / §6 ("Update only references in code and other docs"). |
| `docs/proposal_rev9.md` | 376 | HISTORICAL | "Frozen artifacts" path block; locked. |
| `docs/proposal_rev9.md` | 410 | HISTORICAL | "agent's suggested wording in `docs/p1_hardening/results_report.md` §8" — locked. After dissolution the suggested wording lives in r4_diagnostic per `r4_diagnostic.md` (which is now §4 of `docs/p1_project_b/results_report.md`). |
| `docs/p1_hardening/results_report.md` | all | DISSOLVED — file deleted | content moved into the three target docs per spec §3.4 |

## Audit / maintenance

| File | Note |
|---|---|
| `scripts/maintenance/merge_hardening_pre_inventory.md` | this audit |
| `scripts/maintenance/merge_hardening_pre_hashes.md` | this audit |
| `scripts/maintenance/merge_hardening_references.md` | this file |

## Summary

- 1 historical doc with 4 references (proposal Rev9): leave as-is.
- 3 source files dissolved (config.py, build_results_report.py, p1_hardening/run_all.py).
- 5 runner scripts to be moved + updated.
- 2 helper modules to be moved + updated.
- 1 results_report.md dissolved into 3 target docs.

After Stage 1, the only remaining matches for `p1_hardening` should be:
1. The deprecation shim at `scripts/p1_hardening/__init__.py`.
2. The four locked references in `docs/proposal_rev9.md`.
3. The three audit files in `scripts/maintenance/`.
