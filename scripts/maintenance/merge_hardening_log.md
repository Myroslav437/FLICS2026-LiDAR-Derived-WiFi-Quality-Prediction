# Hardening merge — Stage 1 audit log

**Date:** 2026-04-28.
**Operator:** automated agent under user `myroslav.m437498@gmail.com`, `auto` mode, on branch `flics-2026`.
**Spec:** `Merge p1_hardening into Project A / Project B / dataset analysis, then regenerate the unified report` (Stage 1).

The hardening package `scripts/p1_hardening/` and its companion docs `docs/p1_hardening/` were dissolved in place into Project A, Project B, and the dataset analysis. No model was re-fit; no number changed. This log is the audit trail.

## 1. Method

`scripts/p1_hardening/` and `docs/p1_hardening/` were not yet tracked in git at the time of merge (`git status` reported them as `??`), so `git mv` was unavailable. Used plain `mv` (Unix shell `mv` via the Bash tool) for every file move. No dual-write `cp` + `rm` step was needed because `mv` is atomic on the same filesystem; SHA-256 verification in §3 confirms no data corruption.

## 2. Files moved

48 byte-stable artifacts were moved (15 model files + 15 prediction caches + 9 result parquets/json + 3 docs assets + 6 other = 48 total below). 7 source `.py` files were moved AND edited in Stage 1.3 to update import paths. 2 consolidated parquets were split into per-project halves. 5 source files were dissolved (deleted, with their content reproduced inside other docs/code).

### 2.1 Helper modules (moved + edited)

| Source | Destination | Reason |
|---|---|---|
| `scripts/p1_hardening/placebo.py` | `scripts/p1_project_a/placebo.py` | Used by both Hardening A and Hardening E. Project A is the natural home (Project B → Project A is the established import direction). Imports updated to `from . import config`. |
| `scripts/p1_hardening/lightgbm_training.py` | `scripts/p1_project_a/lightgbm_training.py` | Used only by Hardening B (LightGBM cross-check on F-B). Imports updated. |

### 2.2 Runner scripts (moved + edited + renamed)

| Source | Destination | Renamed because |
|---|---|---|
| `run_a_cross_session_placebo.py` | `scripts/p1_project_a/run_hardening_placebo.py` | naturalized name, no longer "experiment A" |
| `run_b_lightgbm.py` | `scripts/p1_project_a/run_hardening_lightgbm.py` | naturalized name |
| `run_c_r4_combined.py` | `scripts/p1_project_b/run_hardening_combined.py` | naturalized name |
| `run_e_within_session_placebo.py` | `scripts/p1_project_b/run_hardening_placebo.py` | naturalized name; symmetric with Project A's |
| `run_d_dataset_noise_floor.py` | `scripts/p1_dataset_analysis/run_noise_floor.py` | dataset-level property, not a Hardening fit |

Each had its `from . import config` rewritten to import from the new package's config (`scripts.p1_project_a.config` or `scripts.p1_project_b.config`). The naturalized names match the existing `run_*` style of each project's package.

### 2.3 Models (15 files, byte-stable verified)

| Source | Destination |
|---|---|
| `scripts/p1_hardening/models/A_B5_placebo_locked.json` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/A_B5_placebo_H1.json` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/B_B1_lgb_default.txt` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/B_B1_lgb_H1_equiv.txt` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/B_B5_lgb_default.txt` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/B_B5_lgb_H1_equiv.txt` | `scripts/p1_project_a/models/` |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W{0,1,2,3,4,4pp}.json` (6 files) | `scripts/p1_project_b/models/` |
| `scripts/p1_hardening/models/E_R-4_W2_real_locked.json` | `scripts/p1_project_b/models/` |
| `scripts/p1_hardening/models/E_R-4_W4_real_locked.json` | `scripts/p1_project_b/models/` |
| `scripts/p1_hardening/models/E_R-4_W4_placebo_locked.json` | `scripts/p1_project_b/models/` |

Note: filenames in the pre-merge package use experiment-letter prefixes (`A_*`, `B_*`, `C_*`, `E_*`) rather than the spec-suggested fold-prefixed names (`F-B_B5_placebo_locked`, etc.). Filenames preserved as-is; routing was by filename prefix (A,B → Project A; C,E → Project B), per spec §2.3 ("all R-4_* files go to Project B, all F-B_* files go to Project A" — applied by experiment letter since that's how the source uses).

### 2.4 Prediction caches (15 files, byte-stable verified)

A,B → `scripts/p1_project_a/cache/`; C,E → `scripts/p1_project_b/cache/`. Every file in `scripts/p1_hardening/cache/` was moved.

### 2.5 Per-experiment result parquets (9 files, byte-stable verified, renamed at destination)

| Source name | Destination name | Reason |
|---|---|---|
| `experiment_a_integrity.parquet` | `hardening_a_integrity.parquet` | match Project A's `<role>_<descriptor>.parquet` style |
| `experiment_a_metrics.parquet` | `hardening_a_metrics.parquet` | as above |
| `experiment_b_metrics.parquet` | `hardening_b_metrics.parquet` | as above |
| `experiment_c_metrics.parquet` | `hardening_c_metrics.parquet` | match Project B style |
| `experiment_e_integrity.parquet` | `hardening_e_integrity.parquet` | as above |
| `experiment_e_metrics.parquet` | `hardening_e_metrics.parquet` | as above |
| `experiment_d_per_cell_sigma.parquet` | `noise_floor_per_cell_sigma.parquet` (in `scripts/p1_dataset_analysis/results/`) | dataset-level naming |
| `experiment_d_same_cell_deltas.parquet` | `noise_floor_same_cell_deltas.parquet` | as above |
| `experiment_d_summary.json` | `noise_floor_summary.json` | as above |

### 2.6 Consolidated parquets (split rather than moved 1:1)

| Source | Destinations | How split |
|---|---|---|
| `scripts/p1_hardening/results/hardening_metrics.parquet` (75 rows) | `scripts/p1_project_a/results/hardening_metrics.parquet` (42 rows: A, B), `scripts/p1_project_b/results/hardening_metrics.parquet` (33 rows: C, E) | by `experiment` column |
| `scripts/p1_hardening/results/fit_inventory.parquet` (15 rows) | `scripts/p1_project_a/results/hardening_fit_inventory.parquet` (6 rows: A, B), `scripts/p1_project_b/results/hardening_fit_inventory.parquet` (9 rows: C, E) | by `experiment` column |

Split tool: `scripts/maintenance/_split_hardening_results.py`. Verified that no rows had an experiment value other than `{A, B, C, E}` (D is editorial; no D rows in either source parquet).

### 2.7 Docs assets (2 files, byte-stable verified)

| Source | Destination |
|---|---|
| `docs/p1_hardening/figures/dataset_noise_floor.png` | `docs/p1_dataset_analysis/figures/dataset_noise_floor.png` |
| `docs/p1_hardening/tables/dataset_noise_floor.md` | `docs/p1_dataset_analysis/tables/dataset_noise_floor.md` |

### 2.8 Files dissolved (deleted; content reproduced elsewhere)

| Source | Where its content lives now |
|---|---|
| `scripts/p1_hardening/__init__.py` | rewritten as deprecation shim per spec §2.4 |
| `scripts/p1_hardening/config.py` | constants absorbed into `scripts/p1_project_a/config.py` (Hardening A, B section) and `scripts/p1_project_b/config.py` (Hardening C, E section) |
| `scripts/p1_hardening/build_results_report.py` | replaced by extension of Project A and Project B `build_results_report.py` (the dissolved sections are now §11/§12 in Project A and §4.5/§4.6 + §8.3 in Project B) |
| `scripts/p1_hardening/run_all.py` | split into `scripts/p1_project_a/run_hardening_all.py` (A, B) and `scripts/p1_project_b/run_hardening_all.py` (C, E) |
| `docs/p1_hardening/results_report.md` | dissolved into `docs/p1_project_a/results_report.md` (Hardening A and B as §11–§12), `docs/p1_project_b/results_report.md` (Hardening C, E as §4.5–§4.6 + §8.3), `docs/p1_dataset_analysis/report.md` (Hardening D as §9) |

The empty `docs/p1_hardening/` directory was deleted. The empty subdirs `scripts/p1_hardening/{models, cache, results}` were deleted. Only `scripts/p1_hardening/__init__.py` (the deprecation shim) remains.

## 3. SHA-256 verification

Pre-move hashes captured to `scripts/maintenance/merge_hardening_pre_hashes.md` (57 entries).

Post-move verification at `scripts/maintenance/merge_hardening_post_hashes.md`:

- **MATCH (byte-stable file unchanged): 41** — all 15 models + 15 prediction caches + 9 result/summary files + 2 docs assets. None corrupted in transit.
- **EDITED-EXPECTED (.py source updated in Stage 1.3): 7** — placebo.py, lightgbm_training.py, and the 5 runner scripts. Pre-move hashes recorded; post-edit hashes are the new source of truth.
- **MISMATCH (byte-stable file silently changed — CRITICAL): 0**. ✓
- **SPLIT: 2** — `hardening_metrics.parquet` and `fit_inventory.parquet`, split by `experiment` column.
- **DISSOLVED: 5** — `__init__.py`, `config.py`, `build_results_report.py`, `run_all.py`, `docs/p1_hardening/results_report.md` (deleted; content reproduced).
- **MISSING/UNMAPPED: 0**. ✓

Verification tool: `scripts/maintenance/_verify_post_hashes.py`.

## 4. Cross-reference updates

Pre-merge grep for `p1_hardening`: 8 files matched.
Post-merge grep: 21 files match — every match is one of:

| Category | Where | Rationale |
|---|---|---|
| Deprecation shim | `scripts/p1_hardening/__init__.py` (4 references) | Required: keeps proposal Rev9 references discoverable instead of hitting an ImportError. |
| Provenance comments / docstrings | 11 files in `scripts/p1_project_a/`, `scripts/p1_project_b/`, `scripts/p1_dataset_analysis/` (1–2 references each) | Each moved/dissolved file has one "Originally `scripts/p1_hardening/...`; merged on 2026-04-28" line. Useful for future readers; kept intentionally. |
| Doc provenance | `docs/p1_project_b/results_report.md` (3 references) | The TL;DR header sentence and §8.3 inventory note explain where the content used to live. Kept intentionally. |
| Audit / maintenance | `scripts/maintenance/{merge_hardening_*.md, _verify_post_hashes.py, _split_hardening_results.py, _hardening_*_appendix.md}` | The merge log itself; never "updated" — these files document the pre-merge state. |
| Locked historical | `docs/proposal_rev9.md` (4 references) | Proposal Rev9 §3.7, §7 "Frozen artifacts", §11 "agent's suggested wording in `docs/p1_hardening/results_report.md` §8". Locked-by-revision per spec §3.3. The §8 wording referenced from Rev9 §11 now lives verbatim in `docs/p1_project_b/results_report.md` §4.6.4 (Hardening E). |

No leftover live code or doc reference to `scripts/p1_hardening/` outside the deprecation shim, the provenance comments, the locked proposal, and the maintenance/audit files.

## 5. Smoke tests

```
$ PYTHONPATH=. .venv/Scripts/python.exe -c "import scripts.p1_project_a; import scripts.p1_project_b; import scripts.p1_dataset_analysis"
packages import OK
```

```
$ PYTHONPATH=. .venv/Scripts/python.exe -c "
import scripts.p1_project_a.run_hardening_placebo
import scripts.p1_project_a.run_hardening_lightgbm
import scripts.p1_project_a.run_hardening_all
import scripts.p1_project_a.placebo
import scripts.p1_project_a.lightgbm_training
import scripts.p1_project_b.run_hardening_combined
import scripts.p1_project_b.run_hardening_placebo
import scripts.p1_project_b.run_hardening_all
import scripts.p1_dataset_analysis.run_noise_floor
"
all hardening modules import OK
```

```
$ PYTHONPATH=. .venv/Scripts/python.exe -c "import scripts.p1_hardening" 2>&1
ImportError: scripts.p1_hardening was merged into p1_project_a and p1_project_b. See the deprecation notice in scripts/p1_hardening/__init__.py for the new locations.
```

The deprecation shim raises ImportError as designed.

## 6. Open issues / decisions

1. **Source filenames preserved.** The pre-merge models use experiment-letter prefixes (`A_B5_placebo_locked.json`, `C_R-4_buffer_H1_W0.json`, etc.). The spec example used fold-prefixed names (`F-B_B5_placebo_locked.json`). Kept the actual filenames as-is to preserve byte-stability of the SHA-256 hashes and avoid disrupting any future code that locates them by filename. Routing decided by the experiment-letter prefix per `experiment ∈ {A, B}` → Project A and `experiment ∈ {C, E}` → Project B; D goes to dataset analysis.

2. **`r4_diagnostic.md` does not exist as a separate file.** The spec said "update `docs/p1_project_b/r4_diagnostic.md`" but the R-4 diagnostic was already merged into `docs/p1_project_b/results_report.md` as §4. Adapted the spec by adding Hardening C as §4.5 and Hardening E as §4.6 inside the same file, then updating §4.7 (renumbered from §4.4) to be the four-diagnostic combined verdict. The narrative remains internally consistent: §4.2 buffer alone, §4.3 H1 alone, §4.5 H1+buffer combined, §4.6 placebo, §4.7 four-diagnostic verdict.

3. **`p0_analysis/config.py` import in the moved noise-floor runner.** `scripts/p1_dataset_analysis/run_noise_floor.py` imports `from scripts.p0_analysis import config as p0_config` for the cell-size and same-map-pair constants. This dependency was already in the pre-merge code; preserved.

4. **`scripts/p1_dataset_analysis/__init__.py` is empty.** No project-level config module exists. The moved `run_noise_floor.py` defines its own `RESULTS_DIR`, `FIGURES_DIR`, `TABLES_DIR` module-level constants (mirrored from the deleted `scripts.p1_hardening.config` shape).

5. **Project B's `run_all.py` now also runs `run_r4_diagnostic.py` and `run_hardening_all.py`** — previously the runner only chained modeling → XAI → report. The spec said "Each project's existing `run_all.py` (if present) should call its own `run_hardening_all.py` after the main run." Added.

6. **Project A's `run_all.py`** likewise extended with `run_hardening_all` between `run_robustness` and `build_results_report`.

7. **`scripts/p1_dataset_analysis/run_all.py`** extended with `run_noise_floor` as the third stage (after build + validate).

8. **No fits re-run.** Stage 1 was purely organizational. Existing models retained byte-stable SHA-256.

## 7. Outcome

Stage 1 of the merge is complete:

1. ✓ `scripts/p1_hardening/` contains only the deprecation shim. All subdirectories deleted.
2. ✓ `docs/p1_hardening/` directory deleted entirely.
3. ✓ Every model and cached prediction file is at its new home with SHA-256 unchanged.
4. ✓ `docs/p1_project_a/results_report.md` includes Hardening A (§11) and B (§12).
5. ✓ `docs/p1_project_b/results_report.md` includes Hardening C (§4.5), E (§4.6), the four-diagnostic combined verdict (§4.7), and the C+E inventory (§8.3).
6. ✓ `docs/p1_dataset_analysis/report.md` includes Hardening D (§9).
7. ✓ Every cross-reference outside the deprecation shim and Rev9 has been updated or annotated as provenance/audit.
8. ✓ `scripts/maintenance/merge_hardening_log.md` (this file) exists.
9. ✓ Top-level package imports succeed.

Stage 2 (regenerate `docs/unified_report.md` against the post-merge layout, using Rev9 as the framing source) is the next step.
