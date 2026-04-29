# Pre-merge inventory — `scripts/p1_hardening/` and `docs/p1_hardening/`

Snapshot taken 2026-04-28, before Stage 1 of the hardening merge.

## scripts/p1_hardening/

```
scripts/p1_hardening/__init__.py
scripts/p1_hardening/build_results_report.py
scripts/p1_hardening/cache/predictions_A_B5_placebo_H1.parquet
scripts/p1_hardening/cache/predictions_A_B5_placebo_locked.parquet
scripts/p1_hardening/cache/predictions_B_B1_lgb_H1_equiv.parquet
scripts/p1_hardening/cache/predictions_B_B1_lgb_default.parquet
scripts/p1_hardening/cache/predictions_B_B5_lgb_H1_equiv.parquet
scripts/p1_hardening/cache/predictions_B_B5_lgb_default.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W0.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W1.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W2.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W3.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4.parquet
scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4pp.parquet
scripts/p1_hardening/cache/predictions_E_R-4_W2_real_locked.parquet
scripts/p1_hardening/cache/predictions_E_R-4_W4_placebo_locked.parquet
scripts/p1_hardening/cache/predictions_E_R-4_W4_real_locked.parquet
scripts/p1_hardening/config.py
scripts/p1_hardening/lightgbm_training.py
scripts/p1_hardening/models/A_B5_placebo_H1.json
scripts/p1_hardening/models/A_B5_placebo_locked.json
scripts/p1_hardening/models/B_B1_lgb_H1_equiv.txt
scripts/p1_hardening/models/B_B1_lgb_default.txt
scripts/p1_hardening/models/B_B5_lgb_H1_equiv.txt
scripts/p1_hardening/models/B_B5_lgb_default.txt
scripts/p1_hardening/models/C_R-4_buffer_H1_W0.json
scripts/p1_hardening/models/C_R-4_buffer_H1_W1.json
scripts/p1_hardening/models/C_R-4_buffer_H1_W2.json
scripts/p1_hardening/models/C_R-4_buffer_H1_W3.json
scripts/p1_hardening/models/C_R-4_buffer_H1_W4.json
scripts/p1_hardening/models/C_R-4_buffer_H1_W4pp.json
scripts/p1_hardening/models/E_R-4_W2_real_locked.json
scripts/p1_hardening/models/E_R-4_W4_placebo_locked.json
scripts/p1_hardening/models/E_R-4_W4_real_locked.json
scripts/p1_hardening/placebo.py
scripts/p1_hardening/results/experiment_a_integrity.parquet
scripts/p1_hardening/results/experiment_a_metrics.parquet
scripts/p1_hardening/results/experiment_b_metrics.parquet
scripts/p1_hardening/results/experiment_c_metrics.parquet
scripts/p1_hardening/results/experiment_d_per_cell_sigma.parquet
scripts/p1_hardening/results/experiment_d_same_cell_deltas.parquet
scripts/p1_hardening/results/experiment_d_summary.json
scripts/p1_hardening/results/experiment_e_integrity.parquet
scripts/p1_hardening/results/experiment_e_metrics.parquet
scripts/p1_hardening/results/fit_inventory.parquet
scripts/p1_hardening/results/hardening_metrics.parquet
scripts/p1_hardening/run_a_cross_session_placebo.py
scripts/p1_hardening/run_all.py
scripts/p1_hardening/run_b_lightgbm.py
scripts/p1_hardening/run_c_r4_combined.py
scripts/p1_hardening/run_d_dataset_noise_floor.py
scripts/p1_hardening/run_e_within_session_placebo.py
```

## docs/p1_hardening/

```
docs/p1_hardening/figures/dataset_noise_floor.png
docs/p1_hardening/results_report.md
docs/p1_hardening/tables/dataset_noise_floor.md
```

## Per-file destination

| Source | Destination | Reason |
|---|---|---|
| `scripts/p1_hardening/__init__.py` | rewrite as deprecation shim in place | per spec §2.4 |
| `scripts/p1_hardening/config.py` | dissolved — content split into a/b runners; no consolidated replacement | per spec §3 |
| `scripts/p1_hardening/placebo.py` | `scripts/p1_project_a/placebo.py` | helper used by both A and E |
| `scripts/p1_hardening/lightgbm_training.py` | `scripts/p1_project_a/lightgbm_training.py` | LightGBM helper for B |
| `scripts/p1_hardening/build_results_report.py` | dissolved — replaced by extension of A and B `build_results_report.py` | per spec §3 |
| `scripts/p1_hardening/run_a_cross_session_placebo.py` | `scripts/p1_project_a/run_hardening_placebo.py` | per spec §2.3 |
| `scripts/p1_hardening/run_b_lightgbm.py` | `scripts/p1_project_a/run_hardening_lightgbm.py` | per spec §2.3 |
| `scripts/p1_hardening/run_c_r4_combined.py` | `scripts/p1_project_b/run_hardening_combined.py` | per spec §2.3 |
| `scripts/p1_hardening/run_d_dataset_noise_floor.py` | `scripts/p1_dataset_analysis/run_noise_floor.py` | per spec §2.1 (D dataset-level) |
| `scripts/p1_hardening/run_e_within_session_placebo.py` | `scripts/p1_project_b/run_hardening_placebo.py` | per spec §2.3 |
| `scripts/p1_hardening/run_all.py` | split into `scripts/p1_project_a/run_hardening_all.py` + `scripts/p1_project_b/run_hardening_all.py` | per spec §2.3 |
| `scripts/p1_hardening/models/A_*.json` | `scripts/p1_project_a/models/` | F-B fits |
| `scripts/p1_hardening/models/B_*.txt` | `scripts/p1_project_a/models/` | F-B LightGBM fits |
| `scripts/p1_hardening/models/C_R-4_buffer_*.json` | `scripts/p1_project_b/models/` | R-4 buffer+H1 fits |
| `scripts/p1_hardening/models/E_R-4_*.json` | `scripts/p1_project_b/models/` | R-4 placebo fits |
| `scripts/p1_hardening/cache/predictions_A_*.parquet` | `scripts/p1_project_a/cache/` | F-B prediction caches |
| `scripts/p1_hardening/cache/predictions_B_*.parquet` | `scripts/p1_project_a/cache/` | F-B prediction caches |
| `scripts/p1_hardening/cache/predictions_C_*.parquet` | `scripts/p1_project_b/cache/` | R-4 prediction caches |
| `scripts/p1_hardening/cache/predictions_E_*.parquet` | `scripts/p1_project_b/cache/` | R-4 prediction caches |
| `scripts/p1_hardening/results/experiment_a_*.parquet` | `scripts/p1_project_a/results/` | A metrics |
| `scripts/p1_hardening/results/experiment_b_*.parquet` | `scripts/p1_project_a/results/` | B metrics |
| `scripts/p1_hardening/results/experiment_c_*.parquet` | `scripts/p1_project_b/results/` | C metrics |
| `scripts/p1_hardening/results/experiment_d_*.parquet` | `scripts/p1_dataset_analysis/results/` | D dataset-level outputs |
| `scripts/p1_hardening/results/experiment_d_summary.json` | `scripts/p1_dataset_analysis/results/` | D dataset-level outputs |
| `scripts/p1_hardening/results/experiment_e_*.parquet` | `scripts/p1_project_b/results/` | E metrics |
| `scripts/p1_hardening/results/hardening_metrics.parquet` | split → `scripts/p1_project_a/results/hardening_metrics.parquet` (A,B rows) and `scripts/p1_project_b/results/hardening_metrics.parquet` (C,E rows) | per spec §2.3 |
| `scripts/p1_hardening/results/fit_inventory.parquet` | split → `scripts/p1_project_a/results/hardening_fit_inventory.parquet` (A,B rows) and `scripts/p1_project_b/results/hardening_fit_inventory.parquet` (C,E rows) | per spec §2.3 |
| `docs/p1_hardening/figures/dataset_noise_floor.png` | `docs/p1_dataset_analysis/figures/` | per spec §2.3 |
| `docs/p1_hardening/tables/dataset_noise_floor.md` | `docs/p1_dataset_analysis/tables/` | per spec §2.3 |
| `docs/p1_hardening/results_report.md` | dissolved into `docs/p1_project_a/results_report.md`, `docs/p1_project_b/results_report.md`, `docs/p1_dataset_analysis/report.md` | per spec §3.4 |
