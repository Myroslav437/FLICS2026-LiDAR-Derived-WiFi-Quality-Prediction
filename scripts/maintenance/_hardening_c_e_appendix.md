
### 8.3 Hardening C + E model SHA-256 inventory (9 new fits)

Originally `scripts/p1_hardening/results/fit_inventory.parquet` (15 fits combined across A, B, C, E); split into Project A's `hardening_fit_inventory.parquet` (A, B = 6 fits) and Project B's `hardening_fit_inventory.parquet` (C, E = 9 fits) on 2026-04-28.

| Experiment | Fold | Variant | Descriptor | Framework | Config | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| C | R-4_buffer | W0 | R-4_buffer_H1_W0 | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 4.5 | `ea239acbdcfc3adc…` |
| C | R-4_buffer | W1 | R-4_buffer_H1_W1 | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 5.9 | `7497344ea17c54ec…` |
| C | R-4_buffer | W2 | R-4_buffer_H1_W2 | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 8.6 | `ecf292ad7c4f9659…` |
| C | R-4_buffer | W3 | R-4_buffer_H1_W3 | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 10.3 | `3827f34bccc398b1…` |
| C | R-4_buffer | W4 | R-4_buffer_H1_W4 | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 13.8 | `d91c0479a54ca1a1…` |
| C | R-4_buffer | W4pp | R-4_buffer_H1_W4pp | xgboost | H1 | 1999 | 183944 | 20438 | 24593 | 10.8 | `dec53224a148f563…` |
| E | R-4 | W2 | R-4_W2_real_locked | xgboost | locked | 1999 | 186917 | 20769 | 24593 | 11.5 | `c32c5db1afa12bd8…` |
| E | R-4 | W4 | R-4_W4_real_locked | xgboost | locked | 1999 | 186917 | 20769 | 24593 | 17.9 | `89148526e3737b53…` |
| E | R-4 | W4 | R-4_W4_placebo_locked | xgboost | locked | 1999 | 186917 | 20769 | 24593 | 18.5 | `670b02dfed6d8277…` |

Total wall-clock for Hardening C + E fits: 101.8 s.

Output paths (rooted at the project):

- Models: `scripts/p1_project_b/models/{C_R-4_buffer_H1_*.json, E_R-4_*.json}`.
- Predictions: `scripts/p1_project_b/cache/{predictions_C_*.parquet, predictions_E_*.parquet}`.
- Per-experiment metrics: `scripts/p1_project_b/results/hardening_c_metrics.parquet`, `scripts/p1_project_b/results/hardening_e_metrics.parquet`.
- Integrity report (Hardening E placebo): `scripts/p1_project_b/results/hardening_e_integrity.parquet`.
- Consolidated metrics + inventory: `scripts/p1_project_b/results/hardening_metrics.parquet` (rows = experiments C and E), `scripts/p1_project_b/results/hardening_fit_inventory.parquet`.
