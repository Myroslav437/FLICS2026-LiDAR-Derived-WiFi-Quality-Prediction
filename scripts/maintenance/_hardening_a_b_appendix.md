
## 11. Hardening A — Cross-session LiDAR placebo on F-B

Pre-empts the "your features were just bad / your model couldn't learn from them" reviewer objection. Originally reported as Experiment A in the dissolved hardening package; merged into Project A on 2026-04-28 because the placebo refits B5 on F-B and reuses Project A's caches and fold construction.

**Hypothesis tested.** A B5 model trained with the 19 ego-frame LiDAR columns block-shuffled within each training session (preserving each LiDAR feature's marginal distribution and the joint distribution among LiDAR features, but breaking row-level alignment with telemetry / position / AP-relative / target) achieves Δ_LiDAR comparable to the real-LiDAR fit. If true, the real LiDAR features were not contributing a row-aligned signal.

### 11.1 Methodology

- Test set: F-B test rows (15.03), unmodified.

- Training set: 24.03 + 25.02 anomaly-filtered rows. Per session, one permutation π is sampled (seed = SEED + session_offset) and applied as a *block* to all 19 LiDAR columns simultaneously, preserving the joint distribution among LiDAR features.

- 19 LiDAR columns shuffled: `mean_dist_mm`, `dist_p90_mm`, `clutter_frac`, `openness_frac`, `mean_front_mm`, `mean_dist_sector_{1..7}_mm`, `clutter_frac_sector_{1..7}`. `clutter_frac_toward_AP` and `is_AP_in_FOV` are AP-relative and NOT shuffled.

- Validation split: chronological-per-session 90/10 on the post-shuffle pool (matches Project A).

- B1 and B5 (real) predictions reused from Project A's main cache (locked) and diagnostic cache (H1).


### 11.2 Results

RMSE (dB) per (descriptor, config, variant, stratum), bootstrap 95% CI in brackets.

| Descriptor | Config | Variant | Stratum | RMSE | 95% CI |
|---|---|---|---|---:|---|
| real | locked | B1 | overall | 7.935 | [7.91, 7.96] |
| real | locked | B1 | in_fov | 8.014 | [7.98, 8.05] |
| real | locked | B1 | out_of_fov | 7.823 | [7.79, 7.85] |
| real | locked | B5 | overall | 9.304 | [9.28, 9.33] |
| real | locked | B5 | in_fov | 8.529 | [8.49, 8.56] |
| real | locked | B5 | out_of_fov | 10.278 | [10.24, 10.32] |
| real | H1 | B1 | overall | 8.167 | [8.15, 8.19] |
| real | H1 | B1 | in_fov | 7.856 | [7.82, 7.88] |
| real | H1 | B1 | out_of_fov | 8.578 | [8.55, 8.61] |
| real | H1 | B5 | overall | 8.212 | [8.19, 8.23] |
| real | H1 | B5 | in_fov | 8.146 | [8.11, 8.18] |
| real | H1 | B5 | out_of_fov | 8.303 | [8.27, 8.33] |
| placebo | locked | B5 | overall | 8.903 | [8.88, 8.92] |
| placebo | locked | B5 | in_fov | 8.381 | [8.35, 8.41] |
| placebo | locked | B5 | out_of_fov | 9.576 | [9.54, 9.61] |
| placebo | H1 | B5 | overall | 8.397 | [8.37, 8.42] |
| placebo | H1 | B5 | in_fov | 7.846 | [7.81, 7.88] |
| placebo | H1 | B5 | out_of_fov | 9.102 | [9.07, 9.14] |

### 11.3 Δ_LiDAR comparison on F-B in-FOV (the diagnostic stratum)

| Config | Δ_LiDAR (real) | Δ_LiDAR (placebo) | Δ_real − Δ_placebo |
|---|---:|---:|---:|
| locked | -0.52 dB | -0.37 dB | -0.15 dB |
| H1 | -0.29 dB | +0.01 dB | -0.30 dB |

Full Δ_LiDAR by stratum:

| Config | Stratum | Δ_LiDAR (real) | Δ_LiDAR (placebo) |
|---|---|---:|---:|
| locked | overall | -1.37 dB | -0.97 dB |
| locked | in_fov | -0.52 dB | -0.37 dB |
| locked | out_of_fov | -2.45 dB | -1.75 dB |
| H1 | overall | -0.05 dB | -0.23 dB |
| H1 | in_fov | -0.29 dB | +0.01 dB |
| H1 | out_of_fov | +0.27 dB | -0.52 dB |

### 11.4 Verdict

Tolerance: |Δ_real − Δ_placebo| < 0.5 dB ⇒ placebo confirms negative. Δ_real − Δ_placebo > 0.5 dB ⇒ partial rescue.

- **locked**: Δ_real = -0.52 dB, Δ_placebo = -0.37 dB ⇒ **PLACEBO_CONFIRMS_NEGATIVE**.
- **H1**: Δ_real = -0.29 dB, Δ_placebo = +0.01 dB ⇒ **PLACEBO_CONFIRMS_NEGATIVE**.

### 11.5 Reproducibility (Hardening A)

- 2 model files: `scripts/p1_project_a/models/A_B5_placebo_{locked, H1}.json`.
- Predictions: `scripts/p1_project_a/cache/predictions_A_B5_placebo_{locked, H1}.parquet`.
- Metrics: `scripts/p1_project_a/results/hardening_a_metrics.parquet`.
- Integrity report: `scripts/p1_project_a/results/hardening_a_integrity.parquet`.
- Re-run: `python -m scripts.p1_project_a.run_hardening_placebo` (or via `run_hardening_all`).

## 12. Hardening B — LightGBM cross-check on F-B

Pre-empts the "your conclusion is XGBoost-specific" reviewer objection. Originally reported as Experiment B in the dissolved hardening package; merged into Project A on 2026-04-28 because the cross-check refits the F-B B1 vs B5 comparison and reuses Project A's fold construction and reference XGBoost predictions.

**Hypothesis tested.** A LightGBM model with default hyperparameters reaches the same conclusion as XGBoost on F-B: Δ_LiDAR (in-FOV) does not clear the +1 dB practical-relevance threshold.

### 12.1 Methodology

- F-B fold construction, anomaly filter, validation split, FOV stratification: identical to Project A.

- LightGBM defaults — `learning_rate=0.05`, `num_leaves=31`, `min_data_in_leaf=20`, `seed=20260427`, `deterministic=True`. H1-equiv replaces `num_leaves=31` with `num_leaves=15` (≈ max_depth=4).

- Same `num_estimators=2000`, `early_stopping_rounds=100` as XGBoost.

- 4 new LightGBM fits ({B1, B5} × {default, H1-equiv}). XGBoost rows reuse Project A caches.


### 12.2 RMSE (dB) per (framework, config, variant, stratum)

| Framework | Config | Variant | Stratum | RMSE |
|---|---|---|---|---:|
| xgboost | locked | B1 | overall | 7.935 |
| xgboost | locked | B1 | in_fov | 8.014 |
| xgboost | locked | B1 | out_of_fov | 7.823 |
| xgboost | locked | B5 | overall | 9.304 |
| xgboost | locked | B5 | in_fov | 8.529 |
| xgboost | locked | B5 | out_of_fov | 10.278 |
| xgboost | H1 | B1 | overall | 8.167 |
| xgboost | H1 | B1 | in_fov | 7.856 |
| xgboost | H1 | B1 | out_of_fov | 8.578 |
| xgboost | H1 | B5 | overall | 8.212 |
| xgboost | H1 | B5 | in_fov | 8.146 |
| xgboost | H1 | B5 | out_of_fov | 8.303 |
| lightgbm | locked | B1 | overall | 8.699 |
| lightgbm | locked | B1 | in_fov | 8.276 |
| lightgbm | locked | B1 | out_of_fov | 9.251 |
| lightgbm | locked | B5 | overall | 9.183 |
| lightgbm | locked | B5 | in_fov | 8.857 |
| lightgbm | locked | B5 | out_of_fov | 9.615 |
| lightgbm | H1 | B1 | overall | 8.380 |
| lightgbm | H1 | B1 | in_fov | 7.758 |
| lightgbm | H1 | B1 | out_of_fov | 9.170 |
| lightgbm | H1 | B5 | overall | 8.740 |
| lightgbm | H1 | B5 | in_fov | 8.685 |
| lightgbm | H1 | B5 | out_of_fov | 8.814 |

### 12.3 Δ_LiDAR by framework × config × stratum

| Framework | Config | Stratum | Δ_LiDAR (dB) | clears 1 dB (in-FOV)? |
|---|---|---|---:|:---:|
| xgboost | locked | overall | -1.37 | — |
| xgboost | locked | in_fov | -0.52 | ✗ |
| xgboost | locked | out_of_fov | -2.45 | — |
| xgboost | H1 | overall | -0.05 | — |
| xgboost | H1 | in_fov | -0.29 | ✗ |
| xgboost | H1 | out_of_fov | +0.27 | — |
| lightgbm | locked | overall | -0.48 | — |
| lightgbm | locked | in_fov | -0.58 | ✗ |
| lightgbm | locked | out_of_fov | -0.36 | — |
| lightgbm | H1 | overall | -0.36 | — |
| lightgbm | H1 | in_fov | -0.93 | ✗ |
| lightgbm | H1 | out_of_fov | +0.36 | — |

### 12.4 Verdict

- LightGBM Δ_LiDAR (in-FOV, default) = -0.58 dB.
- LightGBM Δ_LiDAR (in-FOV, H1-equiv) = -0.93 dB.
- **NEGATIVE_RESULT_NOT_FRAMEWORK_SPECIFIC**: LightGBM does NOT clear the +1 dB threshold.

### 12.5 Reproducibility (Hardening B)

- 4 model files: `scripts/p1_project_a/models/B_{B1, B5}_lgb_{default, H1_equiv}.txt`.
- Predictions: `scripts/p1_project_a/cache/predictions_B_{B1, B5}_lgb_{default, H1_equiv}.parquet`.
- Metrics: `scripts/p1_project_a/results/hardening_b_metrics.parquet`.
- Re-run: `python -m scripts.p1_project_a.run_hardening_lightgbm` (or via `run_hardening_all`).

### 12.6 Combined Hardening A+B fit inventory (6 new fits)

| Experiment | Fold | Variant | Descriptor | Framework | Config | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| A | F-B | B5 | B5_placebo_locked | xgboost | locked | 31 | 337390 | 37487 | 232279 | 1.8 | `dade19df5f1e23d4…` |
| A | F-B | B5 | B5_placebo_H1 | xgboost | H1 | 49 | 337390 | 37487 | 232279 | 1.8 | `c4f1e7f7c0f8c2c2…` |
| B | F-B | B1 | B1_lgb_default | lightgbm | default | 29 | 337390 | 37487 | 232279 | 0.5 | `39f72d43fed9e3bf…` |
| B | F-B | B5 | B5_lgb_default | lightgbm | default | 37 | 337390 | 37487 | 232279 | 1.7 | `5d7bff046d58de32…` |
| B | F-B | B1 | B1_lgb_H1_equiv | lightgbm | H1_equiv | 44 | 337390 | 37487 | 232279 | 0.5 | `526f093c05832450…` |
| B | F-B | B5 | B5_lgb_H1_equiv | lightgbm | H1_equiv | 38 | 337390 | 37487 | 232279 | 1.4 | `28276de3967af2b9…` |

Total wall-clock for Hardening A+B fits: 7.7 s.

Combined orchestrator: `python -m scripts.p1_project_a.run_hardening_all` writes the consolidated `scripts/p1_project_a/results/hardening_metrics.parquet` (rows = experiments A and B) and `scripts/p1_project_a/results/hardening_fit_inventory.parquet`.
