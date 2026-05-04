# Phase 1 / Project A — results report

Project A is the cross-session leave-one-run-out (LORO) ablation. Project B (within-session leave-region-out) is a separate work line and not reported here.

## 0. TL;DR

**Feature stack: leakage-fixed.** This report uses the leakage-fixed feature stack: 3 telemetry features (`speed_mps`, `turn_rate`, `momentary_current_consumption`) plus AP-relative geometry and LiDAR. Five features from the original locked feature stack were removed post-hoc as either router-side (target leakage), within-session-only (deployment leakage), or constant sentinel (no information). See `MIGRATION_LOG.md` for the full audit trail. (`feature_stack_version = "leakage_fixed_v1"`).

- **B5 vs B1 Δ_RMSE per fold (overall / in-FOV / out-of-FOV)**:
  - F-A: overall +0.06 dB [+0.06, +0.07]; in-FOV +0.55 dB [+0.54, +0.56]; out-of-FOV -0.44 dB [-0.45, -0.43]
  - F-B: overall -0.69 dB [-0.71, -0.67]; in-FOV -0.19 dB [-0.21, -0.16]; out-of-FOV -1.35 dB [-1.38, -1.32]
  - F-C: overall +0.71 dB [+0.69, +0.72]; in-FOV +1.20 dB [+1.18, +1.23]; out-of-FOV +0.22 dB [+0.20, +0.24]
- **Disambiguation verdict**: LiDAR removable.
- **RQ4 verdict**: residual session effect is **negligible** (max same-map fraction 4.80%).
- **All 28 model fits succeeded**: yes (n_fit rows in inventory = 28).
- **Project A status**: PIVOT TO PROJECT B.
  - Pivot reason: F-B in-FOV Δ_LiDAR = -0.19 dB ≤ 1 dB threshold. This is the diagnostic fold; small Δ_LiDAR there indicates LiDAR features are not adding physical structure beyond AP-geometry. Recommend Project B (within-session deep dive).
- **Hyperparameter robustness diagnostic** (§10): PIVOT_JUSTIFIED (locked F-B in-FOV Δ_LiDAR = -0.19 dB; best alternative `H1` = +0.10 dB).

## 1. Inputs and provenance

- Dataset: `data/phase1/dataset.parquet` (SHA-256 `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270`).
  - Expected SHA-256 from `dataset.sha256`: `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270` (match).
- Phase 0 artifacts referenced (read-only): `analysis/p0/artifacts/{ap_coords.json, lidar_fov.json, anomaly_threshold.json}`.
- Total non-anomaly rows used: 607,156.
- LORO fold row counts (post-anomaly-filter; F-C training sessions are decimated by k=7):
- **F-A**: train n=403876 (15.03.2026: 209051, 25.02.2026: 194825); val n=44875 (15.03.2026: 23228, 25.02.2026: 21647); test n=158405 (test session 24.03.2026)
- **F-B**: train n=337390 (24.03.2026: 142565, 25.02.2026: 194825); val n=37487 (24.03.2026: 15840, 25.02.2026: 21647); test n=232279 (test session 15.03.2026)
- **F-C**: train n=50232 (15.03.2026: 29865, 24.03.2026: 20367); val n=5581 (15.03.2026: 3318, 24.03.2026: 2263); test n=216472 (test session 25.02.2026)

## 2. LORO ablation results

### 2.1 Headline ablation table (3 folds × 3 strata × 6 variants)

# LORO ablation — RMSE (dB) per fold × variant × stratum

Bootstrap 95% CI on RMSE in brackets (B = 1000).

| Variant | Stratum | F-A | F-B | F-C |
|---|---|---|---|---|
| B0 | overall | 7.22 [7.19, 7.25] | 8.18 [8.16, 8.21] | 7.56 [7.54, 7.58] |
| B0 | in_fov | 6.80 [6.76, 6.85] | 8.13 [8.10, 8.17] | 5.85 [5.82, 5.87] |
| B0 | out_of_fov | 7.92 [7.88, 7.96] | 8.25 [8.22, 8.28] | 9.31 [9.27, 9.34] |
| B1 | overall | 8.45 [8.42, 8.48] | 7.93 [7.91, 7.96] | 9.71 [9.68, 9.73] |
| B1 | in_fov | 6.71 [6.68, 6.74] | 8.01 [7.98, 8.05] | 8.87 [8.84, 8.91] |
| B1 | out_of_fov | 10.93 [10.88, 10.98] | 7.82 [7.79, 7.85] | 10.69 [10.65, 10.72] |
| B2 | overall | 8.63 [8.60, 8.66] | 8.00 [7.98, 8.02] | 9.09 [9.07, 9.11] |
| B2 | in_fov | 7.03 [7.00, 7.06] | 8.25 [8.22, 8.28] | 8.23 [8.20, 8.26] |
| B2 | out_of_fov | 10.97 [10.92, 11.02] | 7.63 [7.60, 7.66] | 10.10 [10.06, 10.13] |
| B3 | overall | 8.80 [8.77, 8.83] | 8.13 [8.11, 8.15] | 8.88 [8.86, 8.90] |
| B3 | in_fov | 7.16 [7.13, 7.19] | 8.18 [8.15, 8.21] | 7.32 [7.29, 7.35] |
| B3 | out_of_fov | 11.20 [11.15, 11.25] | 8.07 [8.04, 8.10] | 10.56 [10.53, 10.59] |
| B4 | overall | 8.55 [8.52, 8.58] | 8.49 [8.46, 8.51] | 9.01 [8.99, 9.03] |
| B4 | in_fov | 6.50 [6.47, 6.54] | 8.18 [8.15, 8.21] | 7.67 [7.64, 7.70] |
| B4 | out_of_fov | 11.37 [11.32, 11.42] | 8.88 [8.85, 8.92] | 10.49 [10.46, 10.53] |
| B5 | overall | 8.39 [8.35, 8.42] | 8.62 [8.60, 8.65] | 9.00 [8.98, 9.02] |
| B5 | in_fov | 6.16 [6.13, 6.19] | 8.20 [8.17, 8.23] | 7.67 [7.64, 7.70] |
| B5 | out_of_fov | 11.37 [11.32, 11.42] | 9.17 [9.14, 9.21] | 10.47 [10.44, 10.50] |


### 2.2 Δ_LiDAR per fold per stratum

# Δ_LiDAR — RMSE(B1) − RMSE(B5) (dB)

Positive => LiDAR features improve over AP-geometry-only.

| Fold | Stratum | n | Δ_LiDAR | 95% CI |
|---|---|---:|---:|---|
| F-A | overall | 158405 | 0.065 | [0.06, 0.07] |
| F-A | in_fov | 102354 | 0.548 | [0.54, 0.56] |
| F-A | out_of_fov | 56051 | -0.439 | [-0.45, -0.43] |
| F-B | overall | 232279 | -0.689 | [-0.71, -0.67] |
| F-B | in_fov | 134692 | -0.187 | [-0.21, -0.16] |
| F-B | out_of_fov | 97587 | -1.352 | [-1.38, -1.32] |
| F-C | overall | 216472 | 0.706 | [0.69, 0.72] |
| F-C | in_fov | 121831 | 1.201 | [1.18, 1.23] |
| F-C | out_of_fov | 94641 | 0.219 | [0.20, 0.24] |


### 2.3 Δ_angle per fold per stratum

# Δ_angle — RMSE(B0) − RMSE(B1) (dB)

Positive => angle-to-AP improves on distance alone.

| Fold | Stratum | n | Δ_angle | 95% CI |
|---|---|---:|---:|---|
| F-A | overall | 158405 | -1.231 | [-1.26, -1.21] |
| F-A | in_fov | 102354 | 0.092 | [0.06, 0.13] |
| F-A | out_of_fov | 56051 | -3.012 | [-3.05, -2.98] |
| F-B | overall | 232279 | 0.249 | [0.24, 0.26] |
| F-B | in_fov | 134692 | 0.121 | [0.10, 0.14] |
| F-B | out_of_fov | 97587 | 0.428 | [0.41, 0.45] |
| F-C | overall | 216472 | -2.151 | [-2.18, -2.12] |
| F-C | in_fov | 121831 | -3.025 | [-3.06, -2.99] |
| F-C | out_of_fov | 94641 | -1.382 | [-1.42, -1.34] |


### 2.4 Δ_FOV per fold (B5)

# Δ_FOV — B5 RMSE: out-of-FOV minus in-FOV (dB)

Positive => out-of-FOV is harder.

| Fold | RMSE in-FOV | RMSE out-of-FOV | Δ_FOV |
|---|---:|---:|---:|
| F-A | 6.164 | 11.372 | 5.208 |
| F-B | 8.202 | 9.175 | 0.973 |
| F-C | 7.671 | 10.470 | 2.798 |


### 2.5 Per-fold narrative

#### F-A
- Δ_LiDAR overall = +0.06 dB [95% CI +0.06, +0.07].
- Δ_LiDAR in-FOV = +0.55 dB.
- Δ_LiDAR out-of-FOV = -0.44 dB.
- Δ_angle overall = -1.23 dB.

#### F-B
- Δ_LiDAR overall = -0.69 dB [95% CI -0.71, -0.67].
- Δ_LiDAR in-FOV = -0.19 dB.
- Δ_LiDAR out-of-FOV = -1.35 dB.
- Δ_angle overall = +0.25 dB.
- F-B is the diagnostic fold; in-FOV Δ_LiDAR threshold check: **fails (≤1 dB)** — flag for Project B pivot.

#### F-C
- Δ_LiDAR overall = +0.71 dB [95% CI +0.69, +0.72].
- Δ_LiDAR in-FOV = +1.20 dB.
- Δ_LiDAR out-of-FOV = +0.22 dB.
- Δ_angle overall = -2.15 dB.
- F-C is the cross-frame test (25.02 in Map B). Training cadence matched to 4.4 Hz via k=7 decimation. A small or positive Δ_LiDAR here is evidence that the LiDAR-derived structure transfers across map frames.


## 3. Disambiguation experiment (B5 / B5' / B5'')

# Disambiguation — overall RMSE (dB) for B5 / B5' / B5''

B5 = full 32-feature model. B5' = telemetry + AP-relative (no LiDAR). B5'' = telemetry + LiDAR (no AP-relative).

| Variant | F-A | F-B | F-C |
|---|---|---|---|
| B5 | 8.39 [8.35, 8.42] | 8.62 [8.60, 8.65] | 9.00 [8.98, 9.02] |
| B5' | 8.36 [8.33, 8.39] | 8.84 [8.82, 8.86] | 8.98 [8.96, 9.00] |
| B5'' | 8.89 [8.86, 8.92] | 11.76 [11.74, 11.79] | 10.45 [10.43, 10.48] |

Per-fold judgments (threshold = 0.5 dB):
- F-A: **LiDAR removable**
- F-B: **LiDAR removable**
- F-C: **LiDAR removable**

**Overall verdict: LiDAR removable**


Discussion: LiDAR features behave as a **position proxy** in this comparison: replacing them with explicit AP-relative features (B5') closes most of the gap. The paper should weaken its claim from 'LiDAR carries propagation information' to 'AP geometry suffices when known'.

## 4. RQ4 — residual session effect

# RQ4 — residual session effect via SHAP

Variants: rq4a = B5 + per-session intercept (one-hot); rq4b = B5 + integer session_id.
Datasets: same_map = 15.03 + 24.03 (Map A); full = all three sessions.
Fraction = mean|SHAP|(session features) / mean|SHAP|(all features) on the test split.

| Variant | Dataset | n_test | RMSE | MAE | R² | bias | mean|SHAP|_session | mean|SHAP|_total | fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rq4a | same_map | 58602 | 5.953 | 4.881 | 0.536 | 2.634 | 0.541 | 11.264 | 4.800% |
| rq4a | full | 91073 | 6.186 | 4.989 | 0.402 | 2.406 | 1.374 | 10.862 | 12.651% |
| rq4b | same_map | 58602 | 5.953 | 4.881 | 0.536 | 2.634 | 0.541 | 11.264 | 4.800% |
| rq4b | full | 91073 | 5.936 | 4.725 | 0.449 | 2.442 | 1.417 | 10.330 | 13.721% |

Maximum fraction on same-map subset: **4.800%** → verdict: **negligible**.
- Negligible (<5%): residual session effect is small after conditioning on geometry.
- Bounded (5–20%): meaningful but bounded effect; per-session n_d disagreement is real.
- Large (>20%): model needs session awareness; deployment requires per-session calibration.


Discussion: On the same-map subset (15.03 + 24.03), the maximum SHAP fraction attributed to session-id features is 4.80%. On the full pool, it is 13.72%. The Phase-0 n_d disagreement does not appear to manifest as a significant residual session effect once the multivariate model conditions on geometry — encouraging for cross-session deployment.

## 5. XAI analyses

### 5.1 XAI-1: feature-group importance and beeswarms

![XAI-1 feature-group importance](figures/xai_1_feature_group_importance.png)

# XAI-1: feature-group mean|SHAP| (dB) per fold

| Fold | Telemetry | LiDAR scalar | LiDAR sectoral | AP-relative |
|---|---:|---:|---:|---:|
| F-A | 0.599 | 0.739 | 1.441 | 7.672 |
| F-B | 1.600 | 1.740 | 1.399 | 6.884 |
| F-C | 1.086 | 0.815 | 2.028 | 6.671 |


![XAI-1 beeswarm F-A](figures/xai_1_beeswarm_F-A.png)

![XAI-1 beeswarm F-B](figures/xai_1_beeswarm_F-B.png)

![XAI-1 beeswarm F-C](figures/xai_1_beeswarm_F-C.png)

### 5.2 XAI-2: sign-of-effect consistency

# XAI-2: sign-of-effect consistency across LORO folds

Sign of Spearman ρ(feature, SHAP). `0` if |ρ| < 0.05.
**Consistent** = same non-zero sign across all 3 folds.

| Feature | F-A | F-B | F-C | ρ(F-A) | ρ(F-B) | ρ(F-C) | Consistent |
|---|:---:|:---:|:---:|---:|---:|---:|:---:|
| `mean_dist_mm` | + | + | + | +0.67 | +0.39 | +0.82 | yes |
| `dist_p90_mm` | + | + | - | +0.27 | +0.83 | -0.15 | no |
| `clutter_frac` | - | - | - | -0.66 | -0.40 | -0.49 | yes |
| `openness_frac` | - | + | - | -0.34 | +0.11 | -0.30 | no |
| `mean_front_mm` | - | + | + | -0.29 | +0.32 | +0.53 | no |
| `mean_dist_sector_1_mm` | + | - | + | +0.64 | -0.06 | +0.42 | no |
| `mean_dist_sector_2_mm` | - | + | + | -0.30 | +0.25 | +0.48 | no |
| `mean_dist_sector_3_mm` | + | + | + | +0.64 | +0.39 | +0.66 | yes |
| `mean_dist_sector_4_mm` | - | + | + | -0.33 | +0.53 | +0.36 | no |
| `mean_dist_sector_5_mm` | - | - | + | -0.39 | -0.39 | +0.48 | no |
| `mean_dist_sector_6_mm` | + | + | + | +0.63 | +0.45 | +0.11 | yes |
| `mean_dist_sector_7_mm` | 0 | 0 | + | +0.02 | +0.01 | +0.37 | no |
| `clutter_frac_sector_1` | - | + | - | -0.66 | +0.20 | -0.42 | no |
| `clutter_frac_sector_2` | - | + | - | -0.74 | +0.69 | -0.39 | no |
| `clutter_frac_sector_3` | + | + | - | +0.21 | +0.30 | -0.80 | no |
| `clutter_frac_sector_4` | - | - | - | -0.64 | -0.38 | -0.66 | yes |
| `clutter_frac_sector_5` | + | - | + | +0.47 | -0.14 | +0.63 | no |
| `clutter_frac_sector_6` | + | - | + | +0.37 | -0.76 | +0.78 | no |
| `clutter_frac_sector_7` | - | - | + | -0.22 | -0.52 | +0.23 | no |
| `dist_to_AP` | - | - | - | -0.84 | -0.84 | -0.87 | yes |
| `sin_angle_to_AP` | + | + | + | +0.79 | +0.56 | +0.82 | yes |
| `cos_angle_to_AP` | - | - | - | -0.87 | -0.82 | -0.81 | yes |
| `clutter_frac_toward_AP` | + | - | + | +0.66 | -0.09 | +0.76 | no |

**Headline**: 5 / 19 LiDAR features sign-consistent; 3 / 4 AP-relative features sign-consistent.


### 5.3 XAI-3: spatial maps

![XAI-3 combined](figures/xai_3_spatial_combined.png)

![XAI-3 F-A](figures/xai_3_spatial_F-A.png)

![XAI-3 F-B](figures/xai_3_spatial_F-B.png)

![XAI-3 F-C](figures/xai_3_spatial_F-C.png)

Cross-fold comparison: see whether AP-relative dominates near LOS regions and LiDAR groups dominate in obstructed/transition regions.


### 5.4 XAI-4: SHAP × FOV interaction

![XAI-4 F-A](figures/xai_4_shap_fov_F-A.png)

![XAI-4 F-B](figures/xai_4_shap_fov_F-B.png)

![XAI-4 F-C](figures/xai_4_shap_fov_F-C.png)


Qualitative assessment: a strong negative slope on the in-FOV stratum and a flatter slope on out-of-FOV is the expected physical signature (more directional clutter → weaker predicted signal when the AP is in the FOV; clutter towards the AP carries no information when the AP is behind the AGV). See figures above per fold.


## 6. Synthesis: paper claims supported by Phase 1

- **Claim 1 (LiDAR helps)**: Δ_LiDAR > 0 on {F-A, F-C} (overall stratum). Disambiguation says: LiDAR removable.
- **Claim 2 (frame-independence)**: F-C (Δ_LiDAR_overall = +0.71 dB) is the cross-frame check. A non-degenerate positive value supports the claim; a near-zero or negative value weakens it.
- **Claim 3 (physical interpretability)**: sign-consistency = 5/19 LiDAR features and 3/4 AP-relative continuous features.
- **Disambiguation outcome**: LiDAR removable.
- **RQ4**: residual session effect is **negligible**.


## 7. Limitations and caveats

- High operational-anomaly rate on 15.03 (74,437 anomalies removed across all sessions).

- Three sessions, single AP, single facility, single AGV; generalisation to other deployments unverified.

- 25.02 has 4.4 Hz native cadence; F-C training is decimated to match (k=7) — this halves the effective sample volume.

- `clutter_frac_toward_AP` is NaN for ~21% of non-anomaly rows (out of FOV); XGBoost handles this via default-direction-at-split, but interpretation must respect that the in-FOV stratum is where this feature carries information.

- LORO uses *session* as the leave-out unit; spatial leave-region-out within a session is out of scope for Phase 1.


## 8. Recommendation

- **PIVOT TO PROJECT B**.
- Pivot reason: F-B in-FOV Δ_LiDAR = -0.19 dB ≤ 1 dB threshold. This is the diagnostic fold; small Δ_LiDAR there indicates LiDAR features are not adding physical structure beyond AP-geometry. Recommend Project B (within-session deep dive).
- Project B path: within-session leave-region-out spatial split on 15.03 only, with the same ablation ladder; report as a focused single-session feasibility paper.

## 9. Reproducibility

- Seed: `SEED = 20260427` everywhere (XGBoost, NumPy, bootstrap).
- Run end-to-end: `python -m scripts.p1_project_a.run_all`.
- Stages: `run_modeling` → `run_xai` → `run_robustness` → `build_results_report`.
- Wall-clock breakdown is recorded in `scripts/p1_project_a/results/fit_inventory.parquet`.
- Total wall-clock for fits: 49.6 s.

### Model SHA-256 inventory

| fold/tag | variant | n_features | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F-A | B0 | 1 | 74 | 403876 | 44875 | 158405 | 0.9 | `a78fcbfc87d16bee…` |
| F-A | B1 | 3 | 23 | 403876 | 44875 | 158405 | 0.8 | `bc54e940f042cc2d…` |
| F-A | B2 | 8 | 24 | 403876 | 44875 | 158405 | 1.0 | `5a8a727554c864d3…` |
| F-A | B3 | 22 | 24 | 403876 | 44875 | 158405 | 1.4 | `2a8f049377f44c06…` |
| F-A | B4 | 25 | 27 | 403876 | 44875 | 158405 | 1.6 | `1a9875c49c63a917…` |
| F-A | B5 | 27 | 27 | 403876 | 44875 | 158405 | 1.9 | `4b99cae6cb28c00c…` |
| F-A | B5p | 8 | 25 | 403876 | 44875 | 158405 | 1.1 | `01c5307d7127f068…` |
| F-A | B5pp | 22 | 53 | 403876 | 44875 | 158405 | 1.6 | `aab9a542f6326568…` |
| F-B | B0 | 1 | 81 | 337390 | 37487 | 232279 | 0.9 | `b013350674bad64d…` |
| F-B | B1 | 3 | 46 | 337390 | 37487 | 232279 | 0.9 | `5ab4c5e025db7ccd…` |
| F-B | B2 | 8 | 27 | 337390 | 37487 | 232279 | 0.9 | `d5b037aa1eb82051…` |
| F-B | B3 | 22 | 34 | 337390 | 37487 | 232279 | 1.5 | `ce7743836b6e5127…` |
| F-B | B4 | 25 | 78 | 337390 | 37487 | 232279 | 1.8 | `0802fcf537746bd8…` |
| F-B | B5 | 27 | 127 | 337390 | 37487 | 232279 | 2.5 | `fe1d4aab00e3d70c…` |
| F-B | B5p | 8 | 42 | 337390 | 37487 | 232279 | 1.1 | `0b70a024afeec9fb…` |
| F-B | B5pp | 22 | 141 | 337390 | 37487 | 232279 | 2.3 | `274037f520bdb85f…` |
| F-C | B0 | 1 | 117 | 50232 | 5581 | 216472 | 0.4 | `b05a1af46b5ebfbd…` |
| F-C | B1 | 3 | 35 | 50232 | 5581 | 216472 | 0.3 | `c2be61c4d51f9da2…` |
| F-C | B2 | 8 | 37 | 50232 | 5581 | 216472 | 0.5 | `44137f12bb67aeb0…` |
| F-C | B3 | 22 | 28 | 50232 | 5581 | 216472 | 0.5 | `d0ee20f1e5e72e4b…` |
| F-C | B4 | 25 | 34 | 50232 | 5581 | 216472 | 0.6 | `5809ba6369cc2b2b…` |
| F-C | B5 | 27 | 35 | 50232 | 5581 | 216472 | 0.7 | `94e726cbf698433e…` |
| F-C | B5p | 8 | 63 | 50232 | 5581 | 216472 | 0.4 | `0eb45881e38630cf…` |
| F-C | B5pp | 22 | 37 | 50232 | 5581 | 216472 | 0.5 | `c454e01461dc7571…` |
| rq4_rq4a_same_map | rq4a | 29 | 36 | 273479 | 58603 | 58602 | 4.6 | `8e14d93c8ae3be1f…` |
| rq4_rq4a_full | rq4a | 29 | 38 | 425009 | 91074 | 91073 | 7.1 | `32132e5139801ccc…` |
| rq4_rq4b_same_map | rq4b | 28 | 36 | 273479 | 58603 | 58602 | 4.5 | `cd8e93477da47bb8…` |
| rq4_rq4b_full | rq4b | 28 | 38 | 425009 | 91074 | 91073 | 7.2 | `5d3cd7b00a92d8d2…` |


## 10. Hyperparameter robustness diagnostic

Tests whether the F-B in-FOV Δ_LiDAR ≤ 1 dB pivot trigger from §2 is robust to hyperparameter and validation-protocol choice. Refits B1 and B5 on F-B only; all other Phase 1 / Project A artifacts (models, predictions, SHAP) are untouched.

### 10.1 TL;DR

F-B in-FOV Δ_LiDAR by config (positive = LiDAR helps; >1 dB clears the pivot threshold):
- **locked** (max_depth=6, eta=0.05): -0.19 dB
- **H1** (max_depth=4, eta=0.05, λ=1.0): +0.10 dB
- **H2** (max_depth=6, eta=0.02, λ=5.0): -0.07 dB
- **H3** (max_depth=8, eta=0.05, λ=20.0): -0.47 dB
- **H1_randval** (H1 hyperparams + random val (max_depth=4, eta=0.05)): -0.03 dB
- **H2_randval** (H2 hyperparams + random val (max_depth=6, eta=0.02)): -0.48 dB

**Diagnostic verdict: PIVOT_JUSTIFIED**

### 10.2 Configurations

| Tag | max_depth | eta | min_child_weight | reg_lambda | subsample | colsample_bytree | n_estimators | early_stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| locked (frozen) | 6 | 0.05 | 1 | 1.0 | 1.0 | 1.0 | 2000 | 100 |
| H1 | 4 | 0.05 | 1 | 1.0 | 1.0 | 1.0 | 2000 | 100 |
| H2 | 6 | 0.02 | 10 | 5.0 | 0.9 | 0.9 | 5000 | 200 |
| H3 | 8 | 0.05 | 5 | 20.0 | 0.9 | 0.9 | 2000 | 100 |

`H1_randval` and `H2_randval` use the same hyperparameters as `H1` / `H2` but a random-10%-per-session validation split (seed=20260427) instead of chronological-last-10%. B1 is not refit with random val; the Δ_LiDAR for these uses the B1 fit from the matching chronological config.

### 10.3 Per-fit RMSE (dB) on F-B

Bootstrap 95% CI on RMSE in brackets (B = 1000).

| Variant | Config | Val protocol | best_iter | Stratum | n | RMSE | 95% CI |
|---|---|---|---:|---|---:|---:|---|
| B1 | locked | chronological | 46 | overall | 232279 | 7.935 | [7.91, 7.96] |
| B1 | locked | chronological | 46 | in_fov | 134692 | 8.014 | [7.98, 8.05] |
| B1 | locked | chronological | 46 | out_of_fov | 97587 | 7.823 | [7.79, 7.85] |
| B1 | H1 | chronological | 34 | overall | 232279 | 8.167 | [8.15, 8.19] |
| B1 | H1 | chronological | 34 | in_fov | 134692 | 7.856 | [7.82, 7.88] |
| B1 | H1 | chronological | 34 | out_of_fov | 97587 | 8.578 | [8.55, 8.61] |
| B1 | H2 | chronological | 126 | overall | 232279 | 8.818 | [8.79, 8.84] |
| B1 | H2 | chronological | 126 | in_fov | 134692 | 8.145 | [8.11, 8.17] |
| B1 | H2 | chronological | 126 | out_of_fov | 97587 | 9.670 | [9.64, 9.71] |
| B1 | H3 | chronological | 35 | overall | 232279 | 8.929 | [8.91, 8.95] |
| B1 | H3 | chronological | 35 | in_fov | 134692 | 8.575 | [8.54, 8.60] |
| B1 | H3 | chronological | 35 | out_of_fov | 97587 | 9.397 | [9.37, 9.43] |
| B5 | locked | chronological | 127 | overall | 232279 | 8.624 | [8.60, 8.65] |
| B5 | locked | chronological | 127 | in_fov | 134692 | 8.202 | [8.17, 8.23] |
| B5 | locked | chronological | 127 | out_of_fov | 97587 | 9.175 | [9.14, 9.21] |
| B5 | H1 | chronological | 84 | overall | 232279 | 8.144 | [8.12, 8.16] |
| B5 | H1 | chronological | 84 | in_fov | 134692 | 7.752 | [7.72, 7.78] |
| B5 | H1 | chronological | 84 | out_of_fov | 97587 | 8.655 | [8.62, 8.68] |
| B5 | H2 | chronological | 167 | overall | 232279 | 8.834 | [8.81, 8.86] |
| B5 | H2 | chronological | 167 | in_fov | 134692 | 8.217 | [8.19, 8.25] |
| B5 | H2 | chronological | 167 | out_of_fov | 97587 | 9.621 | [9.59, 9.65] |
| B5 | H3 | chronological | 35 | overall | 232279 | 9.619 | [9.60, 9.64] |
| B5 | H3 | chronological | 35 | in_fov | 134692 | 9.044 | [9.01, 9.07] |
| B5 | H3 | chronological | 35 | out_of_fov | 97587 | 10.360 | [10.32, 10.39] |
| B5 | H1_randval | random | 1999 | overall | 232279 | 8.638 | [8.61, 8.66] |
| B5 | H1_randval | random | 1999 | in_fov | 134692 | 7.891 | [7.86, 7.92] |
| B5 | H1_randval | random | 1999 | out_of_fov | 97587 | 9.575 | [9.54, 9.61] |
| B5 | H2_randval | random | 4999 | overall | 232279 | 10.514 | [10.49, 10.54] |
| B5 | H2_randval | random | 4999 | in_fov | 134692 | 8.624 | [8.59, 8.65] |
| B5 | H2_randval | random | 4999 | out_of_fov | 97587 | 12.668 | [12.62, 12.72] |

### 10.4 Δ_LiDAR by config × stratum

Δ_LiDAR = RMSE(B1) − RMSE(B5). Positive => LiDAR features improve over AP-geometry-only.

| Config | Stratum | RMSE(B1) | RMSE(B5) | Δ_LiDAR (dB) | clears 1 dB? |
|---|---|---:|---:|---:|:---:|
| locked | overall | 7.935 | 8.624 | -0.689 | — |
| locked | in_fov | 8.014 | 8.202 | -0.187 | ✗ |
| locked | out_of_fov | 7.823 | 9.175 | -1.352 | — |
| H1 | overall | 8.167 | 8.144 | +0.023 | — |
| H1 | in_fov | 7.856 | 7.752 | +0.104 | ✗ |
| H1 | out_of_fov | 8.578 | 8.655 | -0.077 | — |
| H2 | overall | 8.818 | 8.834 | -0.016 | — |
| H2 | in_fov | 8.145 | 8.217 | -0.073 | ✗ |
| H2 | out_of_fov | 9.670 | 9.621 | +0.050 | — |
| H3 | overall | 8.929 | 9.619 | -0.689 | — |
| H3 | in_fov | 8.575 | 9.044 | -0.469 | ✗ |
| H3 | out_of_fov | 9.397 | 10.360 | -0.962 | — |
| H1_randval | overall | 8.167 | 8.638 | -0.472 | — |
| H1_randval | in_fov | 7.856 | 7.891 | -0.035 | ✗ |
| H1_randval | out_of_fov | 8.578 | 9.575 | -0.997 | — |
| H2_randval | overall | 8.818 | 10.514 | -1.697 | — |
| H2_randval | in_fov | 8.145 | 8.624 | -0.480 | ✗ |
| H2_randval | out_of_fov | 9.670 | 12.668 | -2.998 | — |

### 10.5 best_iteration distribution per config

If slow-eta H2 reaches a much higher iteration count, the locked config was stopping prematurely.

| Config | B1 best_iter | B5 best_iter |
|---|---:|---:|
| locked | 46 | 127 |
| H1 | 34 | 84 |
| H2 | 126 | 167 |
| H3 | 35 | 35 |
| H1_randval | — | 1999 |
| H2_randval | — | 4999 |

### 10.6 Verdict criteria

- **PIVOT_JUSTIFIED**: F-B in-FOV Δ_LiDAR ≤ 1 dB on all 6 configs (locked + 5 alternatives). Result is robust; the proposal's pivot trigger fires.
- **RECOVERABLE_HYPERPARAMS**: at least one of {H1, H2, H3} produces F-B in-FOV Δ_LiDAR > 1 dB. Result is fragile to hyperparameter choice.
- **RECOVERABLE_VALIDATION_PROTOCOL**: a random-val variant differs from its chronological-val counterpart by >0.5 dB AND clears the 1 dB threshold. Chronological-per-session early-stopping was the issue.

### 10.7 Discussion

**best_iteration distribution.** H2 (slow eta) reached B5 best_iter=167, vs the locked config's 127 — comparable depth of training. The locked config was not stopping conspicuously prematurely.

**Out-of-FOV under H2 (slowest learning rate).** H2 raised B5 out-of-FOV RMSE from 9.17 to 9.62 dB (+0.45 dB). Slower learning underfits the harder stratum here.

**Random-val sensitivity.**
- **H1**: chrono Δ_LiDAR_in_fov = +0.10 dB; randval = -0.03 dB; |Δ| = 0.14 dB.
- **H2**: chrono Δ_LiDAR_in_fov = -0.07 dB; randval = -0.48 dB; |Δ| = 0.41 dB.

Neither randval variant moves Δ_LiDAR by >0.5 dB from its chronological counterpart — the validation protocol is not the issue.

### 10.8 Recommendation

**PIVOT TO PROJECT B.** The pivot trigger is robust across all six configurations. The original Phase 1 / Project A conclusion stands; LiDAR-derived structure does not transfer across sessions on the diagnostic fold under any reasonable hyperparameter choice tested. Proceed with the within-session leave-region-out path (15.03 only).

### 10.9 Diagnostic artifacts
- 8 model files: `scripts/p1_project_a/diagnostic/models/F-B_{variant}_{config}.json`
- Predictions: `scripts/p1_project_a/diagnostic/cache/predictions_F-B_{variant}_{config}.parquet`
- Metrics parquet: `scripts/p1_project_a/diagnostic/robustness_metrics.parquet`
- Fit inventory: `scripts/p1_project_a/diagnostic/robustness_fit_inventory.parquet`
- Re-run: `python -m scripts.p1_project_a.run_robustness`


### 10.10 Diagnostic fit inventory

| variant | config | val | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|
| B1 | H1 | chronological | 34 | 337390 | 37487 | 232279 | 0.8 | `e8c443a308a190e0…` |
| B5 | H1 | chronological | 84 | 337390 | 37487 | 232279 | 2.0 | `de7d723eac49fdbb…` |
| B1 | H2 | chronological | 126 | 337390 | 37487 | 232279 | 2.5 | `fed9a06080c3f004…` |
| B5 | H2 | chronological | 167 | 337390 | 37487 | 232279 | 4.9 | `769c7a18757549c7…` |
| B1 | H3 | chronological | 35 | 337390 | 37487 | 232279 | 1.4 | `bfe0d1a49d6e4d69…` |
| B5 | H3 | chronological | 35 | 337390 | 37487 | 232279 | 2.7 | `28e461b42c8bbd95…` |
| B5 | H1_randval | random | 1999 | 337390 | 37487 | 232279 | 17.0 | `3fc845f6776e1c78…` |
| B5 | H2_randval | random | 4999 | 337390 | 37487 | 232279 | 63.3 | `228f65c8afa5542b…` |

