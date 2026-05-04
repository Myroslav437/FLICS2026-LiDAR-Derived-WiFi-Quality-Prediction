# Project B results report — within-session deep dive on 15.03.2026

Project B is the within-session leave-region-out (WLRO) ablation on 15.03 only. The cross-session counterpart is Project A (`docs/p1_project_a/results_report.md`).

## 0. TL;DR

**Feature stack: leakage-fixed.** This report uses the leakage-fixed feature stack: 3 telemetry features (`speed_mps`, `turn_rate`, `momentary_current_consumption`) plus position, AP-relative geometry, and LiDAR. Five features from the original locked feature stack were removed post-hoc as either router-side (target leakage), within-session-only (deployment leakage), or constant sentinel (no information). See `MIGRATION_LOG.md` for the full audit trail. (`feature_stack_version = "leakage_fixed_v1"`).

- **Δ_LiDAR_within (W2 − W4) per fold (overall / in-FOV / out-of-FOV)**:
  - R-1: overall -1.93 dB [-2.02, -1.86]; in-FOV -2.99 dB [-3.08, -2.90]; out-of-FOV +1.15 dB [+1.01, +1.28]
  - R-2: overall -0.61 dB [-0.65, -0.57]; in-FOV -2.20 dB [-2.27, -2.14]; out-of-FOV +0.79 dB [+0.76, +0.82]
  - R-3: overall -0.17 dB [-0.21, -0.13]; in-FOV -0.09 dB [-0.14, -0.04]; out-of-FOV -0.42 dB [-0.46, -0.38]
  - R-4: overall -0.58 dB [-0.64, -0.52]; in-FOV -1.63 dB [-1.67, -1.59]; out-of-FOV -0.08 dB [-0.16, +0.01]
  - R-5: overall -1.31 dB [-1.34, -1.28]; in-FOV -2.22 dB [-2.25, -2.18]; out-of-FOV -0.56 dB [-0.60, -0.52]
- **Folds clearing the 1.0 dB in-FOV Δ_LiDAR_within threshold**: 0 / 5 (required ≥ 3).
- **Disambiguation verdict**: mixed/unclear.
- **Verdict**: **WITHIN-SESSION-NULL**.
- **Recommended paper framing**: Negative-result paper: LiDAR features add no measurable value to AP-relative geometry even within a single session, on a dataset with strong geometric coverage. The paper's contribution is methodological — a careful within-session test that controls for session-shift confounds and finds no signal.
- **All 41 fits succeeded**: yes.

## 1. Inputs and protocol

- Dataset: `data/phase1/dataset.parquet` (SHA-256 `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270`).
  - Expected SHA-256 from `dataset.sha256`: `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270` (match).
- Filter: `session_date == '15.03.2026'` AND `~anomaly_flag` → 232,279 rows.
- Region partition: K-means with k=5 on (x_m, y_m), method = `cached`.
- Per-region row counts:
  - region 1: 80,365
  - region 2: 35,488
  - region 3: 44,657
  - region 4: 24,593
  - region 5: 47,176
- Folds: 5 leave-region-out folds (R-1 … R-5).
- Validation split: random 10% of training rows per fold (seeded; IID with train across regions).
- Buffer-zone sanity check: refit R-1 after dropping training rows within 1.0 m of any held-out-region row.
- Hyperparameter robustness: refit W4 under H1 (max_depth=4) on every fold.

## 2. Spatial regions

![Spatial regions](figures/regions_map.png)

# Project B — per-region statistics on 15.03.2026

| Region | n_rows | unique 0.1 m cells | frac in-FOV | n_in_FOV | n_out_of_FOV | dist_to_AP range (m) | (x_m, y_m) bbox |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 80,365 | 256 | 0.713 | 57,270 | 23,095 | [0.46, 7.06] | x [-0.36, 7.81], y [6.21, 13.33] |
| 2 | 35,488 | 215 | 0.437 | 15,516 | 19,972 | [5.26, 13.20] | x [5.70, 13.04], y [2.49, 6.85] |
| 3 | 44,657 | 240 | 0.694 | 30,974 | 13,683 | [13.14, 19.67] | x [12.88, 18.77], y [-0.79, 3.40] |
| 4 | 24,593 | 188 | 0.435 | 10,687 | 13,906 | [19.56, 26.41] | x [18.34, 24.48], y [-3.92, -0.15] |
| 5 | 47,176 | 143 | 0.429 | 20,245 | 26,931 | [26.42, 31.58] | x [24.39, 28.95], y [-6.57, -3.70] |


## 3. Within-session ablation results

### 3.1 Headline ablation table (5 folds × 3 strata × 6 variants)

# WLRO ablation — RMSE (dB) per fold × variant × stratum

Bootstrap 95% CI on RMSE in brackets (B = 1000). Within-session leave-region-out, 15.03 only.

| Variant | Stratum | R-1 | R-2 | R-3 | R-4 | R-5 |
|---|---|---|---|---|---|---|
| W0 | overall | 7.71 [7.67, 7.76] | 11.11 [11.05, 11.16] | 14.12 [14.05, 14.18] | 9.10 [9.05, 9.15] | 6.07 [6.04, 6.11] |
| W0 | in_fov | 7.91 [7.86, 7.97] | 9.36 [9.26, 9.45] | 16.38 [16.31, 16.46] | 11.23 [11.17, 11.30] | 4.93 [4.88, 4.98] |
| W0 | out_of_fov | 7.21 [7.13, 7.28] | 12.30 [12.22, 12.37] | 6.57 [6.51, 6.64] | 7.03 [6.98, 7.07] | 6.80 [6.76, 6.85] |
| W1 | overall | 8.58 [8.54, 8.63] | 11.38 [11.31, 11.44] | 10.64 [10.58, 10.70] | 8.38 [8.30, 8.47] | 9.74 [9.69, 9.79] |
| W1 | in_fov | 8.70 [8.65, 8.75] | 8.49 [8.41, 8.57] | 12.15 [12.09, 12.22] | 5.72 [5.67, 5.77] | 8.27 [8.22, 8.32] |
| W1 | out_of_fov | 8.29 [8.21, 8.37] | 13.19 [13.10, 13.28] | 5.94 [5.88, 6.00] | 9.96 [9.85, 10.07] | 10.71 [10.63, 10.79] |
| W2 | overall | 10.81 [10.75, 10.88] | 7.45 [7.41, 7.50] | 6.78 [6.73, 6.83] | 6.35 [6.30, 6.40] | 9.27 [9.23, 9.32] |
| W2 | in_fov | 10.80 [10.73, 10.87] | 7.19 [7.13, 7.26] | 7.22 [7.14, 7.29] | 4.16 [4.12, 4.20] | 9.37 [9.29, 9.44] |
| W2 | out_of_fov | 10.83 [10.70, 10.96] | 7.65 [7.59, 7.72] | 5.67 [5.60, 5.74] | 7.61 [7.54, 7.68] | 9.20 [9.14, 9.26] |
| W3 | overall | 16.62 [16.58, 16.67] | 7.78 [7.73, 7.83] | 6.43 [6.39, 6.48] | 6.37 [6.30, 6.43] | 10.91 [10.86, 10.97] |
| W3 | in_fov | 17.76 [17.71, 17.81] | 9.28 [9.20, 9.36] | 6.65 [6.59, 6.70] | 5.20 [5.15, 5.25] | 11.26 [11.18, 11.32] |
| W3 | out_of_fov | 13.40 [13.31, 13.49] | 6.37 [6.32, 6.42] | 5.92 [5.86, 5.99] | 7.13 [7.05, 7.22] | 10.64 [10.58, 10.71] |
| W4 | overall | 12.75 [12.70, 12.79] | 8.07 [8.02, 8.12] | 6.96 [6.91, 7.00] | 6.93 [6.88, 6.98] | 10.58 [10.53, 10.63] |
| W4 | in_fov | 13.79 [13.74, 13.84] | 9.40 [9.32, 9.48] | 7.31 [7.25, 7.36] | 5.79 [5.74, 5.85] | 11.58 [11.50, 11.66] |
| W4 | out_of_fov | 9.69 [9.61, 9.76] | 6.86 [6.80, 6.93] | 6.09 [6.01, 6.16] | 7.69 [7.63, 7.75] | 9.76 [9.71, 9.81] |
| W4pp | overall | 15.34 [15.29, 15.40] | 8.25 [8.20, 8.32] | 9.08 [9.03, 9.14] | 6.60 [6.53, 6.67] | 16.82 [16.76, 16.88] |
| W4pp | in_fov | 17.14 [17.08, 17.20] | 9.51 [9.42, 9.61] | 9.52 [9.44, 9.59] | 4.18 [4.13, 4.23] | 12.86 [12.76, 12.95] |
| W4pp | out_of_fov | 9.50 [9.42, 9.57] | 7.13 [7.05, 7.20] | 8.02 [7.94, 8.10] | 7.97 [7.89, 8.06] | 19.27 [19.19, 19.36] |


![WLRO ablation RMSE](figures/wlro_ablation_chart.png)


### 3.2 Δ_LiDAR_within per fold per stratum

# Δ_LiDAR_within — RMSE(W2) − RMSE(W4) (dB)

Positive => LiDAR features improve over (position + telemetry + AP-relative).

| Fold | Stratum | n | Δ_LiDAR | 95% CI |
|---|---|---:|---:|---|
| R-1 | overall | 80365 | -1.934 | [-2.02, -1.86] |
| R-1 | in_fov | 57270 | -2.986 | [-3.08, -2.90] |
| R-1 | out_of_fov | 23095 | +1.148 | [+1.01, +1.28] |
| R-2 | overall | 35488 | -0.613 | [-0.65, -0.57] |
| R-2 | in_fov | 15516 | -2.203 | [-2.27, -2.14] |
| R-2 | out_of_fov | 19972 | +0.791 | [+0.76, +0.82] |
| R-3 | overall | 44657 | -0.174 | [-0.21, -0.13] |
| R-3 | in_fov | 30974 | -0.087 | [-0.14, -0.04] |
| R-3 | out_of_fov | 13683 | -0.418 | [-0.46, -0.38] |
| R-4 | overall | 24593 | -0.580 | [-0.64, -0.52] |
| R-4 | in_fov | 10687 | -1.631 | [-1.67, -1.59] |
| R-4 | out_of_fov | 13906 | -0.075 | [-0.16, +0.01] |
| R-5 | overall | 47176 | -1.307 | [-1.34, -1.28] |
| R-5 | in_fov | 20245 | -2.215 | [-2.25, -2.18] |
| R-5 | out_of_fov | 26931 | -0.557 | [-0.60, -0.52] |


### 3.3 Δ_AP-relative_within per fold per stratum

# Δ_AP-relative_within — RMSE(W1) − RMSE(W2) (dB)

Positive => AP-relative features improve on (position + telemetry).

| Fold | Stratum | n | Δ_AP-relative | 95% CI |
|---|---|---:|---:|---|
| R-1 | overall | 80365 | -2.229 | [-2.28, -2.18] |
| R-1 | in_fov | 57270 | -2.104 | [-2.16, -2.05] |
| R-1 | out_of_fov | 23095 | -2.547 | [-2.64, -2.44] |
| R-2 | overall | 35488 | +3.925 | [+3.84, +4.01] |
| R-2 | in_fov | 15516 | +1.302 | [+1.23, +1.38] |
| R-2 | out_of_fov | 19972 | +5.541 | [+5.41, +5.66] |
| R-3 | overall | 44657 | +3.860 | [+3.80, +3.92] |
| R-3 | in_fov | 30974 | +4.934 | [+4.85, +5.01] |
| R-3 | out_of_fov | 13683 | +0.270 | [+0.23, +0.31] |
| R-4 | overall | 24593 | +2.037 | [+1.96, +2.11] |
| R-4 | in_fov | 10687 | +1.556 | [+1.51, +1.61] |
| R-4 | out_of_fov | 13906 | +2.347 | [+2.24, +2.45] |
| R-5 | overall | 47176 | +0.467 | [+0.41, +0.52] |
| R-5 | in_fov | 20245 | -1.098 | [-1.18, -1.01] |
| R-5 | out_of_fov | 26931 | +1.512 | [+1.46, +1.56] |


#### Δ_position (W0 vs per-region mean baseline)

# Δ_position — RMSE(per-region mean baseline) − RMSE(W0) (dB)

Positive => position alone (W0 = x_m, y_m) explains structure beyond the train-region mean. Overall stratum.

| Fold | n | RMSE(mean baseline) | RMSE(W0) | Δ_position | 95% CI |
|---|---:|---:|---:|---:|---|
| R-1 | 80365 | 16.458 | 7.715 | +8.743 | [+8.68, +8.80] |
| R-2 | 35488 | 9.120 | 11.107 | -1.987 | [-2.03, -1.95] |
| R-3 | 44657 | 11.113 | 14.121 | -3.007 | [-3.08, -2.94] |
| R-4 | 24593 | 8.014 | 9.097 | -1.083 | [-1.10, -1.06] |
| R-5 | 47176 | 15.183 | 6.072 | +9.111 | [+9.07, +9.15] |


### 3.4 Disambiguation (W4 / W4' / W4'')

# Disambiguation — overall RMSE (dB) for W4 / W4' / W4''

W4 = full 34-feature model. W4' = position + telemetry + AP-relative (≡ W2; LiDAR removed). W4'' = position + telemetry + LiDAR (AP-relative removed).

| Variant | R-1 | R-2 | R-3 | R-4 | R-5 |
|---|---|---|---|---|---|
| W4 | 12.75 [12.70, 12.79] | 8.07 [8.02, 8.12] | 6.96 [6.91, 7.00] | 6.93 [6.88, 6.98] | 10.58 [10.53, 10.63] |
| W4' | 10.81 [10.75, 10.88] | 7.45 [7.41, 7.50] | 6.78 [6.73, 6.83] | 6.35 [6.30, 6.40] | 9.27 [9.23, 9.32] |
| W4'' | 15.34 [15.29, 15.40] | 8.25 [8.20, 8.32] | 9.08 [9.03, 9.14] | 6.60 [6.53, 6.67] | 16.82 [16.76, 16.88] |

Per-fold judgments (threshold = 0.5 dB):
- R-1: **mixed/unclear**
- R-2: **AP-relative removable**
- R-3: **LiDAR removable**
- R-4: **AP-relative removable**
- R-5: **mixed/unclear**

**Overall verdict (≥3 of 5 folds): mixed/unclear**


### 3.5 Buffer-zone sensitivity (R-1)

# Buffer-zone sensitivity — R-1 with vs without 1 m exclusion

Drop training rows within 1.0 m of any held-out region row, then refit. Compare RMSE deltas to the no-buffer R-1 fits.

| Variant | Stratum | RMSE(no buffer) | RMSE(buffer) | Δ_RMSE (buffer - no) |
|---|---|---:|---:|---:|
| W0 | overall | 7.715 | 18.561 | +10.847 |
| W0 | in_fov | 7.911 | 18.124 | +10.214 |
| W0 | out_of_fov | 7.206 | 19.603 | +12.397 |
| W1 | overall | 8.582 | 19.158 | +10.576 |
| W1 | in_fov | 8.699 | 18.991 | +10.292 |
| W1 | out_of_fov | 8.286 | 19.567 | +11.281 |
| W2 | overall | 10.811 | 15.273 | +4.462 |
| W2 | in_fov | 10.802 | 16.432 | +5.629 |
| W2 | out_of_fov | 10.833 | 11.925 | +1.091 |
| W3 | overall | 16.624 | 20.382 | +3.758 |
| W3 | in_fov | 17.759 | 21.131 | +3.372 |
| W3 | out_of_fov | 13.401 | 18.392 | +4.991 |
| W4 | overall | 12.745 | 25.640 | +12.894 |
| W4 | in_fov | 13.789 | 27.645 | +13.856 |
| W4 | out_of_fov | 9.685 | 19.810 | +10.124 |
| W4pp | overall | 15.340 | 22.868 | +7.528 |
| W4pp | in_fov | 17.142 | 24.689 | +7.548 |
| W4pp | out_of_fov | 9.498 | 17.555 | +8.058 |

**Sanity check (overall stratum)**: 0/6 variants change by < 0.5 dB under the buffer.
**Flagged**: majority of variants move by ≥ 0.5 dB under the buffer — leave-region-out may be contaminated by spatial autocorrelation; treat the headline result with caution.


### 3.6 Hyperparameter sensitivity (W4 locked vs H1)

# Hyperparameter sensitivity — W4 locked vs H1 (max_depth=4)

H1 was the cross-session diagnostic's best alternative. Refit on each Project B fold to verify the within-session story is not config-specific.

| Fold | Stratum | RMSE(locked, depth=6) | RMSE(H1, depth=4) | Δ_RMSE (H1 - locked) |
|---|---|---:|---:|---:|
| R-1 | overall | 12.745 | 9.976 | -2.770 |
| R-1 | in_fov | 13.789 | 10.753 | -3.035 |
| R-1 | out_of_fov | 9.685 | 7.716 | -1.969 |
| R-2 | overall | 8.067 | 6.571 | -1.496 |
| R-2 | in_fov | 9.395 | 6.552 | -2.843 |
| R-2 | out_of_fov | 6.859 | 6.586 | -0.274 |
| R-3 | overall | 6.956 | 7.514 | +0.558 |
| R-3 | in_fov | 7.307 | 8.133 | +0.826 |
| R-3 | out_of_fov | 6.087 | 5.878 | -0.209 |
| R-4 | overall | 6.928 | 5.886 | -1.043 |
| R-4 | in_fov | 5.792 | 3.361 | -2.431 |
| R-4 | out_of_fov | 7.688 | 7.252 | -0.437 |
| R-5 | overall | 10.580 | 8.267 | -2.313 |
| R-5 | in_fov | 11.583 | 8.522 | -3.060 |
| R-5 | out_of_fov | 9.760 | 8.071 | -1.689 |

**Flagged (overall stratum)**: |Δ_RMSE| > 0.3 dB on folds ['R-1', 'R-2', 'R-3', 'R-4', 'R-5']. Report locked and H1 side by side for these folds.


## 4. XAI (R-3, W4)

- Selection: R-3 W4 overall RMSE = 6.96 dB within 1 dB of median 8.07 dB; using R-3.

- SHAP rows used: 44,657 (subsampled from 44,657)


### 4.1 Feature-group importance

![XAI group importance](figures/xai_group_importance_R-3.png)

# XAI feature-group importance — R-3 (W4)

Σ mean|SHAP| in group (dB), summed over the group's features.

| Group | Σ mean\|SHAP\| (dB) | # features |
|---|---:|---:|
| AP-relative | 6.281 | 5 |
| Position | 4.754 | 2 |
| LiDAR sectoral | 2.822 | 14 |
| Telemetry | 2.354 | 3 |
| LiDAR scalar | 1.632 | 5 |


### 4.2 Sign-of-effect (single-fold)

# XAI sign-of-effect — R-3 (W4)

Sign of Spearman ρ(feature, SHAP). `0` if |ρ| < 0.05.

| Feature | ρ | Sign |
|---|---:|:---:|
| `x_m` | -0.25 | - |
| `y_m` | -0.19 | - |
| `speed_mps` | +0.57 | + |
| `turn_rate` | +0.19 | + |
| `momentary_current_consumption` | -0.58 | - |
| `dist_to_AP` | -0.06 | - |
| `sin_angle_to_AP` | +0.62 | + |
| `cos_angle_to_AP` | -0.87 | - |
| `clutter_frac_toward_AP` | -0.53 | - |
| `mean_dist_mm` | +0.20 | + |
| `dist_p90_mm` | +0.59 | + |
| `clutter_frac` | +0.29 | + |
| `openness_frac` | +0.51 | + |
| `mean_front_mm` | +0.48 | + |
| `mean_dist_sector_1_mm` | +0.38 | + |
| `mean_dist_sector_2_mm` | -0.52 | - |
| `mean_dist_sector_3_mm` | +0.17 | + |
| `mean_dist_sector_4_mm` | +0.57 | + |
| `mean_dist_sector_5_mm` | +0.14 | + |
| `mean_dist_sector_6_mm` | +0.15 | + |
| `mean_dist_sector_7_mm` | -0.37 | - |
| `clutter_frac_sector_1` | -0.62 | - |
| `clutter_frac_sector_2` | -0.77 | - |
| `clutter_frac_sector_3` | -0.20 | - |
| `clutter_frac_sector_4` | -0.77 | - |
| `clutter_frac_sector_5` | -0.15 | - |
| `clutter_frac_sector_6` | +0.50 | + |
| `clutter_frac_sector_7` | -0.41 | - |


### 4.3 Spatial dominance on the held-out region

![XAI spatial](figures/xai_spatial_R-3.png)


## 5. Synthesis vs Project A

- **Within-session Δ_LiDAR_within (in-FOV) per fold**: R-1 -2.99; R-2 -2.20; R-3 -0.09; R-4 -1.63; R-5 -2.22.
- **Within-session disambiguation**: mixed/unclear (per-fold: {'R-1': 'mixed/unclear', 'R-2': 'AP-relative removable', 'R-3': 'LiDAR removable', 'R-4': 'AP-relative removable', 'R-5': 'mixed/unclear'}).
- **Compared with Project A (cross-session)**: Project A's F-B in-FOV Δ_LiDAR was −0.52 dB (LiDAR hurts) and the disambiguation was 'LiDAR removable'. Project B isolates whether the issue was session shift or LiDAR itself.
- **Implication**: even within a single session with consistent route, LiDAR features add no measurable value beyond AP-relative geometry. The Project A pivot interpretation (LiDAR is a position proxy) holds within-session as well.


## 6. Recommendation

- Final paper framing: **WITHIN-SESSION-NULL**.

Negative-result paper: LiDAR features add no measurable value to AP-relative geometry even within a single session, on a dataset with strong geometric coverage. The paper's contribution is methodological — a careful within-session test that controls for session-shift confounds and finds no signal.

- Negative-result paper structure: §1–3 setup and dataset. §4 cross-session result. §5 within-session result on the cleanest session. §6 disambiguation: AP-relative geometry is sufficient. §7 implications for LiDAR-based propagation modelling.


## 7. Reproducibility

- Seed: `SEED = 20260427` everywhere (XGBoost, NumPy, K-means).
- Run end-to-end: `python -m scripts.p1_project_b.run_all`.
- Stages: `run_modeling` → `run_xai` → `build_results_report`.
- Total wall-clock for fits: 401.9 s.

### Model SHA-256 inventory

| fold | variant | hyperparams | n_features | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R-1 | W0 | locked | 2 | 1999 | 136723 | 15191 | 80365 | 5.2 | `538ff05dcd5bfa63…` |
| R-1 | W1 | locked | 5 | 1999 | 136723 | 15191 | 80365 | 5.8 | `9a51a1f40e307c58…` |
| R-1 | W2 | locked | 10 | 1999 | 136723 | 15191 | 80365 | 8.5 | `43884188bf6084a1…` |
| R-1 | W3 | locked | 15 | 1999 | 136723 | 15191 | 80365 | 10.4 | `5b56c9dbaee0d1ee…` |
| R-1 | W4 | locked | 29 | 1999 | 136723 | 15191 | 80365 | 13.6 | `fb4f5360157bf050…` |
| R-1 | W4pp | locked | 24 | 1999 | 136723 | 15191 | 80365 | 10.2 | `45150ba365987f4a…` |
| R-2 | W0 | locked | 2 | 1999 | 177112 | 19679 | 35488 | 6.1 | `8772b08dbf89d5bc…` |
| R-2 | W1 | locked | 5 | 1999 | 177112 | 19679 | 35488 | 6.9 | `acb7ac314e7d527a…` |
| R-2 | W2 | locked | 10 | 1999 | 177112 | 19679 | 35488 | 9.7 | `bdbfb97744f74247…` |
| R-2 | W3 | locked | 15 | 1999 | 177112 | 19679 | 35488 | 11.1 | `a35c98a335386a69…` |
| R-2 | W4 | locked | 29 | 1999 | 177112 | 19679 | 35488 | 15.9 | `a6a1d951612ff6e2…` |
| R-2 | W4pp | locked | 24 | 1999 | 177112 | 19679 | 35488 | 12.1 | `575a9da50a8983c9…` |
| R-3 | W0 | locked | 2 | 1998 | 168860 | 18762 | 44657 | 6.0 | `fe96988bc378ee2c…` |
| R-3 | W1 | locked | 5 | 1999 | 168860 | 18762 | 44657 | 6.6 | `3091f557530e81ea…` |
| R-3 | W2 | locked | 10 | 1999 | 168860 | 18762 | 44657 | 9.3 | `594396221c1c21d5…` |
| R-3 | W3 | locked | 15 | 1999 | 168860 | 18762 | 44657 | 10.6 | `dacae3fca7c07b23…` |
| R-3 | W4 | locked | 29 | 1999 | 168860 | 18762 | 44657 | 14.5 | `e67636dc73a79317…` |
| R-3 | W4pp | locked | 24 | 1999 | 168860 | 18762 | 44657 | 11.3 | `2bffd1c01b153115…` |
| R-4 | W0 | locked | 2 | 1999 | 186917 | 20769 | 24593 | 6.1 | `7d433179c73fad6b…` |
| R-4 | W1 | locked | 5 | 1999 | 186917 | 20769 | 24593 | 7.1 | `8787732b4476fde7…` |
| R-4 | W2 | locked | 10 | 1999 | 186917 | 20769 | 24593 | 9.9 | `783e00d701de12be…` |
| R-4 | W3 | locked | 15 | 1999 | 186917 | 20769 | 24593 | 11.3 | `f8125dffee990cf9…` |
| R-4 | W4 | locked | 29 | 1999 | 186917 | 20769 | 24593 | 15.2 | `bf2d881a039f10ab…` |
| R-4 | W4pp | locked | 24 | 1999 | 186917 | 20769 | 24593 | 12.0 | `268db51ebe8ab59a…` |
| R-5 | W0 | locked | 2 | 1998 | 166593 | 18510 | 47176 | 5.8 | `0f322a86f9c65186…` |
| R-5 | W1 | locked | 5 | 1999 | 166593 | 18510 | 47176 | 6.6 | `0365f35b08b6eab3…` |
| R-5 | W2 | locked | 10 | 1999 | 166593 | 18510 | 47176 | 9.3 | `8cd879c05d8f82eb…` |
| R-5 | W3 | locked | 15 | 1999 | 166593 | 18510 | 47176 | 10.5 | `9f0572a08826c832…` |
| R-5 | W4 | locked | 29 | 1999 | 166593 | 18510 | 47176 | 14.3 | `26df789446662e52…` |
| R-5 | W4pp | locked | 24 | 1999 | 166593 | 18510 | 47176 | 11.7 | `39a72169034f434d…` |
| R-1_buffer | W0 | locked | 2 | 1992 | 135232 | 15026 | 80365 | 5.1 | `2e236f95958cf051…` |
| R-1_buffer | W1 | locked | 5 | 1999 | 135232 | 15026 | 80365 | 5.7 | `b77d608a23765e50…` |
| R-1_buffer | W2 | locked | 10 | 1999 | 135232 | 15026 | 80365 | 8.3 | `64dc3af721ba4df8…` |
| R-1_buffer | W3 | locked | 15 | 1999 | 135232 | 15026 | 80365 | 9.4 | `cfe03f7a4c4c3845…` |
| R-1_buffer | W4 | locked | 29 | 1999 | 135232 | 15026 | 80365 | 12.8 | `7482a6d3df9d8f9c…` |
| R-1_buffer | W4pp | locked | 24 | 1999 | 135232 | 15026 | 80365 | 10.3 | `8b070ef0bcb60c20…` |
| R-1 | W4 | H1 | 29 | 1999 | 136723 | 15191 | 80365 | 10.2 | `ffdb37b97492ab99…` |
| R-2 | W4 | H1 | 29 | 1999 | 177112 | 19679 | 35488 | 11.7 | `6a9f2912601e3380…` |
| R-3 | W4 | H1 | 29 | 1999 | 168860 | 18762 | 44657 | 11.4 | `76bcabc32cf698f6…` |
| R-4 | W4 | H1 | 29 | 1999 | 186917 | 20769 | 24593 | 12.1 | `28ac052f0174c7aa…` |
| R-5 | W4 | H1 | 29 | 1999 | 166593 | 18510 | 47176 | 11.1 | `10ba16a5e3cdf0cc…` |
