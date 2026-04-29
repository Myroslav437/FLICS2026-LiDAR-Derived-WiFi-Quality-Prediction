# Unified report — source inventory

This file documents the source-by-source inventory built during Pass 1 of `docs/unified_report.md` construction. It serves as a traceability ledger for the unified report.

**Updated 2026-04-28** for Stage 2 of the hardening merge: Rev9 replaces Rev8 as the framing source; the standalone `docs/p1_hardening/results_report.md` no longer exists post-merge — its content has been dissolved into the four target reports below.

## 1. Source documents read (in order of priority)

| # | Path | Role | Read state |
|---|---|---|---|
| 1 | `docs/proposal_rev9.md` | Narrative backbone (post-hardening framing); contribution claims; deployment guidance; paper structure; risk register with eight-experiment, five-orthogonal-axis convergence | full |
| 2 | `docs/initial_dataset_analysis/report.md` | Exploratory data analysis (motion, WiFi, LiDAR distributions; trajectories; correlations) | full |
| 3 | `docs/p0_analysis/report.md` | Phase 0 v1/v2/v3 (anomaly mask, AGV-body mask, FOV, AP path-loss fits, gates) | full |
| 4 | `docs/p1_dataset_analysis/report.md` | Phase 1 dataset construction & validation; **§9 dataset noise floor (Hardening D — σ_intra = 4.95 dB at 0.5 m granularity; cleanest model 3 dB below at RMSE = 1.99 dB) merged in 2026-04-28** | full |
| 5 | `docs/p1_project_a/results_report.md` | Cross-session LORO ablation, B0–B5, disambiguation, XAI, RQ4, hyperparameter diagnostic (inline as §10); **§11 Hardening A (cross-session placebo on F-B) and §12 Hardening B (LightGBM cross-check on F-B) merged in 2026-04-28** | full |
| 6 | `docs/p1_project_b/results_report.md` | Within-session leave-region-out, W0–W4, disambiguation, buffer/H1 sensitivity (§§1–3); R-4 buffer + H1 robustness diagnostic (§4 individual diagnostics; **§4.5 Hardening C combined H1+buffer; §4.6 Hardening E within-session placebo; §4.7 four-diagnostic combined verdict — all merged in 2026-04-28**); XAI (§5); synthesis and recommendation (§§6–7) | full |
| 7 | `docs/time_sync/report.md` | Per-day τ̂ calibration methodology | full |

> **Note 1 (post-compose merge, 2026-04-28).** The R-4 robustness diagnostic was originally a separate file `docs/p1_project_b/r4_diagnostic.md`. After the unified report was first composed, the diagnostic was merged into `docs/p1_project_b/results_report.md` as §4. All citations in `docs/unified_report.md` were updated from `r4_diagnostic.md §X` to `results_report.md §Y`. The unified report's §7.13 narrative is unchanged — only citation paths were rewritten.
>
> **Note 2 (Stage 2 regeneration, 2026-04-28).** The `scripts/p1_hardening/` package and `docs/p1_hardening/` directory were dissolved into `scripts/p1_project_a/`, `scripts/p1_project_b/`, `scripts/p1_dataset_analysis/`, `docs/p1_project_a/`, `docs/p1_project_b/`, `docs/p1_dataset_analysis/`. Stage 1 of the merge moved 48 byte-stable artifacts (15 models + 15 caches + 9 results + 2 docs assets + ...) at unchanged SHA-256, edited 7 source files for import-path updates, dissolved 5 source files (`__init__.py` rewritten as deprecation shim; `config.py`, `build_results_report.py`, `run_all.py`, `docs/p1_hardening/results_report.md` deleted with content reproduced in the four target docs). Stage 2 (this file's update) regenerated `docs/unified_report.md` to use Rev9 framing and pull from the post-merge file layout. See `scripts/maintenance/merge_hardening_log.md` for the full audit trail.

## 2. Hyperparameter diagnostic location resolution

**Status**: `docs/p1_project_a/hyperparam_diagnostic.md` does NOT exist. `docs/hyperparam_diagnostic.md` does NOT exist.

**Canonical location**: inline at `docs/p1_project_a/results_report.md` §10 (sub-sections §10.1 through §10.10). The unified report §6.13 reproduces this content directly from `results_report.md` §10. No reconstruction from the proposal Rev9 was needed; the primary source is the inline §10. Numbers in proposal Rev9 §1, §5.5, §11 are consistent with the inline §10.

## 3. Hardening location resolution (post-2026-04-28 merge)

**Status**: `docs/p1_hardening/results_report.md` no longer exists. `scripts/p1_hardening/` is a deprecation shim only (raises ImportError when imported).

**Canonical locations** for hardening content:

| Hardening experiment | Source (current) | Section in unified report |
|---|---|---|
| **A** — Cross-session LiDAR placebo on F-B | `docs/p1_project_a/results_report.md` §11 | unified §6.14 |
| **B** — LightGBM cross-check on F-B | `docs/p1_project_a/results_report.md` §12 | unified §6.15 |
| **C** — Combined H1 + 1m buffer on R-4 | `docs/p1_project_b/results_report.md` §4.5 | unified §7.13.4 |
| **D** — Dataset noise floor | `docs/p1_dataset_analysis/report.md` §9 | unified §3.8 |
| **E** — Within-session LiDAR placebo on R-4 | `docs/p1_project_b/results_report.md` §4.6 | unified §7.13.5 |

The unified report's §7.13.6 contains the four-diagnostic combined verdict (buffer + H1 + Hardening C + Hardening E), pulled from `docs/p1_project_b/results_report.md` §4.7.

## 4. Table inventory (full-fidelity reproductions required)

### Project A tables — `docs/p1_project_a/tables/`

| Table file | Purpose | Reproduced in unified report at |
|---|---|---|
| `loro_ablation_rmse.md` | Headline ablation 3 folds × 3 strata × 6 variants (B0–B5) | §6.4 |
| `delta_lidar.md` | Δ_LiDAR per fold per stratum (B5 vs B1) | §6.5 |
| `delta_angle.md` | Δ_angle per fold per stratum (B0 vs B1) | §6.6 |
| `delta_fov.md` | Δ_FOV per fold (B5) | §6.6 |
| `disambig_summary.md` | Disambiguation B5/B5'/B5'' overall RMSE per fold | §6.7 |
| `rq4_summary.md` | RQ4 same-map and full pool with mean\|SHAP\|_session | §6.8 |
| `xai_1_group_importance.md` | XAI-1 feature-group importance per fold | §6.9 |
| `xai_2_sign_consistency.md` | XAI-2 sign-of-effect consistency table | §6.10 |

All tables are also reproduced inline in `docs/p1_project_a/results_report.md` §2–§5; the unified report cites the report file directly because the content matches exactly. Hardening A tables reproduced from `docs/p1_project_a/results_report.md` §11.2 / §11.3 in unified §6.14; Hardening B tables from §12.2 / §12.3 in unified §6.15.

### Project B tables — `docs/p1_project_b/tables/`

| Table file | Purpose | Reproduced in unified report at |
|---|---|---|
| `regions_summary.md` | Per-region statistics (n_rows, FOV split, dist range, bbox) | §7.1, §7.2 |
| `wlro_ablation_rmse.md` | Headline ablation 5 folds × 3 strata × 6 variants (W0–W4'') | §7.4 |
| `delta_lidar_within.md` | Δ_LiDAR_within per fold per stratum | §7.5 |
| `delta_ap_relative_within.md` | Δ_AP-relative_within per fold per stratum | §7.6 |
| `delta_position_within.md` | Δ_position (W0 vs per-region mean baseline) | §7.7 |
| `disambig_summary.md` | Disambiguation W4/W4'/W4'' overall RMSE per fold | §7.8 |
| `buffer_sensitivity.md` | R-1 buffer-zone sensitivity per variant per stratum | §7.9 |
| `hyperparam_sensitivity.md` | W4 locked vs H1 (max_depth=4) per fold | §7.10 |
| `xai_group_importance_R-3.md` | XAI feature-group importance on R-3 W4 | §7.11 |
| `xai_sign_R-3.md` | XAI sign-of-effect single-fold (R-3 W4) | §7.11 |

Hardening C tables reproduced from `docs/p1_project_b/results_report.md` §4.5.2 / §4.5.3 in unified §7.13.4; Hardening E tables from §4.6.2 / §4.6.3 in unified §7.13.5.

### Dataset analysis tables — `docs/p1_dataset_analysis/tables/`

| Table file | Purpose | Reproduced in unified report at |
|---|---|---|
| `dataset_noise_floor.md` | Hardening D summary: per-cell |Δ| stats; per-session σ_intra; model-vs-σ_intra comparison | §3.8 |

## 5. Figure inventory (relative paths from `docs/unified_report.md`)

### Initial dataset analysis — `docs/initial_dataset_analysis/figures/`

- `session_overview.png` — per-session matched pairs and motion fraction (§3.1)
- `dist_motion.png` — motion-signal distributions (§3.1 supplementary; §3.2 motion subsection)
- `dist_wifi.png` — WiFi metric distributions (§3.3)
- `dist_lidar.png` — LiDAR feature distributions (§3.4)
- `traj_signal_power.png` — trajectory by signal_power (§3.5)
- `traj_ping.png` — trajectory by ping (§3.5)
- `traj_clutter.png` — trajectory by clutter_frac (§3.5)
- `signal_vs_speed.png` — signal vs speed (§3.6)
- `signal_power_vs_lidar.png` — signal_power vs LiDAR features (§3.6)
- `ping_vs_lidar.png` — ping vs LiDAR features (§3.6)
- `corr_pearson.png` — Pearson correlations (§3.7)
- `corr_spearman.png` — Spearman correlations (§3.7)

### Phase 0 — `docs/p0_analysis/figures/`

- `p0_7_v3_confidence_hist.png` — confidence histogram with saddle (§4.1)
- `p0_7_v3_gmm.png` — GMM fit (§4.1 sanity)
- `p0_7_v3_roc.png` — ROC analysis (§4.1 sanity)
- `p0_7_manual_reposition_example.png` — example anomaly trace (§4.1)
- `p0_7_motor_overheat_example.png` — example anomaly trace (§4.1)
- `p0_3_polar.png` — AGV-body LiDAR mask polar (§4.2)
- `p0_0_polar_overlay_15-03-2026_vs_24-03-2026.png` — frame-sharing overlay (§4.3)
- `p0_2_residual_map_v3_15-03-2026.png` — per-session residual map (§4.5)
- `p0_2_residual_map_v3_24-03-2026.png` — per-session residual map (§4.5)
- `p0_2_residual_map_v3_25-02-2026.png` — per-session residual map (§4.5)
- `p0_1_overlay.png` — spatial overlap overlay (§4.6)
- `p0_1_overlap_heatmap.png` — overlap heatmap (§4.6)
- `p0_5_semivariograms.png` — spatial autocorrelation (§4.8)
- `p0_6_delta_histogram.png` — same-cell |Δ| histogram (§4.9)

### Phase 1 dataset — `docs/p1_dataset_analysis/figures/`

- `lidar_scalars_15_03_2026.png`, `lidar_scalars_24_03_2026.png`, `lidar_scalars_25_02_2026.png` — per-session LiDAR scalar histograms (§5.2)
- **`dataset_noise_floor.png` — same-cell |Δ| histogram (Hardening D), referenced from §3.8 of the unified report. Originally `docs/p1_hardening/figures/dataset_noise_floor.png`; moved with byte-stable SHA-256 on 2026-04-28.**

### Project A — `docs/p1_project_a/figures/`

- `xai_1_feature_group_importance.png` — group importance bar chart (§6.9)
- `xai_1_beeswarm_F-A.png`, `xai_1_beeswarm_F-B.png`, `xai_1_beeswarm_F-C.png` — beeswarms (§6.9)
- `xai_3_spatial_F-A.png`, `xai_3_spatial_F-B.png`, `xai_3_spatial_F-C.png`, `xai_3_spatial_combined.png` — spatial maps (§6.11)
- `xai_4_shap_fov_F-A.png`, `xai_4_shap_fov_F-B.png`, `xai_4_shap_fov_F-C.png` — SHAP × FOV (§6.12)

### Project B — `docs/p1_project_b/figures/`

- `regions_map.png` — spatial regions K-means partition (§7.2)
- `wlro_ablation_chart.png` — ablation bar chart (§7.4)
- `xai_group_importance_R-3.png` — XAI group importance (§7.11)
- `xai_spatial_R-3.png` — XAI spatial dominance (§7.11)

### Time sync — `docs/time_sync/figures/`

- `xcorr_per_day.png` — per-day cross-correlation (§3.2 supplementary)
- `forest_per_day.png` — per-day τ̂ forest plot (§3.2)

## 6. Cross-source inconsistency resolutions

The Phase 0 path-loss exponents and R² values evolved through three revisions (v1, v2, v3). The unified report uses the **v3** numbers throughout because (a) Phase 0 `report.md` §14 explicitly designates v3 as canonical, and (b) Phase 1 was built on the v3 anomaly mask.

| Quantity | v1 (free fit / prior) | v2 (truth AP, anomaly v2) | v3 (truth AP, anomaly v3) — **canonical** | Used in unified report |
|---|---|---|---|---|
| n_d 15.03 | 1.237 | 1.217 | **1.224** | 1.224 |
| n_d 24.03 | 1.000 (prior) | 0.220 | **0.239** | 0.239 |
| n_d 25.02 | 1.000 (prior) | 0.377 | **0.379** | 0.379 |
| R² 15.03 | 0.355 | 0.352 | **0.357** | 0.357 |
| R² 24.03 | −0.033 | 0.003 | **0.003** | 0.003 |
| R² 25.02 | −0.176 | 0.086 | **0.088** | 0.088 |
| P0_d 15.03 (dB) | — | −25.52 | **−25.49** | −25.49 |
| P0_d 24.03 (dB) | — | −38.02 | **−37.83** | −37.83 |
| P0_d 25.02 (dB) | — | −27.89 | **−27.89** | −27.89 |
| n_d 15.03 95% CI | — | [1.189, 1.258] | **[1.181, 1.268]** | [1.181, 1.268] |
| n_d 24.03 95% CI | — | [0.153, 0.380] | **[0.178, 0.406]** | [0.178, 0.406] |
| n_d 25.02 95% CI | — | [0.349, 0.426] | **[0.366, 0.426]** | [0.366, 0.426] |

Other resolved inconsistencies:

- **Anomaly counts**: proposal Rev9 §3.6 cites "74,437 of 681,593"; `docs/p1_dataset_analysis/report.md` §3 reports 74,437 total. Per-session breakdown: 47,746 + 10,535 + 16,156 = 74,437. Phase 0 §14.3: 47,738 + 10,535 + 16,156 = 74,429 — the 8-row gap is `telemetry_nan` rows on 15.03. Unified report uses Phase 1 reconciled total **74,437**.
- **Gate C status**: Phase 0 v3 §14.5.2 downgraded GREEN → YELLOW. Proposal Rev9 §7 lists Gate C as YELLOW. Unified report uses **YELLOW**.
- **F-B in-FOV Δ_LiDAR (real)**: proposal Rev9 §1 says "−0.52 dB"; Project A §0 says "−0.515 dB" (rounded to −0.52). Unified report uses **−0.52 dB** (or −0.515 verbatim when reproducing tables).
- **R-4 in-FOV Δ_LiDAR (main)**: proposal Rev9 §6.2 = "+1.09"; results_report.md §0 = "+1.085". Unified report uses **+1.09 dB** (or +1.085 verbatim).
- **R-4 H1 in-FOV Δ_LiDAR**: proposal Rev9 = "−0.33 dB"; results_report.md §4.3.2 = "−0.331". Unified report uses **−0.33 dB** (or −0.331 verbatim).
- **R-4 buffer in-FOV Δ_LiDAR**: proposal Rev9 = "−1.27 dB"; results_report.md §4.2.2 = "−1.267". Unified report uses **−1.27 dB** (or −1.267 verbatim).
- **R-4 H1+buffer combined Δ_LiDAR (Hardening C)**: proposal Rev9 §6.4 = "−0.83 dB"; results_report.md §4.5.2 = "−0.83". Unified report uses **−0.83 dB** consistently.
- **R-4 within-session placebo (Hardening E)**: proposal Rev9 §6.5 = "Δ_real (+1.11) vs Δ_placebo (+0.13); gap = 0.98"; results_report.md §4.6.3 reproduces these numbers exactly. Unified report uses **+1.11 / +0.13 / 0.98 dB**.
- **σ_intra (Hardening D)**: proposal Rev9 §3.7 = "4.95 dB"; dataset_analysis report.md §9.2 = "4.953" (rounded to 4.95). Per-session: 15.03 = 4.901; 24.03 = 5.005; consistent across both sources. Unified report uses **4.95 dB** (or 4.953 / 4.901 / 5.005 verbatim where the precision matters).
- **Cleanest model RMSE**: proposal Rev9 §6.6 = "1.99 dB"; Project B §4.3.1 = "1.995 dB". Unified report uses **1.99 dB** rounded; or 1.995 dB verbatim where the precision matters.
- **Empirical FOV width**: Phase 0 v2 §3 = "221.6° wide"; Phase 0 v2 §0 = "≈ 222° wide"; proposal Rev9 = "222°". Unified report uses **222°** as the rounded value with `[−110.0°, 111.6°]` as the precise sector.

## 7. Pages of out-of-scope material consulted but not used

- `docs/obsolete/AIDI2026_Methodology_v2.txt`, `docs/obsolete/AIDI2026_Research_Proposal_v2.txt`, `docs/obsolete/Lidar Data Description.txt`, `docs/obsolete/var_list_Myroslav_25-02_2026.txt` — pre-Rev8 materials; superseded by Rev9.
- `docs/proposal_rev*.md` revisions earlier than Rev9 — superseded.
- `docs/paper/{IEEEtran.cls, paper.tex}` — paper template scaffold; not source content.

## 8. Pandoc availability

`pandoc` is not on PATH on the working machine. Pass 4 (PDF render) is therefore **skipped** without error. The unified report is delivered as `docs/unified_report.md` only. The verification report documents this skip explicitly.
