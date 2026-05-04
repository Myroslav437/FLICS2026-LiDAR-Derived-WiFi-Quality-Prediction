# Project A — F-C focused diagnostic

Tests whether F-C's leakage-fixed +1.20 dB Δ_LiDAR (in-FOV) survives three orthogonal corrections: hyperparameter robustness, cross-session LiDAR placebo, and LightGBM framework-agnosticism cross-check.

## 0. TL;DR

- **Feature stack**: `leakage_fixed_v1` (telemetry: speed_mps, turn_rate, momentary_current_consumption — same 3-telemetry stack as the leakage-fixed Project A run).
- **Hyperparameter Δ_LiDAR (in-FOV)** across {locked, H1, H2, H3}: +1.20, +1.11, +1.51, +2.29 dB.
- **Placebo gap (Δ_real − Δ_placebo, in-FOV)**: locked = +0.46, H1 = +0.68 dB.
- **LightGBM Δ_LiDAR (in-FOV)** across {default, H1-equiv}: +1.13, +1.26 dB.
- **Diagnostic verdicts**: A = GREEN, B = MARGINAL_DISTRIBUTION_DRIVEN, C = FRAMEWORK_ROBUST.
- **Combined verdict**: **PARTIALLY_ROBUST**.
- **Paper framing recommendation**: F-C's +1.20 dB holds under some but not all robustness checks. Discuss as a candidate for follow-up work in §V; do not headline as a positive result.
- **All 13 fits succeeded**: yes.

## 1. Context

F-C is the leakage-fixed cross-frame fold (Map B held out; trained on Map A only — sessions 15.03.2026 and 24.03.2026). It is the only fold across the entire leakage-fixed Phase 1 run — three Project A folds and five Project B folds — that nominally clears the +1 dB LiDAR-helps threshold (in-FOV Δ_LiDAR = +1.20 dB; B1 = 8.87, B5 = 7.67). This report subjects that headline to three independent robustness checks, mirroring the R-4 diagnostic on Project B's within-session fold and the Hardening A/B suite on F-B.

Locked F-C disambiguation reference (in-FOV): B5' (no LiDAR) = 8.15 dB, B5'' (no AP-relative) = 8.82 dB. B5 (full) being below B5' indicates LiDAR is **complementary**, not removable.

## 2. Diagnostic A — Hyperparameter robustness

Refits B1 and B5 on F-C under three alternative XGBoost hyperparameter sets, holding the chronological-per-session 10% validation split, seed (20260427), and early-stopping rounds (= 100) fixed. Configurations:

| Config | max_depth | eta | reg_lambda | n_estimators_cap |
|---|---:|---:|---:|---:|
| locked | 6 | 0.05 | 1.0 | 2000 |
| H1 | 4 | 0.05 | 1.0 | 2000 |
| H2 | 6 | 0.01 | 20.0 | 5000 |
| H3 | 8 | 0.05 | 20.0 | 2000 |

Per-configuration in-FOV RMSE on F-C:

| Config | B1 in-FOV RMSE | B5 in-FOV RMSE | Δ_LiDAR (in-FOV, dB) | clears +0.5? |
|---|---:|---:|---:|:---:|
| locked | 8.873 | 7.671 | +1.20 | ✓ |
| H1 | 8.279 | 7.170 | +1.11 | ✓ |
| H2 | 8.988 | 7.480 | +1.51 | ✓ |
| H3 | 9.417 | 7.128 | +2.29 | ✓ |

**Disambiguation under H1.** B5' (no-LiDAR) in-FOV RMSE under H1 = 8.077 dB. Compared to B5 under H1 = 7.170 dB.

**Verdict A = GREEN.** ≥ 2 of 3 H-configs clear +0.5 dB Δ_LiDAR — F-C is hyperparameter-robust.

## 3. Diagnostic B — Cross-session LiDAR placebo

Per-session block-shuffle of the 19 LiDAR feature columns (mean_dist_mm…clutter_frac_sector_7) within the F-C training set (train+val pool re-split chronologically post-shuffle). The test set is unshuffled. `clutter_frac_toward_AP` and `is_AP_in_FOV` are AP-relative and not shuffled. Refits B5 under both locked and H1 hyperparameter configurations.

Integrity check (LiDAR pool, train+val, F-C):
- ρ(mean_dist_mm, signal_power) before shuffle: 0.0451
- ρ(mean_dist_mm, signal_power) after shuffle:  -0.0016
- max |Δ mean| across 19 LiDAR columns post-shuffle: 0
- expected: ρ_after ≈ 0; max |Δ mean| ≈ 0 (block shuffle preserves marginals).

Δ_LiDAR comparison:

| Config | Δ_LiDAR (real, dB) | Δ_LiDAR (placebo, dB) | Δ_real − Δ_placebo (dB) | RMSE(B5, placebo, in-FOV) | 95% CI |
|---|---:|---:|---:|---:|---|
| locked | +1.20 | +0.74 | +0.46 | 8.132 | [8.10, 8.16] |
| H1 | +1.11 | +0.43 | +0.68 | 7.847 | [7.82, 7.88] |

**Sanity-check vs Hardening A on F-B.** F-B placebo gap (|Δ_real − Δ_placebo|): 0.27 dB (locked), 0.18 dB (H1) — both below 0.5 dB, consistent with F-B's negative result. F-C placebo gap on this run: +0.46 dB (locked), +0.68 dB (H1).

**Verdict B = MARGINAL_DISTRIBUTION_DRIVEN.** Δ_placebo itself ≥ +0.5 dB — even shuffled LiDAR achieves a positive Δ on F-C; the headline is driven by marginal-distribution effects, not row-aligned signal.

## 4. Diagnostic C — LightGBM cross-check

Refits B1 and B5 on F-C using LightGBM under two parameter sets (default: num_leaves=31; H1-equivalent: num_leaves=15). Same training set, validation split, and seed as the locked XGBoost run. n_estimators=2000, early_stopping=100.

| Framework | Config | B1 in-FOV RMSE | B5 in-FOV RMSE | Δ_LiDAR (in-FOV, dB) | clears +0.5? |
|---|---|---:|---:|---:|:---:|
| XGBoost | locked | 8.873 | 7.671 | +1.20 | ✓ |
| XGBoost | H1 | 8.279 | 7.170 | +1.11 | ✓ |
| LightGBM | default | 8.965 | 7.839 | +1.13 | ✓ |
| LightGBM | H1_equiv | 8.824 | 7.565 | +1.26 | ✓ |

**Verdict C = FRAMEWORK_ROBUST.** At least one LightGBM config clears +0.5 dB Δ_LiDAR — F-C is framework-robust.

## 5. Combined verdict

- Diagnostic A (hyperparameter): **GREEN**
- Diagnostic B (cross-session placebo): **MARGINAL_DISTRIBUTION_DRIVEN**
- Diagnostic C (LightGBM cross-check): **FRAMEWORK_ROBUST**

**Combined: PARTIALLY_ROBUST.**

F-C's +1.20 dB holds under some but not all robustness checks. Discuss as a candidate for follow-up work in §V; do not headline as a positive result.

### 5.1 Verdict thresholds

- **A** GREEN if Δ_LiDAR ≥ +0.5 dB on ≥ 2 of {H1, H2, H3}; RED if < +0.5 dB on ≥ 2 of 3.
- **B** REAL_SIGNAL if Δ_real − Δ_placebo ≥ +0.5 dB; MARGINAL_DISTRIBUTION_DRIVEN if Δ_placebo ≥ +0.5 dB; otherwise NOT_SIGNAL.
- **C** FRAMEWORK_ROBUST if any LightGBM config Δ_LiDAR ≥ +0.5 dB; FRAMEWORK_FRAGILE if all LightGBM Δ_LiDAR < 0.
- **Combined**: ROBUST_POSITIVE = all three pass; ILLUSORY = none pass; PARTIALLY_ROBUST = 1 or 2 of 3 pass.

## 6. Implications for downstream work

- **Rev10 amendment** in §1, §5.5, §11: F-C's +1.20 dB holds under some but not all corrections; flag in §11 risk register.
- **Unified report**: small update to the cross-session subsection.
- **Paper §V Discussion** includes a negative-with-caveat F-C paragraph; do not claim positive headline.

## 7. Reproducibility

- **End-to-end wall-clock** (this run): 42.3 seconds.
- **Reproduce**: `python -m scripts.p1_project_a.run_fc_diagnostic`
- **Seed**: 20260427 (matches the leakage-fixed rerun).
- **Total fits**: 13.

### 7.1 Fit inventory

| Diag | Variant | Descriptor | Framework | Config | best_iter | wall (s) | model SHA-256 |
|---|---|---|---|---|---:|---:|---|
| A | B1 | B1_H1 | xgboost | H1 | 90 | 0.3 | `abd080efc57d9b09…` |
| A | B5 | B5_H1 | xgboost | H1 | 93 | 0.6 | `b46886df5b4bd998…` |
| A | B1 | B1_H2 | xgboost | H2 | 229 | 0.6 | `d9b472e0960d8366…` |
| A | B5 | B5_H2 | xgboost | H2 | 184 | 1.1 | `ef41dfe8ceb80924…` |
| A | B1 | B1_H3 | xgboost | H3 | 46 | 0.4 | `34779ffffeef379c…` |
| A | B5 | B5_H3 | xgboost | H3 | 35 | 1.0 | `de5c89b415c07470…` |
| A | B5p | B5p_H1 | xgboost | H1 | 162 | 0.5 | `db5cc3b44b040b3d…` |
| B | B5 | B5_placebo_locked | xgboost | locked | 57 | 0.7 | `cd60a3b251f9fc3b…` |
| B | B5 | B5_placebo_H1 | xgboost | H1 | 86 | 0.6 | `9d7aae78e8003fa7…` |
| C | B1 | B1_lgb_default | lightgbm | default | 43 | 0.3 | `67d005d58d8bdc28…` |
| C | B5 | B5_lgb_default | lightgbm | default | 47 | 0.5 | `8096edefc732e109…` |
| C | B1 | B1_lgb_H1_equiv | lightgbm | H1_equiv | 68 | 0.3 | `b32073c269bcb157…` |
| C | B5 | B5_lgb_H1_equiv | lightgbm | H1_equiv | 60 | 0.4 | `f03de380f9283fcd…` |

**Locked-artefact integrity.** A pre/post SHA-256 comparison over `scripts/p1_project_a/{models,cache,results,diagnostic}/` (excluding the new `fc_diagnostic/` subdirectories) and `scripts/p1_project_b/` is recorded in `scripts/p1_project_a/fc_diagnostic_log.md`. No locked artefact was modified.
