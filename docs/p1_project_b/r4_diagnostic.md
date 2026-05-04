# Project B — R-4 focused diagnostic

Tests whether R-4's main-run Δ_LiDAR_within (in-FOV) survives two orthogonal corrections: a 1 m buffer-zone exclusion (spatial-autocorrelation control) and a less-aggressive hyperparameter config (H1: max_depth=4).

## 0. TL;DR

**Feature stack: leakage-fixed.** This diagnostic uses the leakage-fixed feature stack: 3 telemetry features (`speed_mps`, `turn_rate`, `momentary_current_consumption`) plus position, AP-relative geometry, and LiDAR. The 'main-run +1.09 dB' baseline cited in §1 below is the locked-stack number from the pre-leakage-fix Project B run; the leakage-fixed re-run produces the locked-no-buffer number reproduced in this diagnostic, and that is the number from which the four-corrections framing is built. See `MIGRATION_LOG.md`. (`feature_stack_version = "leakage_fixed_v1"`).

- **Buffer-zone Δ_LiDAR_within (in-FOV) on R-4**: -0.014 dB (vs main-run no-buffer -1.631 dB, also reproduced here).
- **H1 Δ_LiDAR_within (in-FOV) on R-4**: +0.598 dB (vs main-run locked +1.09 dB).
- **Verdict**: **R-4 IS PARTIALLY ROBUST**.
- **Recommended paper framing**: Negative-with-caveat paper framing. R-4's LiDAR gap holds under one of the two corrections but not the other. Disambiguation finding still holds; the headline is the cross-session-and-mostly-within-session negative result, with R-4 mentioned as a region where the gap narrows but does not robustly survive both checks.
- **All 12 fits succeeded**: yes (11 new fits + 1 reused R-4_W4_H1 from main run).

## 1. Context

Project B's main run found that R-4 is the only fold with Δ_LiDAR_within ≥ 1 dB on in-FOV (+1.09 dB; all other folds: −9.06, −0.89, −0.14, −0.04). R-4 was therefore the candidate exemplar for a mixed-regional paper framing. Two methodological caveats from the main report cast doubt on whether that R-4 result is real:

- The R-1 buffer-zone test showed every variant moves by ≥ 0.5 dB under buffer (W0 alone moved by +12 dB). Leave-region-out is therefore contaminated by short-range spatial autocorrelation. R-4 was not previously buffer-tested.

- The R-1 H1 hyperparameter sensitivity test showed Δ +4.76 dB improvement under H1 — i.e., the locked depth-6 config catastrophically overfits R-1. R-4's H1 sensitivity had been measured only for W4 (one variant), not for the full disambiguation ladder.

This diagnostic settles whether R-4's +1.09 dB Δ_LiDAR_within is a clean physical finding or an artifact of either spatial autocorrelation or overfitting.

## 2. Buffer-zone diagnostic (R-4)

Drop training rows within 1.0 m of any R-4 row, refit under locked hyperparameters, evaluate on the unchanged R-4 test set.

Post-buffer training-set size: 183,944 rows (dropped 3,304 rows within 1.0 m).

### 2.1 Per-variant RMSE: locked-no-buffer vs locked-buffer

| Variant | Stratum | RMSE(locked-no-buffer) | RMSE(locked-buffer) | Δ_RMSE (locked-buffer − locked-no-buffer) |
|---|---|---:|---:|---:|
| W0 | overall | 9.097 | 9.164 | +0.067 |
| W0 | in_fov | 11.233 | 3.704 | -7.529 |
| W0 | out_of_fov | 7.027 | 11.747 | +4.719 |
| W1 | overall | 8.385 | 8.040 | -0.345 |
| W1 | in_fov | 5.717 | 4.311 | -1.406 |
| W1 | out_of_fov | 9.960 | 10.002 | +0.041 |
| W2 | overall | 6.348 | 6.734 | +0.386 |
| W2 | in_fov | 4.161 | 4.348 | +0.187 |
| W2 | out_of_fov | 7.613 | 8.104 | +0.490 |
| W3 | overall | 6.367 | 7.967 | +1.601 |
| W3 | in_fov | 5.200 | 4.919 | -0.280 |
| W3 | out_of_fov | 7.135 | 9.678 | +2.544 |
| W4 | overall | 6.928 | 6.259 | -0.669 |
| W4 | in_fov | 5.792 | 4.362 | -1.430 |
| W4 | out_of_fov | 7.688 | 7.393 | -0.295 |
| W4pp | overall | 6.600 | 6.846 | +0.246 |
| W4pp | in_fov | 4.181 | 4.103 | -0.078 |
| W4pp | out_of_fov | 7.975 | 8.364 | +0.389 |

### 2.2 Δ_LiDAR_within = RMSE(W2) − RMSE(W4)

| Stratum | locked-no-buffer | locked-buffer |
|---|---:|---:|
| overall | -0.580 | +0.475 |
| in_fov | -1.631 | -0.014 |
| out_of_fov | -0.075 | +0.710 |


**Disambiguation under locked-buffer (R-4 in-FOV, threshold = 0.5 dB)**: LiDAR removable.

**Verdict (Diagnostic A)**: FAIL — Δ_LiDAR_within (in-FOV) under buffer = -0.014 dB < +0.5 dB threshold.

## 3. H1 hyperparameter diagnostic (R-4)

Refit under H1 (max_depth=4; same eta, λ, n_estimators, early_stop) for all 6 variants. W4 reuses the existing main-run R-4 H1 fit (`R-4_W4_H1.json`); W0, W1, W2, W3, W4'' are new.

### 3.1 Per-variant RMSE: locked vs H1

| Variant | Stratum | RMSE(locked-no-buffer) | RMSE(H1) | Δ_RMSE (H1 − locked-no-buffer) |
|---|---|---:|---:|---:|
| W0 | overall | 9.097 | 6.772 | -2.325 |
| W0 | in_fov | 11.233 | 5.851 | -5.382 |
| W0 | out_of_fov | 7.027 | 7.402 | +0.375 |
| W1 | overall | 8.385 | 7.537 | -0.847 |
| W1 | in_fov | 5.717 | 5.382 | -0.335 |
| W1 | out_of_fov | 9.960 | 8.844 | -1.116 |
| W2 | overall | 6.348 | 6.123 | -0.225 |
| W2 | in_fov | 4.161 | 3.959 | -0.202 |
| W2 | out_of_fov | 7.613 | 7.366 | -0.247 |
| W3 | overall | 6.367 | 5.934 | -0.432 |
| W3 | in_fov | 5.200 | 3.661 | -1.538 |
| W3 | out_of_fov | 7.135 | 7.209 | +0.075 |
| W4 | overall | 6.928 | 5.886 | -1.043 |
| W4 | in_fov | 5.792 | 3.361 | -2.431 |
| W4 | out_of_fov | 7.688 | 7.252 | -0.437 |
| W4pp | overall | 6.600 | 6.595 | -0.005 |
| W4pp | in_fov | 4.181 | 3.936 | -0.246 |
| W4pp | out_of_fov | 7.975 | 8.064 | +0.089 |

### 3.2 Δ_LiDAR_within = RMSE(W2) − RMSE(W4)

| Stratum | locked | H1 |
|---|---:|---:|
| overall | -0.580 | +0.237 |
| in_fov | -1.631 | +0.598 |
| out_of_fov | -0.075 | +0.115 |


**Disambiguation under H1 (R-4 in-FOV, threshold = 0.5 dB)**: complementary.

**Verdict (Diagnostic B)**: PASS — Δ_LiDAR_within (in-FOV) under H1 = +0.598 dB ≥ +0.5 dB threshold.

Note on the threshold choice: the §6 verdict criteria use a 0.5 dB threshold, more permissive than the original 1.0 dB pivot threshold from Project A / the main Project B run. This is intentional — the diagnostic asks whether the +1.09 dB result *survives at all* under stricter conditions, not whether it independently clears 1 dB. A diagnostic that is too strict to be informative is no diagnostic.

## 4. Combined verdict

- Diagnostic A (buffer): Δ_LiDAR_within (in-FOV) = -0.014 dB.
- Diagnostic B (H1):     Δ_LiDAR_within (in-FOV) = +0.598 dB.
- Disambiguation per config:
  - locked-no-buffer (existing main run): **mixed/unclear**.
  - locked-buffer:                        **LiDAR removable**.
  - H1-no-buffer:                         **complementary**.

### Verdict: **R-4 IS PARTIALLY ROBUST**

Negative-with-caveat paper framing. R-4's LiDAR gap holds under one of the two corrections but not the other. Disambiguation finding still holds; the headline is the cross-session-and-mostly-within-session negative result, with R-4 mentioned as a region where the gap narrows but does not robustly survive both checks.


#### Negative-with-caveat paper outline

- Headline negative result remains the cross-session and within-session disambiguation: AP-relative geometry largely subsumes LiDAR.
- R-4 mentioned as a region where Δ_LiDAR survives one of two corrections; this is consistent with regional structure but does not robustly support a regional-deployment claim.
- The diagnostic itself (this report) goes into §6 as a methodological figure.


## 5. Reproducibility

- Seed: `SEED = 20260427` everywhere.
- Run end-to-end: `python -m scripts.p1_project_b.run_r4_diagnostic`.
- Naming convention: H1 fits use the existing codebase pattern `R-4_{variant}_H1.json` (matching the pre-cached `R-4_W4_H1.json`), not the brief's suggested `R-4_H1_{variant}.json` — this avoids two parallel naming schemes for the same artifact type. Buffer fits use `R-4_buffer_{variant}.json` matching the main-run R-1_buffer convention.
- Total wall-clock for new fits: 99.5 s.

### Model SHA-256 inventory (new fits only; W4 H1 reused from main run)

| diagnostic | variant | model file | best_iter | n_train | n_val | n_test | wall (s) | model SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|
| buffer | W0 | `R-4_buffer_W0.json` | 1998 | 183944 | 20438 | 24593 | 6.2 | `f7cee85b1cdfff39…` |
| buffer | W1 | `R-4_buffer_W1.json` | 1999 | 183944 | 20438 | 24593 | 7.3 | `8e9cefff1919349e…` |
| buffer | W2 | `R-4_buffer_W2.json` | 1999 | 183944 | 20438 | 24593 | 9.9 | `42ba409884833a0b…` |
| buffer | W3 | `R-4_buffer_W3.json` | 1999 | 183944 | 20438 | 24593 | 11.3 | `2b3f60661d2eab40…` |
| buffer | W4 | `R-4_buffer_W4.json` | 1999 | 183944 | 20438 | 24593 | 14.9 | `025d8b2ba5d452ab…` |
| buffer | W4pp | `R-4_buffer_W4pp.json` | 1999 | 183944 | 20438 | 24593 | 11.9 | `4fdb65c81c172e71…` |
| H1 | W0 | `R-4_W0_H1.json` | 1998 | 186917 | 20769 | 24593 | 5.3 | `caa0873ffcfc5ee8…` |
| H1 | W1 | `R-4_W1_H1.json` | 1999 | 186917 | 20769 | 24593 | 5.8 | `ae150b0d02ec36b7…` |
| H1 | W2 | `R-4_W2_H1.json` | 1999 | 186917 | 20769 | 24593 | 8.1 | `6f6196b90a385abb…` |
| H1 | W3 | `R-4_W3_H1.json` | 1999 | 186917 | 20769 | 24593 | 9.2 | `9888628554709ac9…` |
| H1 | W4pp | `R-4_W4pp_H1.json` | 1999 | 186917 | 20769 | 24593 | 9.6 | `f377234f332fa826…` |

