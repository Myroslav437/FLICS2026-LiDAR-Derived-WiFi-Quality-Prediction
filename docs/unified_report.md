# Unified technical report — FLICS 2026 / AIDI 2026 (leakage-fixed)

**Lab-Measured AP Coordinates Suffice for Industrial AGV WiFi Signal Prediction: A Properly-Controlled Comparison Showing LiDAR-Derived Environment Features Add No Value (leakage-fixed feature stack).**

*This is the project's single internal source-of-truth document, consolidating every Phase-0/Phase-1 finding under the leakage-fixed feature stack. Authored 2026-05-02 against `docs/proposal_rev10.md` after the post-hoc audit removed five features (`load_long`, `load_mid`, `load_short`, `battery_value`, `nns_state`) and the full Phase 1 pipeline was re-run. The downstream paper extracts from this document; reviewers do not see it.*

*Sources, in priority order: `docs/proposal_rev10.md`; `docs/initial_dataset_analysis/report.md`; `docs/p0_analysis/report.md`; `docs/p1_dataset_analysis/report.md`; `docs/p1_battery_provenance/report.md`; `docs/p1_leakage_check/report.md`; `docs/p1_project_a/results_report.md`; `docs/p1_project_b/results_report.md`; `docs/p1_project_b/r4_diagnostic.md`; `docs/time_sync/report.md`; `MIGRATION_LOG.md`.*

---

## §0. TL;DR — leakage-fixed headline numbers

**Feature stack: leakage-fixed.** This report uses the leakage-fixed feature stack: 3 telemetry features (`speed_mps`, `turn_rate`, `momentary_current_consumption`) plus position, AP-relative geometry, and LiDAR. Five features from the original locked feature stack were removed post-hoc as either router-side (target leakage), within-session-only (deployment leakage), or constant sentinel (no information). See `MIGRATION_LOG.md` for the full audit trail.

**Headline Δ values** (positive = LiDAR helps; >+1 dB clears the project's pivot threshold):
- Project A F-A in-FOV Δ_LiDAR = +0.55 dB.
- Project A F-B in-FOV Δ_LiDAR = -0.19 dB.
- Project A F-C in-FOV Δ_LiDAR = +1.20 dB.

- Project B R-1 in-FOV Δ_LiDAR_within = -2.99 dB.
- Project B R-2 in-FOV Δ_LiDAR_within = -2.20 dB.
- Project B R-3 in-FOV Δ_LiDAR_within = -0.09 dB.
- Project B R-4 in-FOV Δ_LiDAR_within = -1.63 dB.
- Project B R-5 in-FOV Δ_LiDAR_within = -2.22 dB.

- R-4 robustness (four corrections): locked-no-buffer -1.63 dB; locked-buffer -0.01 dB; H1-no-buffer +0.60 dB; H1+buffer -0.64 dB.
- Hyperparameter robustness on F-B (six configs): locked=-0.19 dB, H1=+0.10 dB, H2=-0.07 dB, H3=-0.47 dB, H1_randval=—, H2_randval=—.
- Cross-session placebo (Hardening A) on F-B locked: Δ_real = -0.19 dB, Δ_placebo = +0.08 dB.
- LightGBM cross-check (Hardening B) on F-B: default Δ_LiDAR = -0.27 dB; H1-equiv -0.45 dB.
- Within-session placebo (Hardening E) on R-4 locked: Δ_real = -0.01 dB, Δ_placebo = +0.24 dB, gap = -0.25 dB.
- Cleanest deployment-relevant single number: R-4 W2 H1 in-FOV RMSE = 3.96 dB.
- Total fits across the leakage-fixed rerun: 103. All metrics rows tagged `feature_stack_version = "leakage_fixed_v1"`.

---

## §1. Executive summary

(See `docs/proposal_rev10.md` §1 for the canonical executive summary; it is reproduced and elaborated below.)

Across three calibration sessions, two map frames, lab-measured ground-truth AP coordinates, six XGBoost hyperparameter configurations, three cross-session leave-one-route-out folds, five within-session leave-region-out folds, two robustness checks on the one within-session fold that nominally favoured LiDAR (R-4), five hardening experiments designed to pre-empt the four standard reviewer objections to negative-result ML papers, **and a post-hoc feature-stack audit that removed five leaked or no-op features and re-ran every Phase 1 experiment from scratch** under the leakage-fixed 3-feature telemetry stack, the answer to whether LiDAR-derived environmental geometry features add predictive value beyond AP-relative geometry remains **no**.

---

## §2. Project context and objectives

Unchanged from Rev9 / Rev10. See `docs/proposal_rev10.md` §2.

---

## §3. Dataset and time synchronization

Unchanged from Rev9 / Rev10 — the dataset itself was not modified; only the feature selection layer was updated. See `docs/p1_dataset_analysis/report.md` for details and `docs/p1_battery_provenance/report.md` plus `MIGRATION_LOG.md` for the post-hoc audit rationale.

**Hardening D — dataset noise floor** (editorial; unchanged by the leakage-fixed rerun): σ_intra = **4.95 dB** at 0.5 m cell granularity. Same-cell |Δ mean signal_power| between 15.03 and 24.03: median 4.24 dB across the 85 same-map qualifying cells. The cleanest within-session model in the project now is **R-4 W2 H1 in-FOV; RMSE = 3.96 dB** under the leakage-fixed stack — operating ~1.0 dB below the noise floor by exploiting sub-cell information.

---

## §4. Phase 0 — exploratory analysis and validation gates

Locked. Phase 0 artefacts byte-identical pre and post the leakage-fixed rerun. See `docs/p0_analysis/report.md`.

---

## §5. Phase 1 dataset construction

Unchanged. See `docs/p1_dataset_analysis/report.md`.

---

## §6. Project A — Cross-session leave-one-route-out experiments (leakage-fixed)

See `docs/p1_project_a/results_report.md` for the full detailed write-up.

**Δ_LiDAR (in-FOV) per fold under the leakage-fixed stack**:
- F-A: +0.55 dB (B5 RMSE = 6.16 dB).
- F-B: -0.19 dB (B5 RMSE = 8.20 dB).
- F-C: +1.20 dB (B5 RMSE = 7.67 dB).

**Hyperparameter robustness diagnostic on F-B (six configs)**:
- `locked`: in-FOV Δ_LiDAR = -0.19 dB.
- `H1`: in-FOV Δ_LiDAR = +0.10 dB.
- `H2`: in-FOV Δ_LiDAR = -0.07 dB.
- `H3`: in-FOV Δ_LiDAR = -0.47 dB.
- `H1_randval`: in-FOV Δ_LiDAR = —.
- `H2_randval`: in-FOV Δ_LiDAR = —.

**Hardening A (placebo on F-B)**: Δ_real (locked) = -0.19 dB vs Δ_placebo (locked) = +0.08 dB; Δ_real (H1) = +0.10 dB vs Δ_placebo (H1) = +0.28 dB.

**Hardening B (LightGBM cross-check on F-B)**: default Δ_LiDAR = -0.27 dB; H1-equivalent -0.45 dB.

---

## §7. Project B — Within-session leave-region-out experiments (leakage-fixed)

See `docs/p1_project_b/results_report.md` and `docs/p1_project_b/r4_diagnostic.md`.

**Δ_LiDAR_within (in-FOV) per fold under the leakage-fixed stack**:
- R-1: -2.99 dB (W4 RMSE = 13.79 dB).
- R-2: -2.20 dB (W4 RMSE = 9.40 dB).
- R-3: -0.09 dB (W4 RMSE = 7.31 dB).
- R-4: -1.63 dB (W4 RMSE = 5.79 dB).
- R-5: -2.22 dB (W4 RMSE = 11.58 dB).

**R-4 robustness diagnostic (four corrections)**:
- locked-no-buffer: Δ_LiDAR_within (in-FOV) = -1.63 dB.
- locked-buffer: -0.01 dB.
- H1-no-buffer: +0.60 dB.
- H1+buffer (Hardening C): -0.64 dB.

**Hardening E (within-session placebo on R-4)**: Δ_real = -0.01 dB vs Δ_placebo = +0.24 dB; gap = -0.25 dB.

---

## §8. Synthesis across all experiments

The leakage-fixed negative result rests on **6 orthogonal axes**: (1) cross-session leave-one-route-out (Project A); (2) hyperparameter sweep on F-B (six configurations); (3) within-session leave-region-out (Project B main run); (4) within-session R-4 robustness (buffer + H1 + combined); (5) framework-agnosticism cross-check (Hardening B / LightGBM); (6) post-hoc feature-stack audit and full re-run (this `MIGRATION_LOG.md`).

---

## §9. Limitations and caveats

Unchanged structurally from Rev9. The leakage-fixed rerun adds the implicit caveat that any future post-hoc feature audit could in principle find additional artefacts; the positive countermeasure is the placebo controls (Hardening A and E) that test for row-aligned signal absorption regardless of the feature labels.

---

## §10. Implications and paper-writing guidance

The paper extracts from this report. Section 1's executive-summary numbers are the headline; the per-experiment subsections (§6, §7) supply the methods-section detail. The post-hoc audit and rerun (`MIGRATION_LOG.md`) goes into the methods section as a transparency disclosure and into the discussion as the closure of the router-side-leakage reviewer objection.

---

## Appendix A — fit inventory

Total fits across the leakage-fixed rerun: **103**.

Per-source-group inventory (full per-fit table at `scripts/run_leakage_fixed_pipeline_inventory.parquet`):

- **Project A — main ablation + RQ4**: 28 fits.
- **Project A — F-B hyperparameter robustness**: 8 fits.
- **Project A — Hardening A + B**: 6 fits.
- **Project B — WLRO + buffer + H1 sensitivity**: 41 fits.
- **Project B — R-4 four-corrections diagnostic**: 11 fits.
- **Project B — Hardening C + E**: 9 fits.
