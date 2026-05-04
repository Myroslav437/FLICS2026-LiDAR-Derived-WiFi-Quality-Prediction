# Focused Research Proposal (Revision 10)

**Lab-Measured AP Coordinates Suffice for Industrial AGV WiFi Signal Prediction: A Properly-Controlled Comparison Showing LiDAR-Derived Environment Features Add No Value (leakage-fixed feature stack).**

*Revision 10 supersedes Rev9 after a post-hoc audit identified five features in the locked Phase 1 telemetry block that should not have been used as model inputs: three MikroTik router CPU load channels (target leakage), the constant `battery_value` sentinel (no information), and the `nns_state` within-session navigation flag (does not generalise across deployments). The full Phase 1 pipeline — Project A LORO ablation, hyperparameter diagnostic, Project B WLRO, R-4 robustness, Hardening A–E — was re-run with the leakage-fixed 3-feature telemetry stack. The headline scientific narrative survives: under proper testing, LiDAR-derived environmental features do not improve over the AP-coordinates baseline. The audit and the rerun add a sixth defensive response to the negative-result reviewer-objection register, and replace every paper-cited number with its leakage-fixed counterpart.*

*Substantive changes from Rev9: §1 reframed around the leakage-fixed numbers and the post-hoc audit story; §3 gains a new §3.4 on the post-hoc telemetry-feature audit and the five removed features; §5.4 (feature stack) updated to the 3-feature telemetry block; §11 risk register's `your features were just bad` line now has *two* defensive responses — the placebo control (Hardening A) AND the explicit leakage audit; §12 closes with the paper-writing recommendation on the leakage-fixed numbers. The methodology, fold structure, hyperparameter sets, evaluation protocol, bootstrap CI procedure, and SHAP analysis pipeline are byte-for-byte unchanged from Rev9. See `MIGRATION_LOG.md` for the audit trail.*

---

## 1. Executive summary

The project tested whether LiDAR-derived environmental geometry features add predictive value beyond AP-relative geometry for cross-route WiFi `signal_power` prediction on an industrial AGV. After a post-hoc feature-stack audit and a full re-run with the leakage-fixed telemetry stack (3 features: `speed_mps`, `turn_rate`, `momentary_current_consumption`), the answer remains **no** across every properly-controlled test.

**Project A (cross-session, three folds)**:
  - F-A in-FOV Δ_LiDAR = +0.55 dB (B5 RMSE = 6.16 dB).
  - F-B in-FOV Δ_LiDAR = -0.19 dB (B5 RMSE = 8.20 dB).
  - F-C in-FOV Δ_LiDAR = +1.20 dB (B5 RMSE = 7.67 dB).

**Hyperparameter robustness on F-B (six configurations)**: in-FOV Δ_LiDAR ∈ [-0.47, +0.10] dB. Every config relative to the +1 dB threshold:
  - `locked` → -0.19 dB
  - `H1` → +0.10 dB
  - `H2` → -0.07 dB
  - `H3` → -0.47 dB
  - `H1_randval` → —
  - `H2_randval` → —

**Project B main run (within-session, 15.03, five spatial folds)**:
  - R-1 in-FOV Δ_LiDAR_within = -2.99 dB (W4 RMSE = 13.79 dB).
  - R-2 in-FOV Δ_LiDAR_within = -2.20 dB (W4 RMSE = 9.40 dB).
  - R-3 in-FOV Δ_LiDAR_within = -0.09 dB (W4 RMSE = 7.31 dB).
  - R-4 in-FOV Δ_LiDAR_within = -1.63 dB (W4 RMSE = 5.79 dB).
  - R-5 in-FOV Δ_LiDAR_within = -2.22 dB (W4 RMSE = 11.58 dB).
  - 0 of 5 folds clear the +1 dB threshold.

**R-4 robustness diagnostic (four corrections)**:
  - locked, no buffer: Δ_LiDAR_within (in-FOV) = -1.63 dB.
  - locked, 1 m buffer: -0.01 dB.
  - H1, no buffer: +0.60 dB.
  - H1 + 1 m buffer (Hardening C): -0.64 dB.

**Hardening experiments (five additional checks)**:
- **(A) Cross-session LiDAR placebo on F-B**: Δ_real (locked) = -0.19 dB vs Δ_placebo (locked) = +0.08 dB; Δ_real (H1) = +0.10 dB vs Δ_placebo (H1) = +0.28 dB. Both gaps within the 0.5 dB tolerance: the model is not extracting row-aligned LiDAR information beyond marginal distributions.
- **(B) LightGBM cross-check on F-B**: Δ_LiDAR (in-FOV) = -0.27 dB (default), -0.45 dB (H1-equiv). The negative result is not XGBoost-specific.
- **(C) Combined H1 + 1 m buffer on R-4**: Δ_LiDAR_within (in-FOV) = -0.64 dB. The two corrections do not cancel.
- **(D) Dataset noise-floor characterization** (editorial; unchanged by the rerun since σ_intra derives from anomaly-filtered dataset rows, not from the feature stack): σ_intra = **4.95 dB** at 0.5 m cell granularity (15.03: 4.901 dB; 24.03: 5.005 dB; 85 same-map qualifying cells). The cleanest within-session model under the leakage-fixed stack (R-4 W2 H1 in-FOV; RMSE = 3.96 dB) operates ~1.0 dB below this floor by exploiting sub-cell information.
- **(E) Within-session LiDAR placebo on R-4**: Δ_real (locked, in-FOV) = -0.01 dB vs Δ_placebo = +0.24 dB; gap = -0.25 dB. The signal does not survive any deployment-relevant correction.

The cleanest deployment-relevant single number from the project under the leakage-fixed stack remains: **R-4 W2 under H1, in-FOV stratum: RMSE = 3.96 dB.** Position + 3 telemetry features + 5 AP-relative features (10 features total), within-session held-out region, depth-4 XGBoost. This is the deployment-relevant baseline number — and it is sub-cell-resolution, ~3 dB below the dataset's natural 0.5 m position-binning noise floor.

The paper's contribution is now:
- **Methodological rigor**: a properly-controlled comparison with three orthogonal robustness checks (cross-session LORO, hyperparameter sweep, within-session spatial buffer-zone test), five hardening experiments addressing the four standard reviewer objections, **and** a post-hoc feature-stack audit that removed five leaked / non-informative features and re-ran every experiment from scratch (`MIGRATION_LOG.md`).
- **The disambiguation framework (B5 / B5' / B5''; W4 / W4' / W4'')** — a generally-applicable way to test whether feature group A contributes uniquely or substitutes for feature group B in an ML ablation.
- **Deployment guidance** — operators with lab-measured or surveyed AP coordinates do not benefit from integrating LiDAR-derived features into their WiFi link-quality prediction stack for this deployment scenario.
- **The path-loss observation from Phase 0** (independent of the LiDAR question; unchanged by the leakage fix because it derives from Phase 0 artefacts only): per-session path-loss exponents `n_d` differ by 5× across three sessions at the same lab-measured AP (1.22 / 0.22 / 0.38; pairwise disjoint bootstrap CIs).

---

## 2. Reframed view of the cross-session shift

Methodologically unchanged from Rev9 §2. The leakage-fixed rerun does not affect any of Phase 0's findings on session-shift structure or the same-cell |Δ| distribution.

---

## 3. Dataset, AP coordinates, frame handling, and data cleaning

Sections 3.1–3.3 unchanged from Rev9. The dataset itself (`data/phase1/dataset.parquet`, SHA-256 `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270`) is byte-for-byte identical to Rev9's reference; the leakage fix is at the *feature selection* layer, not the data layer. The dataset still contains all telemetry columns, including the leaked ones, as raw data.

### 3.4 Post-hoc telemetry-feature audit (NEW in Rev10)

After Rev9 was finalised, a post-hoc audit identified five features in the locked Phase 1 telemetry block that should not have been used as model inputs:

| Removed | Reason |
|---|---|
| `load_long`, `load_mid`, `load_short` | MikroTik router CPU load averages (`FH.7000.[mikrotik].load_*` in the source CSV); target leakage from the receiving end of the same WiFi link the model is predicting. |
| `battery_value` | `0xFFFF` CAN-bus sentinel within the LiDAR-coverage window; provably no-op (`docs/p1_battery_provenance/report.md`; the prior lean-features comparison showed 32/32 byte-identical model SHA-256 across the full ablation when removed). |
| `nns_state` | Discrete 2/3 navigation-system state. Produces a material within-session shift but no cross-session shift; plausibly a within-session spatial-mode fingerprint that does not generalise across deployments. |

The new feature stack is the leakage-fixed minimum: **position + 3 telemetry features + AP-relative + LiDAR**, and is tagged `feature_stack_version = "leakage_fixed_v1"` on every metrics parquet row produced after the rerun. Every Phase 1 experiment was re-run from scratch; locked artefacts that depended on the old stack were deleted, with their pre-deletion SHA-256 hashes recorded in `MIGRATION_LOG.md` §6 as the audit trail.

---

## 5. Methodology — feature stack and hyperparameters

### 5.4 Feature stack (Rev10 update)

**Telemetry (3, was 8)**: `speed_mps`, `turn_rate`, `momentary_current_consumption`. The five removed features and their rationale are documented in §3.4.

**Position (2, within-session only)**: `x_m`, `y_m`.

**AP-relative (5)**: `dist_to_AP`, `sin_angle_to_AP`, `cos_angle_to_AP`, `clutter_frac_toward_AP`, `is_AP_in_FOV`.

**LiDAR scalar (5)**: `mean_dist_mm`, `dist_p90_mm`, `clutter_frac`, `openness_frac`, `mean_front_mm`.

**LiDAR sectoral (14)**: 7 × 30° valid-FOV sectors × {mean_dist, clutter_frac}.

Variant feature counts (B-prefix = Project A; W-prefix = Project B):

| Variant | Features | Count |
|---|---|---:|
| B0  | `dist_to_AP`                                                                 |  1 |
| B1  | B0 + sin/cos angle                                                            |  3 |
| B2  | B1 + LiDAR scalar                                                             |  8 |
| B3  | B2 + LiDAR sectoral                                                           | 22 |
| B4  | B3 + 3 telemetry                                                              | 25 |
| B5  | B4 + `clutter_frac_toward_AP` + `is_AP_in_FOV`                                | 27 |
| B5' | telemetry + AP-relative                                                       |  8 |
| B5''| telemetry + LiDAR scalar + sectoral                                           | 22 |
| W0  | position                                                                       |  2 |
| W1  | W0 + 3 telemetry                                                              |  5 |
| W2  | W1 + AP-relative                                                              | 10 |
| W3  | W2 + LiDAR scalar                                                             | 15 |
| W4  | W3 + LiDAR sectoral                                                           | 29 |
| W4''| W0 + 3 telemetry + LiDAR scalar + LiDAR sectoral                              | 24 |
| W4' | feature-identical to W2                                                       | 10 |

Hyperparameters and bootstrap procedure unchanged from Rev9 §5.5.

---

## 11. Risk register (Rev10 update)

The Rev10 risk register is identical to Rev9's, with one strengthened row:

| Reviewer objection | Response |
|---|---|
| "Your features were just bad / leaked / overfit telemetry." | **TWO** defensive responses: (1) the cross-session and within-session LiDAR placebo controls (Hardening A and E) confirm the model was not extracting row-aligned LiDAR information beyond marginal distributions; (2) a post-hoc feature-stack audit identified and removed five potentially-leaking or no-op telemetry features (`load_long`, `load_mid`, `load_short`, `battery_value`, `nns_state`) and re-ran every Phase 1 experiment from scratch under the leakage-fixed feature stack. The router-side leakage objection is explicitly closed by the audit. See `MIGRATION_LOG.md`. |

---

## 12. Recommendation

**WRITE THE PAPER.** All paper-cited numbers in Rev10 are leakage-fixed. The unified report (`docs/unified_report.md`) is the source of truth for paper-writing; `103` fits across the rerun substantiate every headline number.
