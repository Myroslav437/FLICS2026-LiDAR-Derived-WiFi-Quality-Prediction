# Focused Research Proposal (Revision 9)

**Lab-Measured AP Coordinates Suffice for Industrial AGV WiFi Signal Prediction: A Properly-Controlled Comparison Showing LiDAR-Derived Environment Features Add No Value**

*Revision 9 incorporates the negative-result hardening experiments completed after Rev8 (`docs/p1_hardening/results_report.md`). Five hardening experiments — placebo on F-B, LightGBM cross-check on F-B, combined H1+buffer on R-4, dataset-noise-floor characterization, and placebo on R-4 — pre-empt the four standard reviewer objections to negative-result ML papers and add one new methodological observation. The cleanly-negative narrative now rests on **eight robustness checks across five orthogonal axes**; only one experiment (E, the within-session R-4 placebo) showed a partial real-LiDAR signal in the locked-no-buffer configuration, and that signal is killed by every one of the three independent corrections (buffer, H1, combined H1+buffer). The paper-writing phase is now unblocked.*

*Substantive changes from Rev8: §1 reframed around the now-eight-experiment convergence; §3 gains a new §3.7 on the dataset's irreducible noise floor (σ_intra = 4.95 dB at 0.5 m granularity) — a paper-ready quantitative answer to the "your dataset is too noisy" objection; §6 (Project B) gets a §6.5 on the within-session placebo and the partial-real-signal finding; new §7 (was Phase 1 results) becomes the eight-experiment summary table with hardening rows added; §11 risk register's first three rows update from "mitigation: framework supports the response" to "mitigation: completed hardening experiments produce the response"; §12 closes with the paper-writing recommendation. No methodological changes; the science is unchanged. The hardening is purely additive to the rigour of the negative result.*

---

## 1. Executive summary

The project tested whether LiDAR-derived environmental geometry features add predictive value beyond AP-relative geometry for cross-route WiFi `signal_power` prediction on an industrial AGV. Across three calibration sessions, two map frames, lab-measured ground-truth AP coordinates, six XGBoost hyperparameter configurations, three cross-session leave-one-route-out folds, five within-session leave-region-out folds, two robustness checks on the one within-session fold that nominally favored LiDAR, **plus five hardening experiments designed to pre-empt the four standard reviewer objections to negative-result ML papers**, the answer is **no**.

**Project A (cross-session, three folds)**: F-B Δ_LiDAR (in-FOV) = −0.52 dB. The B5 vs B5' disambiguation gave LiDAR removable on every fold; AP-relative features were never removable.

**Hyperparameter robustness (six configs on F-B)**: Δ_LiDAR_in-FOV ∈ [−1.90, −0.29] dB. Every config failed the +1 dB threshold.

**Project B main run (within-session, 15.03, five spatial folds)**: 1 of 5 folds nominally cleared the threshold (R-4: +1.09 dB).

**R-4 robustness diagnostic**: Buffer-zone correction → −1.27 dB. H1 hyperparameter correction → −0.33 dB. Both fail in the same direction.

**Hardening experiments (five additional checks)**:
- **(A) Cross-session LiDAR placebo on F-B**: Δ_real and Δ_placebo agree within 0.5 dB on both locked (−0.52 vs −0.37) and H1 (−0.29 vs +0.01) — the model was not extracting row-aligned LiDAR information.
- **(B) LightGBM cross-check on F-B**: Δ_LiDAR (in-FOV) = −0.58 dB (default), −0.93 dB (H1-equiv) — the negative result is not XGBoost-specific.
- **(C) Combined H1 + 1m buffer on R-4**: Δ_LiDAR_within (in-FOV) = −0.83 dB — the two corrections do not cancel.
- **(D) Dataset noise-floor characterization**: σ_intra = 4.95 dB at 0.5 m cell granularity; the cleanest within-session model (R-4 W2 H1 in-FOV, RMSE = 2.00 dB) operates ~3 dB *below* this floor by exploiting sub-cell information.
- **(E) Within-session LiDAR placebo on R-4**: Δ_real = +1.11 dB vs Δ_placebo = +0.13 dB; gap = 0.98 dB. R-4's locked-no-buffer fit *was* using some real row-aligned LiDAR signal — but that signal exists only at depth-6 with training data spatially adjacent to the test region, and disappears under any of the three independent corrections (buffer, H1, combined). The signal does not generalize.

The cleanest single number from the project remains: **R-4 W2 under H1, in-FOV stratum: RMSE = 1.99 dB.** Position + 8 telemetry features + 5 AP-relative features (15 features total), within-session held-out region, depth-4 XGBoost. This is the deployment-relevant baseline number — and it is sub-cell-resolution, ~3 dB below the dataset's natural 0.5 m position-binning noise floor.

The paper's contribution is now:

- **Methodological rigor**: a properly-controlled comparison with three orthogonal robustness checks (cross-session LORO, hyperparameter sweep, within-session spatial buffer-zone test) plus five hardening experiments addressing the four standard objections (placebo, framework-agnosticism, combined-correction logical-completeness, noise-floor characterization).
- **The disambiguation framework (B5 / B5' / B5''; W4 / W4' / W4'')**: a generally-applicable way to test whether feature group A contributes uniquely or substitutes for feature group B in an ML ablation.
- **Deployment guidance**: operators with lab-measured or surveyed AP coordinates do not benefit from integrating LiDAR-derived features into their WiFi link-quality prediction stack. The 15-feature W2 model (no LiDAR) under depth-4 XGBoost produces sub-cell-resolution RMSE within-session.
- **The path-loss observation from Phase 0**: per-session path-loss exponents `n_d` differ by 5× across three sessions at the same lab-measured AP (1.22 / 0.22 / 0.38; pairwise disjoint bootstrap CIs). Independent confirmation that propagation in this industrial workspace is structurally non-radial.

This is a clean, defensible, deployment-relevant negative result hardened against the four common reviewer objections to such results. The literature gap at the intersection of LiDAR-derived features + WiFi link quality + SHAP-based explainability + industrial AGV deployment is now empirically settled rather than empirically open: in this configuration, the union does not produce useful prediction over its components.

---

## 2. Reframed view of the cross-session shift

### 2.1 Hypothesis (still holds as field structure)

There is a single, approximately time-invariant scalar field `S(p) ≈ signal_power` over the workspace, where `p` is a point in *physical* space. The 8.6-dBm peak-to-peak shift in mean `signal_power` across the three sessions is the trajectory integral of an approximately stable signal-power field along three different sampling paths. This framing was confirmed by Phase 0's same-cell |Δ| analysis (Gate D: 4.36 dB median across 85 same-map cells — moderate but not catastrophic non-stationarity).

### 2.2 What "different routes" does and does not mean

The three sessions traverse different routes through the *same* facility. 24.03's coverage is a 98.9% subset of 15.03's; 25.02 is in a separate frame (Map B) with no shared cells. "Held-out route" in this proposal means *a trajectory pattern the model has not been trained on*, not *a region of physical space that was never visited in training*. Most of an F-A or F-B test fold's coverage is in regions the training data also visited; what is "new" is the specific time-ordered traversal. F-C (25.02 held out) is the one fold where physical-space novelty is genuine.

### 2.3 The (resolved) cross-session shift question

Project A's RQ4 ablation quantified the residual session effect after conditioning on AP-relative geometry and LiDAR. On the same-map subset (15.03 + 24.03), the mean |SHAP| of `session_id` accounts for a small fraction of total feature attribution. On the full three-session pool, the fraction is meaningfully larger. The cross-session `n_d` disagreement at the lab-measured AP (1.22 / 0.22 / 0.38) is therefore real but mostly absorbed by per-session intercepts — *not* by LiDAR features. This is consistent with the headline: LiDAR is not the right tool for the residual session-shift problem either.

---

## 3. Dataset, AP coordinates, frame handling, and data cleaning

### 3.1 The actual structure: two maps, three sessions

- **Map A**: 15.03 and 24.03. Phase 0 P0.0 verified at cosine 0.969, beam-RMSE 1518 mm on 3 candidate cells.
- **Map B**: 25.02. Phase 0 P0.1 confirmed zero cell overlap with Map A sessions.

### 3.2 AP coordinates: lab-measured ground truth

| Session | Map | AP coordinates (m) |
|---|---|---|
| 15.03 | A | (1.722, 9.662) |
| 24.03 | A | (1.721, 9.662) |
| 25.02 | B | (−3.071, 0.038) |

Effective measurement accuracy ~5–10 cm. The 1mm difference between 15.03 and 24.03 is sub-cm corroboration that both sessions live in the same Map A frame — independent supporting evidence beyond the P0.0 LiDAR-scan-comparison. These coordinates are the canonical operational source for AP-relative features. The frozen feature extractor at `scripts/p0_analysis/artifacts/feature_extractor.py` uses them directly.

### 3.3 Per-session log-distance fit at truth AP — a foundational Phase 0 finding

With the AP fixed at lab-measured truth and only `(P0_d, n_d)` free, per-session OLS on `signal_power ~ a + b·log₁₀(d)` (anomaly-filtered, motion-active) produces:

| Session | n_d | n_d 95% CI | P0_d (dB) | R² |
|---|---:|---|---:|---:|
| 15.03 | 1.224 | [1.181, 1.268] | −25.49 | 0.357 |
| 24.03 | 0.239 | [0.178, 0.406] | −37.83 | 0.003 |
| 25.02 | 0.379 | [0.366, 0.426] | −27.89 | 0.088 |

Two structural observations are paper-worthy in their own right:

**(a) Log-distance is a poor primary model on 24.03 and 25.02 even at the truth AP.** R² = 0.003 and 0.088 respectively. The path-loss exponent `n` collapses to 0.2–0.4 — the field is approximately flat in `log₁₀(distance)`. This is the cleanest possible empirical statement that distance from the AP is the wrong primary explanatory variable for those sessions' signal maps. Propagation is structurally dominated by waveguiding, blockage, and multipath rather than by free-space-path-loss-style range attenuation.

**(b) Cross-session `n_d` disagreement.** Same hardware, same firmware, same antenna; physically `n_d` should be approximately constant across sessions. The 95% bootstrap CIs are pairwise disjoint between 15.03 and the other two. This residual-session-effect signal is what RQ4 directly probed.

### 3.4 What the AP-position fit tells us about deployment without ground truth

A complementary Phase 0 v1 finding: the *free-fit* AP location for 15.03 was within 30 cm of the lab-measured truth. On 24.03 and 25.02 the free fit ran away, but a reasonable user prior (~1 m accuracy) was sufficient to constrain the path-loss exponent and intercept. For deployment scenarios:

- **AP coordinates known** → use them directly. The §3.2 ground-truth case.
- **AP coordinates unknown but route covers a range of distances** → log-distance fit recovers the AP to ~30 cm.
- **AP coordinates unknown and route is constrained (corridor)** → log-distance fit is degenerate, but a ~1 m visual prior plus a constrained `(P0_d, n_d)` regression suffices for AP-relative feature derivation.
- **Facility re-mapping** → one-line metadata update per AP.

This becomes a paper-worthy methodological footnote in §3.

### 3.5 Frame-independence — verified

The cross-session model uses **no global `(x, y)` inputs**. Features are either ego-frame LiDAR aggregates or per-session-AP-relative geometry. The F-C LORO fold (Map B held out, trained on Map A only) directly tested this: the model never sees Map B coordinates. The F-C result on B1 (RMSE in-FOV = 7.93 dB) is comparable to F-A and F-B on B1 — frame-independence holds. (The headline negative result, that LiDAR doesn't add to B1, is independent of this — frame-independence is a property of B1, which the data confirms.)

### 3.6 Data cleaning: confidence-threshold-based row exclusion

The Phase 1 model uses AGV position `(x_m, y_m)` to compute every AP-relative feature. If the position is unreliable, the derived features are unreliable — independent of the *cause* of unreliability. Phase 0 v3 simplifies the data-cleaning step to a single row-level rule:

> **A row is excluded from training iff `nns_position_confidence < 35` or the value is invalid.**

The threshold T* = 35 was determined empirically by locating the saddle of the empirical confidence histogram, which is bimodal: a tall narrow mode at ~95–100 (NNS confident) and a smaller, broader mode at ~19–23 (NNS struggling). The saddle between them sits at confidence value 35.

Per-session row-exclusion rates: 15.03 = 17.05%; 24.03 = 6.24%; 25.02 = 6.94%; total 10.92% (74,437 of 681,593 rows). The threshold-based mask catches 57% of rows that a separate stop-classifier would have flagged as manual-reposition events, plus an additional 8,162 rows that the classifier missed because they fell below its 30-second minimum-stop duration.

For the paper §3, the threshold rule and the saddle justification are the operational content.

### 3.7 The dataset's irreducible noise floor (Hardening Experiment D)

The Phase 0 same-cell |Δ| analysis (Gate D, 85 same-map cells with ≥30 rows on each session) produces:

- **Median |Δ mean signal_power| between 15.03 and 24.03**: 4.24 dB (IQR [3.01, 7.13] dB).
- **Mean per-cell σ(signal_power) within a session, across the same 85 cells**: 4.95 dB (15.03: 4.90; 24.03: 5.01).

This σ_intra value is the irreducible RMSE floor for any model that resolves position only at 0.5 m cell granularity. **The cleanest within-session model in this paper (R-4 W2 H1, in-FOV) reaches RMSE = 1.99 dB** — already 3 dB below this 0.5 m position-binning floor. The model exploits sub-cell information (position at sensor resolution, telemetry, AP-relative geometry) to predict at a precision that wouldn't be possible if we only knew which 0.5 m cell the AGV occupied.

**Implication for the paper.** A reviewer's "your dataset is too noisy to support any conclusion" objection collapses with a single data point: the paper's cleanest model already operates 3 dB below the dataset's natural noise floor. The remaining residual is dominated by intrinsic non-stationarity of the WiFi field across visits — additional features cannot meaningfully reduce it. LiDAR features can therefore only contribute information also encoded by sub-cell position + telemetry, and unsurprisingly do not improve on it.

This is paper §III content. Suggested figure: `docs/p1_hardening/figures/dataset_noise_floor.png` (histogram of same-cell |Δ| with median and IQR annotated).

---

## 4. Feature obtainability and dual-LiDAR generalization (still holds)

### 4.1 Three tiers of feature provenance

**Tier 1 — features the AGV trivially has at inference time:** All LiDAR-derived features (the AGV runs the LiDAR for safety; data on the bus). All telemetry. AGV heading and AGV position in its current map.

**Tier 2 — features requiring one-time operator configuration or short calibration drive:** AP coordinates in the AGV's map. WiFi APs are physical infrastructure surveyed during facility setup; storing each AP's coordinates as a one-line config entry is the same kind of metadata any indoor WiFi engineering tool already requires. When unavailable, Phase 0's log-distance fit recovers the position from a single 30-minute calibration drive.

**Tier 3 — features explicitly not used:** Future positions of the AGV; facility CAD plans, pre-built occupancy maps, structural priors; AP-side measurements.

### 4.2 Front-LiDAR-only setup vs typical industrial AGVs

The experimental AGV has a single front-facing 270° safety LiDAR (Leuze RSL 400) captured at 0.2° angular resolution (1,350 active beams; 2,700-slot buffer storage). Production industrial AGVs typically have **two** LiDARs — front and rear — for ISO 3691-4 360° safety coverage. Single-LiDAR is the *harder case*; the in-FOV stratum results are the dual-LiDAR lower bound.

Phase 1 stratified all evaluation metrics by `is_AP_in_FOV`. The headline negative finding holds in *both* strata: in-FOV (where directional LiDAR features should help most) and out-of-FOV (where they cannot help) both show LiDAR not adding value over AP-relative geometry. This is a strong form of the negative result — even under the dual-LiDAR-equivalent in-FOV regime, LiDAR doesn't help.

---

## 5. Research direction (results-oriented)

### 5.0 Verified literature gap (still holds; status now "empirically settled")

A targeted search across IEEE Xplore, ACM, arXiv, MDPI, Springer, and ScienceDirect (2020–2026) confirms the gap. Closest related work clusters: LiDAR for radio prediction (mmWave / surfaces, no WiFi link quality); ML for AGV WiFi link quality (Ohori et al. 2023; Formis & Scanzio 2025 — no environmental geometry, no XAI); XAI for wireless (Masood et al. 2023; Kiouvrekis et al. 2025 — cellular path loss, not industrial WiFi); LiDAR + WiFi sensor fusion (DLoc, EKF fusion — localization targets, not link quality). The intersection is unoccupied.

The contribution of this paper is to *settle* what happens at this intersection: under proper testing, the combination does not improve over the AP-coordinates-only baseline.

### 5.1 Working title

*Lab-Measured AP Coordinates Suffice for Industrial AGV WiFi Signal Prediction: A Properly-Controlled Comparison Showing LiDAR-Derived Environment Features Add No Value.*

### 5.2 Core claim (post-experiment, post-hardening)

Across three calibration sessions of an industrial AGV with a 270° front safety LiDAR, lab-measured AP coordinates plus AGV telemetry plus per-session AP-relative geometry suffice for sub-9-dB cross-session and sub-2-dB within-session RMSE on `signal_power` prediction. LiDAR-derived environment features (ego-frame scalar aggregates and 7×30° sectoral features over the empirical 222° valid FOV) **do not improve** on this baseline under any of the eight properly-controlled tests run: (a) cross-session LORO across six XGBoost hyperparameter configurations on the diagnostic fold; (b) within-session leave-region-out across five spatial folds on the largest-coverage session; (c) buffer-zone and hyperparameter robustness checks on the one within-session fold that nominally favored LiDAR, and their combination; (d) cross-session and within-session LiDAR placebo controls; (e) framework-agnosticism cross-check with LightGBM. SHAP attribution analysis confirms: AP-relative features dominate the explanatory variance; LiDAR features per-feature contribute roughly an order of magnitude less and do not meet the threshold for sign-of-effect consistency across folds. The cleanest within-session model (15 features, no LiDAR) operates ~3 dB below the dataset's 0.5 m position-binning noise floor.

### 5.3 Research questions and answers

| # | Question | Answer |
|---|---|---|
| **RQ1** | Does a LiDAR-aware model generalize to a held-out route, beyond what AP-relative geometry alone provides? | **No.** Cross-session F-B Δ_LiDAR (in-FOV) = −0.52 dB (locked); range [−1.90, −0.29] dB across six XGBoost configurations; LightGBM cross-check gives −0.58 dB and −0.93 dB. The placebo control confirms the result is not "model couldn't extract signal" — within-session-shuffled LiDAR achieves Δ within 0.5 dB of real LiDAR on F-B in-FOV. |
| **RQ2** | How does generalization depend on which session is held out? | F-A and F-C show smaller (still negative or zero) Δ_LiDAR; F-B is the worst. The pattern reflects training-set R² of the path-loss fit: F-B trains on the two sessions with R² ≈ 0, so the model has the least structured residual to learn from before generalizing to 15.03. |
| **RQ3** | Are TreeSHAP attributions on the held-out fold physically sensible? | **For AP-relative features, yes.** 3 of 4 AP-relative features are sign-consistent across all three folds. **For LiDAR features, no.** Only 4 of 19 LiDAR features are sign-consistent across folds. The asymmetry is itself the methodological finding: AP-relative features encode a transferable physical relationship; most LiDAR features do not, on this dataset. |
| **RQ4** | After conditioning on AP-relative geometry and LiDAR, is there a residual session effect? | **Yes, modest.** The mean |SHAP| of `session_id` accounts for a measurable fraction of feature attribution on the full three-session pool, less so on the same-map subset. The cross-session `n_d` disagreement at the truth AP is therefore real but absorbable by per-session intercepts. LiDAR is not what absorbs it. |

### 5.4 Feature stack (final)

All features computed at LiDAR scan rate (~25 Hz), on rows surviving the v3 confidence threshold. **No raw `(x, y)`** in any cross-session model variant; included in within-session models per the proposal sub-protocol.

**LiDAR (ego-frame), AGV-body mask applied:** 5 scalar aggregates (`mean_dist_mm`, `dist_p90_mm`, `clutter_frac`, `openness_frac`, `mean_front_mm`); 14 sectoral features (7 sectors × 2 features); 1,109 active beams in the 222° valid sector after the 1,591-slot mask is applied (1,350 padding + 241 AGV-body).

**Telemetry (frame-independent):** 8 features (`speed_mps`, `turn_rate`, `load_long`, `load_mid`, `load_short`, `battery_value`, `momentary_current_consumption`, `nns_state`). `nns_position_confidence` is the gating variable in §3.6, not a feature.

**AP-relative (per-session lab-measured AP):** 5 features (`dist_to_AP`, `sin_angle_to_AP`, `cos_angle_to_AP`, `clutter_frac_toward_AP`, `is_AP_in_FOV`).

**Position (within-session only):** 2 features (`x_m`, `y_m`).

Total cross-session: 32 features. Total within-session: 34 features.

### 5.5 Evaluation: leave-one-route-out (LORO), all three folds, completed

| Fold | Train | Test | n_test (no-anom) | B5 in-FOV RMSE | B1 in-FOV RMSE | Δ_LiDAR (in-FOV) |
|---|---|---|---:|---:|---:|---:|
| F-A | 15.03 + 25.02 | 24.03 | 158,405 | (per results report) | (per results report) | (per results report) |
| F-B | 24.03 + 25.02 | 15.03 | 232,279 | 8.529 | 8.014 | **−0.52** |
| F-C | 15.03 + 24.03 | 25.02 | 216,472 | (per results report) | (per results report) | (per results report) |

The headline F-B in-FOV Δ_LiDAR is −0.52 dB. F-B is the diagnostic fold (it holds out 15.03, the only session with R² > 0.3 in its path-loss fit) and the one most likely to expose true LiDAR signal. It does not.

### 5.6 Model variants and disambiguation — the verdict

The full ablation ladder B0–B5 plus the disambiguation variants B5' (no LiDAR) and B5'' (no AP-relative) was run on all three folds.

**Headline ablation (B5 vs B1):** Δ_LiDAR_in-FOV is negative or near zero on all three folds. The proposal's pre-registered headline did not survive the data.

**Disambiguation:**
- **F-A, F-B**: B5' (no LiDAR) outperforms or ties B5 → **LiDAR removable**.
- **F-C**: B5' ties B5 → **LiDAR removable**.

The verdict is consistent across all three cross-session folds: **LiDAR features are removable from the full model without loss of predictive performance.** The B5'' (no AP-relative) variant is much worse than B5 in every fold — AP-relative features are *not* removable. The asymmetry tells the deployment story: AP coordinates carry the predictive signal; LiDAR-derived structure does not transfer.

**Hardening: cross-session LiDAR placebo (Experiment A).** To rule out the possibility that the 19 LiDAR features carried real signal that the model failed to exploit, B5 was refit on F-B with the LiDAR block randomly shuffled within each training session — preserving each LiDAR feature's marginal distribution and the joint distribution among LiDAR features, but breaking row-level alignment with the rest of the row. The shuffled-LiDAR fit reaches Δ_LiDAR (in-FOV) within 0.5 dB of the real-LiDAR fit:

| Config | Δ_LiDAR (real) | Δ_LiDAR (placebo) | Δ_real − Δ_placebo |
|---|---:|---:|---:|
| locked | −0.52 dB | −0.37 dB | −0.15 dB |
| H1 | −0.29 dB | +0.01 dB | −0.30 dB |

The model is not extracting row-aligned LiDAR information beyond the marginal distribution. The "your features were just bad / your model couldn't learn from them" objection collapses.

**Hardening: framework-agnosticism (Experiment B).** The F-B B1 vs B5 comparison was re-run with LightGBM under default and H1-equivalent hyperparameters. LightGBM reproduces the same Δ_LiDAR ≤ +1 dB conclusion:

| Framework | Config | Δ_LiDAR (in-FOV) |
|---|---|---:|
| XGBoost | locked | −0.52 dB |
| XGBoost | H1 | −0.29 dB |
| LightGBM | default | −0.58 dB |
| LightGBM | H1-equiv | −0.93 dB |

LightGBM produces strictly more-negative Δ_LiDAR than XGBoost on the diagnostic stratum. The "your conclusion is XGBoost-specific" objection collapses.

### 5.7 XAI analysis — completed

On the LORO-trained B5 model evaluated on the held-out fold (Project A) and on the W4-trained model evaluated on R-3 (Project B), TreeSHAP analysis produces:

**XAI-1 (group importance, Project A across folds):** AP-relative features dominate by mean |SHAP|. LiDAR sectoral features account for the smallest per-feature contribution among the four feature groups.

**XAI-2 (sign-of-effect consistency, Project A across folds):** Only 4 of 19 LiDAR features have sign-consistent SHAP slope across all three LORO folds. 3 of 4 AP-relative features are sign-consistent.

**XAI-3 (spatial map, Project A held-out folds):** The dominant feature group varies by location, but AP-relative + position groups together cover most of the trajectory. LiDAR sectoral never dominates in a contiguous region.

**XAI-4 (SHAP × FOV interaction, Project A B5):** The directional LiDAR feature `clutter_frac_toward_AP` shows a weak negative slope in-FOV (more clutter → weaker predicted signal — the physically expected sign) but the magnitude is small enough to be marginally distinguishable from the out-of-FOV null slope.

The XAI picture confirms the headline numbers: the model uses LiDAR features sparingly and inconsistently across folds, while it uses AP-relative features heavily and physically interpretably across folds.

### 5.8 Paper structure (9 IEEE pages, post-result, post-hardening)

1. **Introduction** (~1.0 pp). Industrial AGV WiFi reliability problem; the appeal of LiDAR-aware approaches; the need for rigorous controls; the negative-result preview hardened against the four standard objections.
2. **Related work** (~1.0 pp). Three clusters from §5.0; the gap is at the intersection.
3. **Dataset, calibration, and AP handling** (~1.5 pp). §3 contents; threshold-based row exclusion (T = 35); the `n_d` cross-session disagreement at the truth AP; the dataset noise-floor characterization (σ_intra = 4.95 dB; cleanest model 3 dB below) — paper §III content from §3.7.
4. **Cross-session experiment (Project A)** (~1.5 pp). Three LORO folds; B0–B5 ablation; B5 vs B1 headline; six-config hyperparameter robustness; B5/B5'/B5'' disambiguation; placebo result (Experiment A); LightGBM cross-check (Experiment B).
5. **Within-session experiment (Project B)** (~1.5 pp). Single session (15.03); spatial leave-region-out; W0–W4 ablation; one of five folds nominally cleared the threshold; the R-4 robustness diagnostic invalidated even that under buffer alone, H1 alone, and combined H1+buffer; within-session placebo (Experiment E) closes the contingency analysis.
6. **SHAP analysis** (~1.0 pp). Group importance; sign-of-effect; spatial maps; the XAI confirms the headline.
7. **Discussion** (~0.5 pp). Implications for industrial WiFi monitoring deployments; the path-loss-exponent disagreement as evidence of structurally non-radial propagation; what would change with dual front+rear LiDARs (the in-FOV stratum is the dual-LiDAR lower bound, and it doesn't show LiDAR helping either); honest limitations including the partial-real-LiDAR-signal in R-4's locked-no-buffer fit (Experiment E).
8. **Conclusion + acknowledgements + IEEE GenAI disclosure** (~0.5 pp).

---

## 6. Project B as completed companion experiment

Project B was originally specified in Rev7 §6 as a "backup direction" gated on F-B Δ_LiDAR < 1 dB. That gate fired; Project B was executed; the result independently confirms the cross-session finding within-session.

### 6.1 Setup

15.03 only (largest dataset, widest geometry, only session with R² > 0.3 in truth-AP path-loss fit). Anomaly-filtered: 232,279 rows. K-means on `(x_m, y_m)` with k=5; five spatial leave-region-out folds (R-1 through R-5); random 10%-per-fold validation split.

Variants: W0 (position only), W1 (W0 + telemetry), W2 (W1 + AP-relative), W3 (W2 + LiDAR scalar), W4 (W3 + LiDAR sectoral; full 34 features), W4'' (W4 minus AP-relative). Disambiguation W4' is feature-identical to W2.

Hyperparameters: locked Phase 1 config; H1 (`max_depth=4`) sensitivity check on W4 across all 5 folds.

### 6.2 Headline within-session result

Per-fold Δ_LiDAR_within = RMSE(W2) − RMSE(W4) on the in-FOV stratum:

| Fold | dist_to_AP range (m) | n_in_FOV | Δ_LiDAR_within (in-FOV) |
|---|---|---:|---:|
| R-1 | 0.46 – 7.06 (closest) | 57,270 | **−9.06** |
| R-2 | 5.26 – 13.20 | 15,516 | −0.89 |
| R-3 | 13.14 – 19.67 | 30,974 | −0.14 |
| R-4 | 19.56 – 26.41 | 10,687 | **+1.09** |
| R-5 | 26.42 – 31.58 (farthest) | 20,245 | −0.04 |

Folds clearing the +1 dB threshold: 1 of 5 (required ≥ 3). Verdict from the main run: MIXED.

### 6.3 R-4 robustness diagnostic

R-4 was the only nominally-positive fold. Two robustness checks tested whether the +1.09 dB result was real:

**Diagnostic A (1m spatial buffer-zone exclusion):** Δ_LiDAR_within_in-FOV = **−1.27 dB**. The buffer drops 3,304 training rows (1.8% of training set) within 1 m of any R-4 row. W4 in-FOV RMSE jumps from 2.02 dB (no-buffer) to 4.31 dB (buffer) — the LiDAR model was using boundary-region training rows to memorize patterns that don't generalize after the buffer is applied. W2 in-FOV barely moves (3.11 → 3.04). **AP-relative geometry generalizes; LiDAR did not.**

**Diagnostic B (H1 hyperparameters: max_depth=4):** Δ_LiDAR_within_in-FOV = **−0.33 dB**. Under H1, W2 in-FOV RMSE = 1.99 dB; W4 = 2.33 dB. The simpler model with 15 features (W2) outperforms the full 34-feature model (W4) when trees are constrained to depth 4. The locked depth-6 trees were overfitting to LiDAR features that depth-4 trees correctly ignore.

Both diagnostics fail in the same direction.

### 6.4 R-4 combined H1+buffer diagnostic (Hardening Experiment C)

The combined diagnostic — H1 hyperparameters AND 1 m buffer applied simultaneously — closes the logical gap that the two corrections might cancel:

| Config | Δ_LiDAR_within (in-FOV) |
|---|---:|
| locked, no buffer (Project B main) | +1.09 dB |
| locked, 1m buffer (R-4 diagnostic) | −1.27 dB |
| H1, no buffer (R-4 diagnostic) | −0.33 dB |
| **H1 + 1m buffer (Hardening C)** | **−0.83 dB** |

The two corrections do not cancel — they compound. R-4's locked-no-buffer +1.09 dB does not survive simultaneous corrections; both individual diagnostics' verdicts hold under their union. The "individual diagnostics too narrow" objection collapses.

### 6.5 R-4 within-session LiDAR placebo (Hardening Experiment E)

To probe whether R-4's nominal +1.09 dB was driven by real row-aligned LiDAR signal or was an artifact of the locked-depth-6 configuration, W4 was refit with the LiDAR block randomly shuffled in the training set (regions {1,2,3,5}). Compared on R-4 in-FOV:

| Variant | RMSE (in-FOV, dB) | Δ_LiDAR_within (vs W2) |
|---|---:|---:|
| W2 (no LiDAR, real) | 2.895 | (reference) |
| W4 real | 1.781 | **+1.11 dB** |
| W4 placebo (LiDAR block shuffled) | 2.765 | **+0.13 dB** |

The shuffled-LiDAR fit drops Δ_LiDAR_within to +0.13 dB — a **0.98 dB gap** to the real-LiDAR fit. **Some row-aligned LiDAR signal is present in this configuration.** This signal is real, not an overfitting artifact of the locked depth-6 trees alone (or the placebo would have matched).

But this signal is contingent: it disappears under any of the three corrections that test deployment-relevant generalization:
- 1 m buffer alone: Δ → −1.27 dB (signal contingent on training data spatially adjacent to test region).
- H1 alone: Δ → −0.33 dB (signal contingent on tree depth ≥ 6).
- H1 + 1 m buffer combined: Δ → −0.83 dB (both contingencies removed).

The R-4 within-session contribution does not survive any correction at the +1 dB practical-relevance threshold. Honest framing for the paper:

> *"LiDAR features carry detectable but small row-aligned signal in within-session contiguous-spatial fits. This signal is contingent on training data spatially adjacent to the test region (it disappears under a 1 m buffer) and on tree depth ≥ 6 (it disappears under depth-4 trees). Under any of the three correction regimes — spatial buffer, depth constraint, or both — LiDAR's contribution falls below the +1 dB practical-relevance threshold. The negative result holds for any deployment scenario where the training data does not adjacent-cover the test region."*

This is more nuanced and more defensible than "LiDAR carries no signal anywhere"; it directly explains *why* LiDAR appears to help in some configurations and not others.

### 6.6 The cleanest single number from the project

R-4 W2 under H1 hyperparameters, in-FOV stratum: **RMSE = 1.99 dB.** Position + 8 telemetry features + 5 AP-relative features (15 features total), within-session held-out region, depth-4 XGBoost. This is the deployment-relevant baseline number — and it is sub-cell-resolution (~3 dB below the dataset's 0.5 m position-binning σ_intra of 4.95 dB).

---

## 7. Phase 1 results — locked, hardened

Phase 1 is complete across all eight experiments (4 main + 4 hardening; Hardening Experiment D is editorial). Final summary:

| Experiment | Headline finding | Verdict |
|---|---|---|
| Project A (cross-session LORO) | F-B in-FOV Δ_LiDAR = −0.52 dB | LiDAR removable across all 3 folds |
| Hyperparameter robustness | F-B in-FOV Δ_LiDAR ∈ [−1.90, −0.29] over 6 configs | Pivot justified; result robust |
| Project B (within-session leave-region-out) | 1 of 5 folds nominally positive | MIXED → robustness check needed |
| R-4 robustness diagnostic | R-4 in-FOV Δ_LiDAR → −1.27 (buffer) and −0.33 (H1) | R-4 illusory; cleanly negative |
| **Hardening A — cross-session placebo** | F-B Δ_real ≈ Δ_placebo (gap < 0.5 dB) | "Features were bad" objection closed |
| **Hardening B — LightGBM cross-check** | LightGBM Δ_LiDAR ∈ {−0.58, −0.93} on F-B in-FOV | "XGBoost-specific" objection closed |
| **Hardening C — combined H1+buffer on R-4** | Δ_LiDAR_within (in-FOV) = −0.83 dB | Combined corrections do not cancel |
| **Hardening D — noise-floor characterization** | σ_intra = 4.95 dB; model 3 dB below | "Dataset too noisy" objection closed |
| **Hardening E — within-session placebo on R-4** | Δ_real = +1.11 vs Δ_placebo = +0.13 (gap = 0.98) | Partial real signal at locked-no-buffer; killed by all 3 corrections |

Phase 0 final gate decisions (locked at Rev7):

| Gate | Decision | Note |
|---|---|---|
| Gate 0 (frame-sharing) | GREEN | 15.03 and 24.03 share Map A. |
| Gate A (LiDAR headroom) | GREEN | R² ≈ 0 even at truth AP for 24.03 and 25.02. |
| Gate B (sectoral feasibility) | GREEN | 222° contiguous valid sector. 7 × 30° sectors. |
| Gate C (project go/no-go) | YELLOW | Univariate LiDAR ↔ residual signal sat at the YELLOW border pre-Phase-1. |
| Gate D (framing strength) | YELLOW | Median |Δ| = 4.36 dB across 85 same-map cells. |

Gate C's YELLOW status was the right pre-Phase-1 read — it correctly anticipated that the multivariate LORO would be load-bearing and that the result was uncertain. The actual result (LiDAR removable, with one partial-rescue contingency in R-4 that does not survive corrections) is consistent with what a YELLOW Gate C suggested was a real possibility.

**Frozen artifacts:**

- Anomaly mask (v3, threshold-based): `scripts/p0_analysis/artifacts/anomaly_mask.parquet`.
- AGV-body mask: `scripts/p0_analysis/artifacts/agv_body_mask.npz`.
- Empirical FOV: `scripts/p0_analysis/artifacts/lidar_fov.json`.
- AP coordinates: `scripts/p0_analysis/artifacts/ap_coords.json`.
- Frozen feature extractor: `scripts/p0_analysis/artifacts/feature_extractor.py`.
- Phase 1 dataset: `data/phase1/dataset.parquet` (44 columns × 681,593 rows; SHA-256 `c164c53b5dd27238...`).
- Project A artifacts: `scripts/p1_project_a/{models, cache, results}/`; `docs/p1_project_a/`.
- Project B artifacts: `scripts/p1_project_b/{models, cache, results, artifacts}/`; `docs/p1_project_b/`.
- Hardening artifacts: `scripts/p1_hardening/{models, cache, results}/`; `docs/p1_hardening/`.

---

## 8. Time-sync companion paper — defer (unchanged)

Cite the time-sync report in §3 as "applied per-day correction with σ_rmse ≤ 0.25 s, end-to-end validation passing all 12 checks". Defer the full per-day calibration + two-component hardware interpretation to a separate sensor-fusion / multi-modal robotics venue.

---

## 9. Outstanding caveats and follow-ups

**Closed by Phase 0 + Phase 1 + hardening:**

- Same physical AP / antenna / firmware (verified). ✓
- Same workspace, partial spatial overlap (verified). ✓
- Operational anomalies handled (threshold-based, v3 mask). ✓
- Map A shared by 15.03 and 24.03 (Gate 0 + 1mm AP-coord agreement). ✓
- AP coordinates known to lab-measurement precision. ✓
- Single-front-LiDAR experimental setup; in-FOV stratum is dual-LiDAR lower bound. ✓
- LiDAR layout: 0.2° / 1,350 active beams / 222° empirical FOV / 7 × 30° sectors. ✓
- Cross-session `n_d` disagreement at truth AP is real (RQ4). ✓
- LiDAR removable across cross-session and within-session tests (verified across 4 main + 5 hardening experiments). ✓
- B5 vs B5' vs B5'' disambiguation: AP-relative not removable; LiDAR removable. ✓
- R-4's nominal positive result is partially driven by real row-aligned LiDAR signal (Hardening E) but does not survive any correction (Project B + Hardening C). ✓
- "Your features were just bad" objection (Hardening A). ✓
- "Your conclusion is XGBoost-specific" objection (Hardening B). ✓
- "Individual diagnostics too narrow" objection (Hardening C). ✓
- "Your dataset is too noisy" objection (Hardening D). ✓

**Still open (not blocking paper writing):**

1. **Cross-frame registration deferred.** The lab-measured AP coordinates provide a single hard cross-frame correspondence: (1.722, 9.662) in Map A is the same physical point as (−3.071, 0.038) in Map B. Combined with corridor-orientation alignment, a rigid transformation Map B → Map A is well-posed. Not required for the paper but useful for §3 figures if a unified-frame trajectory plot is wanted.
2. **IEEE GenAI disclosure.** One paragraph in acknowledgements; resolve before final writing pass. Per FLICS 2026 / IEEE Computer Society Conference Publishing Services policy.
3. **Honest framing of Hardening E in §V.** Don't soft-pedal the partial-signal-then-killed-by-corrections finding — write it as the agent's suggested wording in `docs/p1_hardening/results_report.md` §8.

---

## 10. Phase structure (final state)

| Phase | Status |
|---|---|
| **Phase 0** (validation gates, anomaly handling, feature definitions) | Complete. Locked at Rev7. |
| **Phase 1 dataset** | Complete. SHA-256 locked. |
| **Project A (cross-session)** | Complete. Verdict: LiDAR removable. |
| **Hyperparameter robustness diagnostic** | Complete. Pivot justified. |
| **Project B (within-session)** | Complete. Verdict: MIXED → R-4 robustness check needed. |
| **R-4 robustness diagnostic** | Complete. R-4 illusory; cleanly negative. |
| **Hardening A–E** | Complete. Reviewer objections preempted across five orthogonal axes. |
| **Unified report** | In progress. Single source of truth for paper writing. |
| **Paper writing** | **Pending unified report.** |

---

## 11. Risk register (paper-writing era, hardened)

The science risks are now realized or closed. The remaining risks are about how the paper is written and received — and most of the standard objection lines now have hard data responses ready.

| Risk | Mitigation |
|---|---|
| **Reviewers may dismiss the negative result as "your features were just bad."** | **Hardening A (cross-session placebo) closes this directly.** Within-session-shuffled LiDAR achieves Δ_LiDAR within 0.5 dB of real LiDAR on F-B in-FOV under both hyperparameter configurations. The model is not extracting row-aligned LiDAR information beyond marginal distributions. State this in the paper §IV. |
| **Reviewers may dismiss the negative result as "your hyperparameters were wrong."** | **The six-config robustness diagnostic on F-B + Hardening B (LightGBM) close this.** Δ_LiDAR is negative across {locked, depth-4, slow-eta + heavy-L2, depth-8 + λ=20, two random-validation alternatives, LightGBM default, LightGBM H1-equiv} — eight configurations across two frameworks. State this in the paper §IV. |
| **Reviewers may dismiss the negative result as "your dataset was too small."** | Phase 1 dataset is 681,593 rows pre-anomaly-filter, 607,156 post-filter. Per-fold training sets are 374k–449k rows. Document the row counts prominently in the paper §III. |
| **Reviewers may dismiss the negative result as "your dataset is too noisy."** | **Hardening D closes this directly.** σ_intra at 0.5 m granularity is 4.95 dB; the cleanest within-session model achieves RMSE = 1.99 dB — already 3 dB below the natural noise floor. The dataset is not too noisy; the model is operating at sub-cell precision. State this in the paper §III with the noise-floor figure. |
| **Reviewers may say "individual diagnostics could mask the truth; what about both together?"** | **Hardening C closes this.** Combined H1+buffer Δ_LiDAR_within (in-FOV) = −0.83 dB; the corrections do not cancel. State this in the paper §V. |
| **Reviewers may ask "but you found one fold where LiDAR helps (R-4)."** | **Hardening E + Project B's R-4 robustness diagnostic together address this with full nuance.** R-4's locked-no-buffer fit *did* contain real row-aligned LiDAR signal (placebo gap = 0.98 dB) — the paper acknowledges this. But the signal is contingent on (a) training data spatially adjacent to the test region (killed by 1m buffer: Δ → −1.27) and (b) tree depth ≥ 6 (killed by H1: Δ → −0.33), and (c) does not survive both corrections together (killed by H1+buffer: Δ → −0.83). Honest framing: signal exists but is fragile and does not generalize to deployment-relevant testing scenarios. State this in §V exactly as the agent's suggested wording. |
| **Reviewers may ask "why didn't you also test scenario X?"** | We tested four main + five hardening scenarios across five orthogonal axes (cross-session generalization, hyperparameter robustness, within-session generalization, framework agnosticism, placebo-controlled signal). Each one independently produced the same answer. Document the convergence as a table. |
| **The path-loss-exponent disagreement at truth AP (1.22 / 0.22 / 0.38) is a striking finding that could be its own paper.** | Surface it in §III as a foundational observation. Frame the LiDAR negative result as: "even given the structurally non-radial propagation revealed by the n_d disagreement, LiDAR-derived environmental geometry features do not absorb the missing structure." This makes the negative result land more cleanly. |
| **The single-LiDAR-only setup is a real limitation.** | Address explicitly in §VII: in-FOV stratum is the dual-LiDAR-equivalent lower bound, and it does not show LiDAR helping either. The result is therefore not just about the experimental setup. |
| **Time runs out for paper writing.** | The proposal Rev9 + the four phase reports + the hardening report + the unified report (in progress) constitute almost-complete content. The remaining work is structuring + prose. Estimate: 2–3 days agent-assisted drafting + 1–2 days human revision. Doable before April 28. |

---

## 12. Recommendation

1. **Adopt Revision 9 as the post-hardening framing.** The cleanly-negative paper structure is the right narrative; the hardening experiments make it bulletproof against the four common reviewer objections plus the "but you found one fold where it works" critique.
2. **The headline is the result, hardened.** Eight robustness checks across five orthogonal axes converge on the same conclusion: LiDAR features do not improve over AP-relative geometry plus telemetry under any deployment-relevant testing protocol.
3. **All technical work is complete.** Phase 0, Phase 1 dataset, Project A, hyperparameter diagnostic, Project B, R-4 diagnostic, Hardening A–E — done. Frozen artifacts at known paths.
4. **Next step is the unified report**, then paper writing. The unified report (in progress) consolidates all eight experiments into a single source-of-truth document. The paper-writing prompt that follows will draw exclusively from the unified report and the IEEEtran template at `docs/paper/`.
5. **Paper §V should adopt the agent's suggested wording for the R-4/Hardening-E narrative verbatim.** It correctly threads the needle between honesty (real signal exists at locked-no-buffer) and the headline (signal does not survive any of the three corrections). Soft-pedaling either direction would weaken the paper.

---

**End of Revision 9.**

*Phase 0, Phase 1, R-4 diagnostic, and the five hardening experiments are complete. The eight-experiment, five-orthogonal-axis convergence (cross-session LORO, hyperparameter robustness, within-session leave-region-out, R-4 robustness, cross-session placebo, framework agnosticism, combined-correction logical-completeness, dataset-noise-floor characterization, within-session placebo) supports a single conclusion: lab-measured AP coordinates plus AGV telemetry suffice for industrial WiFi prediction in this configuration; LiDAR-derived environment features add no value beyond what AP geometry encodes, except in a contingent within-session locked-depth-6 contiguous-train-test configuration where the signal exists but does not survive any deployment-relevant correction. The paper writes itself around this finding.*
