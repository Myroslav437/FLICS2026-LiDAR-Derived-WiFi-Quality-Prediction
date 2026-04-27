# Focused Research Proposal (Revision 7)

**LiDAR-Aware Cross-Route Generalization of WiFi Signal Prediction for Industrial AGVs, with Frame-Independent Features and SHAP-Based Explainability**

*Revision 7 incorporates the Phase 0 v3 update (`docs/p0_analysis/report.md` §14): anomaly handling simplified to a single per-row threshold on `nns_position_confidence` (T* = 35); Gate C downgraded from GREEN to YELLOW after the cleaner mask removes inflation in 25.02's headline correlation. The substantive changes from Rev6: (a) §3 rewritten around threshold-based row exclusion; (b) §1 and §7 reflect Gate C YELLOW; (c) the B5 / B5' / B5'' disambiguation experiment elevated from auxiliary to essential since the multivariate LORO is now the load-bearing test of the LiDAR-helps hypothesis; (d) the Phase 1 pivot criterion (F-B Δ_LiDAR < 1 dB → Project B) is now more consequential and explicitly highlighted; (e) anomaly percentages updated to v3 values (17.05% / 6.24% / 6.94%).*

---

## 1. Executive summary

Three structural facts about the dataset drive the proposal, all confirmed in Phase 0:

1. **Cross-session shift is a trajectory effect, not a session bias.** Three sessions, one workspace, one physical AP (same hardware, firmware, antenna; lab-measured AP coordinates agree to 1 mm between 15.03 and 24.03 in the same map frame). The 8.6-dBm peak-to-peak shift in mean `signal_power` is the trajectory integral of an approximately stable signal-power field along three different sampling paths.

2. **The dataset uses two maps, not one.** 15.03 and 24.03 share Map A (Phase 0 Gate 0 GREEN at cosine similarity 0.969; lab-measured AP coordinates agree to 1 mm). 25.02 was recorded against a different Map B; Phase 0 confirmed zero spatial cell overlap with the others.

3. **The experimental AGV has only a front LiDAR; production industrial AGVs typically have two.** ISO 3691-4 360° safety-coverage requirements drive dual-LiDAR deployments. Single-LiDAR is the *harder case*; dual-LiDAR results are bounded below by our in-FOV stratum results.

Phase 0 v2 added a fourth fact, with substantive consequences for the paper's framing:

4. **Log-distance from the AP is empirically a poor primary model on 2 of the 3 routes, even with the AP coordinates known to lab-measurement precision.** Per-session R² of the log-distance fit at the lab-measured AP: 0.357 (15.03), 0.003 (24.03), 0.088 (25.02) (v3 values; v2 numbers within rounding). On 24.03 and 25.02, the path-loss exponent collapses to ~0.2–0.4 — the field is approximately flat in log10(distance). Propagation in those sessions is structurally dominated by waveguiding, blockage, and multipath rather than by free-space-path-loss-style range attenuation. **LiDAR-derived environment features therefore carry the predictive load by physical necessity, not by accident.**

These together define the headline experiment:

> *Can a model trained on AGV ego-frame LiDAR features and AP-relative geometry (using lab-measured AP coordinates; no global map (x, y) inputs) predict WiFi `signal_power` on a held-out route, when log-distance from the AP is by itself a poor primary predictor on 2 of 3 routes?*

The contribution claim has three pillars:

- **Frame-independence by design** — no global `(x, y)` inputs; robust to facility re-mapping. The 25.02 fold (Map B held out from Map A training) directly tests this.
- **Feature obtainability triaged into clear tiers** — every input is either trivially available on the AGV bus, recoverable from infrastructure metadata, or explicitly avoided. AP coordinates are treated as known infrastructure metadata; we *additionally* show that when the AP coordinates are unknown, log-distance fitting recovers them to ~30 cm on routes with sufficient geometric coverage (a Phase 0 finding).
- **Deployment-relevant lower bound** — single-LiDAR experimental setup yields a lower bound on what dual-LiDAR production deployments can achieve, made quantitative through FOV-stratified evaluation.

The verified literature gap (§5.0) confirms that this combination — frame-independent LiDAR-derived features, WiFi link quality as the prediction target, SHAP-based explainability, and cross-route generalization on real industrial AGV data — is unoccupied in the 2020–2026 literature.

**Gate C status update.** Phase 0 v3's confidence-threshold-based anomaly mask downgraded Gate C from GREEN to YELLOW: 25.02's headline |ρ| moved from 0.255 (just above the GREEN 0.25 threshold) to 0.230 (just below) under the cleaner mask, leaving only 24.03 above the threshold — and 24.03 still carries the R² ≈ 0 confound (its high |ρ| partly reflects "LiDAR encodes position; raw signal correlates with position"). The univariate LiDAR ↔ residual signal sits at the YELLOW border. **The methodological consequence: the multivariate LORO experiment in Phase 1 is the load-bearing test of the LiDAR-helps hypothesis, not the univariate Phase 0 correlations.** The B5 / B5' / B5'' disambiguation experiment (§5.6) is correspondingly elevated from auxiliary to essential.

---

## 2. Reframed view of the cross-session shift

### 2.1 Hypothesis

There is a single, approximately time-invariant scalar field `S(p) ≈ signal_power` over the workspace, where `p` is a point in *physical* space (independent of map coordinates). The field is determined by AP geometry, walls, fixed clutter, and to a smaller extent by transient day-specific obstacles. Each session is a sampling path along a different physical trajectory `Γ_d(t)`. The per-session mean `(1/T_d) ∫ S(Γ_d(t)) dt` therefore differs across days because the trajectories cover different parts of the workspace — not because the field changes.

### 2.2 Evidence from the initial analysis

The trajectory plots show visually consistent spatial structure: 25.02 traces a tight high-signal corridor near the AP (mean −31 dBm, std 6.2); 24.03 spends most of its time in a region behind some obstruction (mean −40 dBm, std 7.4); 15.03 spans both regimes (mean −37 dBm, std 10.5 — the largest spread because it samples the widest range of geometries). The std ordering is a falsifiable prediction: a session whose route covers a wider range of `S(p)` values must have a larger within-session std. The data agrees.

### 2.3 What "approximately" means

Three reasons the field is not exactly time-invariant, all small:

- **Day-specific obstacles.** People, carts, pallets in different positions on different days — varying but not drastically. The LiDAR registers this directly.
- **Ambient 2.4 GHz traffic.** Affects `signal_noise` more than `signal_power`; the initial analysis shows `signal_noise` varying within a 3-dB band.
- **AP-side load.** Same hardware and firmware, but other clients may be associated with the AP; affects `ping` more than `signal_power`.

Phase 0 v2 quantified these: same-cell |Δ signal_power| between 15.03 and 24.03 has median 4.36 dB across 85 overlapping cells (Gate D YELLOW). The field is mostly stable but with material per-cell drift across the 9-day gap. RQ4 (§5.3) directly investigates this.

### 2.4 What "different routes" does and does not mean

The three sessions traverse different routes through the *same* facility. Phase 0 confirmed: 24.03's coverage is a 98.9% subset of 15.03's; 25.02 is in a separate frame with no shared cells. "Held-out route" in this proposal means *a trajectory pattern the model has not been trained on*, not *a region of physical space that was never visited in training*. Most of an F-A or F-B test fold's coverage will be in regions the training data also visited; what's "new" is the specific time-ordered traversal. F-C (25.02 held out) is the one fold where physical-space novelty is genuine — and even there, the same physical AP and same general workspace layout apply.

---

## 3. AP coordinates, frame handling, and data cleaning

### 3.1 The actual structure: two maps, three sessions

- **Map A**: 15.03 and 24.03. Phase 0 P0.0 verified (cosine 0.969, beam-RMSE 1518 mm on 3 candidate cells).
- **Map B**: 25.02. Phase 0 P0.1 confirmed zero cell overlap with Map A sessions.

Within-session position-confidence drift exists (NNS reports it, correction algorithms run continuously) and is small under normal operation. Operational anomalies are handled by the threshold-based exclusion described in §3.7.

### 3.2 AP coordinates: lab-measured ground truth

The user measured the AP position in the lab in each session's coordinate frame:

| Session | Map | AP coordinates (m) |
|---|---|---|
| 15.03 | A | (1.722, 9.662) |
| 24.03 | A | (1.721, 9.662) |
| 25.02 | B | (−3.071, 0.038) |

Effective measurement accuracy ~5–10 cm (tape-measure / map-reference uncertainty). The 1 mm difference between 15.03 and 24.03 is sub-cm corroboration that both sessions live in the same Map A frame — independent supporting evidence beyond the P0.0 LiDAR-scan-comparison Gate 0.

These coordinates are the canonical operational source for AP-relative features in Phase 1. The frozen feature extractor at `analysis/p0/artifacts/feature_extractor.py` uses them directly.

### 3.3 Per-session log-distance fit at truth AP — a Phase 0 finding

With the AP fixed at lab-measured truth and only `(P0_d, n_d)` free, the per-session log-distance fit (Phase 0 v3) produces:

| Session | n_d | n_d 95% CI | P0_d (dB) | R² |
|---|---:|---|---:|---:|
| 15.03 | 1.224 | [1.181, 1.268] | −25.49 | 0.357 |
| 24.03 | 0.239 | [0.178, 0.406] | −37.83 | 0.003 |
| 25.02 | 0.379 | [0.366, 0.426] | −27.89 | 0.088 |

Two structural observations:

**(a) Log-distance is a poor primary model on 24.03 and 25.02 even at the truth AP.** R² = 0.003 and 0.088 respectively. The path-loss exponent `n` collapses to 0.2–0.4 — the field is approximately flat in log10(distance). This is the cleanest possible empirical statement that distance from the AP is the wrong primary explanatory variable for those sessions' signal maps. LiDAR-derived structural features have maximum headroom by physical necessity.

**(b) Cross-session `n_d` disagreement is the cleanest RQ4 lead.** Same hardware, same firmware, same antenna; physically `n_d` should be approximately constant across sessions. The 95% bootstrap CIs are pairwise disjoint between 15.03 and the other two. This is a residual-session-effect signal that no single-`n` baseline can absorb. RQ4 (§5.3) tests it explicitly via per-session intercept and session-indicator ablations.

### 3.4 What the AP-position fit tells us about deployment without ground truth

A complementary finding from Phase 0 v1 (which used user priors with ±3 m search boxes rather than ground truth): the *free-fit* AP location for 15.03 was within 30 cm of the lab-measured truth. On 24.03 and 25.02 the free fit ran away because the route geometry didn't span enough of the path-loss curve to constrain the AP location, but a reasonable user prior (~1 m accuracy, achievable from a facility map or visual placement) was sufficient to constrain the path-loss exponent and intercept.

For deployment:

- **AP coordinates known** → use them directly. The §3.2 ground-truth case.
- **AP coordinates unknown but route covers a range of distances** → log-distance fit recovers the AP to ~30 cm; treat the fitted value as the operational value.
- **AP coordinates unknown and route is constrained (corridor)** → log-distance fit is degenerate, but a ~1 m visual prior plus a constrained `(P0_d, n_d)` regression suffices for AP-relative feature derivation.
- **Facility re-mapping** → one-line metadata update per AP.

This becomes a paper-worthy methodological footnote in §3, not a load-bearing element of the proposal.

### 3.5 Frame-independence is the real contribution

The model uses **no global `(x, y)` inputs** in any cross-session variant. The features are either ego-frame LiDAR aggregates or per-session-AP-relative geometry, computed with lab-measured AP coordinates. This makes the model robust to facility re-mapping, which routinely changes raw map coordinates — a deployment-relevant property that any model depending on raw `(x, y)` does not have.

The F-C LORO fold (25.02 held out, trained on 15.03 + 24.03) is the direct experimental test of this property: training data lives entirely in Map A, the test fold lives entirely in Map B, and the model has never seen Map B coordinates. If the feature stack is genuinely frame-independent, F-C should not collapse for frame reasons (it might still collapse for genuine physical-environment-difference reasons — that's a separate question RQ2 addresses).

### 3.6 Cross-frame registration — deferred

The lab-measured AP coordinates provide a single hard cross-frame correspondence: (1.722, 9.662) in Map A is the same physical point as (−3.071, 0.038) in Map B. Combined with corridor-orientation alignment, the rigid transformation Map B → Map A is well-posed. It is *not required* for the LORO experiment because the LORO model has no global-frame inputs. Left for Phase 1 if a unified-frame visualization for the paper's §3 figures is wanted.

### 3.7 Data cleaning: confidence-threshold-based row exclusion

The Phase 1 model uses AGV position `(x_m, y_m)` to compute every AP-relative feature. If the position is unreliable, the derived features are unreliable — independent of the *cause* of unreliability. Phase 0 v3 simplifies the data-cleaning step to a single row-level rule:

> **A row is excluded from training iff `nns_position_confidence < 35` or the value is invalid.**

The threshold T* = 35 was determined empirically in Phase 0 v3 by locating the saddle of the empirical confidence histogram, which is bimodal: a tall narrow mode at ~95–100 (NNS confident) and a smaller, broader mode at ~19–23 (NNS struggling). The saddle between them sits at confidence value 35.

A complementary ROC analysis using row-level position discontinuities was inconclusive on this dataset — manual repositioning here presents as long stretches of stuck-at-wrong-position rows rather than row-level position jumps (the actual jump happens only at the recovery moment). This is itself a non-obvious observation about how NNS systems behave during failure modes; it is reported as a methodological side note in the paper. The histogram-saddle threshold is the operational source.

**Per-session row-exclusion summary (Phase 0 v3):**

| Session | n_rows | n_excluded (v3) | % excluded |
|---|---:|---:|---:|
| 15.03 | 280,025 | 47,738 | 17.05% |
| 24.03 | 168,940 | 10,535 | 6.24% |
| 25.02 | 232,628 | 16,156 | 6.94% |
| **Total** | 681,593 | 74,429 | 10.92% |

The threshold-based mask catches 57% of rows that a separate stop-classifier (run as a Phase 0 validation reference) flagged as manual-reposition events. The remaining 43% of those rows had high NNS confidence — they were rows of legitimate stationary operation that the classifier had over-tagged at the episode level. The v3 mask additionally flags 8,162 rows that the stop-classifier missed because they fell below its 30-second minimum stop duration. The threshold-based approach is therefore both more conservative (does not exclude valid stationary rows) and more complete (catches short low-confidence stretches the classifier missed).

For the paper's §3 (data cleaning), the threshold rule and the saddle justification are the operational content. The stop-classifier reference and the ROC-degeneracy note are background that goes into a methodology footnote.

---

## 4. Feature obtainability and deployment realism

### 4.1 Three tiers of feature provenance

**Tier 1 — features the AGV trivially has at inference time:**

- All LiDAR-derived features (the AGV runs the LiDAR for safety; data on the bus).
- All telemetry (speed, turn rate, load, battery, navigation flags including `nns_position_confidence`).
- AGV heading and AGV position in its current map.

**Tier 2 — features requiring one-time operator configuration or short calibration drive:**

- AP coordinates in the AGV's map. WiFi APs are physical infrastructure surveyed during facility setup; storing each AP's coordinates as a one-line config entry per AP is the same kind of metadata any indoor WiFi engineering tool already requires. When unavailable, Phase 0's log-distance fit (§3.4) recovers the position from a single 30-minute calibration drive. Re-mapping requires only a one-line metadata update per AP, not a full re-collection of training data.

**Tier 3 — features explicitly not used:**

- *Future positions of the AGV.* The model is point-in-time.
- *Facility CAD plans, pre-built occupancy maps, structural priors.* No such inputs.
- *AP-side measurements.* All WiFi metrics come from the AGV's MikroTik client.

### 4.2 Front-LiDAR-only setup vs typical industrial AGVs

The experimental AGV has a single front-facing 270° safety LiDAR (Leuze RSL 400) captured at 0.2° angular resolution (1 350 active beams; 2 700-slot buffer storage). Production industrial AGVs typically have **two** LiDARs — front and rear — for ISO 3691-4 360° safety coverage. With both LiDARs every bearing relative to the AP heading is observable, and directional features carry physical signal regardless of the AP's relative direction.

The single-LiDAR setup is the *harder case*. Two practical consequences:

(a) The results we report are a *lower bound* on the predictive performance achievable in standard industrial deployments.

(b) We stratify all evaluation metrics by `is_AP_in_FOV` — a boolean derived from `angle_to_AP` (using the lab-measured AP) and the empirically-determined LiDAR valid-angle range `[−110.0°, 111.6°]` (Phase 0 P0.3, a 222° contiguous valid sector after the AGV-body mask is applied):

- **In-FOV**: directional LiDAR features add substantial predictive value. `clutter_toward_AP` reliably encodes LOS blockage.
- **Out-of-FOV**: directional LiDAR features add little; the model leans on `dist_to_AP` and full-sweep aggregates.

In-FOV results are directly representative of dual-LiDAR coverage.

### 4.3 SHAP-by-FOV interaction analysis

The XAI analysis (§5.7) includes a SHAP × FOV interaction (XAI-4): for each directional LiDAR feature, plot SHAP value vs feature value separately for in-FOV and out-of-FOV samples. Expected: strong negative slope for `clutter_toward_AP` when the AP is in FOV (more clutter on the LOS path → weaker predicted signal); flat / null slope when out of FOV. Confirming this signature on held-out folds is direct evidence that the model has learned route-invariant propagation physics.

---

## 5. Research direction

### 5.0 Verified literature gap

A targeted search across IEEE Xplore, ACM, arXiv, MDPI, Springer, and ScienceDirect (2020–2026) confirms the gap. Closest related work clusters and what each does not address: LiDAR for radio prediction (mmWave / surfaces, no WiFi link quality, no SHAP); ML for AGV WiFi link quality (Ohori et al. 2023; Formis & Scanzio 2025 — no environmental geometry, no XAI); XAI for wireless (Masood et al. 2023; Kiouvrekis et al. 2025 — cellular path loss or HAR, not industrial WiFi quality); LiDAR + WiFi sensor fusion (DLoc, EKF fusion — localization targets, not link quality). The intersection is unoccupied.

### 5.1 Working title

*LiDAR-Aware Cross-Route Generalization of WiFi Signal Prediction for Industrial AGVs, with Frame-Independent Features and SHAP-Based Explainability of Geometric Propagation Effects.*

### 5.2 Core thesis

A model trained on ego-frame LiDAR features and per-session AP-relative geometry (using lab-measured AP coordinates), with **no global map coordinates** as inputs, predicts WiFi `signal_power` on AGV routes the model has not been trained on, in workspaces where log-distance from the AP is by itself a poor primary predictor (R² ≈ 0 on 2 of 3 routes at the truth AP — Phase 0 v3). TreeSHAP attributions decompose the prediction into physically interpretable contributions from distance, line-of-sight openness, and front-cone obstruction, with directional contributions appropriately gated by whether the AP is in the LiDAR's field of view — quantifying for the first time on real industrial AGV data which environmental geometry features carry which fraction of the explanatory power, in which physical regimes.

### 5.3 Research questions

| # | Question | Experiment |
|---|---|---|
| **RQ1** | Does a LiDAR-aware model generalize to a held-out route, beyond what AP-relative geometry alone provides — without using any global-frame features — and does the LiDAR contribution differ between in-FOV and out-of-FOV regimes? | LORO with all three folds (§5.5). Headline: **B5 vs B1** per fold, broken down by `is_AP_in_FOV`. |
| **RQ2** | How does generalization depend on which session is held out? | All three LORO folds reported with equal status, accompanied by per-fold metadata (spatial overlap, frame status, path-loss-fit R² of training-set sessions). |
| **RQ3** | Are TreeSHAP attributions on the held-out fold physically sensible — do directional LiDAR features push predictions in propagation-physics-required directions when the AP is in FOV, and become appropriately uninformative when out of FOV? | Sign-of-effect analysis on the LORO-trained model, stratified by FOV. |
| **RQ4** | After conditioning on AP-relative geometry and LiDAR features, is there a residual session effect? Phase 0 v2/v3 surfaced a specific lead: per-session path-loss exponents at the truth AP differ significantly across sessions (1.22, 0.24, 0.38; pairwise disjoint bootstrap CIs) despite same hardware. | (i) Per-session-intercept ablation: train B5 with vs without per-session intercept; report Δ RMSE. (ii) Session-indicator ablation: train B5 with `session_id` as a categorical feature; report SHAP contribution of `session_id` as the residual-effect estimate. Run on the same-map subset (15.03 + 24.03) and on the full three-session pool separately. |

### 5.4 Feature stack

All features below are computed at LiDAR scan rate (~25 Hz), on rows surviving Phase 0 v3 row exclusion (`nns_position_confidence ≥ 35` AND finite). **No raw `(x, y)`** is included in any cross-session model variant.

**LiDAR (ego-frame), AGV-body mask applied:**

The AGV-body mask (Phase 0 P0.3) marks 1 591 of the 2 700 raw distance slots as invalid: 1 350 are buffer padding (the sensor was captured at 0.2° giving 1 350 active beams stored in a 2 700-slot buffer) and 241 are AGV-body returns at sub-200 mm range. The remaining 1 109 active beams form a contiguous valid sector spanning [−110.0°, +111.6°] — 222° of usable FOV.

- *Scalar aggregates*: `mean_dist_mm`, `dist_p90_mm`, `clutter_frac` (valid beams below 4 m), `openness_frac` (valid beams above 7 m), `mean_front_mm` (front cone, fixed reflector excluded).
- *Sectoral*: the valid 222° sector split into **7 sectors of ~30°** each. Two features per sector → `mean_dist_sector_i_mm` and `clutter_frac_sector_i` for `i ∈ {1..7}`. 14 features.

**Telemetry (frame-independent):**

- `speed_mps`, `turn_rate`, `load_long`, `load_mid`, `load_short`, `battery_value`, `momentary_current_consumption`, `nns_state`. ~8 features. (Note: `nns_position_confidence` is *not* a model feature — it is the gating variable used at the row-exclusion step in §3.7. Including it as a feature would let the model learn to predict less reliably on low-confidence rows, which is the wrong abstraction.)

**AP-relative (using lab-measured AP coordinates per session):**

- `dist_to_AP_d` — Euclidean distance from `(x_m, y_m)` to `(x_AP, y_AP)_d`.
- `sin_angle_to_AP_d`, `cos_angle_to_AP_d` — bearing to AP in AGV ego frame.
- `clutter_frac_toward_AP_d` — `clutter_frac` computed only on the LiDAR sector pointing at the AP.
- `is_AP_in_FOV` — boolean, `(angle_to_AP_deg ∈ [−110.0°, +111.6°])`.

Total: ~32 features, all frame-independent or per-session-AP-relative.

The within-session sanity check (§5.5 sub-protocol) is the one place `(x, y)` may appear — single-frame context, no cross-session generalization claim. Stays out of the LORO model.

### 5.5 Evaluation: leave-one-route-out (LORO), all combinations

Three folds, all reported with equal status — no fold designated as the headline a priori. Per-fold metadata populated from Phase 0 results:

| Fold | Train | Test | Spatial overlap (cells) | Cross-frame status | Train-set R² at truth AP | Notes |
|---|---|---|---|---|---|---|
| **F-A** | 15.03 + 25.02 | 24.03 | 24.03 trajectory ⊂ 15.03 (98.9% of test cells visited in training) | Mixed-frame training (Maps A and B); same-map test (24.03 in Map A) | 15.03: 0.36, 25.02: 0.09 | High train→test cell-overlap. The LiDAR-helps test on this fold has the cleanest expected signal. |
| **F-B** | 24.03 + 25.02 | 15.03 | 0% of 15.03's 225 test cells visited in training (24.03 covers 1 unique cell beyond 15.03; 25.02 disjoint frame) | Mixed-frame training; same-map test (15.03 in Map A) | 24.03: 0.003, 25.02: 0.09 | **Hardest fold.** 15.03's coverage is the largest — most test cells unseen. Training-set log-distance fits are essentially flat. |
| **F-C** | 15.03 + 24.03 | 25.02 | 0% (disjoint frames) | Same-map training (Map A); different-map test (Map B) | 15.03: 0.36, 24.03: 0.003 | **Direct test of frame-independence.** Same physical workspace, different map; model has never seen Map B coordinates. |

**No fold is privileged in advance.** All three are reported in the headline ablation table. The discussion interprets each fold in light of its metadata.

**Reported metrics, stratified.** For each fold: RMSE, MAE, R², bias, bootstrap 95% CIs (B = 1000), in three forms: overall, in-FOV, out-of-FOV. The headline summary table is 3 folds × 3 strata × ~6 model variants = 54 cells, presenting cleanly as three side-by-side per-fold tables.

**Cadence matching for F-C.** 25.02 has telemetry at 4.4 Hz vs 31 Hz on the other days. To prevent F-C from underperforming for cadence reasons rather than frame/geometry reasons, downsample 15.03 and 24.03 to 4.4 Hz before training the F-C model.

**Within-session sanity check (sub-protocol).** Each session in isolation, train on the first 70% chronologically, validate on next 15%, test on the last 15%. May include `(x, y)`. For diagnosing model health and §3 of the paper.

### 5.6 Model variants — disambiguation experiment now essential

Phase 0 v3 demonstrated that log-distance from the AP has R² ≈ 0 on 2 of 3 sessions even at the truth AP location, and Gate C sits at YELLOW with the cleaner anomaly mask. The univariate LiDAR ↔ residual signal cannot carry the LiDAR-helps claim on its own; the multivariate LORO experiment must.

| Variant | Features | Purpose |
|---|---|---|
| **B0** | `dist_to_AP` only | Smoke test. On 24.03 / 25.02, R² ≈ 0 at the truth AP, so B0's RMSE is essentially the per-session standard deviation of `signal_power`. Reported per fold but not the headline. |
| **B1** | B0 + `sin/cos(angle_to_AP)` | **Reference**: full AP-relative geometry without LiDAR. The cleanest "AP-only physics" baseline. |
| **B2** | B1 + LiDAR scalar aggregates | Adds coarse environment context. |
| **B3** | B2 + LiDAR sectoral features | Adds direction-sensitive LiDAR. |
| **B4** | B3 + telemetry | Full LiDAR + telemetry, no AP-targeted features. |
| **B5** | B4 + `clutter_frac_toward_AP` + `is_AP_in_FOV` | **Headline full model**. AP-aware, FOV-aware, ego-frame LiDAR + telemetry. |

**Headline ablation comparisons (in priority order):**

1. **B5 vs B1** — *the* headline result. Does LiDAR + telemetry carry generalizable signal beyond AP-relative geometry alone?
2. **B5 vs B5-without-LiDAR** (= B1 + telemetry + `is_AP_in_FOV`) — LiDAR's *unique* contribution given everything else.
3. **B5(in-FOV) vs B5(out-of-FOV)** — quantifies the dual-LiDAR vs single-LiDAR gap.

**Disambiguation experiment — elevated to essential (Rev7).** With Gate C at YELLOW the univariate LiDAR signal is borderline, so the question of whether LiDAR contributes uniquely to the multivariate model — or whether it serves as a position proxy — must be settled by direct ablation. Train three variants of B5 alongside the standard ablation:

- **B5** — full model (the headline).
- **B5'** — B5 with LiDAR features removed entirely (telemetry + AP-relative + `is_AP_in_FOV` only).
- **B5''** — B5 with AP-relative features removed entirely (LiDAR + telemetry only).

Comparing B5 / B5' / B5'' on the LORO test folds tells us:

- If **B5 ≈ B5''** (LiDAR is removable without loss) → LiDAR was a position proxy. Contribution claim weakens to "explicit AP geometry suffices when known." This is publishable but a different paper.
- If **B5 ≈ B5'** (AP-relative is removable without loss given LiDAR) → LiDAR alone suffices, no AP coordinates needed. This is a *stronger* frame-independence claim than the proposal currently makes — LiDAR + telemetry would predict WiFi without any global infrastructure metadata at all.
- If **B5 < both** (B5's RMSE is meaningfully lower than either reduced model) → LiDAR and AP-relative carry *complementary* information. This is the proposal's expected outcome and the cleanest support for the contribution claim.
- If **B5 ≈ B5' ≈ B5''** (no model variant beats the others meaningfully) → the predictive signal is in the telemetry, not in LiDAR or AP geometry. This would be a pivot signal — Project B for the paper, with the disambiguation result as the main methodological observation.

XGBoost regressors throughout. Target: `signal_power` in dBm. Hyperparameters: `max_depth=6`, `eta=0.05`, `n_estimators=2000`, early stopping with patience 100 on a chronological-split validation set (last 10% of training data by `fh7000_timestamp`).

### 5.7 XAI analysis

On the LORO-trained B5 model evaluated on each held-out fold:

- **XAI-1 (global, per fold).** TreeSHAP beeswarm + feature-group importance (telemetry, LiDAR scalar, LiDAR sectoral, AP-relative). One figure with three panels (one per fold).
- **XAI-2 (sign-of-effect consistency).** For each LiDAR feature and each AP-relative feature, fit a univariate spline through `(feature_value, SHAP_value)` and report the sign of the slope on each fold. A feature whose SHAP slope is sign-consistent across all three folds — *and across map frames where applicable* — is one whose physical interpretation transfers. One table.
- **XAI-3 (spatial map, on the held-out fold).** Bin the test trajectory's `(x, y)` (in *its own* frame) into 0.5 m cells; per cell, plot the dominant feature group by mean |SHAP|. One figure per fold; F-C panel labeled as Map B coordinates.
- **XAI-4 (SHAP × FOV interaction).** For each directional LiDAR feature (`clutter_frac_toward_AP`, per-sector features pointing at the AP), plot SHAP value vs feature value separately for in-FOV and out-of-FOV samples. Expected: strong negative slope in-FOV (more obstruction → weaker predicted signal); flat slope out-of-FOV. Directional features' physical-soundness check.

### 5.8 Paper structure (up to 9 IEEE pages)

1. **Introduction** (~1.0 pp). Industrial AGV WiFi reliability problem; deployment-relevance of frame-independent prediction; verified gap; contribution statement (three pillars from §1).
2. **Related work** (~1.0 pp). Three clusters, one paragraph each, plus distinguishing paragraphs.
3. **Dataset, calibration, and AP handling** (~1.25 pp). Dataset overview; joint-dataset construction (one paragraph + reference to the time-sync companion report); two-maps structure with Phase 0 same-map verification; lab-measured AP coordinates with the path-loss-fit-recovers-AP observation as a methodological footnote; **threshold-based row exclusion** (T = 35 from `nns_position_confidence`, justified by histogram saddle); the `n_d` cross-session disagreement at the truth AP (RQ4 lead).
4. **Feature obtainability and dual-LiDAR generalization** (~0.5 pp). Three-tier framing; single→dual LiDAR generalization argument; FOV stratification rationale.
5. **Method** (~1.5 pp). Feature stack (with the verified 0.2°/1350-active-beam layout, 222° empirical FOV, 7 × 30° sectors); ablation variants (B0 demoted to smoke test; B5 vs B1 as headline; B5 / B5' / B5'' disambiguation); LORO protocol with all three folds; FOV stratification mechanics; XAI procedure (XAI-1 through XAI-4).
6. **Results** (~2.0 pp). Per-session AP calibration table (n_d, P0_d, R²) with the cross-session disagreement explicitly highlighted; row-exclusion summary; LORO ablation table (3 folds × 3 strata × 6 variants); B5 / B5' / B5'' disambiguation summary; sign-of-effect consistency table; XAI-1; XAI-4 (the cleanest physical-soundness check); selected XAI-3 panel for one fold.
7. **Discussion** (~1.0 pp). Per-fold result interpretation in light of spatial overlap and frame status; in-FOV results as lower bound for dual-LiDAR; the RQ4 residual session effect (`n_d` disagreement); honest limitations (3 routes, single AP, single-front-LiDAR experimental setup, single facility, log-distance regime is not what 24.03 / 25.02 follow, Gate C YELLOW so univariate LiDAR signal was borderline pre-multivariate).
8. **Conclusion** (~0.25 pp).
9. **References + acknowledgements + IEEE GenAI disclosure** (~0.5 pp).

---

## 6. Backup direction (Project B)

If Phase 1 reveals a blocker — specifically, if F-B Δ_LiDAR (B5 − B1) on 15.03 held out is < 1 dB — pivot to a single-session deep dive on 15.03.2026 (largest dataset, widest geometry coverage, the only session with R² > 0.3 in its truth-AP path-loss fit). Same B0–B5 ablation logic, applied within-session on a chronological + spatial-region split. Loses the cross-route + frame-independence claim but produces a clean within-session XAI case study, with the FOV stratification still applicable as the structural physics check.

The 15.03 session is also the cleanest substrate for Project B because it is the only session where the post-distance residual carries genuine post-distance signal (R² = 0.36 in the truth-AP fit, with structured residual maps). On 24.03 and 25.02, "residual" is approximately raw signal_power and a single-session within-session model would be operating on a less differentiated target.

**With Gate C at YELLOW, the F-B Δ_LiDAR threshold is more consequential than at Rev6.** A small Δ_LiDAR on F-B is the realistic failure mode the YELLOW gate flags — if the multivariate model cannot extract LiDAR signal where univariate correlation barely exists, Project B is the right move and the disambiguation experiment (§5.6) provides the methodological observation that anchors the alternative paper.

---

## 7. Phase 0 results — locked

Phase 0 is complete (`docs/p0_analysis/report.md`, `analysis/p0/artifacts/`). Final gate decisions (post-v3):

| Gate | Decision | Note |
|---|---|---|
| Gate 0 (frame-sharing) | GREEN | 15.03 and 24.03 share Map A. Cosine similarity 0.969 + 1 mm AP-position agreement at lab-measured truth. |
| Gate A (LiDAR headroom) | GREEN | 24.03 and 25.02 R² ≈ 0 even at truth AP. Maximum LiDAR headroom by physical necessity — propagation is dominated by structure, not range. |
| Gate B (sectoral feasibility) | GREEN | 222° contiguous valid sector. 7 × 30° sectors. |
| Gate C (project go/no-go) | **YELLOW** | Down from v2's GREEN after the v3 mask. Max in-FOV \|ρ\| > 0.25 on only 1 of 3 sessions (24.03, with R² ≈ 0 confound); 25.02 is at \|ρ\| = 0.230 (just below 0.25); 15.03 at \|ρ\| = 0.090. The univariate LiDAR ↔ residual signal sits at the YELLOW border — multivariate LORO is the load-bearing test. |
| Gate D (framing strength) | YELLOW | Median \|Δ\| = 4.36 dB across 85 same-map cells. Soften the time-stable-field claim in the paper. |

**Phase 1 enters with all artifacts locked:**

- Anomaly mask (v3, threshold-based): `analysis/p0/artifacts/anomaly_mask.parquet`.
- Threshold metadata: `analysis/p0/artifacts/anomaly_threshold.json` (T* = 35, methods documented).
- AGV-body mask: `analysis/p0/artifacts/agv_body_mask.npz`.
- Empirical FOV: `analysis/p0/artifacts/lidar_fov.json`, sector [−110.0°, +111.6°].
- AP coordinates (lab-measured truth): `analysis/p0/artifacts/ap_coords.json`.
- Frozen feature extractor: `analysis/p0/artifacts/feature_extractor.py`, `--validate`-tested against 1000-row samples per session.

---

## 8. Time-sync companion paper — defer

Cite the time-sync report in §3 as "applied per-day correction with σ_rmse ≤ 0.25 s, end-to-end validation passing all 12 checks". Defer the full per-day calibration + two-component hardware interpretation to a separate sensor-fusion / multi-modal robotics venue.

---

## 9. Outstanding caveats and follow-ups

**Closed in Rev7 by Phase 0 v2/v3 + lab measurements:**

- Same physical AP / antenna / firmware (verified). ✓
- Same workspace, three different routes per day, partial spatial overlap (24.03 ⊂ 15.03 at 98.9%; 25.02 disjoint frame). ✓
- Operational anomalies handled by single-threshold rule on `nns_position_confidence` (T = 35; v3 row-exclusion rates 17.05% / 6.24% / 6.94%). ✓
- Map A shared by 15.03 and 24.03 (P0.0 GREEN; AP-coordinates agree to 1 mm). ✓
- AP coordinates known to lab-measurement precision per session. ✓
- Single-front-LiDAR experimental setup; production typically dual front+rear. ✓
- LiDAR layout: 0.2° / 1350 active beams / 222° empirical FOV / 7 × 30° sectors. ✓
- Cross-session `n_d` disagreement at truth AP is the load-bearing RQ4 lead. ✓
- Gate C at YELLOW on the cleaner v3 mask; multivariate LORO is the load-bearing test. ✓

**Still open (not blocking Phase 1):**

1. The 24.03 in-FOV correlation `|ρ| = 0.449` (v3) carries a residual-equals-raw-signal caveat (R² ≈ 0 at truth AP means the path-loss fit absorbs essentially nothing, so the LiDAR ↔ residual correlation partially reflects "LiDAR encodes position; raw signal correlates with position"). Honestly reported in Phase 0 v3 §14.5.2 and discussed in the paper's Results.
2. Cross-frame registration deferred. If a unified-frame visualization is needed for the paper's §3, run it post-hoc.
3. IEEE GenAI disclosure plan — one paragraph in acknowledgements; resolve before final writing pass.

---

## 10. Phase structure

| Phase | Tasks |
|---|---|
| **Phase 0** | Complete. See §7 and `docs/p0_analysis/report.md`. |
| **Phase 1 part 1** | Feature extraction across all sessions using the frozen extractor. Train all LORO ablation models (3 folds × 6 variants = 18 fits) plus the disambiguation experiment (3 folds × 3 variants = 9 more fits for B5 / B5' / B5''). TreeSHAP on the 3 LORO B5 models. RQ4 ablations (with vs without per-session intercept; with vs without `session_id`). |
| **Phase 1 part 2** | Generate all figures and tables: LORO ablation (3 × 3 × 6); B5 / B5' / B5'' disambiguation summary; Δ_LiDAR / Δ_angle / Δ_FOV summary; sign-of-effect consistency table (XAI-2); XAI-1 figure; XAI-4 figure; XAI-3 panel for one fold; RQ4 SHAP-of-`session_id` summary. |
| **Writing** | Paper end-to-end in IEEEtran up-to-9 pp template. IEEE GenAI disclosure compliance. Submission. |

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| **F-B Δ_LiDAR (B5 − B1) is < 1 dB on the 15.03 hold-out.** Phase 0 v3 confirmed 15.03's univariate \|ρ\| ≈ 0.09 — the multivariate model has to find combinations of features that carry signal. With Gate C at YELLOW this is now the realistic failure mode the gate flags. | Pivot to Project B (single-session deep dive on 15.03). 15.03 is the only session with R² > 0.3 in its truth-AP path-loss fit, so within-session modelling there has the cleanest substrate. The disambiguation experiment (§5.6) provides the methodological observation anchoring the alternative paper even if Project A fails. |
| **The B5 model relies on AP-relative geometry as a position proxy rather than physics.** | The B5 / B5' / B5'' disambiguation experiment (§5.6) directly tests for this and is now elevated from auxiliary to essential. If B5 ≈ B5'' (LiDAR removable), the contribution claim weakens to "AP-relative geometry suffices when known"; if B5 ≈ B5' (AP-relative removable), the contribution claim becomes "LiDAR alone suffices, no AP coordinates needed" — a stronger frame-independence claim. Either outcome is publishable. |
| **24.03 in-FOV |ρ| = 0.449 is partly artefactual (R² ≈ 0 confound).** | Honestly reported in §3.7 and §9. The §6 paper-results discussion contextualizes 24.03 vs 15.03 vs 25.02 ρ values. If reviewers raise it, the response is: documented, expected given physical regime, and the LORO comparison is the real test (not the univariate Phase-0 correlations). |
| **Cross-session `n_d` disagreement (1.22, 0.24, 0.38) cannot be absorbed by the model.** | RQ4 is designed exactly to test this. Three outcomes: (a) per-session intercept absorbs it cleanly → small residual; (b) `session_id` indicator is needed and SHAP-attributes a measurable share → publishable RQ4 finding (residual session effect is real); (c) neither absorbs it → discussion section names this as the main limitation, paper still stands on RQ1–3. |
| **F-C (25.02 held out) collapses for environment-physics reasons rather than frame reasons.** | Possible — 25.02's environment may be sufficiently different that the model trained on Map A genuinely cannot transfer. Document honestly with per-fold metadata. The frame-independence claim is preserved (failure is not because of frame); the cross-environment-generalization claim is what fails. Interpret in discussion. |
| **Cadence-driven failure on F-C.** | Mitigated by 4.4 Hz downsampling of training data on F-C. |
| **High operational-anomaly rate on 15.03 (17.05% v3).** | Verified to be driven by the manual-reposition events identified in v1/v2 (with the over-aggressive episode-level tagging removed in v3). Surviving row count: 280k × 82.95% = 232k. Adequate. Document the cleaning rate in §3 of the paper as a deployment-realism observation. |
| **LiDAR helps on 2 of 3 folds but not on F-B.** | F-B is the hardest fold (15.03 held out, training-set log-distance fits both flat). Report honestly with per-fold metadata. The discussion section explains why — 15.03 is the only session with structured post-distance residual to learn from, so when it's held out, the training set carries less learnable signal. |
| **Time runs out for paper writing.** | Start writing during Phase 1 part 2 with placeholders; refine numbers as Phase 1 completes. |

---

## 12. Recommendation

1. **Adopt Revision 7 as the final pre-Phase-1 plan.** Phase 0 is locked; the v3 update sharpens the data cleaning methodology and gives Gate C an honest YELLOW.
2. **Headline framing is now precise.** *Log-distance from the AP is empirically inadequate as a primary model on 2 of 3 routes (R² ≈ 0 even at the lab-measured AP); LiDAR-derived environmental geometry features have headroom by physical necessity; SHAP attributions confirm the learned relationships are physically interpretable across folds and across map frames; the multivariate LORO test in Phase 1 is the load-bearing evaluation since the univariate Phase 0 correlations sit at the YELLOW border.*
3. **B5 vs B1 is the sole headline ablation. The B5 / B5' / B5'' disambiguation experiment is essential, not auxiliary.** Phase 0 v3 made it impossible to claim that LiDAR helps based on Phase 0 numbers alone — the disambiguation tells us *how* it helps in the multivariate setting, and which of three publishable outcomes we land in.
4. **RQ4 is sharpened by the n_d disagreement at truth AP.** This is the cleanest residual-session-effect lead in the data and now has its own dedicated experiment in the methodology.
5. **Project B is a real fallback, gated on F-B Δ_LiDAR < 1 dB.** With Gate C at YELLOW, this pivot criterion is more consequential than at Rev6 — and is the right move if it triggers.

---

**End of Revision 7.**

*Phase 0 locked. Phase 1 ready to begin. The frozen feature extractor is the supported entry point; nothing in Phase 1 should re-derive AP-relative features or reload the AGV-body mask from raw inputs. Subsequent revisions, if any, will be triggered by Phase 1 results.*
