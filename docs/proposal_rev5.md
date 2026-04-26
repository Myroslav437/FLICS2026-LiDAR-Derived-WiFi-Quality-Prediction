# Focused Research Proposal (Revision 5)

**LiDAR-Aware Cross-Route Generalization of WiFi Signal Prediction for Industrial AGVs, with Frame-Independent Features and SHAP-Based Explainability**

*Revision 5 incorporates: (a) detection and treatment of operational anomalies (manual-positioning events, motor-overheat stops); (b) the user-provided approximate AP positions as priors for per-session calibration; (c) revised LORO framing that runs all three hold-out combinations symmetrically rather than designating 25.02 as "the" headline fold a priori; (d) corrected language around "unseen routes" — same facility, overlapping coverage, novel trajectories rather than novel terrain; (e) paper structure adjusted to up to 9 IEEE pages.*

---

## 1. Executive summary

Three structural facts about the dataset drive the proposal:

1. **Cross-session shift is a trajectory effect, not a session bias.** Three sessions, one workspace, one physical AP (same hardware, firmware, antenna), three different routes per day. The 8.6-dBm peak-to-peak shift in mean `signal_power` is the trajectory integral of an approximately stable signal-power field along three different sampling paths.

2. **The dataset uses two maps, not one.** 15.03 and 24.03 share Map A (the user states the same map was used; verified empirically by Phase 0 task P0.0). 25.02 was recorded against a different Map B, re-positioned in the mapping software by a different person. The two maps relate by an unknown 2D rigid transformation but cover overlapping physical space.

3. **The experimental AGV has only a front LiDAR; production industrial AGVs typically have two.** Standard industrial AGVs deploy front + rear safety LiDARs to satisfy ISO 3691-4 360° coverage requirements. The single-LiDAR experimental setup is the *harder case*; dual-LiDAR deployments are bounded below by our in-FOV results.

These facts together define the headline experiment:

> *Can a model trained on AGV ego-frame LiDAR features and per-session-calibrated AP-relative features (no global-frame inputs) predict WiFi `signal_power` on a held-out route, when all three hold-out combinations are tested symmetrically?*

The contribution claim has three pillars:

- **Frame-independence by design** — no global `(x, y)` inputs; robust to facility re-mapping.
- **Feature obtainability triaged into clear tiers** — every input is either trivially available on the AGV bus, recoverable from infrastructure metadata, or explicitly avoided. No future-looking features, no AP-side instrumentation.
- **Deployment-relevant lower bound** — single-LiDAR experimental setup yields a lower bound on what dual-LiDAR production deployments can achieve, made quantitative through FOV-stratified evaluation.

The verified literature gap (§5.0) confirms that this combination — frame-independent LiDAR-derived features, WiFi link quality as the prediction target, SHAP-based explainability, and cross-route generalization on real industrial AGV data — is unoccupied in the 2020–2026 literature.

A Phase 0 / Phase 1 split (§7 + §10) reaches a hard go/no-go before any modelling work begins.

---

## 2. Reframed view of the cross-session shift

### 2.1 Hypothesis

There is a single, approximately time-invariant scalar field `S(p) ≈ signal_power` over the workspace, where `p` denotes a point in *physical* space (independent of map coordinates). The field is determined by AP geometry, walls, fixed clutter, and to a smaller extent by transient day-specific obstacles. Each session is a sampling path through this field along a different physical trajectory `Γ_d(t)`. The per-session mean `(1/T_d) ∫ S(Γ_d(t)) dt` therefore differs across days because the trajectories cover different parts of the workspace — not because the field changes.

### 2.2 Evidence already in the initial-analysis report

The "Trajectories coloured by `signal_power`" panel shows visually consistent spatial structure: 25.02 traces a tight high-signal corridor near the AP (mean −31 dBm, std 6.2); 24.03 spends most of its time in a region behind some obstruction (mean −40 dBm, std 7.4); 15.03 spans both regimes (mean −37 dBm, std 10.5 — the largest spread, because it samples the widest range of geometries). The std ordering is a falsifiable prediction of the trajectory-sampling hypothesis: a session whose route covers a wider range of `S(p)` values must have a larger within-session std. The data agrees.

### 2.3 What "approximately" means

Three real reasons the field is not exactly time-invariant, all small:

- **Day-specific obstacles.** People, carts, pallets in different positions on different days — varying but not drastically. The LiDAR registers this directly.
- **Ambient 2.4 GHz traffic.** Affects `signal_noise` more than `signal_power`. The initial analysis shows `signal_noise` varying within a 3-dB band — small but measurable.
- **AP-side load.** Same hardware and firmware, but other clients may be associated with the AP; affects `ping` more than `signal_power`.

These define the residual session effect that should remain after we condition on AP geometry and LiDAR-derived environment features. RQ4 (§5.3) directly quantifies it.

### 2.4 What "different routes" does and does not mean

A clarification carried through the rest of the document: the three sessions traverse three different routes through the *same* facility. The routes are not in disjoint regions of the building. They overlap to varying extents — the long narrow corridor, in particular, appears in all three sessions; broader regions appear in two of three. "Unseen route" in this proposal means *a trajectory pattern that the model has not been trained on*, not *a region of physical space that was never visited in training*. Most of the test fold's coverage will be in regions the training data also visited; what's "new" is the specific time-ordered traversal, the specific (x, y) sequence, and any tiles uniquely visited by the held-out session.

This distinction matters for interpreting LORO results (§5.5). A successful prediction on a held-out route is more like "good interpolation across the same workspace from a different angle" than "extrapolation into terra incognita". This is still deployment-relevant — operators routinely run the same AGV on new mission patterns through familiar facilities — but the framing should not oversell it.

---

## 3. Map frames and AP-position calibration

### 3.1 The actual structure: two maps, not three drifting frames

- **Map A**: used for both 15.03 and 24.03 (verified by P0.0).
- **Map B**: used for 25.02 (different person, different positioning in the mapping software).

Within-session position-confidence drift exists (NNS reports it, correction algorithms run continuously) and is small under normal operation. Two non-routine cases are flagged separately and handled in §3.7 below: occasional auto-positioning failures requiring manual repositioning, and motor-overheat stops.

### 3.2 What survives frame differences

- **All LiDAR-derived features** — computed in the AGV's ego frame, completely independent of the map.
- **Motion and operational telemetry** — ego-frame.
- **AP-relative features** computed with a *per-session-calibrated* AP coordinate.

### 3.3 What does not survive

- **Raw `(x, y)`** as a cross-session feature.
- **A single global AP coordinate** copied across sessions.

### 3.4 Per-session AP calibration with informed priors

The user provides approximate AP positions for each session, in that session's own frame:

| Session | Map | Approximate AP position (user prior) |
|---|---|---|
| 15.03 | A | (x, y) ≈ (2, 10) |
| 24.03 | A | ≈ (2, 10) — should equal 15.03 if the same-map hypothesis holds. P0.0 verifies. |
| 25.02 | B | (x, y) ≈ (−2.5, 0.5) |

These are approximate; the real positions are nearby. The per-session log-distance path-loss fit (P0.2) refines them:

```
signal_power(x, y) = P0_d − 10 n_d log10( ‖(x, y) − (x_AP, y_AP)_d‖ ) + ε
```

with `(x_AP, y_AP)_d` initialized at the user prior, and `P0_d`, `n_d`, `(x_AP, y_AP)_d` fit jointly by nonlinear least squares on motion-active rows (`|speed_mps| > 0.05`). Initialization at the prior is important — the loss surface for `(x_AP, y_AP)` in the presence of multipath and partial blockage can be multi-modal, and a poor random initialization can converge to a local minimum corresponding to a wall-reflection peak rather than the AP itself. Restrict the search to a ±3 m box around the prior to enforce this.

Two consistency checks fall out:

- **Same-map AP agreement**: fitted `(x_AP, y_AP)_15.03` and `(x_AP, y_AP)_24.03` should agree numerically (within ~1 m). Their agreement is a *test* that 15.03 and 24.03 share Map A. Disagreement → either Map A is not actually shared (Gate 0 in P0.0 should also flag this), or the AP fit failed on one session due to insufficient geometric coverage.
- **Frame-independent parameter agreement**: `n_d` and `P0_d` should agree across all three sessions (same physics, same hardware). This check is independent of map frame and is the cleanest test that the field is time-stable. Disagreement → RQ4's first lead.

### 3.5 Frame-registration is *not strictly required* for LORO

Because the LORO model deliberately uses no global `(x, y)` features, registering Map B onto Map A is not a prerequisite for running the headline experiment. It is useful for two secondary purposes: cross-session same-cell consistency tests (§7.7, P0.6) and unified spatial visualization of all three sessions in one figure. P0.0b (§7.1b, optional) recovers the rigid transformation when time permits. The LORO experiment runs without it.

### 3.6 The contribution claim sharpens

The constraint *the model has no global-frame inputs, by design* is itself a deployment-relevant contribution. Industrial AGVs are routinely re-mapped (new workflows, facility expansions, new AGV deployments), and any predictive model that depends on raw map coordinates silently fails the moment the map is regenerated.

### 3.7 Operational anomalies: detection and treatment

Two non-routine event classes occurred during the recording sessions and need explicit handling, both in the analysis and in any deployed system that consumes the model's output.

**(a) Auto-positioning failures with manual repositioning.** When the NNS auto-positioning fails to converge, the AGV stops and is repositioned manually before resuming. The signature in the data: `nns_position_confidence` drops below normal levels (or becomes invalid), `|speed_mps|` is approximately zero for an extended duration, and resumption is marked by a sudden re-anchoring of `x_m, y_m`.

**(b) Motor-overheat stops.** The AGV halts with the motor in a fault state and resumes once the thermal protection clears. The signature in the data: `momentary_current_consumption` patterns and motor-related navigation flags differ from a normal stop; `|speed_mps|` is zero for an extended period; nothing else about the WiFi link or the LiDAR scene needs to be changing during the stop.

A dedicated **Phase 0 task P0.7** (§7.8) systematically detects and quantifies both event classes:

- Iterate through each session and find all stop episodes (`|speed_mps|` continuously below 0.05 m/s for at least 30 s).
- For each stop, classify as **routine** (short stop, normal `nns_position_confidence`, no motor fault flag), **manual reposition** (long stop, low `nns_position_confidence`, post-stop re-anchoring of `x_m, y_m` exceeding the within-confidence band), or **motor-overheat / fault** (long stop, motor-fault flag or characteristic current-consumption pattern, post-stop continuation on the same trajectory).
- Report per-session counts and durations for each category.

**Treatment decision** (made after P0.7 produces the numbers):

- **Routine stops** are kept. They are part of normal operation and contain valid WiFi observations at fixed (x, y); removing them would bias the dataset toward motion.
- **Manual repositioning episodes** are removed. The frame discontinuity at re-anchoring would inject artificial features into the LiDAR-vs-(x, y) relationship, and the AGV-state-during-repositioning is not representative of either driving or normal stopping.
- **Motor-overheat stops** are kept *during* the stop (the WiFi link and LiDAR scene are valid observations of a stationary state) but the immediate post-resumption period is checked for transient effects and removed if the AGV state shows abnormal patterns (e.g., recovery acceleration profiles).

The exact thresholds (stop duration, confidence band, re-anchoring distance) are set after looking at the empirical distributions in P0.7. The decision rule is documented in the paper's §3 alongside the data-cleaning numbers, so the reader can audit what fraction of each session was excluded and why.

A practical side benefit: this analysis produces a directly publishable observation about real industrial AGV deployments — namely, the empirical rate of auto-positioning failures and motor-thermal events on this AGV/facility combination over the recorded sessions. Useful context for any reader deploying similar systems.

---

## 4. Feature obtainability and deployment realism

### 4.1 Three tiers of feature provenance

**Tier 1 — features the AGV trivially has at inference time.**

- All LiDAR-derived features (the AGV runs the LiDAR for safety; the data is on the bus).
- All telemetry (speed, turn rate, load, battery, navigation flags).
- AGV heading and AGV position in its current map.

**Tier 2 — features requiring one-time operator configuration or a short calibration drive.**

- AP coordinates in the AGV's map. WiFi APs are physical infrastructure surveyed during facility setup; storing each AP's coordinates as a one-line config entry is the same kind of metadata any indoor WiFi engineering tool already requires. When unavailable, the per-session log-distance fit (§3.4, P0.2) recovers the position from a single calibration drive. Re-mapping requires only a one-line metadata update per AP, not a full re-collection of training data.

**Tier 3 — features explicitly not used.**

- *Future positions of the AGV.* The model is point-in-time.
- *Facility CAD plans, pre-built occupancy maps, structural priors.* No such inputs are imported.
- *AP-side measurements.* All WiFi metrics come from the AGV's MikroTik client; the system is deployable on any AP.

### 4.2 Front-LiDAR-only setup vs typical industrial AGVs

The experimental AGV in this dataset has a single front-facing 270° safety LiDAR (Leuze RSL 400). The Leuze RSL 400 supports both 0.1° and 0.2° angular resolution; **this dataset was captured at 0.2°**, giving 1 350 active beams per sweep over 270° (the h5 storage allocates 2 700 slots per scan, sized for the 0.1° worst case, so the second half of every scan is buffer-padding zeros — see [docs/p0_analysis/report.md](p0_analysis/report.md) §3 for the empirical confirmation). Production industrial AGVs typically have **two** LiDARs — front and rear — for ISO 3691-4 360° safety coverage. With both LiDARs every bearing relative to the AP heading is observable, and directional features carry physical signal regardless of the AP's relative direction.

The single-LiDAR setup is therefore the *harder case*. Two practical consequences:

(a) The results we report are a *lower bound* on the predictive performance achievable in standard industrial deployments. The discussion section makes this explicit.

(b) We stratify all evaluation metrics by `is_AP_in_FOV` — a boolean derived from `angle_to_AP` and the empirically-determined LiDAR valid-angle range. The headline result is reported separately for in-FOV samples (where the model has direct observation of the AGV→AP propagation path) and out-of-FOV samples (where the model falls back to environment-aggregate features and AP distance):

- **In-FOV**: directional LiDAR features add substantial predictive value. `clutter_toward_AP` reliably encodes LOS blockage.
- **Out-of-FOV**: directional LiDAR features add little; the model leans on `dist_to_AP` and full-sweep aggregates.

In-FOV results are directly representative of dual-LiDAR coverage.

### 4.3 SHAP-by-FOV interaction analysis

The XAI analysis (§5.7) includes a SHAP-by-FOV interaction (XAI-4): for each directional LiDAR feature, plot SHAP value vs feature value separately for in-FOV and out-of-FOV samples. Expected: strong negative slope for `clutter_toward_AP` when the AP is in FOV (more clutter on the LOS path → weaker predicted signal), flat / null slope when the AP is out of FOV. Confirming this signature on held-out folds is direct evidence that the model has learned route-invariant propagation physics.

---

## 5. Research direction

### 5.0 Verified literature gap

A targeted search across IEEE Xplore, ACM, arXiv, MDPI, Springer, and ScienceDirect (2020–2026) confirms the gap. Closest related work clusters and what each does not address: LiDAR for radio prediction (mmWave / surfaces, no WiFi link quality, no SHAP); ML for AGV WiFi link quality (Ohori et al. 2023; Formis & Scanzio 2025 — no environmental geometry, no XAI); XAI for wireless (Masood et al. 2023; Kiouvrekis et al. 2025 — cellular path loss or HAR, not industrial WiFi quality); LiDAR + WiFi sensor fusion (DLoc, EKF fusion — localization targets, not link quality). The intersection is unoccupied.

### 5.1 Working title

*LiDAR-Aware Cross-Route Generalization of WiFi Signal Prediction for Industrial AGVs, with Frame-Independent Features and SHAP-Based Explainability of Geometric Propagation Effects.*

### 5.2 Core thesis

A model trained on ego-frame LiDAR features, per-session-calibrated AP-relative geometry, and frame-independent telemetry — using **no global map coordinates** — predicts WiFi `signal_power` on AGV routes the model has not been trained on, in a facility the model knows from other sessions. TreeSHAP attributions decompose the prediction into physically interpretable contributions from distance, line-of-sight openness, and front-cone obstruction, with directional contributions appropriately gated by whether the AP is in the LiDAR's field of view — quantifying for the first time on real industrial AGV data which environmental geometry features carry which fraction of the explanatory power, in which physical regimes.

### 5.3 Research questions

| # | Question | Experiment |
|---|---|---|
| **RQ1** | Does a LiDAR-aware model generalize to a held-out route, beyond what AP-relative geometry alone provides — without using any global-frame features — and does the LiDAR contribution differ between in-FOV and out-of-FOV regimes? | LORO with all three hold-out combinations (§5.5). B5 vs B1 per fold, broken down by `is_AP_in_FOV`. |
| **RQ2** | How does generalization depend on which session is held out? | All three LORO folds reported with equal status. Per-fold metrics + per-fold spatial-overlap statistic + per-fold cross-frame status. |
| **RQ3** | Are TreeSHAP attributions on the held-out fold physically sensible — do directional LiDAR features push predictions in propagation-physics-required directions when the AP is in FOV, and become appropriately uninformative when out of FOV? | Sign-of-effect analysis on the LORO-trained model, stratified by FOV. |
| **RQ4** | After conditioning on geometry features, is there a residual session effect (interference, AP load, day-specific clutter) that motivates session-aware deployment? | Pooled-with-`session_id` vs pooled-without on the same-map subset (15.03 + 24.03), and on the full three-session pool. SHAP contribution of `session_id` is the residual-effect estimate. |

### 5.4 Feature stack

All features below are computed at LiDAR scan rate (~25 Hz). **No raw `(x, y)`** is included in any cross-session model variant.

**LiDAR (ego-frame), AGV-body mask applied (mask determined by P0.3):**

- *Scalar aggregates*: `mean_dist_mm`, `dist_p90_mm`, `clutter_frac` (valid beams below 4 m), `openness_frac` (valid beams above 7 m), `mean_front_mm` (front cone, |θ| ≤ 30°). All aggregates are computed over the active-beam set with the AGV-body mask (`mask_invalid` from P0.3) applied — buffer-padding slots and beams hitting the AGV chassis are excluded.
- *Sectoral*: the empirical valid angular range from P0.3 (≈ 222° on this dataset, [-110°, +112°]) split into 30° sectors; two features per sector → `mean_dist_sector_i_mm` and `clutter_frac_sector_i`. With the measured FOV this yields **7 sectors / 14 features** (the original proposal-time figure of 6 sectors / 12 features assumed a planned 180° front; Phase 0 expanded the usable sector).

**Telemetry (frame-independent):**

- `speed_mps`, `turn_rate`, `load_long`, `load_mid`, `load_short`, `battery_value`, `momentary_current_consumption`, `nns_state`, `nns_position_confidence`. ~9 features.

**AP-relative (per-session-calibrated):**

- `dist_to_AP_d`, `sin_angle_to_AP_d`, `cos_angle_to_AP_d`, `clutter_frac_toward_AP_d`. 4 features.
- **`is_AP_in_FOV`** — boolean derived from `angle_to_AP` and the empirically-determined LiDAR valid-angle range.

Total: ~33 features (5 scalar + 14 sectoral LiDAR + ~9 telemetry + 5 AP-relative), all frame-independent or per-session-calibrated.

The within-session sanity check (§5.5 sub-protocol) is the one place `(x, y)` may appear — single-frame context, no cross-session generalization claim. Stays out of the LORO model.

### 5.5 Evaluation: leave-one-route-out (LORO), all combinations

Three folds, all reported with equal status — no fold designated as "the headline" a priori. The interpretation of each fold depends on Phase 0 results (frame status from P0.0; spatial overlap from P0.1; same-cell consistency from P0.6) and is summarized in the per-fold metadata column.

| Fold | Train | Test | Per-fold metadata (filled after Phase 0) |
|---|---|---|---|
| **F-A** | 15.03 + 25.02 | 24.03 | Spatial overlap of test with training; cross-frame status (whether 15.03 and 24.03 share Map A; whether 25.02 in training is in Map B). |
| **F-B** | 24.03 + 25.02 | 15.03 | Spatial overlap; cross-frame status. 15.03 covers the largest area of all three sessions, so this fold tests whether a model trained on smaller-coverage sessions can predict on the broader-coverage session. |
| **F-C** | 15.03 + 24.03 | 25.02 | Spatial overlap; cross-frame status. Likely (but to be verified by P0.0) the only fold where training and test span different maps. |

**No fold is privileged in advance.** All three fold metrics are reported in the headline ablation table. A discussion subsection in the paper interprets the result in light of the per-fold metadata: "Fold F-X had ~Y % spatial overlap with training and Z cross-frame status; its RMSE was W". This is the methodologically correct presentation — designating one fold as "the" headline before seeing the Phase 0 metadata would be cherry-picking.

**Reported metrics, stratified.** For each fold: RMSE, MAE, R², bias, bootstrap 95 % CIs (B = 1000), in three forms:

- **Overall** (all rows in the held-out session, post operational-anomaly cleaning).
- **In-FOV** (rows where `is_AP_in_FOV == True`). Directly representative of dual-LiDAR deployment.
- **Out-of-FOV** (rows where `is_AP_in_FOV == False`). Captures the experimental AGV's blind-rear limitation.

The headline summary table is therefore 3 folds × 3 strata × ~6 model variants = 54 cells. Presents cleanly as three side-by-side per-fold tables.

**Cadence matching for the 25.02-test fold.** 25.02 has telemetry at 4.4 Hz vs 31 Hz on the other days. To prevent that fold from underperforming for cadence reasons rather than frame/geometry reasons, downsample 15.03 and 24.03 to 4.4 Hz before training on that fold.

**Within-session sanity check (sub-protocol).** Each session in isolation, train on the first 70 % chronologically, validate on the next 15 %, test on the last 15 %. May include `(x, y)`. For diagnosing model health and for §3 of the paper.

### 5.6 Model variants

| Variant | Features | Purpose |
|---|---|---|
| **B0** | `dist_to_AP` only | Pure log-distance path-loss baseline. The simplest frame-independent prediction. |
| **B1** | B0 + `sin/cos(angle_to_AP)` | Adds angular AP geometry. Tests whether direction-to-AP matters beyond raw distance. |
| **B2** | B0 + LiDAR scalar | Distance + coarse environment. Tests whether LiDAR adds to distance alone. |
| **B3** | B1 + LiDAR scalar | LiDAR added to full AP geometry. |
| **B4** | B3 + LiDAR sectoral | Adds direction-aware LiDAR. |
| **B5** | B4 + telemetry + `clutter_frac_toward_AP` + `is_AP_in_FOV` | Full model. |

Headline ablation comparisons:

- **B5 vs B1** — does LiDAR carry generalizable signal beyond AP geometry?
- **B5 vs B5-without-LiDAR** — LiDAR's *unique* contribution given everything else.
- **B0 vs B1** — does angular geometry to the AP matter, or is this a pure-distance regime?
- **B5(in-FOV) vs B5(out-of-FOV)** — quantifies the dual-LiDAR vs single-LiDAR gap.

XGBoost regressors throughout, target `signal_power` in dBm. Hyperparameters: `max_depth=6`, `eta=0.05`, `n_estimators=2000`, early stopping with patience 100 on a chronological-split validation set (last 10 % of training data by `fh7000_timestamp`).

### 5.7 XAI analysis

On the LORO-trained B5 model evaluated on each held-out fold:

- **XAI-1 (global, per fold).** TreeSHAP beeswarm + feature-group importance (telemetry, LiDAR scalar, LiDAR sectoral, AP-relative). One figure with three panels (one per fold).
- **XAI-2 (sign-of-effect consistency).** For each LiDAR feature and each AP-relative feature, fit a univariate spline through `(feature_value, SHAP_value)` and report the sign of the slope on each fold. A feature whose SHAP slope is sign-consistent across all three folds — *and across map frames where applicable* — is one whose physical interpretation transfers. One table.
- **XAI-3 (spatial map, on the held-out fold).** Bin the test trajectory's `(x, y)` (in *its own* frame) into 0.5 m cells; per cell, plot the dominant feature group by mean |SHAP|. One figure per fold, with the fold's frame clearly labeled.
- **XAI-4 (SHAP × FOV interaction).** For each directional LiDAR feature (`clutter_frac_toward_AP`, per-sector features), plot SHAP value vs feature value separately for in-FOV and out-of-FOV samples. Expected: strong negative slope in-FOV (more obstruction → weaker predicted signal), flat slope out-of-FOV (the feature can't see what it claims to measure). Directional features' physical-soundness check.

### 5.8 Paper structure (up to 9 IEEE pages)

1. **Introduction** (~1.0 pp). Industrial AGV WiFi reliability problem; deployment-relevance of frame-independent prediction; verified gap; contribution statement.
2. **Related work** (~1.0 pp). Three clusters (LiDAR-for-radio, ML-for-AGV-WiFi, XAI-for-wireless), one paragraph each, plus one paragraph distinguishing this work from each cluster.
3. **Dataset, calibration, and frame handling** (~1.25 pp). Dataset overview; joint dataset construction (one paragraph + reference to the time-sync companion report); two-maps structure; per-session AP calibration with user priors and log-distance refinement; operational-anomaly detection and cleaning (counts and durations excluded).
4. **Feature obtainability and dual-LiDAR generalization** (~0.5 pp). Three-tier framing; single→dual LiDAR generalization argument; FOV stratification rationale.
5. **Method** (~1.5 pp). Feature stack, ablation variants, LORO protocol with all three folds, FOV stratification mechanics, XAI procedure (XAI-1 through XAI-4).
6. **Results** (~2.0 pp). Per-session AP calibration table (n_d, P0_d, R²); operational-anomaly cleaning summary; LORO ablation table (3 folds × 3 strata × 6 variants); Δ_LiDAR / Δ_angle / Δ_FOV summary; sign-of-effect consistency table; XAI-1 figure; XAI-4 figure (the cleanest physical-soundness check); selected XAI-3 panel for one fold.
7. **Discussion** (~1.0 pp). Per-fold result interpretation in light of spatial overlap and frame status (avoiding cherry-picked headline framing); in-FOV results as lower bound for dual-LiDAR; residual session effect (RQ4); limitations (3 routes, single AP, single-front-LiDAR experimental setup, single facility).
8. **Conclusion** (~0.25 pp).
9. **References + acknowledgements + IEEE GenAI disclosure** (~0.5 pp).

---

## 6. Backup direction (Project B)

If Phase 0 reveals a blocker (Gate C red, Gate 0 strongly red), pivot to a single-session deep dive on the session that survived Phase 0 best — likely 15.03 (largest area, widest geometry, broadest signal range). Same B0–B5 ablation logic, applied within-session on a chronological + spatial-region split. Loses the cross-route + frame-independence claim but produces a clean within-session XAI case study, with the FOV stratification still applicable as the structural physics check.

---

## 7. Phase 0 — preliminary tests with go/no-go gates

Eight analyses (P0.7 added in Rev5 for operational anomalies). Each output goes into the final paper.

### 7.1 P0.0 — Frame-sharing verification: 15.03 ↔ 24.03

**What.** Pick 5–10 (x, y) cells where both 15.03 and 24.03 spent significant time at low speed. For each cell, compute the per-beam-angle median return distance over all scans falling in that cell, after the AGV-body mask. Compare 15.03 vs 24.03 mean masked scans cell-by-cell — beam-wise RMSE in mm and cosine similarity treating the two scans as `N_ACTIVE_BEAMS`-dim vectors (1 350 on this dataset's 0.2° sensor configuration; the h5 storage width of 2 700 slots includes 1 350 buffer-padding zeros that must be excluded before the comparison).

**Output.** Table + one polar-overlay figure. Per-cell beam-RMSE and similarity.

**Decision gate (Gate 0 — frame-sharing).**
- Median beam-RMSE < 200 mm and cosine similarity > 0.9 → **frame shared**, proceed.
- Median beam-RMSE > 500 mm or similarity < 0.7 → frames differ; methodology revision needed; Project B becomes more realistic.
- Marginal → document, proceed, expect F-A and F-B results slightly worse than ideal.

Same gate is also run on **15.03 ↔ 25.02** and **24.03 ↔ 25.02** for completeness. The user expects 25.02 to differ from both, but this is presented as a hypothesis to test, not as a stipulation. The result populates the cross-frame-status column in the §5.5 LORO fold metadata.

### 7.1b P0.0b — Cross-frame registration (OPTIONAL)

**What.** For any pair of sessions whose frames are confirmed to differ (likely 25.02 vs the others), recover the 2D rigid transformation `(R, t)` aligning one frame onto the other. Two methods; do whichever is faster:

- **ICP** on AGV-body-masked LiDAR scans collected at corresponding parts of the long narrow corridor.
- **Procrustes** alignment of trajectory point-clouds in the corridor segment.

**Output.** Three numbers `(θ, t_x, t_y)` per session-pair. Trajectory overlay before/after. Useful for unified visualization in §3 of the paper. No gate; skip if time-constrained.

### 7.2 P0.1 — Spatial overlap of trajectories

**What.** Bin all three sessions' (x, y) into 0.5 m × 0.5 m cells. Compute pairwise overlap statistics for the same-map session pair (15.03 ↔ 24.03 if Gate 0 passes) directly. For the 25.02 pair-comparisons, note that a meaningful overlap requires P0.0b registration; if available, compute registered-frame overlap, otherwise report 25.02's range separately and qualitatively.

**Output.** Multi-panel figure: trajectory overlay coloured by session; per-cell session count for the same-map subset. Numerical: % of 15.03 ↔ 24.03 cells visited by both; % unique to each.

This is informational and feeds the per-fold metadata in §5.5.

### 7.3 P0.2 — Per-session log-distance path-loss fit (AP calibration)

**What.** Fit `signal_power = P0_d − 10 n_d log10(d) + ε` per session. **Initialize at the user-provided priors** (15.03: (2, 10); 24.03: (2, 10); 25.02: (−2.5, 0.5)). Restrict the search to a ±3 m box around each prior. Use only motion-active rows (`|speed_mps| > 0.05`) and rows surviving the operational-anomaly cleaning from P0.7.

**Output.** Table: session, prior (x_AP, y_AP), fitted (x_AP, y_AP), displacement (m), n_d, P0_d, R². Plus 3 residual-map figures.

**Cross-session compare frame-independent parameters (n_d, P0_d).** Disagreement is RQ4 ammo.

**Decision gate (Gate A — LiDAR headroom).**
- `R² > 0.85` everywhere → distance dominates, small Δ_LiDAR expected.
- `R² ∈ [0.4, 0.85]` → moderate fit, structured residuals, healthy headroom.
- `R² < 0.4` → bad fit (LOS blockage); LiDAR has *more* to do. Use prior position; proceed.
- Fit-vs-prior disagrees by > 3 m on any session → use prior; document as evidence of multi-modal loss surface.
- `n_d` or `P0_d` disagree across sessions by > 2× within-session bootstrap CI → flag in §6 of the paper as evidence of non-trivial residual session effect.

### 7.4 P0.3 — AGV-body LiDAR mask characterization

**What.** Per-beam-angle histograms over a representative sample of active beams (the 1 350 active slots; the trailing 1 350 buffer-padding slots are flagged as `mask_pad` separately, since they are storage artefacts not angular blockages). Two further masks restricted to the active range: `mask_zero[i]` if `frac_zero(i) > 0.95`; `mask_body[i]` if `median(i) < 200 mm` AND `std(i) < 30 mm`. Combined `mask_invalid = mask_pad | mask_zero | mask_body` + the resulting *empirical valid angular range* (used to define `is_AP_in_FOV`). The original brief assumed a stationary baseline scan; in practice a motion-active sample (|speed| > 0.1 m/s) was used because nearby walls in a stationary window confound `mask_body` with environmental clutter.

**Output.** Polar plot. Number: fraction of beams marked invalid. **Critical secondary output: the angular range of valid beams**, used to define `is_AP_in_FOV` for §4.2 and §5.5.

**Decision gate (Gate B — sectoral feasibility).**
- Mask cleanly separates contiguous "valid" sector → use sectoral features and define `is_AP_in_FOV` accordingly.
- Mask fragmented → fall back to scalar aggregates only; document.

### 7.5 P0.4 — LiDAR ↔ residual univariate correlation (the project gate)

**What.** Path-loss residuals from P0.2. Spearman ρ of each LiDAR scalar feature against the residuals, per session. **Stratify by `is_AP_in_FOV` from P0.3.**

**Output.** 3 × 5 × 3 table (3 sessions × 5 headline scalar LiDAR features × 3 strata: overall, in-FOV, out-of-FOV). Headline: max |ρ| in the in-FOV stratum.

**Decision gate (Gate C — project go/no-go).**
- max |ρ| > 0.25 in-FOV on ≥ 2 sessions → **green**.
- max |ρ| ∈ [0.15, 0.25] → marginal; proceed with Project B as backup.
- max |ρ| < 0.15 everywhere → **red**; pivot to Project B.

### 7.6 P0.5 — Spatial autocorrelation of `signal_power`

**What.** Per-session empirical semivariogram. Read off range.

**Output.** 3-curve figure + ranges in meters. Informational; sizes the optional spatial leave-region-out tile.

### 7.7 P0.6 — Cross-session same-cell consistency (15.03 ↔ 24.03)

**What.** For cells visited by both 15.03 and 24.03 (P0.1, conditional on Gate 0 pass), compute Δ = mean_15.03 − mean_24.03 of `signal_power`.

**Output.** Histogram + median |Δ|.

**Decision gate (Gate D — framing strength).**
- median |Δ| < 4 dB → strong support for time-stable-field claim.
- 4–7 dB → mixed; soften claim.
- > 7 dB → genuine non-stationarity; consider Project B.

### 7.8 P0.7 — Operational anomaly detection (NEW in Rev5)

**What.** Detect and classify stop episodes per session:

- **Stop detection.** `|speed_mps| < 0.05` continuously for at least 30 s.
- **Routine vs anomalous.** A stop is anomalous if (a) `nns_position_confidence` drops materially below the session median during the stop, OR (b) post-stop re-anchoring of `(x_m, y_m)` exceeds the within-confidence band, OR (c) motor-related fault flags / abnormal `momentary_current_consumption` patterns are present.
- **Sub-classification of anomalous stops.** Manual repositioning (signature: confidence drop + re-anchoring) vs motor-overheat (signature: motor fault flag / current pattern, no re-anchoring).

**Output.** Per-session table: count and total duration of routine stops, manual-reposition events, motor-overheat events. One example timeline plot per anomaly type showing the relevant signals (`speed_mps`, `nns_position_confidence`, `momentary_current_consumption`, `x_m`, `y_m`) around the event. Decision-rule documentation: which rows are kept, which are removed, and why.

**Why it matters.** Three reasons. (1) Manual-repositioning rows would otherwise inject artificial features into the LiDAR-vs-(x, y) relationship. (2) The empirical event rates are themselves a publishable observation about real industrial AGV deployments. (3) Phase 1 needs to know exactly which rows are valid before training begins.

**Decision rule (informational, not gate).**
- Manual-reposition rows: removed.
- Motor-overheat stops: stop period kept (valid stationary observations); resumption transient (first 10 s post-resume) checked and removed if abnormal.
- Routine stops: kept.

The total fraction of rows removed is reported per session and added to the dataset summary in the paper.

### 7.9 Phase 0 deliverables

A short Phase 0 report containing:

- The figures from P0.0–P0.7.
- Summary tables (per-session AP calibration; same-cell median |Δ| for the same-map subset; LiDAR-mask invalid fraction and valid angular range; operational-anomaly counts and durations).
- Two-line conclusion on each gate: **Gate 0**, **Gate A**, **Gate B**, **Gate C**, **Gate D**. Explicit green / yellow / red.
- Frozen feature-extraction script for Phase 1, validated against a 1000-row sample from each session.

If Gates C and 0 are at least yellow, proceed to Phase 1. If Gate C is red, switch to Project B. If Gate 0 is red, the methodology needs another revision before Phase 1.

---

## 8. Time-sync companion paper — defer

Cite the time-sync report in §3 as "applied per-day correction with σ_rmse ≤ 0.25 s, end-to-end validation passing all 12 checks". The full per-day calibration + two-component hardware interpretation is a separate publishable contribution; target a sensor-fusion / multi-modal robotics venue separately.

---

## 9. Clarifications closed and remaining open

**Closed by Revisions 2–5** (per conversation):

- Same physical AP / antenna / firmware on all three days. ✓
- Same workspace / facility, three different routes per day, partial spatial overlap between routes (not disjoint regions). ✓
- People and movable obstacles vary day-to-day but not drastically. ✓
- Within-session NNS drift is small and continuously corrected; non-routine cases (manual reposition, motor overheat) handled in P0.7. ✓
- 15.03 and 24.03 plausibly share Map A; 25.02 uses Map B (verify in P0.0). ✓
- AP coordinates: user priors provided (15.03: (2,10); 24.03: ≈ same as 15.03 if Gate 0 passes; 25.02: (−2.5, 0.5)); refined per-session by log-distance fit. ✓
- Single-front-LiDAR experimental setup; production AGVs typically dual front+rear. ✓
- Paper structure: up to 9 IEEE pages. ✓

**Still open**:

1. Stationary LiDAR scan recording for P0.3 mask (would improve Gate B and the FOV definition).
2. Per-run metadata (time of day, other equipment, observer notes).
3. Per-session AP-side log (MikroTik connected-clients count, AP CPU) — would let RQ4 directly attribute residual session effect to AP load.
4. IEEE GenAI disclosure plan — one paragraph in acknowledgements; resolve before final writing pass.

---

## 10. Phase structure

| Phase | Tasks |
|---|---|
| **Phase 0** | P0.0 (frame check), P0.0b (registration, optional), P0.1 (spatial overlap), P0.2 (AP calibration with priors), P0.3 (LiDAR mask + FOV definition), P0.4 (LiDAR↔residual correlation), P0.5 (semivariogram), P0.6 (same-cell consistency), P0.7 (operational anomaly detection). Five gate decisions: Gate 0, A, B, C, D. End deliverable: the frozen feature-extraction script and a short report. |
| **Phase 1 part 1** | Feature extraction across all sessions (using mask from P0.3, AP coords from P0.2, FOV definition from P0.3, anomaly mask from P0.7). All LORO ablation models (3 folds × 6 variants = 18 fits). TreeSHAP on the 3 LORO B5 models. |
| **Phase 1 part 2** | Generate all figures and tables: LORO ablation table (3 folds × 3 strata × 6 variants), Δ_LiDAR / Δ_angle / Δ_FOV summary, sign-of-effect consistency table, XAI-1, XAI-4 (headline), XAI-3 (one fold), XAI-2 table. |
| **Writing** | Paper end-to-end in IEEEtran up-to-9 pp template. IEEE GenAI disclosure compliance. Submission. |

---

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Gate 0 red** (15.03 and 24.03 do not share a frame) | Low–medium | Methodology revision: register all sessions before Phase 1, or LORO with all three sessions treated as separate frames. Realistically: Project B becomes the better path. |
| **Gate C red** (LiDAR ↔ residual ρ < 0.15 in-FOV everywhere) | Low–medium | Pivot to Project B. |
| **Gate D yellow/red** (same-cell std > 4 dB on 15.03 ↔ 24.03) | Medium | Soften time-stable-field claim; present generalization as graded. |
| **Per-session AP fit fails on a session** (large displacement from prior) | Low | Use prior position; document as evidence of strong LOS blockage / multi-modal loss surface (LiDAR has more to do). |
| **One LORO fold underperforms substantially** | Medium | Expected if that fold has lowest spatial overlap with training, or differs in frame. Frame-independence preserved by design; large failures are most likely from physical environment differences too far from training data. Report honestly with per-fold metadata explaining the result. |
| **25.02-test fold cadence-driven failure** | Medium | Mitigated by 4.4 Hz downsampling of training data on that fold. |
| **In-FOV vs out-of-FOV gap is small** | Medium | If LiDAR features show little improvement in-FOV over out-of-FOV → either directional features uninformative even when geometrically meaningful, or model is averaging across regimes. Investigate via XAI-4: SHAP slopes should differ visibly. If they don't, the FOV stratification is itself a finding. |
| **`n_d` or `P0_d` differ across sessions** (RQ4 lead) | Medium | Investigate magnitude. Small differences → real residual effect, paper-worthy. Large → AP configuration changed; reframe. |
| **High operational-anomaly rate on a session** | Low–medium | If P0.7 reveals that more than ~10 % of one session's recording time is in manual-reposition / motor-overheat states, the effective sample size for that session shrinks. Document explicitly; if the residual session is too small for stable LORO results, consider dropping from training (but keeping in test) or vice versa, with the choice driven by which usage best preserves statistical power on the headline ablation. |
| **LiDAR helps B4 vs B0 but not B5 vs B5-no-LiDAR** | Medium | Still publishable: "explicit AP geometry suffices when known; LiDAR is the practical alternative when AP coordinates are unlogged." |

---

## 12. Recommendation

1. **Adopt Revision 5 as the working plan.** It correctly frames the LORO experiment as a symmetric all-folds protocol rather than designating a headline a priori; it incorporates user-provided AP priors as initialization for per-session calibration; it adds explicit detection and treatment of operational anomalies; and it removes overclaiming about "unseen routes" implying "unseen terrain".
2. **Run Phase 0 in full before any Phase 1 modeling.** Eight analyses, five gate decisions, all outputs reusable in the final paper.
3. **AP placement: user priors initialize the per-session log-distance fit; restrict the search to a ±3 m box.** Their disagreement with the fitted positions is a paper-worthy methodological note in §3.
4. **Report all three LORO folds with equal status, accompanied by per-fold metadata** (spatial overlap, frame status). Let the data tell us which fold is hardest, rather than designating the headline before seeing the Phase 0 results.
5. **In-FOV stratum results are the lower-bound estimate for dual-LiDAR deployments.** Discussion section makes this explicit.
6. **Project B remains a real fallback.** If Gate C is red, switch.

---

**End of Revision 5.**

*Ready for Phase 0. Suggested next deliverable: the Python module that runs P0.0 through P0.7 end-to-end against `joint_coverage.parquet`, including the AGV-body mask + valid-angular-range computation that defines `is_AP_in_FOV`, the operational-anomaly detector, and the user-prior-initialized per-session log-distance fit.*
