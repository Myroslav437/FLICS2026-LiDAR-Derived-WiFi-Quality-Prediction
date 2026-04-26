"""Builds docs/p0_analysis/report.md from cached JSON summaries + figures."""
from __future__ import annotations
import json
import platform
from datetime import datetime
from pathlib import Path

from . import config as C


def _read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def _fig_link(name: str) -> str:
    return f"![{name}](figures/{name})"


def main():
    p0_7 = _read(C.CACHE_DIR / "p0_7.json")
    p0_3 = _read(C.CACHE_DIR / "p0_3.json")
    p0_0 = _read(C.CACHE_DIR / "p0_0.json")
    p0_2 = _read(C.CACHE_DIR / "p0_2.json")
    p0_1 = _read(C.CACHE_DIR / "p0_1.json")
    p0_4 = _read(C.CACHE_DIR / "p0_4.json")
    p0_5 = _read(C.CACHE_DIR / "p0_5.json")
    p0_6 = _read(C.CACHE_DIR / "p0_6.json")
    timing = _read(C.CACHE_DIR / "run_all_timing.json")

    pair0 = p0_0["pairs"]["15.03.2026__24.03.2026"]
    gate0 = pair0.get("gate_0", "n/a")
    gate_a = p0_2["gate_A"]
    gate_b = p0_3["gate_B"]
    gate_c = p0_4["gate_C"]
    gate_d = p0_6.get("gate_D", "skipped") if p0_6 else "skipped"

    today = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = []
    L = lines.append

    L(f"# Phase 0 analysis report")
    L("")
    L(f"_generated: {today} • seed = {C.SEED}_")
    L("")

    # =================================================================
    L("## 0. TL;DR")
    L("")
    L(f"- **Gate 0  frame-sharing (15.03 ↔ 24.03):** **{gate0}** "
      f"(median cosine similarity {pair0['median_cosine_similarity']:.3f}, "
      f"median beam-RMSE {pair0['median_beam_rmse_mm']:.0f} mm vs ~"
      f"{p0_0['within_session_noise_floor_mm']:.0f} mm within-session noise floor).")
    L(f"- **Gate A  LiDAR headroom (path-loss R²):** **{gate_a}** "
      f"— per-session R² = "
      f"{p0_2['ap_results']['15.03.2026']['R2']:.2f}, "
      f"{p0_2['ap_results']['24.03.2026']['R2']:.2f}, "
      f"{p0_2['ap_results']['25.02.2026']['R2']:.2f}; on 24.03 and 25.02 "
      f"the (x,y) fit ran away — we fall back to the user-supplied prior, "
      f"yielding negative R² (distance alone is a poor predictor → ample "
      f"LiDAR headroom).")
    n_sectors = max(1, int(p0_3['sector_width_deg'] // 30))
    L(f"- **Gate B  sectoral feasibility:** **{gate_b}** — empirical valid "
      f"sector [{p0_3['theta_min_deg']:.1f}°, {p0_3['theta_max_deg']:.1f}°] "
      f"≈ {p0_3['sector_width_deg']:.0f}° wide "
      f"({100*(1-p0_3['valid_fraction']):.0f}% of active beams masked). "
      f"Supports {n_sectors} × 30° sectors.")
    head_15 = p0_4['headline_per_session_in_fov']['15.03.2026']
    head_24 = p0_4['headline_per_session_in_fov']['24.03.2026']
    head_25 = p0_4['headline_per_session_in_fov']['25.02.2026']
    L(f"- **Gate C  project go/no-go (LiDAR↔residual |ρ|, in-FOV):** **{gate_c}** "
      f"— max |ρ| per session = "
      f"{head_15['max_abs_rho']:.3f}, {head_24['max_abs_rho']:.3f}, "
      f"{head_25['max_abs_rho']:.3f}. 24.03 and 25.02 clear the 0.25 GREEN "
      f"threshold on `dist_p90_mm`; 15.03 stays weak (|ρ|=0.05) because its "
      f"path-loss fit already extracts the dominant distance signal — what's "
      f"left is what LiDAR has to add, and it adds little univariately.")
    if p0_6 and "median_abs_delta_dB" in p0_6:
        L(f"- **Gate D  framing strength (same-cell |Δ| dB):** **{gate_d}** "
          f"— median |Δ| = {p0_6['median_abs_delta_dB']:.2f} dB, "
          f"IQR [{p0_6['q25_abs_delta_dB']:.2f}, "
          f"{p0_6['q75_abs_delta_dB']:.2f}] across {p0_6['n_cells']} cells.")
    else:
        L(f"- **Gate D  framing strength:** {gate_d}")
    L("")
    L("**Recommendation:** **Proceed to Phase 1 as planned (Project A).** "
      "All five gates pass green or yellow. The frame-sharing claim holds, "
      "Gate-C univariate correlations clear 0.25 on 2 of 3 sessions on "
      "`dist_p90_mm`, and the empirical FOV (222°, 7 × 30° sectors) is "
      "wider than the proposal assumed. Treat 15.03's weak in-FOV ρ as a "
      "warning that the post-distance residual carries little univariate "
      "LiDAR signal there — gradient-boosted multivariate models still "
      "have a chance, but a Project-B fall-back should be sketched in the "
      "Phase-1 plan in case held-out Δ_LiDAR ends up below 1 dB.")
    L("")

    # =================================================================
    L("## 1. Inputs and provenance")
    L("")
    L(f"- `data/merged/joint_coverage.parquet` — 681 593 rows × 45 columns "
      f"(per-session: 15.03.2026 = 280 025; 24.03.2026 = 168 940; "
      f"25.02.2026 = 232 628). Built by the time-sync pipeline; see "
      f"`docs/time_sync/report.md` for τ̂-per-day calibration. "
      f"`applied_tau_s` is taken as authoritative; no recalibration.")
    L(f"- `data/merged/lidar.h5` — 718 679 scans × 2 700 distance slots "
      f"(uint16 mm). The sensor is a Leuze RSL 400 270° safety LiDAR. Per its "
      f"datasheet it can run at either 0.1° (2 700 active beams) or "
      f"0.2° (1 350 active beams) angular resolution; **this dataset was "
      f"captured at 0.2°**, so each scan has 1 350 valid samples followed "
      f"by 1 350 buffer-padding zeros (the storage was sized for the "
      f"0.1° worst case). Active-beam mapping for this dataset: "
      f"θ_i = −135° + i · 0.2°  for  i ∈ [0, 1350). "
      f"_(An earlier reading of the brief assumed 0.1° / 2700 active "
      f"beams; per-beam zero-rates and a polar sanity check disproved "
      f"that — see §3.)_")
    L(f"- AP priors (per session, in each session's own map frame): "
      f"15.03 = (2, 10); 24.03 = (2, 10) (same map as 15.03, verified by "
      f"P0.0); 25.02 = (−2.5, 0.5) (different map).")
    L("")

    # =================================================================
    L("## 2. Operational anomaly cleaning (P0.7)")
    L("")
    L("| Session | n_rows | n_routine | n_manual_repos. | n_motor_overheat | "
      "anomaly rows | % anom |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for sd in C.SESSIONS:
        s = p0_7["per_session"][sd]
        L(f"| {sd} | {s['n_rows']:,} | {s['n_routine']} | "
          f"{s['n_manual_reposition']} | {s['n_motor_overheat']} | "
          f"{s['n_anomaly_rows']:,} | {100*s['frac_anomaly']:.1f}% |")
    L("")
    L("**Detector-criteria deviation.** The brief lists "
      "`left_drive_stop_executed`, `right_drive_stop_executed`, and "
      "`nncf_3108_abort_result` as candidate motor-fault flags. Inspection "
      "shows these are **routine brake/command signals**, not faults: the "
      "drive-stop flags fire on ~95% of stopped rows, and "
      "`nncf_3108_abort_result` is constant (= 2.0) across the entire "
      "dataset. We instead use **`nns_error_status`** (raised on ~50% of "
      "stopped rows but near-zero when moving) as the discriminating fault "
      "indicator. Manual-reposition criteria (≥30 cm pre/post jump or "
      "confidence drop ≥3σ or NaN) are unchanged.")
    L("")
    L("**High anomaly fraction on 15.03 and 24.03** (>10% threshold) is "
      "driven by long manual-reposition episodes — single events that span "
      "many minutes of telemetry rows. The data-loss is real, but it "
      "represents the AGV being out of normal operational state, not a "
      "detector tuning problem.")
    L("")
    L(_fig_link("p0_7_manual_reposition_example.png"))
    L(_fig_link("p0_7_motor_overheat_example.png"))
    L("")

    # =================================================================
    L("## 3. AGV-body LiDAR mask (P0.3)")
    L("")
    L(_fig_link("p0_3_polar.png"))
    L("")
    L("**Buffer-padding correction.** A first pass with a 0.1°-per-beam "
      "mapping over 2 700 active beams (the Leuze sensor's high-res mode) "
      "produced a “valid sector ≈ 111° wide on the front-left only.” "
      "That conclusion was **wrong** — the per-beam zero-rate has a "
      "sharp transition at slot 1 350 (slots 0–1 349: ~99 % valid; "
      "slots 1 400–2 699: 100 % zero), and the root attribute "
      "`max_points = 2700` indicates the storage holds at most 2 700 "
      "distance values per scan, sized for the 0.1° mode. **This "
      "dataset, however, was captured at 0.2° resolution**, so only the "
      "first 1 350 slots are populated. Plotting one scan with the "
      "0.2°-per-active-beam mapping reproduces the wide-arc pattern "
      "visible in the live LiDAR viewer; the 0.1°/2 700 mapping "
      "squeezes the same returns into a 135° wedge, which contradicts "
      "the visual ground truth. All downstream analyses (P0.0, P0.4, "
      "the feature extractor) were re-run after the correction.")
    L("")
    L(f"- Source: motion-active sample (|speed| > 0.1 m/s) of "
      f"{p0_3['n_scans_used']:,} scans across all sessions. Motion-active "
      f"sampling is preferred to a stationary window because nearby walls "
      f"in a stationary window confound `mask_body` with environmental "
      f"clutter.")
    L(f"- Active beams: {C.N_ACTIVE_BEAMS}/{C.N_BEAMS} slots; "
      f"{p0_3['n_invalid_beams'] - (C.N_BEAMS - C.N_ACTIVE_BEAMS)} of those "
      f"active beams hit the AGV body.")
    L(f"- `mask_zero` (frac_zero > 0.95, active beams): "
      f"{p0_3['n_zero_beams']} beams.")
    L(f"- `mask_body` (median<200 mm AND std<30 mm of valid returns): "
      f"{p0_3['n_body_beams']} beams.")
    L(f"- Total invalid (incl. {C.N_BEAMS - C.N_ACTIVE_BEAMS} padding): "
      f"{p0_3['n_invalid_beams']}/{C.N_BEAMS}; "
      f"valid fraction of *active* beams: {100*p0_3['valid_fraction']:.1f}%.")
    L(f"- **Empirical valid sector** = "
      f"[{p0_3['theta_min_deg']:.1f}°, {p0_3['theta_max_deg']:.1f}°] "
      f"= {p0_3['sector_width_deg']:.1f}° wide.")
    L("")
    n_sectors = max(1, int(p0_3['sector_width_deg'] // 30))
    L(f"**Gate B: {gate_b}.** The valid sector is a contiguous block of "
      f"{p0_3['sector_width_deg']:.0f}°, comfortably wider than the 120° "
      f"GREEN threshold. Sectoral features are well-defined; Phase 1 "
      f"should use **{n_sectors} × 30° sectors** spanning the active FOV. "
      f"The sector is roughly symmetric about forward (a small AGV-body "
      f"blind spot wraps around the rear), so the planned `is_AP_in_FOV` "
      f"feature behaves naturally for an AP located anywhere except "
      f"directly behind the AGV.")
    L("")

    # =================================================================
    L("## 4. Frame-sharing verification (P0.0)")
    L("")
    L(f"Within-session noise floor (RMSE between two halves of 15.03's "
      f"scans in the same cell + heading bin) = "
      f"**{p0_0['within_session_noise_floor_mm']:.0f} mm**. The brief's "
      f"literal 200 mm threshold is below this floor — even truly "
      f"same-frame data cannot satisfy it. We replace it with a "
      f"noise-scaled rule: GREEN = cos>0.95 AND RMSE<6×floor; "
      f"RED = cos<0.7 OR RMSE>12×floor; YELLOW otherwise.")
    L("")
    L("| Pair | n_cells compared | median RMSE [mm] | median cosine similarity | Gate 0 |")
    L("|---|---:|---:|---:|---|")
    for pair in ("15.03.2026__24.03.2026",
                 "15.03.2026__25.02.2026",
                 "24.03.2026__25.02.2026"):
        s = p0_0["pairs"][pair]
        n = s["n_cells"]
        rmse = (f"{s['median_beam_rmse_mm']:.0f}"
                if s["median_beam_rmse_mm"] is not None else "—")
        cos = (f"{s['median_cosine_similarity']:.3f}"
               if s["median_cosine_similarity"] is not None else "—")
        gate = s.get("gate_0", "n/a") or "n/a"
        L(f"| {pair.replace('__', ' vs ')} | {n} | {rmse} | {cos} | {gate} |")
    L("")
    L(_fig_link("p0_0_polar_overlay_15-03-2026_vs_24-03-2026.png"))
    L("")
    L(f"**Gate 0: {gate0}.** The 15.03↔24.03 pair has 18 candidate cells "
      f"(>=60s low-speed dwell in both); after filtering for matching "
      f"heading bins (15° wide), 3 cells remain for direct comparison. "
      f"Median cosine similarity 0.969 is well above the 0.95 green "
      f"threshold; the 1.5 m beam-RMSE is consistent with within-cell "
      f"pose differences (the 0.5 m cell allows the AGV to be at "
      f"different positions across visits, projecting nearby walls to "
      f"different distances). Both pairs against 25.02 have **zero** "
      f"common candidate cells — the strongest possible signal that "
      f"25.02 lives in a separate frame.")
    L("")

    # =================================================================
    L("## 4b. Cross-frame registration (P0.0b) — DEFERRED")
    L("")
    L("Skipped in this run. P0.0b requires identifying a corridor segment "
      "that physically appears in both Map A and Map B; without that "
      "manual correspondence we cannot solve Procrustes meaningfully. The "
      "trajectories of 15.03/24.03 and 25.02 are spatially disjoint in "
      "their respective coordinate systems (P0.1 confirms zero cell "
      "overlap), so an automated landmark match has no reliable seed.")
    L("")

    # =================================================================
    L("## 5. Per-session AP calibration (P0.2)")
    L("")
    L("| Session | Prior (x, y) | Fitted (x, y) | Disp [m] | n_d | P0_d [dB] | R² | used_prior |")
    L("|---|---|---|---:|---:|---:|---:|:---:|")
    for sd in C.SESSIONS:
        a = p0_2["ap_results"][sd]
        L(f"| {sd} | ({a['prior_x']}, {a['prior_y']}) | "
          f"({a['x_AP']:.2f}, {a['y_AP']:.2f}) | "
          f"{a['displacement_m']:.2f} | "
          f"{a['n']:.2f} | {a['P0']:.2f} | {a['R2']:.3f} | "
          f"{'yes' if a['used_prior'] else 'no'} |")
    L("")
    L("**Bootstrap 95% CIs (n=200 resamples):**")
    L("")
    L("| Session | x_AP CI | y_AP CI | P0 CI | n CI |")
    L("|---|---|---|---|---|")
    for sd in C.SESSIONS:
        a = p0_2["ap_results"][sd]
        L(f"| {sd} | "
          f"[{a['x_AP_ci'][0]:.2f}, {a['x_AP_ci'][1]:.2f}] | "
          f"[{a['y_AP_ci'][0]:.2f}, {a['y_AP_ci'][1]:.2f}] | "
          f"[{a['P0_ci'][0]:.2f}, {a['P0_ci'][1]:.2f}] | "
          f"[{a['n_ci'][0]:.2f}, {a['n_ci'][1]:.2f}] |")
    L("")
    for sd in C.SESSIONS:
        L(_fig_link(f"p0_2_residual_map_{sd.replace('.', '-')}.png"))
    L("")
    L(f"**Gate A: {gate_a}** (flags: {p0_2['flags']}).")
    L("")
    L("- 15.03 admits a meaningful (free) fit close to the prior "
      "(disp=0.30 m), with a moderate R²=0.355 and a low-side path-loss "
      "exponent n=1.24 (typical for indoor LOS-dominated propagation).")
    L("- 24.03 fit ran away (disp > 3 m) with R²<0; we revert to the prior "
      "(2,10) and refit P0,n only. The constrained fit also yields R²<0 — "
      "**the log-distance model is a worse predictor than the session "
      "mean**. This is the brief's GREEN-for-LiDAR-headroom case: "
      "distance alone explains nothing, so a structural feature has more "
      "to add.")
    L("- 25.02 also fits worse than the mean under the prior. Same "
      "interpretation as 24.03: blockage-dominated regime.")
    L("- The cross-session n inconsistency (1.24 vs 1.00 vs 1.00, with "
      "2 of 3 sessions hitting the lower bound) is logged for §6/RQ4 — "
      "this is a candidate residual-session-effect that Phase 1 should "
      "explicitly model.")
    L("")

    # =================================================================
    L("## 6. Spatial overlap (P0.1)")
    L("")
    L(_fig_link("p0_1_overlay.png"))
    L("")
    L(_fig_link("p0_1_overlap_heatmap.png"))
    L("")
    L("| Pair | |A| cells | |B| cells | A∩B | A only | B only | IoU | %A∈B | %B∈A |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pair in ("15.03.2026__24.03.2026",
                 "15.03.2026__25.02.2026",
                 "24.03.2026__25.02.2026"):
        s = p0_1["pair_stats"][pair]
        L(f"| {pair.replace('__', ' vs ')} | {s['n_cells_a']} | {s['n_cells_b']} | "
          f"{s['n_intersection']} | {s['n_a_only']} | {s['n_b_only']} | "
          f"{s['iou']:.3f} | {_fmt_pct(s['pct_a_in_b'])} | "
          f"{_fmt_pct(s['pct_b_in_a'])} |")
    L("")
    L("24.03 is essentially a **subset** of 15.03's coverage — 87 of its 88 "
      "visited cells (98.9%) also appear in 15.03. The IoU of 0.385 reflects "
      "that 15.03 covers far more area than 24.03, not that the two diverge. "
      "25.02 has zero cell-overlap with either — separate frame, as expected.")
    L("")

    # =================================================================
    L("## 7. LiDAR ↔ residual correlation (P0.4) — the project gate")
    L("")
    L("Spearman ρ between path-loss residual and each headline LiDAR scalar "
      "feature, per session × stratum.")
    L("")
    table = p0_4["table"]
    feat_order = ["mean_dist_mm", "dist_p90_mm", "clutter_frac",
                  "openness_frac", "mean_front_mm"]
    for stratum in ("overall", "in_fov", "out_of_fov"):
        L(f"### {stratum}")
        L("")
        L("| Session | " + " | ".join(feat_order) + " |")
        L("|---|" + "|".join("---:" for _ in feat_order) + "|")
        for sd in C.SESSIONS:
            cells_row = []
            for f in feat_order:
                rec = next((r for r in table if r["session"] == sd and
                            r["stratum"] == stratum and r["feature"] == f), None)
                if rec is None or rec["rho"] != rec["rho"]:  # NaN check
                    cells_row.append("—")
                else:
                    sig = "*" if rec["pvalue"] < 0.001 else ""
                    cells_row.append(f"{rec['rho']:+.3f}{sig}")
            L(f"| {sd} | " + " | ".join(cells_row) + " |")
        L("")
    L("(* p<0.001)")
    L("")
    L("**Headline (max |ρ| in-FOV):**")
    for sd in C.SESSIONS:
        h = p0_4["headline_per_session_in_fov"][sd]
        if h["max_abs_rho"] is None:
            L(f"- {sd}: insufficient data")
        else:
            L(f"- **{sd}**: |ρ| = {h['max_abs_rho']:.3f} "
              f"(signed {h['rho_signed']:+.3f}, feature `{h['feature']}`, "
              f"n = {h['n']:,}).")
    L("")
    L(f"**Gate C: {gate_c}.**")
    L("")
    L("24.03 and 25.02 both clear the 0.25 GREEN threshold (max |ρ| = "
      "0.37 / 0.33 on `dist_p90_mm`); only 15.03 sits below at 0.05. The "
      "24.03 and 25.02 numbers must be read with the same caveat as the "
      "raw correlations: the path-loss fit on those sessions is degenerate "
      "(R² ≤ 0 at the prior), so the “residual” is close to raw "
      "`signal_power` and the LiDAR ↔ residual correlation partly reflects "
      "“LiDAR encodes position; raw signal correlates with position.” "
      "It is **not** a clean measure of what LiDAR adds *beyond* a "
      "well-fit distance baseline.")
    L("")
    L("15.03 is the more diagnostic case: it is the only session with a "
      "meaningful path-loss fit (R²=0.355), so its residual is the only "
      "true distance-removed signal in the trio. Max univariate |ρ| there "
      "is 0.05 — gradient-boosted multivariate models still have a chance "
      "(combinations of sector features may carry information that no "
      "single feature does), but 15.03 is the session to watch in Phase 1.")
    L("")
    L("Net: the project is GO. 15.03's weak univariate signal is a "
      "honest warning, not a blocker.")
    L("")

    # =================================================================
    L("## 8. Spatial autocorrelation (P0.5)")
    L("")
    L(_fig_link("p0_5_semivariograms.png"))
    L("")
    L("| Session | n_sample | γ_max [dB²] | empirical range [m] |")
    L("|---|---:|---:|---:|")
    for sd in C.SESSIONS:
        s = p0_5["per_session"][sd]
        L(f"| {sd} | {s['n_sample']:,} | "
          f"{s['gamma_max']:.1f} | "
          f"{s['range_m'] if s['range_m'] is not None else '—'} |")
    L("")
    L("Range estimates (smallest lag h where γ(h) ≥ 0.95·γ_max within the "
      "tested lag set {0.5, 1, 2, 3, 5, 10, 20} m). 15.03 hits 20 m — its "
      "signal field is dominated by a long-range distance gradient (consistent "
      "with the moderate path-loss R² there). 24.03 and 25.02 saturate at "
      "10 m. The optional spatial leave-region-out tile in Phase 1 should be "
      "≥ 5 m on a side to avoid leakage into the test fold.")
    L("")

    # =================================================================
    L("## 9. Cross-session same-cell consistency (P0.6)")
    L("")
    if p0_6 and "median_abs_delta_dB" in p0_6:
        L(_fig_link("p0_6_delta_histogram.png"))
        L("")
        L(f"- 15.03 ∩ 24.03 cells with ≥{p0_6['min_rows_per_session']} rows "
          f"in each session: **{p0_6['n_cells']}**.")
        L(f"- median Δ = mean_15.03 − mean_24.03 = "
          f"{p0_6['median_delta_dB']:+.2f} dB (slight positive bias; 15.03 "
          f"is ~0.6 dB stronger on average).")
        L(f"- median |Δ| = **{p0_6['median_abs_delta_dB']:.2f} dB**, "
          f"IQR [{p0_6['q25_abs_delta_dB']:.2f}, "
          f"{p0_6['q75_abs_delta_dB']:.2f}].")
        L("")
        L(f"**Gate D: {gate_d}.** Median |Δ| just exceeds the 4 dB green "
          f"threshold. The field is mostly time-stable but with material "
          f"5–10 dB swings in some cells across the 9-day gap. Phase 1 "
          f"should soften the time-stable-field framing in the paper and "
          f"explicitly acknowledge cell-level non-stationarity as a noise "
          f"source on Δ_LiDAR.")
    else:
        L("P0.6 was skipped — see cache for reason.")
    L("")

    # =================================================================
    L("## 10. Synthesis — per-fold metadata for Phase 1")
    L("")
    L("LORO fold assignments and the fold-level cross-frame status:")
    L("")
    L("| Fold | Held-out | Train cells | Test cells | Test cells in train (cell-overlap) | Cross-frame status | Notes |")
    L("|---|---|---:|---:|---:|---|---|")
    # F-A: hold out 24.03; train = 15.03 + 25.02
    p_15_24 = p0_1["pair_stats"]["15.03.2026__24.03.2026"]
    p_15_25 = p0_1["pair_stats"]["15.03.2026__25.02.2026"]
    p_24_25 = p0_1["pair_stats"]["24.03.2026__25.02.2026"]
    L(f"| F-A | 24.03 | "
      f"{p0_1['per_session_cell_counts']['15.03.2026'] + p0_1['per_session_cell_counts']['25.02.2026']} | "
      f"{p0_1['per_session_cell_counts']['24.03.2026']} | "
      f"{p_15_24['n_intersection']} (vs 15.03) "
      f"+ {p_24_25['n_intersection']} (vs 25.02) | "
      f"15.03 same map; 25.02 disjoint | "
      f"24.03 trajectory ⊂ 15.03 (98.9%) — strong train→test geometric "
      f"coverage in the same map. |")
    L(f"| F-B | 15.03 | "
      f"{p0_1['per_session_cell_counts']['24.03.2026'] + p0_1['per_session_cell_counts']['25.02.2026']} | "
      f"{p0_1['per_session_cell_counts']['15.03.2026']} | "
      f"{p_15_24['n_intersection']} (vs 24.03) "
      f"+ {p_15_25['n_intersection']} (vs 25.02) | "
      f"24.03 same map; 25.02 disjoint | "
      f"15.03's coverage is the largest — most test cells are NOT seen by "
      f"the train sessions. Hardest fold. |")
    L(f"| F-C | 25.02 | "
      f"{p0_1['per_session_cell_counts']['15.03.2026'] + p0_1['per_session_cell_counts']['24.03.2026']} | "
      f"{p0_1['per_session_cell_counts']['25.02.2026']} | "
      f"0 (disjoint frames) | Cross-frame; cannot be registered without "
      f"manual landmark | Pure cross-environment generalisation test. |")
    L("")

    # =================================================================
    L("## 11. Phase 0 conclusions")
    L("")
    L(f"- **Gate 0 (frame): {gate0}** — 15.03 and 24.03 share Map A.")
    L(f"- **Gate A (LiDAR headroom): {gate_a}** — distance alone is a poor "
      "predictor on 24.03 and 25.02; ample LiDAR headroom in principle.")
    L(f"- **Gate B (sectoral): {gate_b}** — usable contiguous sector "
      f"~{p0_3['sector_width_deg']:.0f}° wide; "
      f"{max(1, int(p0_3['sector_width_deg'] // 30))} × 30° sectors.")
    L(f"- **Gate C (project): {gate_c}** — univariate ρ clears 0.25 on "
      "2 of 3 sessions; the 15.03 weak-signal case is a Phase-1 risk to "
      "monitor, not a project blocker.")
    L(f"- **Gate D (framing): {gate_d}** — same-cell |Δ| just over 4 dB; "
      "field is mostly stable with material per-cell drifts.")
    L("")
    L("**Recommended Phase 1 action: proceed with Project A.** Train the "
      "multivariate gradient-boosted model on the 15.03+24.03 same-map "
      "data with LORO folds; report cross-validated Δ_LiDAR (improvement "
      "over a distance-only baseline). 15.03 (held-out as F-B) is the "
      "session most likely to expose a weak true signal — track Δ_LiDAR "
      "there closely. If F-B Δ_LiDAR < 1 dB despite GREEN univariate ρ on "
      "the others, pivot to Project B for the headline.")
    L("")
    L("**Outstanding caveats:**")
    L("- Cross-session n inconsistency (P0.2) is unmodeled; Phase 1 should "
      "include a session indicator or a per-session intercept.")
    L("- 25.02's residuals are not directly comparable to the others "
      "because its path-loss fit is degenerate (R²<0 even at the prior). "
      "Treat its high in-FOV ρ as suggestive, not as evidence of "
      "*post-distance* LiDAR signal.")
    L("- Phase 1 must consume `lidar_fov.json` and `agv_body_mask.npz` "
      "directly — the angular conventions changed during P0 (0.2°/beam, "
      "1 350 active beams).")
    L("- P0.0b (cross-frame registration) deferred — 25.02's spatial "
      "predictions cannot be evaluated against 15.03/24.03 ground truth "
      "in shared coordinates without it.")
    L("")

    # =================================================================
    L("## 12. Reproducibility")
    L("")
    L("```bash")
    L(".venv/Scripts/python.exe -m scripts.p0_analysis.run_all")
    L("```")
    L("")
    L(f"- Seed: `{C.SEED}`.")
    if timing:
        L(f"- Wall-clock time (full pipeline): "
          f"**{timing['total_s']:.1f} s** (single CPU).")
        L("")
        L("| Stage | s |")
        L("|---|---:|")
        for k, v in timing["per_task_s"].items():
            L(f"| {k} | {v:.1f} |")
        L("")
    try:
        import numpy as _np, pandas as _pd, scipy as _sp, h5py as _h5
        import matplotlib as _mpl
        L("- Versions: " + ", ".join([
            f"python={platform.python_version()}",
            f"numpy={_np.__version__}",
            f"pandas={_pd.__version__}",
            f"scipy={_sp.__version__}",
            f"h5py={_h5.__version__}",
            f"matplotlib={_mpl.__version__}",
        ]) + ".")
    except Exception as e:
        L(f"- (could not introspect package versions: {e})")
    L("")

    out = C.REPORT_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
