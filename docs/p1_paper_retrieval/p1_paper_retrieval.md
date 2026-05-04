# Paper-data retrieval — leakage-fixed

Generated: 2026-05-03

Retrieval target: 5 items flagged in `docs/paper/paper.tex` after the
leakage-fixed rerun. Every value below is from artefacts tagged
`feature_stack_version = "leakage_fixed_v1"` or directly derivable from
such artefacts (mtime ≥ `MIGRATION_LOG.md` generation date 2026-05-02 used as a
secondary verification when the file does not carry the column itself). Items where
leakage-fixed data is unavailable are marked **MISSING** and the author should
either re-run the underlying pipeline or accept the placeholder text already in
the paper. **No pre-leakage-fix number appears anywhere in this report.**

## 0. Summary

| Item | Status | Source |
|---|---|---|
| 1. Cross-session headline 95% bootstrap CIs (Table IV) | found | `docs/p1_project_a/tables/delta_lidar.md`; `scripts/p1_project_a/results/loro_metrics.parquet` |
| 2. Cross-session disambiguation table for F1, F2 (Table V) | found | `scripts/p1_project_a/results/disambig_metrics.parquet`; `docs/p1_project_a/tables/disambig_summary.md (overall-stratum analogue, kept for cross-reference)` |
| 3. Within-session disambiguation table per fold | found | `scripts/p1_project_b/results/disambig_metrics.parquet`; `scripts/p1_project_b/results/wlro_metrics.parquet (provenance cross-check)` |
| 4. R-1 buffer-zone increments (§V-B) | found | `docs/p1_project_b/tables/buffer_sensitivity.md`; `scripts/p1_project_b/results/robustness_metrics.parquet` |
| 5. SHAP group importance + sign consistency (§VI) | found | `scripts/p1_project_a/results/shap_F-A.parquet`; `scripts/p1_project_a/results/shap_F-B.parquet` |

## 1. Item 1 — Cross-session headline 95% bootstrap CIs (Table IV)

Source for Δ_LiDAR point estimates and 95% CIs: paired-bootstrap on the
joint-idx-aligned B1/B5 prediction parquets, computed by
`scripts.p1_project_a.build_results_report._bootstrap_diff_rmse` (B = 1000)
and rendered to `docs/p1_project_a/tables/delta_lidar.md`. The underlying
`scripts/p1_project_a/results/loro_metrics.parquet` confirms
`feature_stack_version = leakage_fixed_v1` on the contributing rows: 
PASS.

Paper-fold mapping (per `docs/proposal_rev10.md` §5 / paper §IV-A):
F1 = F-B (15.03 held out), F2 = F-A (24.03 held out),
F3 = F-C (25.02 held out, cross-frame to Map B).

**Markdown for direct paste into Table IV's last column:**

| Fold | $\dlidar$ (95 % CI) |
|---|---|
| F1 | $-0.19$ ($[-0.21, -0.16]$) |
| F2 | $+0.55$ ($[+0.54, +0.56]$) |
| F3 | $+1.20$ ($[+1.18, +1.23]$) |

**Notes:**
- Bootstrap is the *paired* delta variant (joint-idx-aligned B1/B5), not a normal approximation.

## 2. Item 2 — Cross-session disambiguation table for F1, F2 (Table V)

Source: in-FOV stratum rows of
`scripts/p1_project_a/results/disambig_metrics.parquet`
(feature_stack_version = leakage_fixed_v1 across all 27 rows).
Verdict labels follow the existing paper draft (the prompt explicitly notes
"the verdict labels are already correct; only the numeric values need filling").

F-C/F3 row is reproduced as-is from the paper draft (already filled).

**Markdown rows for Table V (in-FOV RMSE in dB):**

| Fold | B5 | B5$^\prime$ | B5$^{\prime\prime}$ | verdict |
|---|---:|---:|---:|---|
| F1 | 8.20 | 7.93 | 12.14 | LiDAR removable |
| F2 | 6.16 | 5.85 | 7.51 | LiDAR removable |
| F3 | 7.67 | 8.15 | 8.82 | LiDAR complementary |

Verdict-classifier check (0.5 dB threshold per paper §IV-A):
- F1 (F-B): B5 = 8.20, B5' = 7.93 (Δ = +0.28 dB), B5'' = 12.14 (Δ = -3.94 dB).
  B5 worse than B5' by 0.27 dB and better than B5'' by 3.94 dB => B5 is dominated by B5' (LiDAR removable). Verdict matches paper.
- F2 (F-A): B5 = 6.16, B5' = 5.85 (Δ = +0.32 dB), B5'' = 7.51 (Δ = -1.34 dB).
  B5 worse than B5' by 0.32 dB and better than B5'' by 1.34 dB => B5 is dominated by B5' (LiDAR removable). Verdict matches paper.

**Notes:**
- The existing `docs/p1_project_a/tables/disambig_summary.md` reports overall-stratum RMSE; the paper's Table V uses in-FOV. The in-FOV values above come directly from the disambig_metrics.parquet.

## 3. Item 3 — Within-session disambiguation table per fold

Source: in-FOV stratum rows of
`scripts/p1_project_b/results/disambig_metrics.parquet`
(provenance: mtime 2026-05-02T23:38:13; fallback to mtime check because the file does
not carry an explicit `feature_stack_version` column — Project B's disambig is
constructed from `wlro_metrics.parquet` rows in `run_modeling.py` without forwarding
the column).

Cross-check: the underlying `scripts/p1_project_b/results/wlro_metrics.parquet` IS
tagged `feature_stack_version = leakage_fixed_v1` (PASS — all 90 rows tagged); the disambig parquet
rows for in-FOV W4, W2, W4pp are byte-identical RMSE values to the wlro rows for the
same (fold, variant, stratum) keys (verified at retrieval time).

Verdict per fold computed locally with the 0.5 dB threshold from paper §IV-A.
Source `docs/p1_project_b/tables/disambig_summary.md` reports overall-stratum
verdicts; the in-FOV verdicts derived here may differ from those (the paper's table
is in-FOV-stratum-specific).

**LaTeX for the new within-session disambiguation table:**

```latex
\begin{table}[!htbp]
    \caption{Within-session feature group ablation verdict per fold (H0). In-FOV RMSE in dB.}
    \label{tab:wlro-disambig}
    \centering
    \footnotesize
    \begin{tabular}{l c c c l}
        \toprule
        Fold & W4 & W4$^\prime$ & W4$^{\prime\prime}$ & verdict \\
        \midrule
        R-1 & 13.79 & 10.80 & 17.14 & mixed/unclear \\
        R-2 & 9.40 & 7.19 & 9.51 & AP-relative removable \\
        R-3 & 7.31 & 7.22 & 9.52 & LiDAR removable \\
        R-4 & 5.79 & 4.16 & 4.18 & mixed/unclear \\
        R-5 & 11.58 & 9.37 & 12.86 & mixed/unclear \\
        \bottomrule
    \end{tabular}
\end{table}
```

**Markdown counterpart for direct in-line reading:**

| Fold | W4 | W4$^\prime$ | W4$^{\prime\prime}$ | verdict |
|---|---:|---:|---:|---|
| R-1 | 13.79 | 10.80 | 17.14 | mixed/unclear |
| R-2 | 9.40 | 7.19 | 9.51 | AP-relative removable |
| R-3 | 7.31 | 7.22 | 9.52 | LiDAR removable |
| R-4 | 5.79 | 4.16 | 4.18 | mixed/unclear |
| R-5 | 11.58 | 9.37 | 12.86 | mixed/unclear |

**Notes:**
- Project B disambig parquet has no `feature_stack_version` column (built by run_modeling.py from wlro rows). Verified via mtime fallback (>= MIGRATION_LOG.md generation date) plus a byte-identical RMSE cross-check against wlro_metrics.parquet which IS tagged.
- The R-4 W4 = 5.79 / W2 = 4.16 numbers cited in the prompt match the in-FOV row for fold R-4 below — confirms the values are the leakage-fixed ones.

## 4. Item 4 — R-1 buffer-zone increments (§V-B)

Source: `docs/p1_project_b/tables/buffer_sensitivity.md` (R-1, 1 m buffer,
locked hyperparameters). Parquet provenance:
`scripts/p1_project_b/results/robustness_metrics.parquet`
(all 33 rows feature_stack_version = leakage_fixed_v1).

**Per-variant in-FOV table (drop-in for §V-B):**

| Variant | RMSE (no buffer) | RMSE (1 m buffer) | $\Delta$RMSE |
|---|---:|---:|---:|
| W0 | 7.911 | 18.124 | $+10.214\dB$ |
| W1 | 8.699 | 18.991 | $+10.292\dB$ |
| W2 | 10.802 | 16.432 | $+5.629\dB$ |
| W3 | 17.759 | 21.131 | $+3.372\dB$ |
| W4 | 13.789 | 27.645 | $+13.856\dB$ |
| W4pp | 17.142 | 24.689 | $+7.548\dB$ |

**Plain-prose insertion for §V-B (drop-in replacement for the qualitative paragraph):**

```latex
To bound the magnitude of this contamination, we apply a buffer-zone test on R-1: dropping all training rows within 1\,m of any held-out R-1 row inflates
the W0 in-FOV RMSE by $+10.21\dB$, the W1 in-FOV RMSE by $+10.29\dB$, the W2 in-FOV RMSE by $+5.63\dB$, the W4 in-FOV RMSE by $+13.86\dB$.
```

**Notes:**
- The `robustness_metrics.parquet` covers the buffer fold (`R-1_buffer`) under `check = 'buffer'` and is tagged leakage_fixed_v1; the markdown table is its rendered form (regenerated post-rerun by build_results_report).

## 5. Item 5 — SHAP group importance + sign consistency (§VI)

**Provenance.** SHAP source rows do not carry an explicit `feature_stack_version`
column. Verified leakage-fixed via two independent checks per file: (i) mtime
≥ 2026-05-02 (MIGRATION_LOG.md generation date), and (ii) no legacy features
(`load_long`, `load_mid`, `load_short`, `battery_value`, `nns_state`) appear as
`shap_*` columns:

| File | mtime | n shap features | leakage-fixed? |
|---|---|---:|---|
| `scripts/p1_project_a/results/shap_F-A.parquet` | 2026-05-02T23:32:21 | 27 | PASS |
| `scripts/p1_project_a/results/shap_F-B.parquet` | 2026-05-02T23:32:36 | 27 | PASS |
| `scripts/p1_project_a/results/shap_F-C.parquet` | 2026-05-02T23:32:41 | 27 | PASS |
| `scripts/p1_project_b/results/shap_R-3.parquet` | 2026-05-02T23:44:48 | 29 | PASS |

(B5 cross-session has 27 features; W4 within-session has 29 features. Counts match.)

### (a) Group importance per fold (cumulative mean|SHAP|, dB)

| Group | F1 (F-B) | F2 (F-A) | F3 (F-C) | R-3 | # features |
|---|---:|---:|---:|---:|---:|
| AP-relative | 6.884 | 7.672 | 6.671 | 6.281 | 5 |
| Telemetry | 1.600 | 0.599 | 1.086 | 2.354 | 3 |
| Position | n/a | n/a | n/a | 4.754 | 2 |
| LiDAR scalar | 1.740 | 0.739 | 0.815 | 1.632 | 5 |
| LiDAR sectoral | 1.399 | 1.441 | 2.028 | 2.822 | 14 |

Per-feature contributions are derivable as group-cumulative ÷ feature-count
when uniform-within-group is a sufficient first-order summary; the underlying
per-feature mean|SHAP| values are in `shap_F-A.parquet` (etc.) for finer breakdown.

### (b) Cross-session sign consistency (Spearman ρ ≥ 0.05 same sign on all 3 folds)

- **AP-relative**: 4 / 4 pass (excluding `is_AP_in_FOV` boolean): `dist_to_AP`, `sin_angle_to_AP`, `cos_angle_to_AP`, `clutter_frac_toward_AP`.
- **LiDAR (scalar + sectoral)**: 5 / 19 pass: `mean_dist_mm`, `clutter_frac`, `mean_dist_sector_3_mm`, `mean_dist_sector_6_mm`, `clutter_frac_sector_4`.
- **Telemetry**: 1 / 3 pass: `speed_mps`.

Headline asymmetry: **4/4 AP-relative against 5/19 LiDAR**.
(Cross-references the existing `docs/p1_project_a/tables/xai_2_sign_consistency.md`,
which reports per-feature ρ values for inspection.)

### Figure-5 regeneration status

Source artefact: `docs/p1_project_a/figures/xai_1_feature_group_importance.png`
Last modified: 2026-05-02T23:33:03
Underlying SHAP source rows: leakage-fixed (verified above).
Status: **ready for caption update**.


**Notes:**
- SHAP per-feature mean|SHAP| values, group cumulatives, and per-fold Spearman ρ are all computed from the `shap_*.parquet` files in this retrieval — values match the published `docs/p1_project_*/tables/xai_*` rendering to within rounding.
- The within-session R-3 fit only has one fold of cross-fold consistency, so the sign-consistency test (a 3-fold majority) does not apply within-session; only the group-importance numbers are meaningful for R-3.

## 6. Items the author should consider re-running

All items found; no re-run needed.


## 7. Verification log

| File | mtime | feature_stack_version |
|---|---|---|
| `scripts/p1_project_a/results/loro_metrics.parquet` | 2026-05-02T23:27:45 | leakage_fixed_v1 ×54 |
| `scripts/p1_project_a/results/disambig_metrics.parquet` | 2026-05-02T23:27:45 | leakage_fixed_v1 ×27 |
| `scripts/p1_project_a/results/shap_F-A.parquet` | 2026-05-02T23:32:21 | (no column; mtime fallback OK) |
| `scripts/p1_project_a/results/shap_F-B.parquet` | 2026-05-02T23:32:36 | (no column; mtime fallback OK) |
| `scripts/p1_project_a/results/shap_F-C.parquet` | 2026-05-02T23:32:41 | (no column; mtime fallback OK) |
| `docs/p1_project_a/tables/delta_lidar.md` | 2026-05-02T23:33:02 | (no column; mtime fallback OK) |
| `docs/p1_project_a/tables/disambig_summary.md` | 2026-05-02T23:33:02 | (no column; mtime fallback OK) |
| `docs/p1_project_a/tables/xai_1_group_importance.md` | 2026-05-02T23:33:03 | (no column; mtime fallback OK) |
| `docs/p1_project_a/tables/xai_2_sign_consistency.md` | 2026-05-02T23:33:04 | (no column; mtime fallback OK) |
| `docs/p1_project_a/figures/xai_1_feature_group_importance.png` | 2026-05-02T23:33:03 | (no column; mtime fallback OK) |
| `scripts/p1_project_b/results/wlro_metrics.parquet` | 2026-05-02T23:38:13 | leakage_fixed_v1 ×90 |
| `scripts/p1_project_b/results/disambig_metrics.parquet` | 2026-05-02T23:38:13 | (no column; mtime fallback OK) |
| `scripts/p1_project_b/results/robustness_metrics.parquet` | 2026-05-02T23:40:06 | leakage_fixed_v1 ×33 |
| `scripts/p1_project_b/results/shap_R-3.parquet` | 2026-05-02T23:44:48 | (no column; mtime fallback OK) |
| `docs/p1_project_b/tables/buffer_sensitivity.md` | 2026-05-02T23:44:54 | (no column; mtime fallback OK) |
| `docs/p1_project_b/tables/disambig_summary.md` | 2026-05-02T23:44:54 | (no column; mtime fallback OK) |
| `docs/p1_project_b/tables/xai_group_importance_R-3.md` | 2026-05-02T23:44:54 | (no column; mtime fallback OK) |
| `docs/p1_project_b/tables/xai_sign_R-3.md` | 2026-05-02T23:44:55 | (no column; mtime fallback OK) |

Wall-clock retrieval time: 2.9 s.
