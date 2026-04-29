# Dataset noise floor (Experiment D)

Same-cell |Δ signal_power| between 15.03 and 24.03 quantifies the cross-visit non-stationarity of the WiFi field. The within-cell σ_intra summarises the irreducible session-internal noise that any model must compete against.

## Same-cell Δ summary

| Metric | Value |
|---|---:|
| n cells (≥30 rows / session, both sessions) | 85 |
| Cell size | 0.50 m × 0.50 m |
| median Δ (signed) | +0.99 dB |
| median \|Δ\| | 4.24 dB |
| IQR \|Δ\| | [3.01, 7.13] dB |

## Within-cell σ_intra (irreducible noise floor)

Mean of per-cell σ(signal_power) over the qualifying cells, computed independently for each session. ddof=1 (sample std).

| Session | mean σ_intra (dB) |
|---|---:|
| 15.03.2026 | 4.901 |
| 24.03.2026 | 5.005 |
| **global (mean of all per-cell σ values)** | **4.953** |

## Comparison to model performance

| Reference | RMSE (dB) |
|---|---:|
| Cleanest within-session model (R-4 W2 H1 in-FOV from Project B) | 1.995 |
| σ_intra (irreducible) | 4.953 |
| Gap (model − σ_intra) | -2.958 |

## Discussion (paper §III draft)

Two fully-mapped passes of the same workspace 9 days apart show median |Δ mean signal_power| = 4.24 dB across 85 0.5 m cells (IQR [3.01, 7.13] dB). Within a single session, the mean per-cell σ(signal_power) is 4.95 dB across the same 85 cells; this σ_intra is the noise floor for any model that resolves position only at 0.5 m granularity. For context, the cleanest within-session model achieves RMSE = 2.00 dB — already below the 0.5 m position-binning σ_intra of 4.95 dB. The model beats σ_intra by exploiting sub-cell position (x_m, y_m at sensor resolution), telemetry, and AP geometry; what remains within the 1.99–5.0 dB window is variance that no 0.5 m position-binning model could touch. Adding LiDAR features does not improve on this — the residual error is dominated by the intrinsic non-stationarity of the WiFi field across visits, not by missing environmental structure.
