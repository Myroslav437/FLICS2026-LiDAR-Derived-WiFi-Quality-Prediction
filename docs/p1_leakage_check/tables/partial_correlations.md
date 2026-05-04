# Partial Spearman correlations

Spearman ρ between each telemetry feature and `signal_power`, with and without controlling for `speed_mps`, `turn_rate`, and `momentary_current_consumption`. CIs from a moving-block bootstrap (block size 200, 200 resamples; respects serial correlation).

**Reading the table.** A large drop from `ρ_marginal` to `ρ_partial` means the feature's apparent association with `signal_power` is explained away by motion variables — i.e. the feature acts as a motion proxy. A small drop means the feature carries signal that motion does not.

| session | feature | ρ_marginal (95% CI) | ρ_partial (95% CI) | n |
|:---|:---|:---|:---|---:|
| 25.02.2026 | load_long | +0.003 (-0.051, +0.052) | +0.033 (-0.020, +0.082) | 216,472 |
| 25.02.2026 | load_mid | +0.009 (-0.047, +0.063) | +0.017 (-0.038, +0.069) | 216,472 |
| 25.02.2026 | load_short | +0.024 (-0.031, +0.073) | +0.044 (-0.014, +0.091) | 216,472 |
| 25.02.2026 | nns_state | +0.055 (+0.011, +0.103) | +0.076 (+0.043, +0.112) | 216,472 |
| 15.03.2026 | load_long | +0.008 (-0.048, +0.066) | -0.030 (-0.077, +0.025) | 232,279 |
| 15.03.2026 | load_mid | +0.016 (-0.040, +0.073) | +0.002 (-0.053, +0.060) | 232,279 |
| 15.03.2026 | load_short | -0.031 (-0.093, +0.030) | -0.060 (-0.120, +0.007) | 232,279 |
| 15.03.2026 | nns_state | -0.001 (-0.046, +0.053) | -0.107 (-0.172, -0.050) | 232,279 |
| 24.03.2026 | load_long | +0.147 (+0.086, +0.211) | -0.077 (-0.140, -0.016) | 158,405 |
| 24.03.2026 | load_mid | -0.008 (-0.085, +0.053) | -0.094 (-0.161, -0.030) | 158,405 |
| 24.03.2026 | load_short | -0.030 (-0.090, +0.031) | -0.166 (-0.229, -0.104) | 158,405 |
| 24.03.2026 | nns_state | -0.075 (-0.128, -0.022) | -0.020 (-0.059, +0.024) | 158,405 |
