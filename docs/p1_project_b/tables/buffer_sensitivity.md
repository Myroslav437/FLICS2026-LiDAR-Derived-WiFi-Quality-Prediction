# Buffer-zone sensitivity — R-1 with vs without 1 m exclusion

Drop training rows within 1.0 m of any held-out region row, then refit. Compare RMSE deltas to the no-buffer R-1 fits.

| Variant | Stratum | RMSE(no buffer) | RMSE(buffer) | Δ_RMSE (buffer - no) |
|---|---|---:|---:|---:|
| W0 | overall | 7.775 | 19.886 | +12.111 |
| W0 | in_fov | 7.951 | 19.429 | +11.478 |
| W0 | out_of_fov | 7.321 | 20.977 | +13.657 |
| W1 | overall | 8.395 | 10.033 | +1.638 |
| W1 | in_fov | 8.437 | 10.283 | +1.846 |
| W1 | out_of_fov | 8.288 | 9.384 | +1.096 |
| W2 | overall | 8.922 | 10.448 | +1.527 |
| W2 | in_fov | 9.512 | 11.336 | +1.824 |
| W2 | out_of_fov | 7.254 | 7.826 | +0.571 |
| W3 | overall | 17.088 | 18.281 | +1.193 |
| W3 | in_fov | 18.136 | 19.233 | +1.097 |
| W3 | out_of_fov | 14.158 | 15.672 | +1.515 |
| W4 | overall | 17.533 | 18.164 | +0.630 |
| W4 | in_fov | 18.568 | 19.440 | +0.873 |
| W4 | out_of_fov | 14.657 | 14.521 | -0.136 |
| W4pp | overall | 16.306 | 17.079 | +0.773 |
| W4pp | in_fov | 17.991 | 18.888 | +0.897 |
| W4pp | out_of_fov | 11.075 | 11.418 | +0.343 |

**Sanity check (overall stratum)**: 0/6 variants change by < 0.5 dB under the buffer.
**Flagged**: majority of variants move by ≥ 0.5 dB under the buffer — leave-region-out may be contaminated by spatial autocorrelation; treat the headline result with caution.
