# Buffer-zone sensitivity — R-1 with vs without 1 m exclusion

Drop training rows within 1.0 m of any held-out region row, then refit. Compare RMSE deltas to the no-buffer R-1 fits.

| Variant | Stratum | RMSE(no buffer) | RMSE(buffer) | Δ_RMSE (buffer - no) |
|---|---|---:|---:|---:|
| W0 | overall | 7.715 | 18.561 | +10.847 |
| W0 | in_fov | 7.911 | 18.124 | +10.214 |
| W0 | out_of_fov | 7.206 | 19.603 | +12.397 |
| W1 | overall | 8.582 | 19.158 | +10.576 |
| W1 | in_fov | 8.699 | 18.991 | +10.292 |
| W1 | out_of_fov | 8.286 | 19.567 | +11.281 |
| W2 | overall | 10.811 | 15.273 | +4.462 |
| W2 | in_fov | 10.802 | 16.432 | +5.629 |
| W2 | out_of_fov | 10.833 | 11.925 | +1.091 |
| W3 | overall | 16.624 | 20.382 | +3.758 |
| W3 | in_fov | 17.759 | 21.131 | +3.372 |
| W3 | out_of_fov | 13.401 | 18.392 | +4.991 |
| W4 | overall | 12.745 | 25.640 | +12.894 |
| W4 | in_fov | 13.789 | 27.645 | +13.856 |
| W4 | out_of_fov | 9.685 | 19.810 | +10.124 |
| W4pp | overall | 15.340 | 22.868 | +7.528 |
| W4pp | in_fov | 17.142 | 24.689 | +7.548 |
| W4pp | out_of_fov | 9.498 | 17.555 | +8.058 |

**Sanity check (overall stratum)**: 0/6 variants change by < 0.5 dB under the buffer.
**Flagged**: majority of variants move by ≥ 0.5 dB under the buffer — leave-region-out may be contaminated by spatial autocorrelation; treat the headline result with caution.
