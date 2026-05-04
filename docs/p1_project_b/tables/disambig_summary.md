# Disambiguation — overall RMSE (dB) for W4 / W4' / W4''

W4 = full 34-feature model. W4' = position + telemetry + AP-relative (≡ W2; LiDAR removed). W4'' = position + telemetry + LiDAR (AP-relative removed).

| Variant | R-1 | R-2 | R-3 | R-4 | R-5 |
|---|---|---|---|---|---|
| W4 | 12.75 [12.70, 12.79] | 8.07 [8.02, 8.12] | 6.96 [6.91, 7.00] | 6.93 [6.88, 6.98] | 10.58 [10.53, 10.63] |
| W4' | 10.81 [10.75, 10.88] | 7.45 [7.41, 7.50] | 6.78 [6.73, 6.83] | 6.35 [6.30, 6.40] | 9.27 [9.23, 9.32] |
| W4'' | 15.34 [15.29, 15.40] | 8.25 [8.20, 8.32] | 9.08 [9.03, 9.14] | 6.60 [6.53, 6.67] | 16.82 [16.76, 16.88] |

Per-fold judgments (threshold = 0.5 dB):
- R-1: **mixed/unclear**
- R-2: **AP-relative removable**
- R-3: **LiDAR removable**
- R-4: **AP-relative removable**
- R-5: **mixed/unclear**

**Overall verdict (≥3 of 5 folds): mixed/unclear**
