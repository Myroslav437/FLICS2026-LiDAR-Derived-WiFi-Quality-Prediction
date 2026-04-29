# Disambiguation — overall RMSE (dB) for W4 / W4' / W4''

W4 = full 34-feature model. W4' = position + telemetry + AP-relative (≡ W2; LiDAR removed). W4'' = position + telemetry + LiDAR (AP-relative removed).

| Variant | R-1 | R-2 | R-3 | R-4 | R-5 |
|---|---|---|---|---|---|
| W4 | 17.53 [17.49, 17.58] | 6.06 [6.01, 6.12] | 4.57 [4.53, 4.60] | 4.69 [4.62, 4.76] | 10.68 [10.62, 10.74] |
| W4' | 8.92 [8.89, 8.96] | 5.49 [5.45, 5.54] | 4.18 [4.15, 4.20] | 5.76 [5.69, 5.82] | 10.60 [10.54, 10.66] |
| W4'' | 16.31 [16.25, 16.36] | 6.67 [6.61, 6.73] | 6.61 [6.56, 6.65] | 5.00 [4.93, 5.07] | 11.40 [11.35, 11.45] |

Per-fold judgments (threshold = 0.5 dB):
- R-1: **mixed/unclear**
- R-2: **mixed/unclear**
- R-3: **LiDAR removable**
- R-4: **AP-relative removable**
- R-5: **LiDAR removable**

**Overall verdict (≥3 of 5 folds): mixed/unclear**
