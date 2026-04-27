# Disambiguation — overall RMSE (dB) for B5 / B5' / B5''

B5 = full 32-feature model. B5' = telemetry + AP-relative (no LiDAR). B5'' = telemetry + LiDAR (no AP-relative).

| Variant | F-A | F-B | F-C |
|---|---|---|---|
| B5 | 8.80 [8.76, 8.83] | 9.30 [9.28, 9.33] | 9.28 [9.26, 9.30] |
| B5' | 8.47 [8.44, 8.50] | 8.91 [8.89, 8.93] | 9.37 [9.35, 9.39] |
| B5'' | 9.13 [9.10, 9.16] | 11.44 [11.41, 11.46] | 10.69 [10.66, 10.71] |

Per-fold judgments (threshold = 0.5 dB):
- F-A: **LiDAR removable**
- F-B: **LiDAR removable**
- F-C: **LiDAR removable**

**Overall verdict: LiDAR removable**
