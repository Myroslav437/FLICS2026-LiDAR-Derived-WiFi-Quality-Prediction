# Disambiguation — overall RMSE (dB) for B5 / B5' / B5''

B5 = full 32-feature model. B5' = telemetry + AP-relative (no LiDAR). B5'' = telemetry + LiDAR (no AP-relative).

| Variant | F-A | F-B | F-C |
|---|---|---|---|
| B5 | 8.39 [8.35, 8.42] | 8.62 [8.60, 8.65] | 9.00 [8.98, 9.02] |
| B5' | 8.36 [8.33, 8.39] | 8.84 [8.82, 8.86] | 8.98 [8.96, 9.00] |
| B5'' | 8.89 [8.86, 8.92] | 11.76 [11.74, 11.79] | 10.45 [10.43, 10.48] |

Per-fold judgments (threshold = 0.5 dB):
- F-A: **LiDAR removable**
- F-B: **LiDAR removable**
- F-C: **LiDAR removable**

**Overall verdict: LiDAR removable**
