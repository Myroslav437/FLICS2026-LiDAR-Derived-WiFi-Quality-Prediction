# RQ4 — residual session effect via SHAP

Variants: rq4a = B5 + per-session intercept (one-hot); rq4b = B5 + integer session_id.
Datasets: same_map = 15.03 + 24.03 (Map A); full = all three sessions.
Fraction = mean|SHAP|(session features) / mean|SHAP|(all features) on the test split.

| Variant | Dataset | n_test | RMSE | MAE | R² | bias | mean|SHAP|_session | mean|SHAP|_total | fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rq4a | same_map | 58602 | 5.953 | 4.881 | 0.536 | 2.634 | 0.541 | 11.264 | 4.800% |
| rq4a | full | 91073 | 6.186 | 4.989 | 0.402 | 2.406 | 1.374 | 10.862 | 12.651% |
| rq4b | same_map | 58602 | 5.953 | 4.881 | 0.536 | 2.634 | 0.541 | 11.264 | 4.800% |
| rq4b | full | 91073 | 5.936 | 4.725 | 0.449 | 2.442 | 1.417 | 10.330 | 13.721% |

Maximum fraction on same-map subset: **4.800%** → verdict: **negligible**.
- Negligible (<5%): residual session effect is small after conditioning on geometry.
- Bounded (5–20%): meaningful but bounded effect; per-session n_d disagreement is real.
- Large (>20%): model needs session awareness; deployment requires per-session calibration.
