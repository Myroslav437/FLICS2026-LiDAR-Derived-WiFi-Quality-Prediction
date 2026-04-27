# RQ4 — residual session effect via SHAP

Variants: rq4a = B5 + per-session intercept (one-hot); rq4b = B5 + integer session_id.
Datasets: same_map = 15.03 + 24.03 (Map A); full = all three sessions.
Fraction = mean|SHAP|(session features) / mean|SHAP|(all features) on the test split.

| Variant | Dataset | n_test | RMSE | MAE | R² | bias | mean|SHAP|_session | mean|SHAP|_total | fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rq4a | same_map | 58602 | 11.623 | 9.620 | -0.769 | 6.769 | 0.164 | 3.969 | 4.132% |
| rq4a | full | 91073 | 6.500 | 5.141 | 0.340 | 2.780 | 1.317 | 12.511 | 10.527% |
| rq4b | same_map | 58602 | 11.623 | 9.620 | -0.769 | 6.769 | 0.164 | 3.969 | 4.132% |
| rq4b | full | 91073 | 7.075 | 5.529 | 0.218 | 3.401 | 1.557 | 12.036 | 12.939% |

Maximum fraction on same-map subset: **4.132%** → verdict: **negligible**.
- Negligible (<5%): residual session effect is small after conditioning on geometry.
- Bounded (5–20%): meaningful but bounded effect; per-session n_d disagreement is real.
- Large (>20%): model needs session awareness; deployment requires per-session calibration.
