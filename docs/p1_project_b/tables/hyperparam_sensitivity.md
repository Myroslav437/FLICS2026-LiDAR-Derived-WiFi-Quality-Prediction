# Hyperparameter sensitivity — W4 locked vs H1 (max_depth=4)

H1 was the cross-session diagnostic's best alternative. Refit on each Project B fold to verify the within-session story is not config-specific.

| Fold | Stratum | RMSE(locked, depth=6) | RMSE(H1, depth=4) | Δ_RMSE (H1 - locked) |
|---|---|---:|---:|---:|
| R-1 | overall | 17.533 | 12.775 | -4.758 |
| R-1 | in_fov | 18.568 | 13.765 | -4.803 |
| R-1 | out_of_fov | 14.657 | 9.902 | -4.755 |
| R-2 | overall | 6.064 | 5.871 | -0.192 |
| R-2 | in_fov | 6.080 | 6.633 | +0.553 |
| R-2 | out_of_fov | 6.050 | 5.203 | -0.848 |
| R-3 | overall | 4.566 | 5.032 | +0.466 |
| R-3 | in_fov | 4.525 | 5.457 | +0.932 |
| R-3 | out_of_fov | 4.657 | 3.903 | -0.755 |
| R-4 | overall | 4.692 | 4.789 | +0.097 |
| R-4 | in_fov | 2.021 | 2.327 | +0.305 |
| R-4 | out_of_fov | 5.983 | 6.033 | +0.051 |
| R-5 | overall | 10.684 | 11.094 | +0.410 |
| R-5 | in_fov | 4.786 | 6.336 | +1.550 |
| R-5 | out_of_fov | 13.519 | 13.617 | +0.098 |

**Flagged (overall stratum)**: |Δ_RMSE| > 0.3 dB on folds ['R-1', 'R-3', 'R-5']. Report locked and H1 side by side for these folds.
