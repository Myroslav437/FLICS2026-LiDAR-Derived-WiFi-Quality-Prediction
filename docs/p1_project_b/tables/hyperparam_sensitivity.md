# Hyperparameter sensitivity — W4 locked vs H1 (max_depth=4)

H1 was the cross-session diagnostic's best alternative. Refit on each Project B fold to verify the within-session story is not config-specific.

| Fold | Stratum | RMSE(locked, depth=6) | RMSE(H1, depth=4) | Δ_RMSE (H1 - locked) |
|---|---|---:|---:|---:|
| R-1 | overall | 12.745 | 9.976 | -2.770 |
| R-1 | in_fov | 13.789 | 10.753 | -3.035 |
| R-1 | out_of_fov | 9.685 | 7.716 | -1.969 |
| R-2 | overall | 8.067 | 6.571 | -1.496 |
| R-2 | in_fov | 9.395 | 6.552 | -2.843 |
| R-2 | out_of_fov | 6.859 | 6.586 | -0.274 |
| R-3 | overall | 6.956 | 7.514 | +0.558 |
| R-3 | in_fov | 7.307 | 8.133 | +0.826 |
| R-3 | out_of_fov | 6.087 | 5.878 | -0.209 |
| R-4 | overall | 6.928 | 5.886 | -1.043 |
| R-4 | in_fov | 5.792 | 3.361 | -2.431 |
| R-4 | out_of_fov | 7.688 | 7.252 | -0.437 |
| R-5 | overall | 10.580 | 8.267 | -2.313 |
| R-5 | in_fov | 11.583 | 8.522 | -3.060 |
| R-5 | out_of_fov | 9.760 | 8.071 | -1.689 |

**Flagged (overall stratum)**: |Δ_RMSE| > 0.3 dB on folds ['R-1', 'R-2', 'R-3', 'R-4', 'R-5']. Report locked and H1 side by side for these folds.
