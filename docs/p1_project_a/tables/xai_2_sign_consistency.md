# XAI-2: sign-of-effect consistency across LORO folds

Sign of Spearman ρ(feature, SHAP). `0` if |ρ| < 0.05.
**Consistent** = same non-zero sign across all 3 folds.

| Feature | F-A | F-B | F-C | ρ(F-A) | ρ(F-B) | ρ(F-C) | Consistent |
|---|:---:|:---:|:---:|---:|---:|---:|:---:|
| `mean_dist_mm` | + | + | + | +0.67 | +0.39 | +0.82 | yes |
| `dist_p90_mm` | + | + | - | +0.27 | +0.83 | -0.15 | no |
| `clutter_frac` | - | - | - | -0.66 | -0.40 | -0.49 | yes |
| `openness_frac` | - | + | - | -0.34 | +0.11 | -0.30 | no |
| `mean_front_mm` | - | + | + | -0.29 | +0.32 | +0.53 | no |
| `mean_dist_sector_1_mm` | + | - | + | +0.64 | -0.06 | +0.42 | no |
| `mean_dist_sector_2_mm` | - | + | + | -0.30 | +0.25 | +0.48 | no |
| `mean_dist_sector_3_mm` | + | + | + | +0.64 | +0.39 | +0.66 | yes |
| `mean_dist_sector_4_mm` | - | + | + | -0.33 | +0.53 | +0.36 | no |
| `mean_dist_sector_5_mm` | - | - | + | -0.39 | -0.39 | +0.48 | no |
| `mean_dist_sector_6_mm` | + | + | + | +0.63 | +0.45 | +0.11 | yes |
| `mean_dist_sector_7_mm` | 0 | 0 | + | +0.02 | +0.01 | +0.37 | no |
| `clutter_frac_sector_1` | - | + | - | -0.66 | +0.20 | -0.42 | no |
| `clutter_frac_sector_2` | - | + | - | -0.74 | +0.69 | -0.39 | no |
| `clutter_frac_sector_3` | + | + | - | +0.21 | +0.30 | -0.80 | no |
| `clutter_frac_sector_4` | - | - | - | -0.64 | -0.38 | -0.66 | yes |
| `clutter_frac_sector_5` | + | - | + | +0.47 | -0.14 | +0.63 | no |
| `clutter_frac_sector_6` | + | - | + | +0.37 | -0.76 | +0.78 | no |
| `clutter_frac_sector_7` | - | - | + | -0.22 | -0.52 | +0.23 | no |
| `dist_to_AP` | - | - | - | -0.84 | -0.84 | -0.87 | yes |
| `sin_angle_to_AP` | + | + | + | +0.79 | +0.56 | +0.82 | yes |
| `cos_angle_to_AP` | - | - | - | -0.87 | -0.82 | -0.81 | yes |
| `clutter_frac_toward_AP` | + | - | + | +0.66 | -0.09 | +0.76 | no |

**Headline**: 5 / 19 LiDAR features sign-consistent; 3 / 4 AP-relative features sign-consistent.
