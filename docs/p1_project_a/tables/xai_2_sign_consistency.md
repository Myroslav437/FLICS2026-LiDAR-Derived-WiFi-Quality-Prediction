# XAI-2: sign-of-effect consistency across LORO folds

Sign of Spearman ρ(feature, SHAP). `0` if |ρ| < 0.05.
**Consistent** = same non-zero sign across all 3 folds.

| Feature | F-A | F-B | F-C | ρ(F-A) | ρ(F-B) | ρ(F-C) | Consistent |
|---|:---:|:---:|:---:|---:|---:|---:|:---:|
| `mean_dist_mm` | + | + | + | +0.73 | +0.79 | +0.78 | yes |
| `dist_p90_mm` | + | + | + | +0.45 | +0.75 | +0.06 | yes |
| `clutter_frac` | + | - | 0 | +0.43 | -0.54 | -0.01 | no |
| `openness_frac` | + | - | - | +0.14 | -0.46 | -0.70 | no |
| `mean_front_mm` | - | + | + | -0.45 | +0.82 | +0.60 | no |
| `mean_dist_sector_1_mm` | - | - | + | -0.14 | -0.18 | +0.36 | no |
| `mean_dist_sector_2_mm` | — | + | + | — | +0.39 | +0.62 | no |
| `mean_dist_sector_3_mm` | + | - | + | +0.64 | -0.32 | +0.72 | no |
| `mean_dist_sector_4_mm` | - | + | + | -0.17 | +0.76 | +0.41 | no |
| `mean_dist_sector_5_mm` | + | - | + | +0.17 | -0.47 | +0.56 | no |
| `mean_dist_sector_6_mm` | + | + | + | +0.54 | +0.86 | +0.39 | yes |
| `mean_dist_sector_7_mm` | + | + | + | +0.60 | +0.45 | +0.40 | yes |
| `clutter_frac_sector_1` | - | + | - | -0.20 | +0.18 | -0.55 | no |
| `clutter_frac_sector_2` | - | + | - | -0.67 | +0.53 | -0.56 | no |
| `clutter_frac_sector_3` | - | - | + | -0.54 | -0.38 | +0.69 | no |
| `clutter_frac_sector_4` | — | - | - | — | -0.77 | -0.54 | no |
| `clutter_frac_sector_5` | + | - | + | +0.41 | -0.58 | +0.45 | no |
| `clutter_frac_sector_6` | + | - | + | +0.43 | -0.07 | +0.84 | no |
| `clutter_frac_sector_7` | 0 | - | - | +0.04 | -0.45 | -0.10 | no |
| `dist_to_AP` | - | - | - | -0.84 | -0.88 | -0.82 | yes |
| `sin_angle_to_AP` | + | + | + | +0.75 | +0.73 | +0.88 | yes |
| `cos_angle_to_AP` | - | - | - | -0.87 | -0.82 | -0.84 | yes |
| `clutter_frac_toward_AP` | + | - | - | +0.71 | -0.70 | -0.68 | no |

**Headline**: 4 / 19 LiDAR features sign-consistent; 3 / 4 AP-relative features sign-consistent.
