"""Phase 1 / Project A configuration: constants, paths, feature lists, hyperparameters."""

from __future__ import annotations

from pathlib import Path

SEED = 20260427

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "phase1" / "dataset.parquet"
DATASET_SHA_PATH = PROJECT_ROOT / "data" / "phase1" / "dataset.sha256"

# Project A code, models, prediction caches, metrics live under scripts/p1_project_a/
# (mirrors the scripts/p0_analysis/ layout: code + outputs in one place).
ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "p1_project_a"
MODELS_DIR = ANALYSIS_DIR / "models"
CACHE_DIR = ANALYSIS_DIR / "cache"
RESULTS_DIR = ANALYSIS_DIR / "results"
DIAGNOSTIC_DIR = ANALYSIS_DIR / "diagnostic"

DOCS_DIR = PROJECT_ROOT / "docs" / "p1_project_a"
FIGURES_DIR = DOCS_DIR / "figures"
TABLES_DIR = DOCS_DIR / "tables"
REPORT_PATH = DOCS_DIR / "results_report.md"

for _d in (MODELS_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# Leakage-fixed telemetry stack (post-hoc audit, see MIGRATION_LOG.md).
#
# Removed (vs the locked full-8 stack):
#   - load_long, load_mid, load_short: MikroTik router CPU load averages
#     (FH.7000.[mikrotik].load_*); target leakage from the receiving end of the
#     same WiFi link the model is predicting.
#   - battery_value: 0xFFFF CAN-bus sentinel that contributes zero information
#     within the LiDAR-coverage window (docs/p1_battery_provenance/report.md).
#   - nns_state: discrete 2/3 navigation-system state that produces a material
#     within-session shift but no cross-session shift; plausibly a within-session
#     spatial-mode fingerprint that does not transfer across deployments.
LEAKAGE_FIXED_TELEMETRY = [
    "speed_mps",
    "turn_rate",
    "momentary_current_consumption",
]
TELEMETRY_FEATURES = list(LEAKAGE_FIXED_TELEMETRY)
FEATURE_STACK_VERSION = "leakage_fixed_v1"

LIDAR_SCALAR_FEATURES = [
    "mean_dist_mm",
    "dist_p90_mm",
    "clutter_frac",
    "openness_frac",
    "mean_front_mm",
]

LIDAR_SECTORAL_FEATURES = [
    "mean_dist_sector_1_mm",
    "mean_dist_sector_2_mm",
    "mean_dist_sector_3_mm",
    "mean_dist_sector_4_mm",
    "mean_dist_sector_5_mm",
    "mean_dist_sector_6_mm",
    "mean_dist_sector_7_mm",
    "clutter_frac_sector_1",
    "clutter_frac_sector_2",
    "clutter_frac_sector_3",
    "clutter_frac_sector_4",
    "clutter_frac_sector_5",
    "clutter_frac_sector_6",
    "clutter_frac_sector_7",
]

AP_RELATIVE_FEATURES = [
    "dist_to_AP",
    "sin_angle_to_AP",
    "cos_angle_to_AP",
    "clutter_frac_toward_AP",
    "is_AP_in_FOV",
]

# Ablation variant feature lists.
B0 = ["dist_to_AP"]
B1 = ["dist_to_AP", "sin_angle_to_AP", "cos_angle_to_AP"]
B2 = B1 + LIDAR_SCALAR_FEATURES
B3 = B2 + LIDAR_SECTORAL_FEATURES
B4 = B3 + TELEMETRY_FEATURES
B5 = B4 + ["clutter_frac_toward_AP", "is_AP_in_FOV"]

# Disambiguation variants.
B5_PRIME = TELEMETRY_FEATURES + AP_RELATIVE_FEATURES                 # 13 features
B5_DBLPRIME = TELEMETRY_FEATURES + LIDAR_SCALAR_FEATURES + LIDAR_SECTORAL_FEATURES  # 27 features

VARIANTS = {
    "B0": B0,
    "B1": B1,
    "B2": B2,
    "B3": B3,
    "B4": B4,
    "B5": B5,
    "B5p": B5_PRIME,
    "B5pp": B5_DBLPRIME,
}

# LORO folds: (train_sessions, test_session)
FOLDS = {
    "F-A": (("15.03.2026", "25.02.2026"), "24.03.2026"),
    "F-B": (("24.03.2026", "25.02.2026"), "15.03.2026"),
    "F-C": (("15.03.2026", "24.03.2026"), "25.02.2026"),
}

CADENCE_DECIMATION_FACTOR = 7  # round(31 / 4.4) — applied to F-C training sessions only.

# Feature groups for SHAP aggregation.
FEATURE_GROUPS = {
    "Telemetry": TELEMETRY_FEATURES,
    "LiDAR scalar": LIDAR_SCALAR_FEATURES,
    "LiDAR sectoral": LIDAR_SECTORAL_FEATURES,
    "AP-relative": AP_RELATIVE_FEATURES,
}

# Frozen XGBoost hyperparameters.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "eval_metric": "rmse",
    "seed": SEED,
    "verbosity": 0,
}
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100

TARGET = "signal_power"

PROVENANCE_COLS = [
    "joint_idx",
    "session_date",
    "fh7000_timestamp",
    "x_m",
    "y_m",
    "is_AP_in_FOV",
]

BOOTSTRAP_B = 1000
SHAP_SUBSAMPLE_LIMIT = 100_000


# ---- Hardening experiments (A, B) -------------------------------------
#
# Originally lived under scripts/p1_hardening/config.py. Merged in 2026-04-28
# (see scripts/maintenance/merge_hardening_log.md). Kept here so Project A's
# hardening runners (run_hardening_placebo.py, run_hardening_lightgbm.py) and
# Project B's run_hardening_placebo.py — which imports placebo helpers from
# Project A — share the same definitions.

# 19 ego-frame LiDAR columns block-shuffled by the placebo helper.
# clutter_frac_toward_AP and is_AP_in_FOV are AP-relative and NOT shuffled.
LIDAR_COLUMNS: list[str] = list(LIDAR_SCALAR_FEATURES) + list(LIDAR_SECTORAL_FEATURES)
assert len(LIDAR_COLUMNS) == 19, f"expected 19 LiDAR columns, got {len(LIDAR_COLUMNS)}"

# XGBoost hyperparameter sets used by Hardening A (placebo) — same as the
# diagnostic robustness configs.
XGB_LOCKED = dict(XGB_PARAMS)
XGB_H1 = {**XGB_PARAMS, "max_depth": 4}
XGB_LABELS = {"locked": XGB_LOCKED, "H1": XGB_H1}

# LightGBM hyperparameter sets for Hardening B (framework-agnosticism).
LGB_DEFAULT = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
}
LGB_H1_EQUIV = {**LGB_DEFAULT, "num_leaves": 15}
LGB_LABELS = {"default": LGB_DEFAULT, "H1_equiv": LGB_H1_EQUIV}

# Verdict thresholds shared by Hardening A, B, C, E.
PLACEBO_TOLERANCE_DB = 0.5
DELTA_LIDAR_HELPS_DB = 1.0
SOFT_HELPS_DB = 0.5
