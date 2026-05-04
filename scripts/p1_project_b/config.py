"""Project B configuration: within-session deep dive on 15.03.2026.

Imports and reuses Project A's config where possible (SEED, dataset paths,
feature lists, XGBoost hyperparameters) and adds Project-B-specific values.
"""

from __future__ import annotations

from pathlib import Path

from scripts.p1_project_a import config as a_config
from scripts.p1_project_a.config import (  # re-exports
    AP_RELATIVE_FEATURES,
    BOOTSTRAP_B,
    EARLY_STOPPING_ROUNDS,
    FEATURE_GROUPS as A_FEATURE_GROUPS,
    FEATURE_STACK_VERSION,
    LEAKAGE_FIXED_TELEMETRY,
    LIDAR_SCALAR_FEATURES,
    LIDAR_SECTORAL_FEATURES,
    N_ESTIMATORS,
    PROVENANCE_COLS,
    SEED,
    TARGET,
    TELEMETRY_FEATURES,
    XGB_PARAMS,
)

PROJECT_ROOT = a_config.PROJECT_ROOT
DATASET_PATH = a_config.DATASET_PATH
DATASET_SHA_PATH = a_config.DATASET_SHA_PATH

# Project B owns its own outputs.
ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "p1_project_b"
MODELS_DIR = ANALYSIS_DIR / "models"
CACHE_DIR = ANALYSIS_DIR / "cache"
RESULTS_DIR = ANALYSIS_DIR / "results"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"

DOCS_DIR = PROJECT_ROOT / "docs" / "p1_project_b"
FIGURES_DIR = DOCS_DIR / "figures"
TABLES_DIR = DOCS_DIR / "tables"
REPORT_PATH = DOCS_DIR / "results_report.md"

for _d in (MODELS_DIR, CACHE_DIR, RESULTS_DIR, ARTIFACTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---- Project B specific constants --------------------------------------

SESSION = "15.03.2026"

POSITION_FEATURES = ["x_m", "y_m"]

# Model-input variants (full feature universe = 34: 2 pos + 8 telem + 5 AP-rel + 19 LiDAR).
W0 = list(POSITION_FEATURES)
W1 = W0 + list(TELEMETRY_FEATURES)
W2 = W1 + list(AP_RELATIVE_FEATURES)
W3 = W2 + list(LIDAR_SCALAR_FEATURES)
W4 = W3 + list(LIDAR_SECTORAL_FEATURES)
# W4' (W4 minus LiDAR) is feature-identical to W2; we reuse W2 fits and label them as W4' for the
# disambiguation table — we do NOT define a separate W4p here to enforce the no-refit rule.
W4_DBLPRIME = W0 + list(TELEMETRY_FEATURES) + list(LIDAR_SCALAR_FEATURES) + list(LIDAR_SECTORAL_FEATURES)

VARIANTS: dict[str, list[str]] = {
    "W0": W0,
    "W1": W1,
    "W2": W2,
    "W3": W3,
    "W4": W4,
    "W4pp": W4_DBLPRIME,
}

MAIN_VARIANTS: list[str] = ["W0", "W1", "W2", "W3", "W4", "W4pp"]

# Disambiguation labels (W4p ≡ W2; reuse the same predictions).
DISAMBIG_LABELS = {"W4": "W4", "W2": "W4'", "W4pp": "W4''"}

# Feature groups for SHAP aggregation. Position is its own group in Project B.
FEATURE_GROUPS = {
    "Position": list(POSITION_FEATURES),
    "Telemetry": list(TELEMETRY_FEATURES),
    "AP-relative": list(AP_RELATIVE_FEATURES),
    "LiDAR scalar": list(LIDAR_SCALAR_FEATURES),
    "LiDAR sectoral": list(LIDAR_SECTORAL_FEATURES),
}

# Region-construction parameters.
N_REGIONS = 5
REGION_SUBSAMPLE_N = 50_000  # rows used for K-means fit
REGION_MIN_ROWS = 5_000      # quality floor; fall back to PCA-trajectory partition below this

FOLD_NAMES = [f"R-{i}" for i in range(1, N_REGIONS + 1)]
FOLD_TO_REGION = {f"R-{i}": i for i in range(1, N_REGIONS + 1)}

# Within-session validation: random 10% (IID with train across regions).
VAL_FRACTION = 0.10

# Buffer-zone sensitivity (R-1 only).
BUFFER_FOLD = "R-1"
BUFFER_DISTANCE_M = 1.0

# Hyperparameter sensitivity check (W4 only, all 5 folds).
HYPERPARAMS_LOCKED = dict(XGB_PARAMS)  # locked Project A choice
HYPERPARAMS_H1 = {**XGB_PARAMS, "max_depth": 4}  # H1 alternative from the diagnostic
HYPERPARAM_LABELS = {"locked": HYPERPARAMS_LOCKED, "H1": HYPERPARAMS_H1}

SHAP_FOLD_DEFAULT = "R-3"
SHAP_SUBSAMPLE_LIMIT = 50_000


# ---- Hardening experiments (C, E) -------------------------------------
#
# Originally lived under scripts/p1_hardening/config.py. Merged in 2026-04-28
# (see scripts/maintenance/merge_hardening_log.md). The XGB hyperparameter
# objects already exist as HYPERPARAMS_* above; mirror them under the names
# the moved runners use, then add the verdict thresholds.

XGB_LOCKED = dict(HYPERPARAMS_LOCKED)
XGB_H1 = dict(HYPERPARAMS_H1)
XGB_LABELS = {"locked": XGB_LOCKED, "H1": XGB_H1}

# Re-export of LIDAR_COLUMNS from Project A so the placebo helper sees the same
# 19 ego-frame columns when imported by Project B's hardening runner.
LIDAR_COLUMNS = list(a_config.LIDAR_COLUMNS)

PLACEBO_TOLERANCE_DB = 0.5
SOFT_HELPS_DB = 0.5

# Verdict thresholds (§10–11 of brief).
DELTA_LIDAR_THRESHOLD_DB = 1.0
WITHIN_HELPS_FOLDS_REQUIRED = 3  # of 5
BUFFER_RMSE_DELTA_THRESHOLD_DB = 0.5
HYPERPARAM_RMSE_DELTA_THRESHOLD_DB = 0.3
