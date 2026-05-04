"""Lean-A and Lean-B feature lists, plus the affected variant feature sets.

HISTORICAL. Pre-leakage-fix this module compared three telemetry stacks:
locked (full-8), lean-A (drops 4 weak features but keeps battery_value), and
lean-B (3-feature minimum). The leakage-fixed rerun made the lean-B stack the
canonical one (see MIGRATION_LOG.md), so the FULL8/LEAN_A vs LEAN_B comparison
no longer reflects active modelling — it is retained only as a methodological
precursor record. The `EXPECTED_COUNTS["lean_b"]` row continues to match the
current canonical telemetry; the `lean_a` and `full8` rows are now informational.
"""

from __future__ import annotations

from scripts.p1_project_a import config as a_config
from scripts.p1_project_b import config as b_config

PROJECT_ROOT = a_config.PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "p1_lean_features"
MODELS_DIR = ANALYSIS_DIR / "models"
CACHE_DIR = ANALYSIS_DIR / "cache"
RESULTS_DIR = ANALYSIS_DIR / "results"

DOCS_DIR = PROJECT_ROOT / "docs" / "p1_lean_features"
TABLES_DIR = DOCS_DIR / "tables"
REPORT_PATH = DOCS_DIR / "comparison_report.md"

for _d in (MODELS_DIR, CACHE_DIR, RESULTS_DIR, DOCS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---- Telemetry feature stacks ------------------------------------------
#
# Historical record of the three telemetry stacks compared in §3 of the
# (now-deleted) lean_features comparison report. Post leakage-fix, the
# canonical project telemetry is LEAN_B_TELEMETRY (see a_config.TELEMETRY_FEATURES).

# Full pre-leakage-fix stack: 3 leakage-fixed features + 5 leaked/sentinel
# features that were removed (load_long/mid/short, battery_value, nns_state).
FULL8_TELEMETRY: list[str] = [
    "speed_mps",
    "turn_rate",
    "load_long",
    "load_mid",
    "load_short",
    "battery_value",
    "momentary_current_consumption",
    "nns_state",
]

# Drops load_long / load_mid / load_short / nns_state.
LEAN_A_TELEMETRY: list[str] = [
    "speed_mps",
    "turn_rate",
    "battery_value",
    "momentary_current_consumption",
]

# Lean-A minus battery_value (constant across the dataset).
# This is now the canonical project telemetry (a_config.TELEMETRY_FEATURES).
LEAN_B_TELEMETRY: list[str] = list(a_config.TELEMETRY_FEATURES)


def _telem_for(variant_set: str) -> list[str]:
    if variant_set == "lean_a":
        return list(LEAN_A_TELEMETRY)
    if variant_set == "lean_b":
        return list(LEAN_B_TELEMETRY)
    if variant_set == "full8":
        return list(FULL8_TELEMETRY)
    raise ValueError(f"unknown variant_set {variant_set!r}; expected full8, lean_a, or lean_b")


# ---- Project A variant sets (B4, B5, B5', B5'') ------------------------


def project_a_variant_features(variant_set: str) -> dict[str, list[str]]:
    """Lean B4/B5/B5'/B5'' feature lists for variant_set ∈ {lean_A, lean_B}."""
    telem = _telem_for(variant_set)
    b3 = list(a_config.B3)  # B1 + LiDAR scalar + LiDAR sectoral; no telemetry
    b4 = b3 + telem
    b5 = b4 + ["clutter_frac_toward_AP", "is_AP_in_FOV"]
    b5p = telem + list(a_config.AP_RELATIVE_FEATURES)
    b5pp = telem + list(a_config.LIDAR_SCALAR_FEATURES) + list(a_config.LIDAR_SECTORAL_FEATURES)
    return {
        "B4": b4,
        "B5": b5,
        "B5p": b5p,
        "B5pp": b5pp,
    }


# ---- Project B variant sets (W1..W4, W4'') -----------------------------


def project_b_variant_features(variant_set: str) -> dict[str, list[str]]:
    """Lean W1/W2/W3/W4/W4'' feature lists for variant_set ∈ {lean_A, lean_B}."""
    telem = _telem_for(variant_set)
    pos = list(b_config.POSITION_FEATURES)
    w1 = pos + telem
    w2 = w1 + list(b_config.AP_RELATIVE_FEATURES)
    w3 = w2 + list(b_config.LIDAR_SCALAR_FEATURES)
    w4 = w3 + list(b_config.LIDAR_SECTORAL_FEATURES)
    w4pp = pos + telem + list(b_config.LIDAR_SCALAR_FEATURES) + list(b_config.LIDAR_SECTORAL_FEATURES)
    return {
        "W1": w1,
        "W2": w2,
        "W3": w3,
        "W4": w4,
        "W4pp": w4pp,
    }


# ---- Sanity check expected feature counts ------------------------------

EXPECTED_COUNTS = {
    "lean_a": {
        "B4": 26, "B5": 28, "B5p": 9, "B5pp": 23,
        "W1": 6, "W2": 11, "W3": 16, "W4": 30, "W4pp": 25,
    },
    "lean_b": {
        "B4": 25, "B5": 27, "B5p": 8, "B5pp": 22,
        "W1": 5, "W2": 10, "W3": 15, "W4": 29, "W4pp": 24,
    },
}


def verify_counts() -> None:
    # Post-leakage-fix only the lean_b stack reflects the canonical telemetry;
    # the lean_a row is informational and we don't enforce its counts here.
    a_feats = project_a_variant_features("lean_b")
    b_feats = project_b_variant_features("lean_b")
    for v, feats in {**a_feats, **b_feats}.items():
        expected = EXPECTED_COUNTS["lean_b"][v]
        assert len(feats) == expected, (
            f"lean_b/{v}: expected {expected} features, got {len(feats)}: {feats}"
        )


verify_counts()
