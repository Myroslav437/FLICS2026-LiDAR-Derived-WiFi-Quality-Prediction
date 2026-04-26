"""Paths, constants, and per-session metadata for Phase 0."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

JOINT_PARQUET = ROOT / "data" / "merged" / "joint_coverage.parquet"
LIDAR_H5 = ROOT / "data" / "merged" / "lidar.h5"
TELEMETRY_CLEANED = ROOT / "data" / "merged" / "telemetry_cleaned.parquet"

P0_DIR = ROOT / "scripts" / "p0_analysis"
CACHE_DIR = P0_DIR / "cache"
ARTIFACTS_DIR = P0_DIR / "artifacts"

REPORT_DIR = ROOT / "docs" / "p0_analysis"
FIGURES_DIR = REPORT_DIR / "figures"

for d in (CACHE_DIR, ARTIFACTS_DIR, REPORT_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260427

SESSIONS = ["15.03.2026", "24.03.2026", "25.02.2026"]
SESSION_ORDER = SESSIONS  # alias

AP_PRIORS = {
    "15.03.2026": (2.0, 10.0),
    "24.03.2026": (2.0, 10.0),
    "25.02.2026": (-2.5, 0.5),
}

# Map A = 15.03 + 24.03 (assumed shared frame, verified by P0.0)
# Map B = 25.02 (different frame)
SAME_MAP_PAIR = ("15.03.2026", "24.03.2026")

# Beam->angle mapping.
#
# The h5 dataset stores `distances` with shape (n_scans, 2700) but only the
# first N_ACTIVE_BEAMS = 1350 slots carry returns; slots [1350, 2700) are
# buffer padding (always zero). The active beams cover a 270° FOV at
# 0.2° angular resolution, with theta_i = -135° + i * 0.2° for
# i in [0, 1350).
#
# Note on sensor capability: the Leuze RSL 400 safety LiDAR used here can be configured
# at up to 0.1° angular resolution per its datasheet (giving 2700 beams
# over 270°). **This dataset, however, was captured at 0.2° resolution**,
# producing 1350 beams per sweep. The h5 storage width was sized for the
# 0.1°-mode worst case (`max_points = 2700`), which is why the second half
# of every scan is always zero.
#
# This mapping was confirmed by inspection: per-beam zero-rate has a
# sharp transition at slot 1350 (slots 0-1349: ~99% valid; slots
# 1400-2699: 100% zero). Plotting one scan with the 0.2°-per-active-beam
# mapping reproduces the wide arc visible in the live LiDAR viewer; the
# 0.1°/2700 mapping squeezes the same returns into a 135° wedge, which
# contradicts the visual ground truth.
N_BEAMS = 2700           # storage width (sized for the sensor's 0.1° mode)
N_ACTIVE_BEAMS = 1350    # active beams produced at this dataset's 0.2° resolution
ANGLE_STEP_DEG = 0.2     # this dataset's resolution; sensor can do 0.1° in other modes
ANGLE_MIN_DEG = -135.0   # beam 0
ANGLE_MAX_DEG = -135.0 + (N_ACTIVE_BEAMS - 1) * ANGLE_STEP_DEG  # = +134.8°
def beam_angle_deg(i: int) -> float:
    return ANGLE_MIN_DEG + i * ANGLE_STEP_DEG

# Sentinel values in distances (uint16, mm)
INVALID_FAR = 65535  # no return
INVALID_NEAR = 0     # no return / blocked

# Cell size for spatial bins (P0.0, P0.1, P0.6)
CELL_SIZE_M = 0.5

# Chunk size for h5 batch reads
H5_CHUNK = 10000
