"""Configuration for the per-DAY time-synchronisation analysis.

Sign convention (used throughout):
    tau_hat > 0  =>  LiDAR clock lags telemetry clock
    Correction   =>  add tau_hat to every LiDAR timestamp.

A +3600 s timezone shift (UTC -> CET) is already baked into
``data/merged/lidar.h5::local_timestamps``. The offsets reported here
are the *residual* offsets on top of that timezone correction.

This pipeline treats each ``session_date`` (calibration day) as the unit
of analysis. The multiple CSV files within a day are concatenated and
sorted chronologically, then a single per-day signature is built. The
previous (obsolete) analysis treated each CSV file as an independent
unit and pooled per-file estimates; see ``docs/obsolete/...``.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- Inputs ----
TELEMETRY_PARQUET = ROOT / "data" / "merged" / "telemetry_cleaned.parquet"
LIDAR_H5 = ROOT / "data" / "merged" / "lidar.h5"
SESSION_INDEX_JSON = ROOT / "data" / "merged" / "session_index.json"

# Reuse cached LiDAR scan-to-scan dissimilarity from the obsolete analysis
# if present (it is purely a function of the H5 file, so the cache is
# valid). Falls back to recomputation when missing.
LIDAR_DISSIM_CACHE_LEGACY = (ROOT / "scripts" / "obsolete" / "time_sync_per_run"
                             / "cache")

# ---- Outputs ----
OUT_DIR = ROOT / "scripts" / "time_sync"
CACHE_DIR = OUT_DIR / "cache"
FIG_DIR = OUT_DIR / "figures"
JOINT_PARQUET = ROOT / "data" / "merged" / "joint_coverage.parquet"
OFFSETS_CSV = OUT_DIR / "offsets_per_day.csv"

# Where the publication-grade report lives.
REPORT_DIR = ROOT / "docs" / "time_sync"
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT_MD = REPORT_DIR / "report.md"

# ---- Common time grid ----
GRID_HZ = 50.0
GRID_DT = 1.0 / GRID_HZ

# ---- Cross-correlation ----
# Physical prior on clock skew: sub-second to a few seconds. The AGV runs
# a near-periodic back-and-forth route (cycle ~25 s); the cross-correlation
# develops route-period sidelobes at multiples of the half-cycle that can
# exceed the true peak in magnitude. We cap the search at +/- 3 s to
# exclude those aliases. If a future dataset needs more, widen with care
# and re-inspect xcorr curves visually.
SEARCH_RANGE_S = 3.0
WIDEN_RANGE_S = 5.0

# ---- Bootstrap ----
BOOTSTRAP_B = 500
BLOCK_LEN_S = 10.0
RNG_SEED = 20260426

# ---- Quality gate ----
PROMINENCE_MIN = 0.20
SECOND_PEAK_MIN_LAG_S = 1.0

# ---- Motion threshold ----
MOTION_SPEED_THRESH = 0.05           # m/s
MIN_MOTION_FOR_ESTIMATION_S = 30.0

# ---- LiDAR signature ----
LIDAR_SENSOR_MAX_MM = 8250.0
LIDAR_INVALID_MARKER = 65535
LIDAR_LOAD_CHUNK = 4000

# ---- Within-day drift test ----
DRIFT_HALF_MIN_MOTION_S = 30.0

# ---- Telemetry signature ----
# Canonical primary is rot_aware = |speed_mps| + R * |omega|, the linear-
# speed magnitude of a point at distance R from the AGV rotation centre
# (i.e. an approximation of the LiDAR mount's own motion). Justified
# empirically and physically in the obsolete report; we keep the same
# canonical here so the per-run vs per-day comparison is apples-to-apples.
ROT_R_EFF = 0.25
PRIMARY_VARIANT = "rot_aware"
BRIEF_VARIANT = "speed_mps"

# Telemetry-gap handling: when concatenating per-day telemetry, real wall-
# clock gaps appear between adjacent CSV files (recorder rotation; up to
# ~90 s). On the 50 Hz resampled grid these gaps would otherwise be
# bridged by linear interpolation, injecting fake "smooth descent" that
# can bias the cross-correlation. We mask grid samples that fall inside a
# telemetry gap > GAP_MASK_S seconds; masked samples are zeroed (after
# z-scoring) so they contribute nothing to the FFT-based cross-correlation
# numerator.
GAP_MASK_S = 1.0

# ---- Homogeneity decision rule (across-DAY pool) ----
HOMOGENEITY_I2_MAX = 25.0            # %
HOMOGENEITY_P_MIN = 0.05

# ---- Reporting ----
N_OVERLAYS_PER_SESSION = 3
