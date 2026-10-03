"""Minimal, self-contained example: read and use the raw AGV LiDAR ``.h5`` captures.

Dependencies: ``numpy`` and ``h5py`` (``matplotlib`` only for ``--plot``).

    python read_lidar_h5.py data/24.03.2026/lidar_data_20260324_130319.h5
    python read_lidar_h5.py data/24.03.2026/lidar_data_20260324_130319.h5 --plot

Context
=======
Sensor      Leuze RSL 400 2D safety laser scanner, mounted front-facing on an
            industrial AGV (Navitrol / NNS navigation).
Field of    270 deg nominal.  About 240 beams at the two ends of the sweep hit
view        the AGV's own body (readings ~80-200 mm), leaving a usable sector
            of about 222 deg, roughly -112 deg ... +110 deg (angles defined below).
Angular     0.2 deg.  The scanner indexes its sweep in 0.1 deg steps and was
resolution  configured to output every 2nd index (``index_interval == 2``).
            This gives 1351 beams per scan covering -135.0 ... +135.0 deg.
Scan rate   ~25 Hz (40 ms scan period).  The measured median interval between
            stored scans is 39.6-39.8 ms (25.1-25.2 Hz).
Range       Distances are integer millimetres.  Valid readings go from ~70 mm
            (AGV body) up to just under 60 m; the median is ~2-3 m and 99 %
            are below ~20-35 m, depending on the session and location.

Sessions (one file per recording session; Poland, CET = UTC+1):
    25.02.2026  lidar_data_20260225_151241.h5  248,471 scans  2.75 h
    15.03.2026  lidar_data_20260315_165958.h5  288,496 scans  3.19 h
    24.03.2026  lidar_data_20260324_130319.h5  181,713 scans  2.01 h (last scan is empty)

File schema
===========
N = number of rows (one row per scan).  All datasets are resizable along axis 0.

dataset              shape      dtype    meaning
-------------------  ---------  -------  ---------------------------------------------
distances            (N, 2700)  uint16   range per beam slot, mm (see "Beams" below);
                                         gzip level 4, chunked one scan per chunk
local_timestamps     (N,)       float64  logging-PC receive time, Unix epoch seconds
                                         in UTC.  Add 3600 s for Polish local time (CET).
device_scan_counter  (N,)       uint32   scanner's scan counter; +1 per scan, a jump
                                         of 2 means one scan was dropped (rare: 0-3 per file)
device_scan_nr       (N,)       uint32   scanner's scan number; equal to the counter
                                         except for a handful of off-by-one rows
start_index          (N,)*      int16    scanner output window, always 0
stop_index           (N,)*      int16    scanner output window, always 2699
index_interval       (N,)*      int16    output every n-th 0.1-deg index, always 2

File attributes: ``created`` (logging start, local CET, "YYYYmmdd_HHMMSS"),
``max_points`` (= 2700, the row width).

* In the 24.03.2026 file these three datasets are one row shorter than the
  others; always truncate to the shortest dataset length.

Beams and angles
================
The row width (2700) was sized for the scanner's 0.1 deg mode.  At the 0.2 deg
resolution used here only the first 1351 slots carry data:

    slot i in [0, 1350]     beam i at angle  theta_i = +135 deg - 0.2 deg * i
    slot i in [1351, 2699]  padding, always 0

so slot 0 is +135 deg, slot 675 is straight ahead (0 deg), and slot 1350 is -135 deg.

theta is measured in the AGV body frame: x points forward, and positive angles
rotate in the same direction as the AGV's ``heading_rad`` from the navigation
system (to the left in a standard right-handed map).  A return at range r
therefore sits at (r cos theta, r sin theta) in the AGV frame, and at
(x + r cos(h + theta), y + r sin(h + theta)) in the map frame for an AGV pose
(x, y, h).  This sign convention was checked against the AGV's odometry by
aligning pairs of scans before and after in-place rotations.  The scanner's
mounting offset from the AGV reference point is not recorded and is ignored here.

Special values in ``distances``
===============================
65535   no valid measurement (no echo / out of range): ~1-2 % of beam readings.
0       invalid: the padding value, plus rare isolated drop-outs on real beams
        (at most one beam per scan, in ~0.2-0.4 % of scans).
< 200   on the beams near the sweep ends these are the AGV's own body.

Gotchas
=======
1. Bad rows: the last row of the 24.03.2026 file was never completed
   (timestamp 0, counter 0, all-zero distances).  Keep rows with
   ``local_timestamps > 1e8``.
2. Uneven dataset lengths, see the * note above.
3. Read speed: ``distances`` is gzip-compressed one scan per chunk, so read
   contiguous blocks (``ds[a:b]``).  h5py fancy indexing (``ds[[i, j, k]]``)
   needs sorted, unique indices and is far slower per row.
4. Timestamps are taken on the logging PC at receive time and jitter by a few
   ms (more on 25.02.2026).  Use ``device_scan_counter`` for scan ordering and
   gap detection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

N_SLOTS = 2700            # row width of `distances`
N_BEAMS = 1351            # slots that carry data at 0.2 deg resolution
ANGLE_STEP_DEG = 0.2
ANGLE_FIRST_DEG = 135.0   # angle of slot 0; angles decrease with slot index
NO_RETURN = 65535         # "no valid measurement" marker
UTC_TO_CET_S = 3600.0     # all sessions were recorded in winter time (CET = UTC+1)
BODY_RANGE_MM = 200       # readings below this are the AGV's own body

PER_SCAN = ("local_timestamps", "device_scan_counter", "device_scan_nr",
            "start_index", "stop_index", "index_interval")


def beam_angles_deg() -> np.ndarray:
    """Angle of every beam slot 0..1350, in degrees (0 = forward, + = left)."""
    return ANGLE_FIRST_DEG - ANGLE_STEP_DEG * np.arange(N_BEAMS)


def read_metadata(path: str | Path) -> dict[str, np.ndarray]:
    """Read all per-scan 1-D datasets, truncated to a common length.

    Adds ``valid`` (bool mask of usable rows) and ``n_rows`` (common length).
    """
    with h5py.File(path, "r") as f:
        n = min(f[name].shape[0] for name in ("distances",) + PER_SCAN)
        meta = {name: f[name][:n] for name in PER_SCAN}
    meta["valid"] = meta["local_timestamps"] > 1e8
    meta["n_rows"] = np.int64(n)
    return meta


def iter_scans(path: str | Path, chunk: int = 5000
               ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(row_index, timestamps_utc, distances)`` for every valid scan, in blocks.

    ``distances`` has shape (n, 1351), dtype uint16, in mm; padding slots are dropped.
    Reading contiguous blocks keeps this fast (~10k scans/s).
    """
    meta = read_metadata(path)
    n, valid, ts = int(meta["n_rows"]), meta["valid"], meta["local_timestamps"]
    with h5py.File(path, "r") as f:
        ds = f["distances"]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            keep = valid[start:stop]
            if not keep.any():
                continue
            block = ds[start:stop, :N_BEAMS][keep]
            rows = np.arange(start, stop)[keep]
            yield rows, ts[start:stop][keep], block


def scan_to_points(distances: np.ndarray, min_range_mm: int = BODY_RANGE_MM) -> np.ndarray:
    """Convert one scan (1351 or 2700 values, mm) to (k, 2) points in metres, AGV frame.

    Drops no-return beams and readings closer than ``min_range_mm`` (AGV body).
    """
    d = np.asarray(distances)[:N_BEAMS].astype(np.float64)
    ok = (d != NO_RETURN) & (d >= max(min_range_mm, 1))
    theta = np.deg2rad(beam_angles_deg()[ok])
    r = d[ok] / 1000.0
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def to_map_frame(points: np.ndarray, x: float, y: float, heading_rad: float) -> np.ndarray:
    """Place AGV-frame points into the navigation map frame for pose (x, y, heading)."""
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x, y])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5", type=Path, help="raw LiDAR capture, e.g. lidar_data_*.h5")
    ap.add_argument("--plot", action="store_true", help="plot one scan (needs matplotlib)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        print(f"{args.h5.name}  attrs={dict(f.attrs)}")
        for name, ds in f.items():
            print(f"  {name:<20} {str(ds.shape):<14} {ds.dtype}")

    meta = read_metadata(args.h5)
    ts = meta["local_timestamps"][meta["valid"]]
    counter = meta["device_scan_counter"][meta["valid"]].astype(np.int64)
    dt = np.diff(ts)
    print(f"\nusable scans : {ts.size:,} of {int(meta['n_rows']):,} rows")
    print(f"time span    : {ts[-1] - ts[0]:.0f} s, from "
          f"{np.datetime64(int((ts[0] + UTC_TO_CET_S) * 1e3), 'ms')} CET")
    print(f"scan rate    : {1 / np.median(dt):.2f} Hz (median interval {np.median(dt) * 1e3:.1f} ms)")
    print(f"dropped scans: {int((np.diff(counter) - 1).clip(min=0).sum())}")
    print(f"output window: start={np.unique(meta['start_index'])}, "
          f"stop={np.unique(meta['stop_index'])}, interval={np.unique(meta['index_interval'])}")

    # Statistics over the first block of scans
    rows, block_ts, block = next(iter_scans(args.h5))
    print(f"\nfirst block  : {len(rows):,} scans, distances {block.shape} {block.dtype}")
    print(f"no-return    : {np.mean(block == NO_RETURN):.2%} of beam readings")
    valid = block[(block != NO_RETURN) & (block >= BODY_RANGE_MM)]
    print(f"valid ranges : median {np.median(valid) / 1000:.2f} m, "
          f"p99 {np.percentile(valid, 99) / 1000:.2f} m")

    scan = block[len(block) // 2]
    pts = scan_to_points(scan)
    print(f"example scan : row {rows[len(rows) // 2]}, {len(pts)} points; "
          f"forward beam (slot 675, {beam_angles_deg()[675]:+.1f} deg) = {scan[675]} mm")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pts[:, 1], pts[:, 0], s=1)   # left on the left, forward up
        ax.plot(0, 0, "r^", ms=10, label="AGV (facing up)")
        ax.invert_xaxis()                        # +y (left) drawn to the left
        ax.set_xlabel("y, left (m)")
        ax.set_ylabel("x, forward (m)")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{args.h5.name}, row {rows[len(rows) // 2]}")
        plt.show()


if __name__ == "__main__":
    main()
