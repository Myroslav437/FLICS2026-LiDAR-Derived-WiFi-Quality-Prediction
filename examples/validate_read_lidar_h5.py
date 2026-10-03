"""Validate ``read_lidar_h5.py`` and the facts in its docstring against this repo's data.

    python examples/validate_read_lidar_h5.py

Checks, per raw session file:
  * schema, attributes and scanner output window as documented;
  * a full ``iter_scans`` pass returns exactly the scans that the project's
    independent merge pipeline kept (``data/merged/lidar.h5``), with the same
    timestamps (+3600 s) and the same distances;
  * value semantics: padding slots, beam 1350, no-return marker, AGV-body beams,
    scan rate and dropped-scan counts.
Geometry (needs ``data/merged/joint_coverage.parquet`` for AGV poses):
  * beam angles rotate in the same direction as ``heading_rad`` and 0 deg is forward,
    tested by aligning scan pairs before and after AGV rotations / translations.

Exits non-zero if any check fails.  Takes ~2-3 minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_lidar_h5 as rl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MERGED_H5 = DATA / "merged" / "lidar.h5"
JOINT = DATA / "merged" / "joint_coverage.parquet"
SESSIONS = {  # session -> (raw file, documented scan count, documented dropped scans)
    "25.02.2026": ("lidar_data_20260225_151241.h5", 248_471, 3),
    "15.03.2026": ("lidar_data_20260315_165958.h5", 288_496, 0),
    "24.03.2026": ("lidar_data_20260324_130319.h5", 181_712, 2),
}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def sample_blocks(ds: h5py.Dataset, n_rows: int, size: int = 3000) -> np.ndarray:
    """Five contiguous blocks spread over the file (full 2700-slot rows)."""
    starts = np.linspace(0, n_rows - size, 5).astype(int)
    return np.concatenate([ds[s:s + size] for s in starts])


def validate_file(session: str, fname: str, n_expected: int, dropped_expected: int,
                  merged_index: dict) -> None:
    path = DATA / session / fname
    print(f"\n== {session}: {fname}")

    # --- schema ----------------------------------------------------------------
    with h5py.File(path, "r") as f:
        dtypes = {k: str(v.dtype) for k, v in f.items()}
        check("datasets and dtypes", dtypes == {
            "distances": "uint16", "local_timestamps": "float64",
            "device_scan_counter": "uint32", "device_scan_nr": "uint32",
            "start_index": "int16", "stop_index": "int16", "index_interval": "int16"},
            str(dtypes))
        check("distances width = 2700", f["distances"].shape[1] == rl.N_SLOTS)
        check("attrs created / max_points",
              set(f.attrs) == {"created", "max_points"} and int(f.attrs["max_points"]) == 2700)
        created = str(f.attrs["created"])
        ds = f["distances"]
        meta = rl.read_metadata(path)
        n_rows = int(meta["n_rows"])
        blocks = sample_blocks(ds, n_rows)

    ts = meta["local_timestamps"][meta["valid"]]
    first_cet = np.datetime64(int(ts[0] + rl.UTC_TO_CET_S), "s").item()
    check("'created' attr = first scan time in CET",
          abs((first_cet - first_cet.strptime(created, "%Y%m%d_%H%M%S")).total_seconds()) <= 2,
          f"created={created}, first scan {first_cet} CET")
    check("output window 0 / 2699 / 2",
          set(np.unique(meta["start_index"])) == {0}
          and set(np.unique(meta["stop_index"])) == {2699}
          and set(np.unique(meta["index_interval"])) == {2})

    # --- timing and counters -------------------------------------------------------
    dt_ms = np.median(np.diff(ts)) * 1e3
    check("scan rate ~25 Hz", 39.0 <= dt_ms <= 41.0, f"median interval {dt_ms:.2f} ms")
    counter = meta["device_scan_counter"][meta["valid"]].astype(np.int64)
    steps = np.diff(counter)
    check("counter increasing, documented dropped scans",
          steps.min() >= 1 and int((steps - 1).sum()) == dropped_expected,
          f"dropped={int((steps - 1).sum())}")
    nr = meta["device_scan_nr"][meta["valid"]].astype(np.int64)
    check("scan_nr == counter except off-by-one rows",
          np.abs(nr - counter).max() <= 1 and np.mean(nr == counter) > 0.9999)

    # --- full pass with iter_scans vs. the project's merged file -------------------
    merged = merged_index[session]
    rows_all, ts_all, n_seen = [], [], 0
    with h5py.File(MERGED_H5, "r") as mf:
        mts = mf["local_timestamps"][merged["start_row"]:merged["stop_row"]]
        mds = mf["distances"]
        check_at = {0, n_expected // 2 // 5000 * 5000, (n_expected - 1) // 5000 * 5000}
        same_dist = True
        for rows, block_ts, block in rl.iter_scans(path):
            if block.shape[1] != rl.N_BEAMS or block.dtype != np.uint16:
                same_dist = False
            if rows[0] in check_at:  # compare whole blocks at start / middle / end
                m0 = merged["start_row"] + n_seen
                same_dist &= np.array_equal(block, mds[m0:m0 + len(block), :rl.N_BEAMS])
            rows_all.append(rows)
            ts_all.append(block_ts)
            n_seen += len(rows)
    check("iter_scans count = documented = merged pipeline",
          n_seen == n_expected == merged["rows"], f"{n_seen:,} scans")
    check("timestamps + 3600 s = merged local_timestamps",
          np.array_equal(np.concatenate(ts_all) + rl.UTC_TO_CET_S, mts))
    check("distances = merged distances (start/middle/end blocks)", bool(same_dist))

    # --- value semantics (15,000 scans sampled across the file) ---------------------
    beams, pad = blocks[:, :rl.N_BEAMS], blocks[:, rl.N_BEAMS:]
    check("slots 1351..2699 always 0", not pad.any())
    check("slot 1350 carries data", np.mean(beams[:, 1350] > 0) > 0.99)
    zeros_per_scan = (beams == 0).sum(axis=1)
    check("zeros on real beams: <= 1 per scan, in < 1 % of scans",
          zeros_per_scan.max() <= 1 and np.mean(zeros_per_scan > 0) < 0.01,
          f"{np.mean(zeros_per_scan > 0):.2%} of scans")
    no_ret = np.mean(beams == rl.NO_RETURN)
    check("no-return 65535 is ~1-2 %", 0.005 <= no_ret <= 0.03, f"{no_ret:.2%}")
    valid = beams[(beams != rl.NO_RETURN) & (beams != 0)]
    p50, p99 = np.percentile(valid, [50, 99]) / 1000
    check("valid range < 60 m, median ~2-3 m, p99 ~20-35 m",
          valid.max() < 60_000 and 1.8 <= p50 <= 3.5 and 18 <= p99 <= 36,
          f"min {valid.min()} mm, median {p50:.2f} m, p99 {p99:.1f} m, max {valid.max()} mm")
    body = np.median(np.where(beams == rl.NO_RETURN, 0, beams), axis=0) < rl.BODY_RANGE_MM
    usable = rl.beam_angles_deg()[~body]
    check("~240 AGV-body beams, usable sector ~ -112..+110 deg",
          abs(int(body.sum()) - 241) <= 10 and abs(usable.min() + 112) <= 2
          and abs(usable.max() - 110) <= 2,
          f"{int(body.sum())} body beams, usable {usable.min():+.1f}..{usable.max():+.1f} deg")


def validate_geometry(merged_index: dict) -> None:
    print("\n== Geometry: beam angles vs. AGV odometry")
    if not JOINT.exists():
        print(f"  [SKIP] {JOINT} not found")
        return
    import pandas as pd
    from scipy.spatial import cKDTree

    jp = pd.read_parquet(JOINT, columns=["session_date", "lidar_row", "heading_rad", "x_m", "y_m"])
    rng = np.random.default_rng(0)
    pairs = []  # (session, kind, raw_row_a, raw_row_b, pose_a, pose_b)
    for session, g in jp.groupby("session_date"):
        g = g.sort_values("lidar_row").drop_duplicates("lidar_row")
        r = g["lidar_row"].to_numpy() - merged_index[session]["start_row"]  # raw file row
        h = np.unwrap(g["heading_rad"].to_numpy())
        x, y = g["x_m"].to_numpy(), g["y_m"].to_numpy()
        j = np.searchsorted(r, r + 15)                  # partner scan 0.6 s later
        i = np.flatnonzero((j < len(r)) & (r[np.minimum(j, len(r) - 1)] == r + 15))
        j = j[i]
        dh, dd = np.rad2deg(h[j] - h[i]), np.hypot(x[j] - x[i], y[j] - y[i])
        for kind, sel in (("rotation", (np.abs(dh) > 4) & (np.abs(dh) < 25) & (dd < 0.15)),
                          ("translation", (np.abs(dh) < 1) & (dd > 0.2) & (dd < 0.6))):
            for k in rng.choice(np.flatnonzero(sel), min(40, sel.sum()), replace=False):
                a, b = i[k], j[k]
                pairs.append((session, kind, r[a], r[b], (x[a], y[a], h[a]), (x[b], y[b], h[b])))

    scans = {}
    for session in SESSIONS:
        rows = sorted({p[2] for p in pairs if p[0] == session} | {p[3] for p in pairs if p[0] == session})
        with h5py.File(DATA / session / SESSIONS[session][0], "r") as f:
            for row, d in zip(rows, f["distances"][rows, :rl.N_BEAMS]):
                scans[(session, row)] = d

    def residual(pair, transform) -> float:
        """Median distance (m) between scan A moved into B's frame by odometry, and scan B."""
        session, _, ra, rb, pa, pb = pair
        pts_a = transform(rl.scan_to_points(scans[(session, ra)]))
        pts_b = transform(rl.scan_to_points(scans[(session, rb)]))
        pts_a, pts_b = pts_a[np.hypot(*pts_a.T) < 15], pts_b[np.hypot(*pts_b.T) < 15]
        world = rl.to_map_frame(pts_a, *pa)
        c, s = np.cos(pb[2]), np.sin(pb[2])
        in_b = (world - pb[:2]) @ np.array([[c, -s], [s, c]])   # inverse of to_map_frame
        return float(np.median(cKDTree(pts_b).query(in_b)[0]))

    documented = lambda p: p                 # noqa: E731
    mirrored = lambda p: p * [1, -1]         # angles rotating against heading_rad
    backwards = lambda p: -p                 # 0 deg pointing backwards
    for kind, alt_name, alt in (("rotation", "mirrored", mirrored),
                                ("translation", "backwards", backwards)):
        sub = [p for p in pairs if p[1] == kind]
        doc = np.median([residual(p, documented) for p in sub])
        other = np.median([residual(p, alt) for p in sub])
        check(f"{kind} pairs: documented convention beats {alt_name}",
              doc < 0.5 * other, f"{len(sub)} pairs, median residual {doc:.3f} m vs {other:.3f} m")

    # Independent of scan_to_points: after the heading increases by dh, scan features
    # must move to higher slots by dh / 0.2 deg (angles decrease with slot index).
    def feature_shift_deg(a: np.ndarray, b: np.ndarray) -> float:
        la = np.log(np.where(a == rl.NO_RETURN, np.nan, np.maximum(a, 1.0)))
        lb = np.log(np.where(b == rl.NO_RETURN, np.nan, np.maximum(b, 1.0)))
        errs = {s: np.nanmedian(np.abs(la[150:1201] - lb[150 + s:1201 + s]))
                for s in range(-150, 151)}
        return rl.ANGLE_STEP_DEG * min(errs, key=errs.get)

    rot = [p for p in pairs if p[1] == "rotation"]
    dh = np.array([np.rad2deg(p[5][2] - p[4][2]) for p in rot])
    shift = np.array([feature_shift_deg(scans[(p[0], p[2])].astype(float),
                                        scans[(p[0], p[3])].astype(float)) for p in rot])
    slope = np.polyfit(dh, shift, 1)[0]
    agree = np.mean(np.sign(shift) == np.sign(dh))
    check("feature shift = +heading change (slope ~ +1)", 0.8 <= slope <= 1.2 and agree > 0.9,
          f"slope {slope:+.3f}, sign agreement {agree:.0%}")


def main() -> None:
    merged_index = json.loads((DATA / "merged" / "session_index.json").read_text())["lidar"]["sessions"]
    for session, (fname, n, dropped) in SESSIONS.items():
        validate_file(session, fname, n, dropped, merged_index)
    validate_geometry(merged_index)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
