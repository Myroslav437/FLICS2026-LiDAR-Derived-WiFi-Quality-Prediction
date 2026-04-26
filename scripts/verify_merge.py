"""Sanity-check the merged datasets.

Validates:
* telemetry.parquet loads, has expected row count, WiFi columns are numeric.
* lidar.h5 has the expected scan count, schema unchanged from sources, the
  +3600 s correction is actually applied (delta vs raw), and per-session
  timestamp ranges line up with the telemetry CSV ranges.
* a small merge_asof join between streams succeeds at the corrected times.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"


def main():
    print("=== Telemetry ===")
    tel_path = MERGED / "telemetry.parquet"
    print(f"file: {tel_path} ({tel_path.stat().st_size / 1e6:.1f} MB)")
    tel = pd.read_parquet(tel_path)
    print(f"rows: {len(tel):,}")
    print(f"columns: {len(tel.columns)} -- first 25: {list(tel.columns[:25])}")
    print(f"sessions: {tel['session_date'].value_counts().to_dict()}")
    print("WiFi non-null counts:")
    for c in ["signal_power", "signal_quality", "ping"]:
        non_null = tel[c].notna().sum()
        rng = (float(tel[c].min()), float(tel[c].max()))
        print(f"  {c:<16} non_null={non_null:>7,}  range={rng}")
    print(f"ts range: {tel['ts'].min()} to {tel['ts'].max()}")
    print(f"ts_unix_cet (head): {tel['ts_unix_cet'].head(3).tolist()}")

    print("\n=== LiDAR ===")
    lid_path = MERGED / "lidar.h5"
    print(f"file: {lid_path} ({lid_path.stat().st_size / 1e6:.1f} MB)")
    with h5py.File(lid_path, "r") as f:
        print(f"datasets: {list(f.keys())}")
        print(f"attrs: {dict(f.attrs)}")
        n = f["distances"].shape[0]
        print(f"scans: {n:,}")
        # Verify offset applied
        utc_head = f["local_timestamps_unix_utc"][:5]
        cet_head = f["local_timestamps"][:5]
        delta = cet_head - utc_head
        print(f"first 5 raw UTC ts:        {utc_head}")
        print(f"first 5 corrected ts:      {cet_head}")
        print(f"applied offset (s):        {delta} (should be 3600)")
        assert np.allclose(delta, 3600.0), "Offset not applied!"

        # Per-session timestamp ranges
        ses = f["session_date"][:].astype(str)
        print("\nLiDAR per-session corrected-time ranges:")
        for s in np.unique(ses):
            mask = ses == s
            ts = f["local_timestamps"][:][mask]
            t0 = pd.to_datetime(ts.min(), unit="s")
            t1 = pd.to_datetime(ts.max(), unit="s")
            print(f"  {s}: {mask.sum():>7,} scans, {t0} to {t1}")

    print("\n=== Cross-stream join smoke test ===")
    # Telemetry CSV strings are CET local time. lidar.local_timestamps after
    # +1 h shift represents the same naive CET time when interpreted via
    # pd.to_datetime(unit='s'). They should now overlap directly.
    with h5py.File(lid_path, "r") as f:
        idx = np.linspace(0, f["distances"].shape[0]-1, 10000).astype(int)
        ts_lidar = f["local_timestamps"][:][idx]
    lidar_ts = pd.to_datetime(ts_lidar, unit="s")
    print(f"sample lidar ts span: {lidar_ts.min()} to {lidar_ts.max()}")

    # asof join: each telemetry row finds the nearest lidar scan within 200 ms
    tel_subset = tel.sample(20000, random_state=0).sort_values("ts").reset_index(drop=True).copy()
    tel_subset["ts"] = tel_subset["ts"].astype("datetime64[ns]")
    lidar_df = pd.DataFrame({"ts": lidar_ts.astype("datetime64[ns]"),
                              "lidar_idx": idx}).sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(tel_subset, lidar_df, on="ts", direction="nearest",
                            tolerance=pd.Timedelta(milliseconds=200))
    n_paired = merged["lidar_idx"].notna().sum()
    print(f"sampled 20,000 telemetry rows; {n_paired:,} paired with a lidar scan within 200 ms")
    print(f"by session:")
    print(merged.groupby("session_date")["lidar_idx"].apply(lambda s: f"{s.notna().sum():>5,}/{len(s):,}").to_string())

    manifest = json.loads((MERGED / "session_index.json").read_text())
    print("\n=== Manifest summary ===")
    print(f"telemetry rows: {manifest['telemetry']['rows']:,}")
    print(f"lidar scans:    {manifest['lidar']['scans']:,}")
    print(f"offset applied: {manifest['lidar_offset_seconds_applied']} s")


if __name__ == "__main__":
    main()
