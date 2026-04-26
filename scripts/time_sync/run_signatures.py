"""Phase 1 + Phase 2: enumerate calibration days and build per-day signatures.

Outputs (idempotent — re-running uses the cache):
    cache/days.json                       per-session_date metadata
    cache/lidar_sig_<session_date>.parquet  per-day LiDAR scan-to-scan dissimilarity
                                          (reused from obsolete cache when present)
    cache/sig_day_<session_date>.npz      50 Hz z-scored, gap-masked per-day signatures
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config as C
from . import lib


def main():
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[Phase 1] Enumerating calibration days ...")
    days, telemetry, lidar_index = lib.enumerate_days(verbose=True)
    days_path = C.CACHE_DIR / "days.json"
    days_path.write_text(json.dumps([d.to_jsonable() for d in days], indent=2))
    print(f"[Phase 1] Wrote {days_path}  ({len(days)} days)")

    print("\n[Phase 2a] Building / reusing LiDAR dissimilarity per day ...")
    lib.build_lidar_dissim_cache(verbose=True)

    print("\n[Phase 2b] Building per-day telemetry/LiDAR signatures (50 Hz, z-scored, "
          "gap-masked) ...")
    for meta in days:
        out_path = C.CACHE_DIR / f"sig_day_{meta.session_date}.npz"
        if out_path.exists():
            print(f"  [skip] {out_path.name}")
            continue
        if not meta.eligible_for_estimation:
            print(f"  [skip] {meta.session_date}: {meta.eligibility_reason}")
            continue
        lid_path = C.CACHE_DIR / f"lidar_sig_{meta.session_date}.parquet"
        lidar_sig = pd.read_parquet(lid_path)
        sig = lib.make_day_signature(meta, telemetry, lidar_sig)
        if not sig:
            print(f"  [empty] {meta.session_date}: signature unavailable")
            continue
        np.savez_compressed(
            out_path,
            t_grid=sig["t_grid"],
            s_t_raw=sig["s_t_raw"],
            s_l_raw=sig["s_l_raw"],
            z_t=sig["z_t"],
            z_l=sig["z_l"],
            gap_mask=sig["gap_mask"],
        )
        masked_pct = 100.0 * sig["n_grid_masked"] / max(sig["n_grid_total"], 1)
        print(f"  [save] {out_path.name}  ({sig['n_grid_total']:,} grid samples, "
              f"{sig['n_grid_masked']:,} masked = {masked_pct:.2f}%)")

    print("\n[Phase 1+2] DONE")


if __name__ == "__main__":
    main()
