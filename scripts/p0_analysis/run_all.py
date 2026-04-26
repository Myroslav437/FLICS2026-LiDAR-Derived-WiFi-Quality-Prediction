"""Driver: run all Phase 0 tasks in dependency order, then write the report.

    .venv/Scripts/python.exe -m scripts.p0_analysis.run_all
"""
from __future__ import annotations
import os
import sys
import time

# Force UTF-8 stdio so the per-task prints with arrow chars don't crash on
# Windows cp1250 consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from . import run_p0_7, run_p0_3, run_p0_0, run_p0_2, run_p0_1, run_p0_4
from . import run_p0_5, run_p0_6, lidar_features, report_writer


TASKS = [
    ("P0.7  anomaly detection",         run_p0_7.main),
    ("P0.3  AGV-body mask",             run_p0_3.main),
    ("P0.0  frame-sharing verify",      run_p0_0.main),
    ("P0.2  AP path-loss fit",          run_p0_2.main),
    ("P0.1  spatial overlap",           run_p0_1.main),
    ("LiDAR scalar feature cache",      lambda: lidar_features.compute_and_cache(force=False)),
    ("P0.4  LiDAR<->residual corr",     run_p0_4.main),
    ("P0.5  semivariogram",             run_p0_5.main),
    ("P0.6  same-cell consistency",     run_p0_6.main),
    ("Report writer",                   report_writer.main),
]


def main():
    t_start = time.time()
    timings: dict[str, float] = {}
    for label, fn in TASKS:
        print(f"\n>>> {label}")
        t0 = time.time()
        fn()
        dt = time.time() - t0
        timings[label] = dt
        print(f"    ({dt:.1f}s)")
    print(f"\nALL DONE in {time.time() - t_start:.1f}s")
    print("Per-task wall time:")
    for k, v in timings.items():
        print(f"  {v:6.1f}s  {k}")
    # Save the timing table for the report
    import json
    from . import config as C
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (C.CACHE_DIR / "run_all_timing.json").write_text(json.dumps({
        "total_s": time.time() - t_start,
        "per_task_s": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
