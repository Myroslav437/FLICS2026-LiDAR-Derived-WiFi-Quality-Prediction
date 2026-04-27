"""Phase 0 delta driver — re-runs only the AP-coordinate-dependent steps.

Re-runs:
    1. P0.7 anomaly mask (regenerates if missing — AP-independent)
    2. P0.3 AGV-body mask + lidar_fov (regenerates if missing — AP-independent)
    3. P0.2 v2 path-loss fit at the lab-measured truth AP
    4. AP-relative feature cache (v2)
    5. P0.4 v2 LiDAR <-> residual_v2 correlations + Gate C re-evaluation
    6. feature_extractor --validate

Then re-renders docs/p0_analysis/report.md from the combined v1 + v2 artifacts.

Run:
    .venv/Scripts/python.exe -m scripts.p0_analysis.run_delta
"""
from __future__ import annotations
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from . import config as C
from . import run_p0_7, run_p0_3, run_p0_2_v2, run_p0_4_v2
from . import ap_relative_features as APR
from . import report_writer_v2
from .artifacts import feature_extractor as FX


def _need_artifact(path) -> bool:
    return not path.exists()


def main() -> None:
    t_start = time.time()
    timings: dict[str, float] = {}

    # 1. Anomaly mask (AP-independent; only re-run if missing)
    label = "P0.7 anomaly mask (if missing)"
    t0 = time.time()
    if _need_artifact(C.ARTIFACTS_DIR / "anomaly_mask.parquet"):
        print(f"\n>>> {label}: regenerating")
        run_p0_7.main()
    else:
        print(f"\n>>> {label}: present, skipping")
    timings[label] = time.time() - t0

    # 2. AGV-body mask + FOV (AP-independent; only re-run if missing)
    label = "P0.3 AGV-body mask + FOV (if missing)"
    t0 = time.time()
    if _need_artifact(C.ARTIFACTS_DIR / "agv_body_mask.npz") or \
       _need_artifact(C.ARTIFACTS_DIR / "lidar_fov.json"):
        print(f"\n>>> {label}: regenerating")
        run_p0_3.main()
    else:
        print(f"\n>>> {label}: present, skipping")
    timings[label] = time.time() - t0

    # 3. P0.2 v2 — refit at fixed truth AP
    label = "P0.2 v2  truth-AP path-loss fit"
    t0 = time.time()
    print(f"\n>>> {label}")
    run_p0_2_v2.main()
    timings[label] = time.time() - t0

    # 4. AP-relative features v2
    label = "AP-relative feature cache v2"
    t0 = time.time()
    print(f"\n>>> {label}")
    APR.compute_and_cache(force=True)
    timings[label] = time.time() - t0

    # 5. P0.4 v2
    label = "P0.4 v2  LiDAR <-> residual_v2 corr"
    t0 = time.time()
    print(f"\n>>> {label}")
    run_p0_4_v2.main()
    timings[label] = time.time() - t0

    # 6. feature_extractor self-validate
    label = "feature_extractor --validate"
    t0 = time.time()
    print(f"\n>>> {label}")
    FX._validate()
    timings[label] = time.time() - t0

    # 7. Report writer v2
    label = "Report writer v2"
    t0 = time.time()
    print(f"\n>>> {label}")
    report_writer_v2.main()
    timings[label] = time.time() - t0

    total = time.time() - t_start
    print(f"\nDELTA DONE in {total:.1f}s")
    for k, v in timings.items():
        print(f"  {v:6.1f}s  {k}")

    import json
    (C.CACHE_DIR / "run_delta_timing.json").write_text(json.dumps({
        "total_s": total,
        "per_task_s": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
