"""Phase 0 v3 delta driver.

Re-runs the v3 update tasks in order:
    1. P0.7 v3  confidence-threshold-based anomaly mask
    2. P0.2 v3  truth-AP path-loss fit on v3-cleaned data
    3. AP-relative feature cache  (mask-independent; copied to v3 name)
    4. P0.4 v3  LiDAR <-> residual_v3 correlation + Gate C re-eval
    5. feature_extractor --validate  (uses v3 mask via canonical path)

Prerequisites: the v2 delta has already run (v2 caches and ap_coords.json
exist). The v3 driver does NOT regenerate v2 artifacts; it only adds the v3
generation alongside.

Run:
    .venv/Scripts/python.exe -m scripts.p0_analysis.run_delta_v3
"""
from __future__ import annotations

import os
import shutil
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from . import config as C
from . import run_p0_2_v3, run_p0_4_v3, run_p0_7_v3
from .artifacts import feature_extractor as FX


def main() -> None:
    t_start = time.time()
    timings: dict[str, float] = {}

    # 1. P0.7 v3 — confidence-threshold-based anomaly mask
    label = "P0.7 v3  confidence-threshold mask"
    t0 = time.time()
    print(f"\n>>> {label}")
    run_p0_7_v3.main()
    timings[label] = time.time() - t0

    # 2. P0.2 v3 — refit at fixed truth AP on v3-cleaned data
    label = "P0.2 v3  truth-AP path-loss fit"
    t0 = time.time()
    print(f"\n>>> {label}")
    run_p0_2_v3.main()
    timings[label] = time.time() - t0

    # 3. AP-relative feature cache (mask-independent values; we just snapshot
    #    the v2 cache under the v3 name so downstream consumers can pin to it)
    label = "AP-relative feature cache v3"
    t0 = time.time()
    print(f"\n>>> {label}")
    src = C.CACHE_DIR / "ap_relative_features_v2.parquet"
    dst = C.CACHE_DIR / "ap_relative_features_v3.parquet"
    shutil.copyfile(src, dst)
    print(f"  copied {src.name} -> {dst.name} (mask-independent values)")
    timings[label] = time.time() - t0

    # 4. P0.4 v3 — LiDAR <-> residual_v3 correlation + Gate C re-eval
    label = "P0.4 v3  LiDAR <-> residual_v3 corr"
    t0 = time.time()
    print(f"\n>>> {label}")
    run_p0_4_v3.main()
    timings[label] = time.time() - t0

    # 5. feature_extractor self-validate
    label = "feature_extractor --validate"
    t0 = time.time()
    print(f"\n>>> {label}")
    FX._validate()
    timings[label] = time.time() - t0

    total = time.time() - t_start
    print(f"\nDELTA v3 DONE in {total:.1f}s")
    for k, v in timings.items():
        print(f"  {v:6.1f}s  {k}")

    import json
    (C.CACHE_DIR / "run_delta_v3_timing.json").write_text(json.dumps({
        "total_s": total,
        "per_task_s": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
