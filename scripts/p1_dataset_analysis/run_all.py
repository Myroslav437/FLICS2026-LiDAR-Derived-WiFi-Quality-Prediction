"""Phase 1 dataset: build + validate in a single entry point.

Equivalent to::

    python -m scripts.p1_dataset_analysis.build_dataset
    python -m scripts.p1_dataset_analysis.validate_dataset

Exits non-zero (with the validator's exit code) if any hard validation check
fails; the report is written either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p1_dataset_analysis import build_dataset, validate_dataset  # noqa: E402


def main() -> None:
    print("\n" + "#" * 70)
    print("# Phase 1: build_dataset")
    print("#" * 70)
    build_dataset.main()

    print("\n" + "#" * 70)
    print("# Phase 1: validate_dataset")
    print("#" * 70)
    validate_dataset.main()


if __name__ == "__main__":
    main()
