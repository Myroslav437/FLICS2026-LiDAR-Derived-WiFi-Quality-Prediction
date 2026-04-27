"""Phase 1 / Project A end-to-end driver: modeling → SHAP → robustness → report."""

from __future__ import annotations

from . import build_results_report, run_modeling, run_robustness, run_xai


def main() -> None:
    print("=== Project A: stage 1 — run_modeling ===")
    run_modeling.main()
    print("=== Project A: stage 2 — run_xai ===")
    run_xai.main()
    print("=== Project A: stage 3 — run_robustness ===")
    run_robustness.main()
    print("=== Project A: stage 4 — build_results_report ===")
    build_results_report.main()
    print("=== Project A complete ===")


if __name__ == "__main__":
    main()
