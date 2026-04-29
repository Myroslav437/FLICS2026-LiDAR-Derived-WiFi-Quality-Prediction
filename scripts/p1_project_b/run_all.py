"""Phase 1 / Project B end-to-end driver: modeling -> SHAP -> R-4 diag -> hardening -> report."""

from __future__ import annotations

from . import build_results_report, run_hardening_all, run_modeling, run_r4_diagnostic, run_xai


def main() -> None:
    print("=== Project B: stage 1 — run_modeling ===")
    run_modeling.main()
    print("=== Project B: stage 2 — run_xai ===")
    run_xai.main()
    print("=== Project B: stage 3 — run_r4_diagnostic ===")
    run_r4_diagnostic.main()
    print("=== Project B: stage 4 — run_hardening_all (Hardening C, E) ===")
    run_hardening_all.main()
    print("=== Project B: stage 5 — build_results_report ===")
    build_results_report.main()
    print("=== Project B complete ===")


if __name__ == "__main__":
    main()
