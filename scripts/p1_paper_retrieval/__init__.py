"""Paper-data retrieval over the leakage-fixed Phase 1 artefacts.

Read-only file inspection only; no model fitting, no SHAP recomputation,
no estimation. Items where leakage-fixed data is unavailable are reported
as MISSING; pre-leakage-fix fall-backs are never substituted.

Run: `python -m scripts.p1_paper_retrieval.run_retrieval`
"""
