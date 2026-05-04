"""Battery channel provenance diagnostic.

Resolves the contradiction between the cleanup audit
(`docs/TELEMETRY_CLEANUP.md`, which kept `battery_value`) and the Phase 1
dataset analysis (`docs/p1_dataset_analysis/report.md`, which reports
`battery_value` as constant 6.554e+04).

Run as a module:
    python -m scripts.p1_battery_provenance.run_diagnostic
"""

SEED = 20260502
