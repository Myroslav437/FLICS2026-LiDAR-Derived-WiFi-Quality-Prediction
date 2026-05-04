"""Phase 1 lean-features rerun.

Tests whether removing post-hoc-confirmed-useless telemetry features
(load_long, load_mid, load_short, nns_state, optionally battery_value)
changes any paper-cited number from the locked Phase 1 experiments.

Two parallel feature stacks are tested:
  - Lean-A: 4 telemetry features (drops the 4 weak ones; keeps battery_value).
  - Lean-B: 3 telemetry features (also drops constant battery_value).

Locked artifacts under scripts/p1_project_a/, scripts/p1_project_b/ are NOT
modified. All new outputs go under scripts/p1_lean_features/ and
docs/p1_lean_features/.
"""
