"""Telemetry leakage diagnostic for load_*, nns_state, battery_value.

No model fitting; descriptive analysis only. Reads
`data/phase1/dataset.parquet`, conditions on `anomaly_flag == False`, and
emits figures, tables, and a verdict report under `docs/p1_leakage_check/`.
"""

SEED = 20260427
