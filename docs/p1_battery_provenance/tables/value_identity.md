# Cross-stage value-identity check

## merged → cleaned (per-row, joined on (session_date, fh7000_timestamp))

| channel | rows compared | NaN-NaN matches | max abs diff (finite) | mismatches (>1e-9) | identical |
|---|---:|---:|---:|---:|---|
| battery_value | 653,702 | 0 | 0 | 0 | OK (identical) |
| battery_cell_voltage | 653,702 | 0 | 0 | 0 | OK (identical) |
| cumulative_energy_consumption | 653,702 | 0 | 0 | 0 | OK (identical) |
| cumulative_energy_consumption_uint | 653,702 | 0 | 0 | 0 | OK (identical) |
| momentary_current_consumption | 653,702 | 0 | 0 | 0 | OK (identical) |

## cleaned → phase1 (per-telemetry-timestamp, max-min within each phase1 group must be zero)

| channel | telemetry rows in cleaned (all sessions) | telemetry rows joined into phase1 | per-group max−min (sup) | constant within group | max abs diff (cleaned vs phase1 first match) |
|---|---:|---:|---:|---|---:|
| battery_value | 653,612 | 489,539 | 0 | OK (yes) | 0 |
| battery_cell_voltage | 653,612 | 0 | n/a | channel not present in phase1 (dropped at column-projection in build_dataset.py) | n/a |
| cumulative_energy_consumption | 653,612 | 0 | n/a | channel not present in phase1 (dropped at column-projection in build_dataset.py) | n/a |
| cumulative_energy_consumption_uint | 653,612 | 0 | n/a | channel not present in phase1 (dropped at column-projection in build_dataset.py) | n/a |
| momentary_current_consumption | 653,612 | 489,539 | 0 | OK (yes) | 0 |