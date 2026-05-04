# Joint-key persistence check

## battery_value: per-telemetry-timestamp persistence into Phase 1

Sample of 100 random telemetry timestamps per session that are also present in the Phase 1 dataset. For each, verify all phase1 rows joined to that timestamp share the same battery_value, and that battery_value matches the value in `telemetry_cleaned`.

| session | sampled tel ts (in cleaned ∩ phase1) | phase1 group rows (mean) | within-group constant? | matches cleaned? | mismatches |
|---|---:|---:|---|---|---:|
| 25.02.2026 | 100 | 5.62 | YES | YES | 0 |
| 15.03.2026 | 100 | 1.02 | YES | YES | 0 |
| 24.03.2026 | 100 | 1.00 | YES | YES | 0 |

## Coverage check: do the variable (≠65535) battery_value rows reach Phase 1?

| session | n rows ≠65535 in cleaned | tel-ts range of variable rows | n of those rows present in phase1 (by ts) |
|---|---:|---|---:|
| 25.02.2026 | 1,605 | 2026-02-25 15:03:44.945000 … 2026-02-25 15:09:36.799000 | 0 |
| 15.03.2026 | 0 | — | 0 |
| 24.03.2026 | 0 | — | 0 |