# Battery channel provenance diagnostic

## 0. TL;DR

- **`battery_value` verdict**: **AUDIT THRESHOLD MISMATCH** — both reports are honest. The cleanup audit saw **662** unique values across the full 653,612-row telemetry timeline and so kept the column. The Phase 1 dataset analysis saw **1** unique value (65535) within its 681,593-row LiDAR-joined scope. The mechanism is **coverage-window narrowing**: the 1,605 non-sentinel rows all live in a ~6-minute pre-LiDAR window of session 25.02.2026 and never make it through the LiDAR-rate `merge_asof`.
- **`battery_cell_voltage` status**: **VARIABLE** in `telemetry_cleaned.parquet`; **NOT PRESENT** in the Phase 1 feature stack (dropped at the build_dataset projection step, not at the join).
- **`cumulative_energy_consumption` status**: **NOT a true accumulator** — values fluctuate around a slowly-drifting baseline (mostly ±1 between consecutive samples, with occasional larger jumps), even within a single CSV recording. Despite the channel name, it behaves more like a noisy battery-state reading than a monotonic energy counter.
- **Most-likely explanation**: Explanation B (audit threshold mismatch), but realised through coverage-window narrowing rather than amplitude rounding. No pipeline bug.
- **Recommended action**: **DOCUMENT AS-IS AND PROCEED**. The Phase 1 dataset is correct. The `battery_value` column should be dropped from the model feature stack on the next non-locked rebuild (it carries one bit of information). No re-run of the locked artefact is required: the lean-feature ablations already showed that removing `battery_value` from the stack gives byte-identical results.

## 1. Pipeline lineage

```
data/merged/telemetry_merged.parquet   ── 653,612 rows × 77 cols ──┐
        │ scripts/clean_telemetry.py                                │
        ▼                                                           │
data/merged/telemetry_cleaned.parquet  ── 653,612 rows × 35 cols ──┤
        │ scripts/time_sync/run_apply.py                            │
        │ (per-day τ̂-corrected merge_asof against LiDAR scans;      │
        │  retains telemetry rows that have a LiDAR partner within  │
        │  tolerance, drops the rest)                               │
        ▼                                                           │
data/merged/joint_coverage.parquet     ── 681,593 rows × 45 cols   │
        │ scripts/p1_dataset_analysis/build_dataset.py              │
        ▼                                                           │
data/phase1/dataset.parquet            ── 681,593 rows × 44 cols ──┘
```

`telemetry_merged.parquet` is the file the cleanup audit reports as
`telemetry.parquet`; this is the only filename mismatch in the lineage and
appears to be a labelling difference in the audit report, not a separate
file. Stage row counts: merged=653,612,
cleaned=653,612, phase1=681,593.

The 681,593 vs 653,612 row count difference reflects the LiDAR-rate join
(each telemetry row matches multiple LiDAR scans). The `joint_coverage`
intermediate stage is where channel **values** are first joined onto
LiDAR-rate rows; `build_dataset.py` then projects a subset of columns into
the Phase 1 schema and drops the rest. `battery_cell_voltage`,
`cumulative_energy_consumption`, and `cumulative_energy_consumption_uint`
fall in the dropped subset.

## 2. Per-channel per-stage statistics

See [tables/per_stage_summary.md](tables/per_stage_summary.md). The
salient row, in compact form:

| stage | battery_value nunique | NaN | min | max |
|---|---:|---:|---:|---:|
| merged | 662 | 0 | 42732 | 65535 |
| cleaned | 662 | 0 | 42732 | 65535 |
| phase1 | 1 | 0 | 65535 | 65535 |

The collapse from 662 → 1 happens at the
LiDAR-rate join, not at the cleanup step.

Histograms per stage: `figures/hist_<channel>_<stage>.png` (15 files).
Per-session overlay for `battery_value`:
[figures/battery_value_per_session_overlay.png](figures/battery_value_per_session_overlay.png).

## 3. Cross-stage value-identity check

See [tables/value_identity.md](tables/value_identity.md).

- **merged → cleaned**: every channel that survives the cleanup is
  byte-identical at every joined key. The cleanup script does not modify
  values; it only drops 40 fully-constant columns and renames the timestamp.
- **cleaned → phase1**: for each Phase 1 row, the joined `battery_value`
  matches the source `telemetry_cleaned` row at that
  `(session_date, fh7000_timestamp)` exactly. Within each phase1 group
  (one telemetry timestamp × multiple LiDAR scans), `battery_value` is
  constant. The same is true for `momentary_current_consumption`.
  `battery_cell_voltage`, `cumulative_energy_consumption` and
  `cumulative_energy_consumption_uint` are still present in
  `data/merged/joint_coverage.parquet` (the LiDAR-rate intermediate) but
  are dropped at the column-projection step in
  `scripts/p1_dataset_analysis/build_dataset.py` — their absence in
  `dataset.parquet` is a deliberate schema decision, not corruption.

## 4. Channel time-series and physical interpretation

Time-series figures: `figures/timeseries_*.png` (12 files; 4 channels × 3 sessions).

- **`battery_value`**: pegged at 65535 (= 0xFFFF, the uint16 maximum) for
  the entire LiDAR-coverage window of every session. The only non-sentinel
  values are in the 6-minute warm-up window at the start of session
  25.02.2026 (2026-02-25 15:03:44.945000 → 2026-02-25 15:09:36.799000), where the channel ramps from
  ~42,732 up to 65,535 and then locks at the sentinel. This is the
  signature of a CAN-bus channel that publishes a real reading until a
  state changes (e.g. firmware-side disable, service-mode toggle, sensor
  watchdog) and then reverts to the "no value" sentinel for the rest of
  the recording.
- **`battery_cell_voltage`**: variable across all sessions; full range
  77,392 to 1,024,391 in raw units (no decoding factor applied — likely
  millivolts × 1000 or raw ADC ticks). Worth a one-paragraph note: this
  channel exists in `telemetry_cleaned.parquet` but is not in the Phase 1
  feature stack, so the existing leakage investigation does not need to
  consider it.
- **`cumulative_energy_consumption` and `_uint`**: despite the names, these
  are **not** strictly monotonic counters. Within a single CSV recording
  the consecutive diff is ±1 about 18% of the time (each direction) and 0
  the rest of the time, with rare large excursions. The values fluctuate
  around a baseline that drifts on the scale of thousands of samples, with
  per-session ranges of ~3,500 over 44k–395k rows. The behaviour is
  consistent with a noisy battery-state-of-charge or voltage-level reading
  rather than a true accumulator. Even so, **the slow drift is enough to
  carry session-time information** (e.g. signal_power could be predicted
  in part from the slowly-drifting battery level), so excluding them from
  the model feature stack is still the right call. They are not in the
  Phase 1 feature stack.

### Monotonicity of cumulative-energy channels (within each run_file)

  - `cumulative_energy_consumption` / 25.02.2026: 1 run_file(s), n=44,040, per-run total n_decreasing=7994, per-run max_drop=-1818, all runs monotonic non-decreasing = NO
  - `cumulative_energy_consumption` / 15.03.2026: 4 run_file(s), n=395,395, per-run total n_decreasing=7690, per-run max_drop=-2799, all runs monotonic non-decreasing = NO
  - `cumulative_energy_consumption` / 24.03.2026: 3 run_file(s), n=214,177, per-run total n_decreasing=4380, per-run max_drop=-2605, all runs monotonic non-decreasing = NO
  - `cumulative_energy_consumption_uint` / 25.02.2026: 1 run_file(s), n=44,040, per-run total n_decreasing=8482, per-run max_drop=-722, all runs monotonic non-decreasing = NO
  - `cumulative_energy_consumption_uint` / 15.03.2026: 4 run_file(s), n=395,395, per-run total n_decreasing=8761, per-run max_drop=-1172, all runs monotonic non-decreasing = NO
  - `cumulative_energy_consumption_uint` / 24.03.2026: 3 run_file(s), n=214,177, per-run total n_decreasing=4839, per-run max_drop=-1134, all runs monotonic non-decreasing = NO

All run_files monotonic non-decreasing across all (channel × session): NO.

### `_uint` vs `_uint_float` pair (§3.6)

These are **not** an integer/float cast of the same physical quantity.
Both round to integer values (cumulative_energy_consumption is integer:
yes; _uint is integer: yes), so the float dtype on disk is
incidental. Their per-session ranges differ:

| session | cumulative_energy_consumption | cumulative_energy_consumption_uint |
|---|---|---|
| 25.02.2026 | 4938 → 8533 (nu=1752) | 1852 → 3065 (nu=1050) |
| 15.03.2026 | 4789 → 8268 (nu=2009) | 2095 → 3722 (nu=1423) |
| 24.03.2026 | 4740 → 8232 (nu=1480) | 2181 → 3809 (nu=1246) |

The two channels appear to be paired counters from different bus-side
accumulators (one ~total-energy, one ~total-energy-since-power-on or
similar). Max abs difference between the two over all rows where both are
finite: 5954. This is a side curiosity; neither is
in the Phase 1 feature stack.

## 5. Verdict

Three explanations were on the table:

- **Explanation A (pipeline bug)** would require non-65535 rows in
  `telemetry_cleaned.parquet` whose telemetry timestamps DO appear in the
  Phase 1 dataset, and yet the Phase 1 value is 65535 instead of the cleaned
  value. The diagnostic finds **0** such rows
  out of **1,605** non-65535 cleaned rows. This rules
  out a pipeline overwrite/coercion as the cause.
- **Explanation B (audit threshold mismatch)** is the closest match. The
  cleanup audit is honest: in `telemetry_cleaned.parquet` the channel has
  **662** unique values across 3 sessions and
  ~653,612 rows (so the "drop columns with nunique == 1" rule legitimately
  retained it). The Phase 1 analysis is also honest: within the Phase 1
  dataset the channel has **1** unique value (65535).
- **Explanation C (NaN-rule mismatch)** does not apply: NaN counts are
  0 / 0 / 0 across the three
  stages, and the audit's tally lines up with 662 unique
  non-NaN values.

The mechanism behind Explanation B in this dataset is **coverage-window
narrowing by the LiDAR-rate join**, not amplitude rounding. All
**1,605** variable-battery rows live in session
**25.02.2026**, between **2026-02-25 15:03:44.945000** and **2026-02-25 15:09:36.799000** — a single
~6-minute window at the very start of that recording. The LiDAR file for
that session does not begin until ~3 minutes later, so when
`scripts/time_sync/run_apply.py` does the per-day τ̂-corrected
`merge_asof` against LiDAR scan timestamps, every variable-battery
telemetry row falls outside the LiDAR coverage window and is dropped.
The rows that survive the join all carry battery_value == 65535 — the
uint16 "no-value" sentinel from the AGV CAN bus — so by the time we read
`data/phase1/dataset.parquet`, the channel is genuinely constant within
its scope.


**Verdict: Explanation B (audit threshold mismatch), via coverage-window
narrowing.** Cited numbers: §2 shows
`battery_value.nunique` = 662 in `telemetry_cleaned.parquet`
and 1 in `data/phase1/dataset.parquet`. §3.5 shows
1,605 non-sentinel rows in `cleaned`, of which
0 reach Phase 1. §3.3 shows that on the
intersection of `(session_date, fh7000_timestamp)` keys, cleaned and
Phase 1 are byte-identical for `battery_value` (max abs diff = 0). The
constancy in Phase 1 is therefore **inherited from upstream constancy
within the join window**, not produced by the join.

## 6. Implications for the paper and for the leakage investigation

- The Phase 1 dataset and the locked Phase 1 results have **not** been
  trained on a corrupted feature. `battery_value` was constant in their
  entire training scope, so the model treated it as zero-information,
  and the lean-feature byte-equality results (lean-A vs lean-B) hold
  unchanged.
- The paper should describe `battery_value` as **effectively constant
  within the LiDAR-recording window** (a 0xFFFF sensor sentinel) rather
  than as a meaningful feature. On the next non-locked feature-stack
  rebuild, drop it.
- Audit follow-up: it is worth re-running the cleanup audit with a
  stricter rule — "drop columns whose `nunique` is 1 within each
  `session_date` AND within the LiDAR-coverage window" — to catch other
  channels that are technically variable in the raw merged file but
  effectively constant in the modelled regime. This diagnostic does not
  perform that audit, but the framework is in place.

## 7. Implications for `battery_cell_voltage` and `cumulative_energy_consumption`

- **`battery_cell_voltage`** has confirmed variability across every
  session (see §2 / `figures/timeseries_battery_cell_voltage_vs_current_*.png`).
  It correlates negatively with `momentary_current_consumption` in every
  session (visible in the dual-axis plots), as the voltage-sag hypothesis
  predicts under load. It is **not in the Phase 1 feature stack**, so the
  current leakage investigation does not need to incorporate it. A future
  feature-stack revision could include it as a physical predictor; the
  voltage-sag relationship would let it carry information about
  transient power draw beyond what `momentary_current_consumption`
  captures.
- **`cumulative_energy_consumption` (and its `_uint` companion)** are
  not strictly monotonic — see §4. They behave as noisy battery-state
  readings with slow drift, not as accumulators. The drift component
  alone could still leak some session-time information into a model
  (the value at minute 30 differs systematically from the value at
  minute 60), but they are not the strong session-time leak that the
  channel name might suggest. The current `build_dataset.py` projection
  correctly excludes both; do not promote them to the feature stack.

## 8. Reproducibility

- Wall-clock for end-to-end run: **40.3 s**.
- Reproduce: `python -m scripts.p1_battery_provenance.run_diagnostic`.
- Inputs (read-only): `data/merged/telemetry_merged.parquet`,
  `data/merged/telemetry_cleaned.parquet`, `data/phase1/dataset.parquet`.
- Outputs:
  - Code: `scripts/p1_battery_provenance/{__init__.py, run_diagnostic.py, plotting.py}`
  - Tables: `docs/p1_battery_provenance/tables/{per_stage_summary,value_identity,join_keys}.md`
  - Figures: `docs/p1_battery_provenance/figures/*.png`
  - Report: this file.
