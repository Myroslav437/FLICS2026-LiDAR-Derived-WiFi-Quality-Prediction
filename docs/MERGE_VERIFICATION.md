# Merged dataset verification report

Generated: 2026-04-25 21:15:29

Sources:
- merged telemetry: `data/merged/telemetry.parquet` (8.7 MB, 653,612 rows × 77 cols)
- merged lidar:     `data/merged/lidar.h5` (1685.2 MB, 718,679 scans)

Verification samples per source: 200 CSV rows, 100 LiDAR scans (deterministic RNG seed = 20260425).

---

## Telemetry verification

- (B) Column count: **77** (expected ≥ 77)
- (B) Missing expected columns: none
- (B) Unexpected extra columns: none
- (B) Anonymous c<i> columns remaining: none ✓

### (A) Row counts vs originals

| Session | File | Raw lines | 73-col | 63-col | Other (dropped) | Expected | Merged | Match |
|---|---|---|---|---|---|---|---|---|
| 25.02.2026 | out_Myroslav_25-02_2026.csv | 44,040 | 44,040 | 0 | 0 | 44,040 | 44,040 | ✓ |
| 15.03.2026 | out_Myroslav_15-03_2026_1.csv | 173,047 | 172,982 | 60 | 5 | 173,042 | 173,042 | ✓ |
| 15.03.2026 | out_Myroslav_15-03_2026_2.csv | 125,633 | 125,617 | 11 | 5 | 125,628 | 125,628 | ✓ |
| 15.03.2026 | out_Myroslav_15-03_2026_3.csv | 35,191 | 35,191 | 0 | 0 | 35,191 | 35,191 | ✓ |
| 15.03.2026 | out_Myroslav_15-03_2026_4.csv | 61,535 | 61,534 | 0 | 1 | 61,534 | 61,534 | ✓ |
| 24.03.2026 | out_Myroslav_24-03_2026_1.csv | 41,399 | 41,399 | 0 | 0 | 41,399 | 41,399 | ✓ |
| 24.03.2026 | out_Myroslav_24-03_2026_2.csv | 35,540 | 35,540 | 0 | 0 | 35,540 | 35,540 | ✓ |
| 24.03.2026 | out_Myroslav_24-03_2026_3.csv | 137,238 | 137,238 | 0 | 0 | 137,238 | 137,238 | ✓ |

### (C) Value-level spot check

For each source CSV we draw 200 random row indices, read those rows from the *original* file, and compare every column against the merged dataset row that has the same `(run_file, ts_str)` key. Mismatches are logged below.

| Session | File | Samples | Mismatched cells | Mismatch % |
|---|---|---|---|---|
| 25.02.2026 | out_Myroslav_25-02_2026.csv | 200 | 0 | 0.0000% |
| 15.03.2026 | out_Myroslav_15-03_2026_1.csv | 200 | 0 | 0.0000% |
| 15.03.2026 | out_Myroslav_15-03_2026_2.csv | 200 | 0 | 0.0000% |
| 15.03.2026 | out_Myroslav_15-03_2026_3.csv | 200 | 0 | 0.0000% |
| 15.03.2026 | out_Myroslav_15-03_2026_4.csv | 200 | 0 | 0.0000% |
| 24.03.2026 | out_Myroslav_24-03_2026_1.csv | 200 | 0 | 0.0000% |
| 24.03.2026 | out_Myroslav_24-03_2026_2.csv | 200 | 0 | 0.0000% |
| 24.03.2026 | out_Myroslav_24-03_2026_3.csv | 200 | 0 | 0.0000% |

**Telemetry summary:** lost rows = **0**, value mismatches = **0**.

> The `Other (dropped)` column in the table above counts non-WiFi/non-telemetry physical lines that the loader explicitly skips: 1-column warm-up rows (timestamp only, no payload) and rare 72-column malformed rows (one trailing field missing, ≤2 per file). These have no telemetry information and are not considered data loss.

## LiDAR verification

- (B) Datasets present: ['device_scan_counter', 'device_scan_nr', 'distances', 'index_interval', 'local_timestamps', 'local_timestamps_unix_utc', 'run_file', 'session_date', 'start_index', 'stop_index']
- (B) Missing expected datasets: none ✓
- (B) Offset attribute: 3600 s (expected 3600)
- (B) +3600 s applied on every sampled scan: bad=0/1000 (expected 0)

### (A) Scan counts vs originals

| Session | Source h5 | Source rows (raw) | Source valid | Merged | Match |
|---|---|---|---|---|---|
| 25.02.2026 | lidar_data_20260225_151241.h5 | 248,471 | 248,471 | 248,471 | ✓ |
| 15.03.2026 | lidar_data_20260315_165958.h5 | 288,496 | 288,496 | 288,496 | ✓ |
| 24.03.2026 | lidar_data_20260324_130319.h5 | 181,713 | 181,712 | 181,712 | ✓ |

### (B) Per-scan session_date matches manifest segments

| Session | Range in merged | session_date all match | run_file all match |
|---|---|---|---|
| 25.02.2026 | [0:248,471) | ✓ | ✓ |
| 15.03.2026 | [248,471:536,967) | ✓ | ✓ |
| 24.03.2026 | [536,967:718,679) | ✓ | ✓ |

### (C) Per-scan value verification

For each session we sample 100 random scans, open the *original* h5 at the corresponding position and compare every numeric field. The merged `local_timestamps` must equal raw `local_timestamps + 3600`.

| Session | Samples | distance_mismatches | meta_mismatches | offset_mismatches |
|---|---|---|---|---|
| 25.02.2026 | 100 | 0 | 0 | 0 |
| 15.03.2026 | 100 | 0 | 0 | 0 |
| 24.03.2026 | 100 | 0 | 0 | 0 |

**LiDAR summary:** lost scans = **0**, value mismatches = **0**.

---

Total verification time: 20.6 s