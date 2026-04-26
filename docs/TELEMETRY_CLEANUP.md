# Telemetry cleanup report

Generated: 2026-04-25 21:44:15
Input:  `data\merged\telemetry.parquet` (8.7 MB, 653,612 rows × 77 cols)

## 1. Redundant timestamp columns

- `ts_str` re-parsed and compared with `ts` element-wise.
- max abs(ts - parse(ts_str)) = **0 ns**, mismatches = **0**
- ⇒ `ts_str` is redundant — DROPPED.
- `ts_unix_cet` reproduced from `ts` and compared element-wise.
- max abs(ts_unix_cet - ts.timestamp) = **0.238 µs**, mismatches (>1 µs) = **0**
- ⇒ `ts_unix_cet` is redundant — DROPPED.

## 2. Rename `ts` to `fh7000_timestamp`

- `ts` (the FH_ID_7000.TimeStamp value, parsed as datetime64) renamed to `fh7000_timestamp`.

## 3. Constant columns (no information)

Identified **40** columns with a single unique (or all-NaN) value across all 653,612 rows. These have zero explanatory power and are dropped.

| Column | Dtype | Sole value |
|---|---|---|
| `reserve` | float64 | np.float64(0.0) + 71 NaN |
| `wifi_state_1` | float64 | np.float64(0.0) + 71 NaN |
| `brake_lock_command_on` | bool | np.False_ |
| `brake_lock_executed` | bool | np.False_ |
| `brake_lock_in_progress` | bool | np.False_ |
| `brake_release_executed` | bool | np.True_ |
| `brake_release_in_progress` | bool | np.False_ |
| `left_drive_activate_command_on` | bool | np.False_ |
| `left_drive_activate_enable_active` | bool | np.False_ |
| `left_drive_activate_executed` | bool | np.False_ |
| `left_drive_activate_in_progress` | bool | np.False_ |
| `left_drive_stop_automatic_permission` | bool | np.False_ |
| `left_drive_stop_command_on` | bool | np.False_ |
| `left_drive_stop_in_progress` | bool | np.False_ |
| `right_drive_activate_command_on` | bool | np.False_ |
| `right_drive_activate_enable_active` | bool | np.False_ |
| `right_drive_activate_executed` | bool | np.False_ |
| `right_drive_activate_in_progress` | bool | np.False_ |
| `right_drive_stop_command_on` | bool | np.False_ |
| `right_drive_stop_in_progress` | bool | np.False_ |
| `inclination_x` | float64 | np.float64(0.0) |
| `inclination_y` | float64 | np.float64(0.0) |
| `nncf_3105_destination_id` | float64 | np.float64(0.0) |
| `nncf_3105_go_to_result` | float64 | np.float64(0.0) |
| `nncf_3106_pause_result` | float64 | np.float64(0.0) |
| `nncf_3107_resume_result` | float64 | np.float64(0.0) |
| `nns_going_to_id` | float64 | np.float64(0.0) |
| `nns_level` | float64 | np.float64(0.0) |
| `nns_status` | bool | np.True_ |
| `nns_target_reached` | float64 | np.float64(0.0) |
| `cumulative_distance_right` | float64 | np.float64(0.0) |
| `encoder_freq_left` | float64 | np.float64(0.0) |
| `encoder_freq_right` | float64 | np.float64(0.0) |
| `front_scanner_warning_zone_not_violated` | bool | np.False_ |
| `warn_front_scanner_warning_zone_active` | bool | np.False_ |
| `warn_rear_scanner_warning_zone_active` | bool | np.False_ |
| `weight_front_left` | float64 | np.float64(0.0) |
| `weight_front_right` | float64 | np.float64(0.0) |
| `weight_rear_left` | float64 | np.float64(0.0) |
| `weight_rear_right` | float64 | np.float64(0.0) |

## 4. Output

- Path: `data\merged\telemetry_cleaned.parquet`
- Size: 4.2 MB
- Rows: 653,612 (unchanged)
- Columns: **35** (was 77; -2 redundant timestamps, -40 constants)

### Final column list

- `fh7000_timestamp` (datetime64[us])
- `session_date` (str)
- `run_file` (str)
- `load_long` (float64)
- `load_mid` (float64)
- `load_short` (float64)
- `ping` (float64)
- `signal_noise` (float64)
- `signal_power` (float64)
- `signal_quality` (float64)
- `uptime` (float64)
- `fh6000_timestamp` (float64)
- `fh6000_number` (float64)
- `battery_cell_voltage` (float64)
- `battery_value` (float64)
- `cumulative_energy_consumption` (float64)
- `cumulative_energy_consumption_uint` (float64)
- `momentary_current_consumption` (float64)
- `brake_release_command_on` (bool)
- `actual_speed_left` (float64)
- `left_drive_stop_executed` (bool)
- `actual_speed_right` (float64)
- `right_drive_stop_executed` (bool)
- `nncf_3108_abort_result` (float64)
- `nns_current_segment` (float64)
- `nns_error_status` (bool)
- `heading_rad` (float64)
- `nns_state` (float64)
- `nns_on_route_status` (bool)
- `nns_position_confidence` (float64)
- `nns_position_initialize_status` (bool)
- `speed_mps` (float64)
- `x_m` (float64)
- `y_m` (float64)
- `front_scanner_safety_zone_not_violated` (bool)

Total cleanup time: 1.0 s