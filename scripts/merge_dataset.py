"""Merge all telemetry CSVs and all LiDAR h5 files into single datasets.

Outputs (in ``data/merged/``):

* ``telemetry.parquet`` — one row per CSV row, with WiFi columns labelled,
  warm-up rows dropped, ``session_date`` and ``run_file`` columns added,
  ``ts`` parsed as a naive datetime in CET local time.
* ``lidar.h5`` — one row per scan, same schema as the source files plus a
  ``session_date`` byte array and a ``run_file`` byte array per scan.
  ``local_timestamps`` is the **corrected** value (original + 3600 s) so it
  represents CET wall clock when interpreted as Unix seconds.
  ``local_timestamps_unix_utc`` retains the original UTC values.
* ``session_index.json`` — start/stop row indices per session in each output,
  for fast slicing downstream.

Run-time: ~30 s for telemetry, a few minutes for LiDAR (gzip recompress).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "merged"
OUT.mkdir(parents=True, exist_ok=True)

WIFI_COLS = [
    "load_long", "load_mid", "load_short", "ping", "reserve",
    "signal_noise", "signal_power", "signal_quality", "uptime", "wifi_state_1",
]

# Full mapping of CSV column index -> (semantic name, source line in
# var_list_Myroslav_25-02_2026.txt). Verified by aligning the row at file
# offset 0 against the var_list and by per-row type checking
# (see verify_merged_dataset.py).
TELEMETRY_NAMES_BY_INDEX: dict[int, tuple[str, int | None, str]] = {
    # idx: (name, var_list_line, units_or_note)
    11: ("fh6000_timestamp",                          23, "frame internal clock"),
    12: ("fh6000_number",                             24, "frame counter"),
    13: ("battery_cell_voltage",                      26, "[ENS] cell voltage"),
    14: ("battery_value",                             27, "[ENS] pack value"),
    15: ("cumulative_energy_consumption",             28, "[ENS]"),
    16: ("cumulative_energy_consumption_uint",        29, "[ENS] high word"),
    17: ("momentary_current_consumption",             30, "[ENS]"),
    18: ("brake_lock_command_on",                     33, "[G1BS] bool"),
    19: ("brake_lock_executed",                       34, "[G1BS] bool"),
    20: ("brake_lock_in_progress",                    35, "[G1BS] count"),
    21: ("brake_release_command_on",                  36, "[G1BS] bool"),
    22: ("brake_release_executed",                    37, "[G1BS] bool"),
    23: ("brake_release_in_progress",                 38, "[G1BS] count"),
    24: ("actual_speed_left",                         39, "[G1LDS]"),
    25: ("left_drive_activate_command_on",            40, "[G1LDS] bool"),
    26: ("left_drive_activate_enable_active",         41, "[G1LDS] bool"),
    27: ("left_drive_activate_executed",              42, "[G1LDS] bool"),
    28: ("left_drive_activate_in_progress",           43, "[G1LDS] count"),
    29: ("left_drive_stop_automatic_permission",      44, "[G1LDS] count"),
    30: ("left_drive_stop_command_on",                45, "[G1LDS] bool"),
    31: ("left_drive_stop_executed",                  46, "[G1LDS] bool"),
    32: ("left_drive_stop_in_progress",               47, "[G1LDS] count"),
    33: ("actual_speed_right",                        48, "[G2RDS]"),
    34: ("right_drive_activate_command_on",           49, "[G2RDS] bool"),
    35: ("right_drive_activate_enable_active",        50, "[G2RDS] bool"),
    36: ("right_drive_activate_executed",             51, "[G2RDS] bool"),
    37: ("right_drive_activate_in_progress",          52, "[G2RDS] count"),
    38: ("right_drive_stop_command_on",               53, "[G2RDS] bool"),
    39: ("right_drive_stop_executed",                 54, "[G2RDS] bool"),
    40: ("right_drive_stop_in_progress",              55, "[G2RDS] count"),
    41: ("inclination_x",                             56, "[IS]"),
    42: ("inclination_y",                             57, "[IS]"),
    43: ("nncf_3105_destination_id",                  58, "[NNCF]"),
    44: ("nncf_3105_go_to_result",                    59, "[NNCF]"),
    45: ("nncf_3106_pause_result",                    60, "[NNCF]"),
    46: ("nncf_3107_resume_result",                   61, "[NNCF]"),
    47: ("nncf_3108_abort_result",                    62, "[NNCF]"),
    48: ("nns_current_segment",                       63, "[NNS]"),
    49: ("nns_error_status",                          64, "[NNS]"),
    50: ("nns_going_to_id",                           65, "[NNS]"),
    51: ("heading_rad",                               66, "[NNS] heading"),
    52: ("nns_level",                                 67, "[NNS]"),
    53: ("nns_state",                                 68, "[NNS]"),
    54: ("nns_status",                                69, "[NNS]"),
    55: ("nns_on_route_status",                       70, "[NNS]"),
    56: ("nns_position_confidence",                   71, "[NNS] %"),
    57: ("nns_position_initialize_status",            72, "[NNS]"),
    58: ("speed_mps",                                 73, "[NNS] speed"),
    59: ("nns_target_reached",                        74, "[NNS]"),
    60: ("x_m",                                       75, "[NNS] X"),
    61: ("y_m",                                       76, "[NNS] Y"),
    62: ("cumulative_distance_right",                 77, "[ODS]"),
    63: ("encoder_freq_left",                         78, "[ODS]"),
    64: ("encoder_freq_right",                        79, "[ODS]"),
    65: ("front_scanner_safety_zone_not_violated",    80, "[SS] bool"),
    66: ("front_scanner_warning_zone_not_violated",   81, "[SS] bool"),
    67: ("warn_front_scanner_warning_zone_active",    86, "[WI] bool"),
    68: ("warn_rear_scanner_warning_zone_active",     87, "[WI] bool"),
    69: ("weight_front_left",                         88, "[WS] strain gauge"),
    70: ("weight_front_right",                        89, "[WS] strain gauge"),
    71: ("weight_rear_left",                          90, "[WS] strain gauge"),
    72: ("weight_rear_right",                         91, "[WS] strain gauge"),
}
TELEMETRY_NAMED = {f"c{idx}": name for idx, (name, _, _) in TELEMETRY_NAMES_BY_INDEX.items()}

# Skipped var_list lines (present in the spec but not in CSV exports):
#   line  1     -> replaced by the literal CSV `ts` string in column 0
#   lines 12-22 -> FH_ID_7000 frame metadata (11 fields)
#   line 25     -> FH_ID_6000.Status
#   lines 31-32 -> Momentary energy / power consumption (2 ENS fields)
#   lines 82-85 -> [TMP] Heading_x1000 / Speed_x1000 / X_x1000 / Y_x1000
# Total skipped: 1 + 11 + 1 + 2 + 4 = 19; var_list 91 - 19 = 72 + 1 (ts) = 73 cols.

LIDAR_OFFSET_SEC = 3600  # +1 h timezone shift (Unix UTC -> Poland CET).
# This is a *timezone* correction, not a residual clock offset; the residual
# sub-second clock offset is calibrated separately in analysis/time_sync/.

SESSIONS = {
    "25.02.2026": {
        "csvs": ["out_Myroslav_25-02_2026.csv"],
        "h5":   "lidar_data_20260225_151241.h5",
    },
    "15.03.2026": {
        "csvs": [
            "out_Myroslav_15-03_2026_1.csv",
            "out_Myroslav_15-03_2026_2.csv",
            "out_Myroslav_15-03_2026_3.csv",
            "out_Myroslav_15-03_2026_4.csv",
        ],
        "h5":   "lidar_data_20260315_165958.h5",
    },
    "24.03.2026": {
        "csvs": [
            "out_Myroslav_24-03_2026_1.csv",
            "out_Myroslav_24-03_2026_2.csv",
            "out_Myroslav_24-03_2026_3.csv",
        ],
        "h5":   "lidar_data_20260324_130319.h5",
    },
}


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def load_csv_robust(path: Path) -> pd.DataFrame:
    """Load a single CSV, dispatching each row by its actual column count.

    Files mix two schemas line by line:
      * 73 columns: ``ts`` + 10 WiFi + 62 telemetry  (full schema)
      * 63 columns: ``ts`` + 62 telemetry            (no WiFi, early warm-up)

    Both are normalised to the full layout (``ts_str`` + 10 WiFi + ``c11``..
    ``c72``); 63-column rows get NaN in their WiFi columns. 1-column warm-up
    rows and rare malformed rows (72-col stragglers) are dropped.
    """
    rows_full: list[list[str]] = []
    rows_nowifi: list[list[str]] = []
    n_dropped = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            parts = line.split(",")
            n = len(parts)
            if n == 73:
                rows_full.append(parts)
            elif n == 63:
                rows_nowifi.append(parts)
            else:
                n_dropped += 1

    frames: list[pd.DataFrame] = []
    if rows_full:
        df_full = pd.DataFrame(rows_full)
        df_full.columns = ["ts_str"] + WIFI_COLS + [f"c{i}" for i in range(11, 73)]
        frames.append(df_full)
    if rows_nowifi:
        df_nw = pd.DataFrame(rows_nowifi)
        df_nw.columns = ["ts_str"] + [f"c{i}" for i in range(11, 73)]
        for w in WIFI_COLS:
            df_nw[w] = np.nan
        df_nw = df_nw[["ts_str"] + WIFI_COLS + [f"c{i}" for i in range(11, 73)]]
        frames.append(df_nw)
    if not frames:
        raise RuntimeError(f"No usable rows in {path}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts_str"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.attrs["n_full_schema_rows"] = len(rows_full)
    df.attrs["n_nowifi_rows"] = len(rows_nowifi)
    df.attrs["n_dropped"] = n_dropped
    return df


def coerce_telemetry_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force every non-meta column to either bool or float.

    Telemetry fields are either booleans ("True"/"False" strings) or numbers.
    For each non-meta column whose dtype is not already a numeric/bool kind
    (``b``, ``i``, ``u``, ``f``) we sample non-null values: if they are all
    boolean strings, the column becomes ``bool``; otherwise it is forced
    numeric (non-parseable values become NaN). Rebuilds the DataFrame so
    no residual string/object dtype survives into Parquet conversion.
    Note: pandas 3.x assigns the ``str`` dtype to former object columns by
    default, so checking ``dtype.kind in 'biuf'`` is the right gate (a plain
    ``dtype == object`` check is too narrow).
    """
    bool_strings = {"True", "False", "true", "false"}
    keep_meta = {"ts", "ts_str", "session_date", "run_file"}
    new_cols = {}
    for col in df.columns:
        if col in keep_meta:
            new_cols[col] = df[col]
            continue
        s = df[col]
        if getattr(s.dtype, "kind", "") in "biuf":
            new_cols[col] = s
            continue
        # Treat object/str/category as text; sample non-null values.
        s_str = s.astype("string").str.strip()
        sample = s_str.dropna().head(500).unique()
        if len(sample) > 0 and set(sample).issubset(bool_strings | {""}):
            new_cols[col] = s_str.isin({"True", "true"}).astype("bool")
        else:
            new_cols[col] = pd.to_numeric(s_str, errors="coerce").astype("float64")
    out = pd.DataFrame(new_cols)
    out.attrs = dict(df.attrs)
    return out


def merge_telemetry() -> dict:
    print("\n=== Merging telemetry ===")
    frames = []
    session_index: dict[str, dict] = {}
    cursor = 0
    for session, conf in SESSIONS.items():
        for csv_name in conf["csvs"]:
            path = DATA / session / csv_name
            t0 = time.time()
            df = load_csv_robust(path)
            df = coerce_telemetry_dtypes(df)
            df["session_date"] = session
            df["run_file"] = csv_name
            n = len(df)
            session_index.setdefault(session, {})[csv_name] = {
                "start_row": cursor,
                "stop_row": cursor + n,
                "rows": n,
                "n_full_schema": int(df.attrs.get("n_full_schema_rows", 0)),
                "n_nowifi_schema": int(df.attrs.get("n_nowifi_rows", 0)),
                "n_dropped": int(df.attrs.get("n_dropped", 0)),
            }
            cursor += n
            print(f"  {session}/{csv_name}: {n:>7,} rows "
                  f"(full={df.attrs.get('n_full_schema_rows', 0):>6,} "
                  f"nowifi={df.attrs.get('n_nowifi_rows', 0):>3} "
                  f"dropped={df.attrs.get('n_dropped', 0):>2})  "
                  f"({time.time()-t0:.1f}s)")
            frames.append(df)

    print("  Concatenating ...")
    merged = pd.concat(frames, ignore_index=True)
    # Concat across files can re-promote some columns back to object if one
    # source had all-NaN (float) and another had string. Re-coerce after.
    merged = coerce_telemetry_dtypes(merged)

    # Rename known logical columns
    merged = merged.rename(columns=TELEMETRY_NAMED)

    # Reorder for readability: ts, session_date, run_file, WiFi, named, others
    front = ["ts", "ts_str", "session_date", "run_file"] + WIFI_COLS + list(TELEMETRY_NAMED.values())
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    # Apply CET to epoch (seconds) for downstream joins. Telemetry strings are
    # parsed as naive CET; we expose ``ts_unix_cet`` = naive_seconds (i.e.
    # treat the naive datetime as if it were Unix UTC). This matches the
    # corrected LiDAR ``local_timestamps`` after +1 h shift, so a single
    # numeric key joins both streams.
    merged["ts_unix_cet"] = merged["ts"].astype("datetime64[ns]").astype("int64") / 1e9

    out_path = OUT / "telemetry.parquet"
    print(f"  Writing {out_path} ...")
    merged.to_parquet(out_path, compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  done. {len(merged):,} rows, {size_mb:.1f} MB on disk.")

    return {"rows": len(merged), "files": session_index, "path": str(out_path)}


# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------

def merge_lidar(chunk_size: int = 5000) -> dict:
    """Concatenate all 3 h5 files into one with the +1h correction applied.

    Output preserves the original schema (so existing readers keep working)
    and adds ``local_timestamps_unix_utc`` (raw) plus ``session_date`` and
    ``run_file`` byte arrays per scan. Bad rows (timestamp == 0, the truncated
    last write of 24.03) are dropped.
    """
    print("\n=== Merging LiDAR ===")

    # Pre-compute total kept-scan count to size the output file
    total = 0
    per_session: dict[str, int] = {}
    for session, conf in SESSIONS.items():
        h5path = DATA / session / conf["h5"]
        with h5py.File(h5path, "r") as src:
            common = min(src["distances"].shape[0],
                         src["local_timestamps"].shape[0],
                         src["start_index"].shape[0],
                         src["stop_index"].shape[0],
                         src["index_interval"].shape[0],
                         src["device_scan_counter"].shape[0],
                         src["device_scan_nr"].shape[0])
            ts = src["local_timestamps"][:common]
            keep = int((ts > 1e8).sum())
            per_session[session] = keep
            total += keep
            print(f"  {session}/{conf['h5']}: {common:>7,} valid + "
                  f"{src['distances'].shape[0]-common} truncated to keep {keep:,}")

    max_pts = 2700  # original schema
    out_path = OUT / "lidar.h5"
    print(f"  Allocating {out_path} ({total:,} scans × {max_pts}) ...")

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w", libver="latest") as dst:
        # Resizable datasets, mirroring storage.py
        dst.create_dataset("distances", shape=(total, max_pts),
                           dtype=np.uint16, chunks=(1, max_pts),
                           compression="gzip", compression_opts=4)
        dst.create_dataset("local_timestamps", shape=(total,),
                           dtype=np.float64)
        dst.create_dataset("local_timestamps_unix_utc", shape=(total,),
                           dtype=np.float64)
        dst.create_dataset("device_scan_counter", shape=(total,),
                           dtype=np.uint32)
        dst.create_dataset("device_scan_nr", shape=(total,),
                           dtype=np.uint32)
        dst.create_dataset("start_index", shape=(total,), dtype=np.int16)
        dst.create_dataset("stop_index", shape=(total,), dtype=np.int16)
        dst.create_dataset("index_interval", shape=(total,), dtype=np.int16)
        # Variable-length UTF-8 string columns for provenance
        str_dt = h5py.string_dtype(encoding="utf-8")
        dst.create_dataset("session_date", shape=(total,), dtype=str_dt)
        dst.create_dataset("run_file", shape=(total,), dtype=str_dt)

        dst.attrs["created"] = time.strftime("%Y%m%d_%H%M%S")
        dst.attrs["max_points"] = max_pts
        dst.attrs["lidar_offset_seconds_applied"] = LIDAR_OFFSET_SEC
        dst.attrs["offset_explanation"] = (
            "local_timestamps = local_timestamps_unix_utc + 3600. The +1 h "
            "shift converts Unix UTC (LiDAR collector clock) into Poland CET "
            "wall clock to match telemetry CSV timestamps. The residual "
            "sub-second clock offset is calibrated in analysis/time_sync/."
        )

        write_cursor = 0
        session_index: dict[str, dict] = {}
        for session, conf in SESSIONS.items():
            h5path = DATA / session / conf["h5"]
            print(f"  Streaming {session}/{conf['h5']} ...")
            session_start = write_cursor
            with h5py.File(h5path, "r") as src:
                common = min(src["distances"].shape[0],
                             src["local_timestamps"].shape[0],
                             src["start_index"].shape[0],
                             src["stop_index"].shape[0],
                             src["index_interval"].shape[0],
                             src["device_scan_counter"].shape[0],
                             src["device_scan_nr"].shape[0])
                # Stream in chunks
                t0 = time.time()
                for i0 in range(0, common, chunk_size):
                    i1 = min(i0 + chunk_size, common)
                    ts_chunk = src["local_timestamps"][i0:i1]
                    keep_mask = ts_chunk > 1e8
                    if not keep_mask.any():
                        continue
                    keep_n = int(keep_mask.sum())
                    sl = slice(write_cursor, write_cursor + keep_n)
                    dst["distances"][sl] = src["distances"][i0:i1][keep_mask]
                    dst["local_timestamps_unix_utc"][sl] = ts_chunk[keep_mask]
                    dst["local_timestamps"][sl] = ts_chunk[keep_mask] + LIDAR_OFFSET_SEC
                    dst["device_scan_counter"][sl] = \
                        src["device_scan_counter"][i0:i1][keep_mask]
                    dst["device_scan_nr"][sl] = \
                        src["device_scan_nr"][i0:i1][keep_mask]
                    dst["start_index"][sl] = src["start_index"][i0:i1][keep_mask]
                    dst["stop_index"][sl] = src["stop_index"][i0:i1][keep_mask]
                    dst["index_interval"][sl] = \
                        src["index_interval"][i0:i1][keep_mask]
                    dst["session_date"][sl] = [session] * keep_n
                    dst["run_file"][sl] = [conf["h5"]] * keep_n
                    write_cursor += keep_n
                print(f"    wrote {write_cursor - session_start:,} scans "
                      f"({time.time()-t0:.1f}s)")
            session_index[session] = {
                "h5": conf["h5"],
                "start_row": session_start,
                "stop_row": write_cursor,
                "rows": write_cursor - session_start,
            }

    size_mb = out_path.stat().st_size / 1e6
    print(f"  done. {write_cursor:,} scans, {size_mb:.1f} MB on disk.")
    return {"scans": write_cursor, "sessions": session_index, "path": str(out_path)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    tel_info = merge_telemetry()
    lid_info = merge_lidar()

    manifest = {
        "lidar_offset_seconds_applied": LIDAR_OFFSET_SEC,
        "offset_explanation": (
            "Unix UTC LiDAR timestamps shifted by +3600 s so that they "
            "represent Poland CET wall clock and align with the telemetry "
            "CSV timestamp strings."
        ),
        "telemetry": tel_info,
        "lidar": lid_info,
    }
    (OUT / "session_index.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
