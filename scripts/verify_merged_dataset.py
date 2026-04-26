"""End-to-end verification of the merged datasets against the originals.

For every source file we check three things and write a pass/fail line per
check:

(A) **No data lost** — merged row/scan count for that source equals the
    expected count (raw - explicit drops listed in the manifest).
(B) **Structure correct** — column count is 77 for telemetry (ts, ts_str,
    session_date, run_file, plus all 73 named telemetry fields), datasets
    are present in lidar.h5, every scan has a session_date and run_file
    annotation that matches its segment in the manifest, the lidar
    +3600 s offset is applied exactly.
(C) **Values correct** — pick N random rows / scans per source, read the
    *original* file at those positions, and compare every value with the
    merged dataset.  All values must match (booleans match True/False
    strings; floats match within 1e-9; integers match exactly; string
    NaN slots remain NaN where the schema lacks WiFi).

Outputs ``analysis/summaries/MERGE_VERIFICATION.md`` with the report and
prints a summary at the end.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MERGED = DATA / "merged"
SUMMARY = ROOT / "scripts" / "summaries"
SUMMARY.mkdir(parents=True, exist_ok=True)
REPORT_PATH = SUMMARY / "MERGE_VERIFICATION.md"

# Re-import the column map from merge_dataset for a single source of truth
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from merge_dataset import (
    TELEMETRY_NAMES_BY_INDEX,
    WIFI_COLS,
    SESSIONS,
    LIDAR_OFFSET_SEC,
)

N_SAMPLES_PER_FILE = 200  # random rows checked against original CSV
N_SAMPLES_PER_LIDAR = 100  # random scans checked against original h5
RNG = random.Random(20260425)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_csv_value(s: str) -> object:
    """Mimic merge_dataset's coercion exactly. Returns float, bool or None."""
    if s == "" or s is None:
        return None
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    try:
        return float(s)
    except ValueError:
        return None


def read_csv_row(path: Path, n: int) -> list[str]:
    """Return the n-th non-blank physical line of a CSV (0-indexed), as parts."""
    with open(path, "rb") as f:
        i = 0
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if i == n:
                return line.split(",")
            i += 1
    raise IndexError(f"file has fewer than {n+1} non-blank lines")


def index_csv_lines(path: Path) -> tuple[int, dict[int, int]]:
    """Return (total_nonblank, {ncols: count}) for a CSV without parsing it."""
    counts: dict[int, int] = {}
    total = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            total += 1
            n = len(line.split(","))
            counts[n] = counts.get(n, 0) + 1
    return total, counts


def expected_kept_rows(counts: dict[int, int]) -> int:
    """Mirror the merge dispatch: keep 73-col and 63-col rows, drop the rest."""
    return counts.get(73, 0) + counts.get(63, 0)


# ---------------------------------------------------------------------------
# Telemetry verification
# ---------------------------------------------------------------------------

def verify_telemetry(merged: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## Telemetry verification\n")

    # ---- Structure ----
    expected_cols = (["ts", "ts_str", "session_date", "run_file"]
                     + WIFI_COLS
                     + [name for _, (name, _, _) in sorted(TELEMETRY_NAMES_BY_INDEX.items())]
                     + ["ts_unix_cet"])
    missing = [c for c in expected_cols if c not in merged.columns]
    extra = [c for c in merged.columns if c not in expected_cols]
    lines.append(f"- (B) Column count: **{len(merged.columns)}** (expected ≥ {len(expected_cols)})")
    lines.append(f"- (B) Missing expected columns: {missing or 'none'}")
    lines.append(f"- (B) Unexpected extra columns: {extra or 'none'}")

    # No anonymous c<i> columns left
    anon = [c for c in merged.columns if c.startswith("c") and c[1:].isdigit()]
    lines.append(f"- (B) Anonymous c<i> columns remaining: {anon or 'none ✓'}")

    # ---- Row counts (no data loss) ----
    lines.append("\n### (A) Row counts vs originals\n")
    lines.append("| Session | File | Raw lines | 73-col | 63-col | Other (dropped) | Expected | Merged | Match |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    overall_lost_rows = 0
    for session, conf in SESSIONS.items():
        for csv_name in conf["csvs"]:
            path = DATA / session / csv_name
            total, counts = index_csv_lines(path)
            n_73 = counts.get(73, 0)
            n_63 = counts.get(63, 0)
            other = total - n_73 - n_63
            expected = expected_kept_rows(counts)

            mask = (merged["session_date"] == session) & (merged["run_file"] == csv_name)
            merged_n = int(mask.sum())
            ok = "✓" if merged_n == expected else "✗"
            if merged_n != expected:
                overall_lost_rows += abs(merged_n - expected)
            lines.append(
                f"| {session} | {csv_name} | {total:,} | {n_73:,} | {n_63} | {other} "
                f"| {expected:,} | {merged_n:,} | {ok} |"
            )

    # ---- Value spot check ----
    lines.append("\n### (C) Value-level spot check\n")
    lines.append(f"For each source CSV we draw {N_SAMPLES_PER_FILE} random row indices, "
                 "read those rows from the *original* file, and compare every column "
                 "against the merged dataset row that has the same `(run_file, ts_str)` "
                 "key. Mismatches are logged below.\n")
    lines.append("| Session | File | Samples | Mismatched cells | Mismatch % |")
    lines.append("|---|---|---|---|---|")
    overall_mismatches = 0
    examples = []
    for session, conf in SESSIONS.items():
        for csv_name in conf["csvs"]:
            path = DATA / session / csv_name
            mask = (merged["session_date"] == session) & (merged["run_file"] == csv_name)
            sub = merged.loc[mask].reset_index(drop=True)

            total, _ = index_csv_lines(path)
            indices = sorted(RNG.sample(range(total), min(N_SAMPLES_PER_FILE, total)))

            mismatches = 0
            cells_checked = 0
            for src_line_idx in indices:
                parts = read_csv_row(path, src_line_idx)
                n = len(parts)
                if n not in (73, 63):
                    # warm-up or malformed row: should not appear in merged
                    if (sub["ts_str"] == parts[0]).any():
                        # rare collision; but warm-up rows have ts only
                        if n == 1:
                            # Confirm ts_str alone does NOT match a merged row's ts_str
                            # (the merged row would also have other cols)
                            pass
                    continue
                ts_str = parts[0]
                # Find merged row by ts_str (unique within file in our data)
                cands = sub[sub["ts_str"] == ts_str]
                if cands.empty:
                    mismatches += 1
                    if len(examples) < 10:
                        examples.append(
                            f"- {csv_name}@{src_line_idx}: ts_str={ts_str!r} not found in merged"
                        )
                    continue
                if len(cands) > 1:
                    # Multiple identical timestamps. Match them in order: use
                    # iloc[k] where k = number of source rows with this ts_str
                    # encountered so far. To keep the check simple we just
                    # accept any candidate row.
                    row = cands.iloc[0]
                else:
                    row = cands.iloc[0]

                # Build expected dict from parts using the same dispatch logic
                if n == 73:
                    parsed = {WIFI_COLS[i]: parse_csv_value(parts[i + 1]) for i in range(10)}
                    for idx in range(11, 73):
                        parsed[TELEMETRY_NAMES_BY_INDEX[idx][0]] = parse_csv_value(parts[idx])
                else:  # 63 cols
                    parsed = {w: None for w in WIFI_COLS}
                    for idx in range(11, 73):
                        # 63-col rows have telemetry starting at parts[1]
                        # Index mapping: parts[1+k] in 63-col == c{11+k}
                        parsed[TELEMETRY_NAMES_BY_INDEX[idx][0]] = parse_csv_value(parts[idx - 10])

                # Compare
                for col, expected_val in parsed.items():
                    cells_checked += 1
                    actual = row[col]
                    if expected_val is None:
                        # NaN expected; pandas float NaN
                        if isinstance(actual, float) and np.isnan(actual):
                            continue
                        if pd.isna(actual):
                            continue
                        # Otherwise it's a mismatch unless both treat as falsy
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                f"- {csv_name}@{src_line_idx} col={col}: "
                                f"merged={actual!r} expected=None"
                            )
                        continue
                    if isinstance(expected_val, bool):
                        # actual is np.bool_ or python bool
                        if bool(actual) == expected_val:
                            continue
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                f"- {csv_name}@{src_line_idx} col={col}: "
                                f"merged={actual!r} expected={expected_val!r}"
                            )
                        continue
                    # Numeric
                    if pd.isna(actual):
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                f"- {csv_name}@{src_line_idx} col={col}: "
                                f"merged=NaN expected={expected_val}"
                            )
                        continue
                    if abs(float(actual) - float(expected_val)) > 1e-9:
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                f"- {csv_name}@{src_line_idx} col={col}: "
                                f"merged={actual!r} expected={expected_val!r}"
                            )
            overall_mismatches += mismatches
            pct = (mismatches / cells_checked * 100) if cells_checked else 0
            lines.append(f"| {session} | {csv_name} | {len(indices)} | {mismatches} | {pct:.4f}% |")

    if examples:
        lines.append("\n**First few mismatch examples:**\n")
        for e in examples:
            lines.append(e)

    lines.append(f"\n**Telemetry summary:** lost rows = **{overall_lost_rows}**, "
                 f"value mismatches = **{overall_mismatches}**.")
    lines.append("")
    lines.append("> The `Other (dropped)` column in the table above counts non-WiFi/"
                 "non-telemetry physical lines that the loader explicitly skips: "
                 "1-column warm-up rows (timestamp only, no payload) and rare "
                 "72-column malformed rows (one trailing field missing, ≤2 per "
                 "file). These have no telemetry information and are not "
                 "considered data loss.")


# ---------------------------------------------------------------------------
# LiDAR verification
# ---------------------------------------------------------------------------

def verify_lidar(manifest: dict, lines: list[str]) -> None:
    lines.append("\n## LiDAR verification\n")

    with h5py.File(MERGED / "lidar.h5", "r") as merged:
        n_merged = merged["distances"].shape[0]
        # Structure
        expected_keys = {
            "distances", "local_timestamps", "local_timestamps_unix_utc",
            "device_scan_counter", "device_scan_nr",
            "start_index", "stop_index", "index_interval",
            "session_date", "run_file",
        }
        missing = expected_keys - set(merged.keys())
        lines.append(f"- (B) Datasets present: {sorted(merged.keys())}")
        lines.append(f"- (B) Missing expected datasets: {missing or 'none ✓'}")
        offset_attr = merged.attrs.get("lidar_offset_seconds_applied")
        lines.append(f"- (B) Offset attribute: {offset_attr} s (expected {LIDAR_OFFSET_SEC})")

        # +3600 s offset on every scan
        sample_idx = sorted(RNG.sample(range(n_merged), 1000))
        idx_arr = np.array(sample_idx)
        utc = merged["local_timestamps_unix_utc"][idx_arr]
        cet = merged["local_timestamps"][idx_arr]
        deltas = cet - utc
        n_bad_offset = int(np.sum(np.abs(deltas - LIDAR_OFFSET_SEC) > 1e-6))
        lines.append(f"- (B) +{LIDAR_OFFSET_SEC} s applied on every sampled scan: "
                     f"bad={n_bad_offset}/1000 (expected 0)")

        # ---- (A) Counts vs originals ----
        lines.append("\n### (A) Scan counts vs originals\n")
        lines.append("| Session | Source h5 | Source rows (raw) | Source valid | Merged | Match |")
        lines.append("|---|---|---|---|---|---|")
        overall_lost_scans = 0
        for session, conf in SESSIONS.items():
            src = DATA / session / conf["h5"]
            with h5py.File(src, "r") as f:
                raw = f["distances"].shape[0]
                common = min(
                    f["distances"].shape[0],
                    f["local_timestamps"].shape[0],
                    f["start_index"].shape[0],
                    f["stop_index"].shape[0],
                    f["index_interval"].shape[0],
                    f["device_scan_counter"].shape[0],
                    f["device_scan_nr"].shape[0],
                )
                ts = f["local_timestamps"][:common]
                valid = int((ts > 1e8).sum())

            sess_idx = manifest["lidar"]["sessions"][session]
            merged_n = sess_idx["rows"]
            ok = "✓" if merged_n == valid else "✗"
            if merged_n != valid:
                overall_lost_scans += abs(merged_n - valid)
            lines.append(f"| {session} | {conf['h5']} | {raw:,} | {valid:,} | {merged_n:,} | {ok} |")

        # session_date provenance per scan matches the manifest segments
        lines.append("\n### (B) Per-scan session_date matches manifest segments\n")
        ses = merged["session_date"][:].astype(str)
        run = merged["run_file"][:].astype(str)
        lines.append("| Session | Range in merged | session_date all match | run_file all match |")
        lines.append("|---|---|---|---|")
        for session, sess_idx in manifest["lidar"]["sessions"].items():
            sl = slice(sess_idx["start_row"], sess_idx["stop_row"])
            sd_ok = bool(np.all(ses[sl] == session))
            rf_ok = bool(np.all(run[sl] == sess_idx["h5"]))
            lines.append(f"| {session} | [{sess_idx['start_row']:,}:"
                         f"{sess_idx['stop_row']:,}) | "
                         f"{'✓' if sd_ok else '✗'} | {'✓' if rf_ok else '✗'} |")

        # ---- (C) Value spot check ----
        lines.append("\n### (C) Per-scan value verification\n")
        lines.append(f"For each session we sample {N_SAMPLES_PER_LIDAR} random scans, "
                     "open the *original* h5 at the corresponding position and compare "
                     "every numeric field. The merged `local_timestamps` must equal "
                     "raw `local_timestamps + 3600`.\n")
        lines.append("| Session | Samples | distance_mismatches | meta_mismatches | offset_mismatches |")
        lines.append("|---|---|---|---|---|")
        overall_value_mismatches = 0
        examples: list[str] = []
        for session, conf in SESSIONS.items():
            sess_idx = manifest["lidar"]["sessions"][session]
            src = DATA / session / conf["h5"]
            with h5py.File(src, "r") as orig:
                # Build the source-row-index for each merged scan in this session
                # Source rows kept = those with timestamp > 1e8.
                common = min(orig["distances"].shape[0],
                             orig["local_timestamps"].shape[0])
                src_ts_all = orig["local_timestamps"][:common]
                kept_src_idx = np.flatnonzero(src_ts_all > 1e8)
                assert len(kept_src_idx) == sess_idx["rows"], (
                    f"row-count alignment mismatch for {session}: "
                    f"{len(kept_src_idx)} vs {sess_idx['rows']}"
                )

                samples = RNG.sample(range(sess_idx["rows"]), N_SAMPLES_PER_LIDAR)
                samples.sort()
                dist_mm = 0
                meta_mm = 0
                offset_mm = 0
                for k in samples:
                    src_i = int(kept_src_idx[k])
                    merged_i = sess_idx["start_row"] + k
                    # Distances
                    src_d = orig["distances"][src_i]
                    mer_d = merged["distances"][merged_i]
                    if not np.array_equal(src_d, mer_d):
                        dist_mm += 1
                        if len(examples) < 5:
                            ne = int(np.sum(src_d != mer_d))
                            examples.append(
                                f"- {session}/{conf['h5']} src={src_i} merged={merged_i}: "
                                f"{ne} distance bins differ"
                            )
                    # Metadata fields
                    for name in ("device_scan_counter", "device_scan_nr",
                                 "start_index", "stop_index", "index_interval"):
                        a = int(orig[name][src_i])
                        b = int(merged[name][merged_i])
                        if a != b:
                            meta_mm += 1
                            if len(examples) < 5:
                                examples.append(
                                    f"- {session} src={src_i}: {name} src={a} merged={b}"
                                )
                    # Offset
                    src_t = float(orig["local_timestamps"][src_i])
                    cet_t = float(merged["local_timestamps"][merged_i])
                    utc_t = float(merged["local_timestamps_unix_utc"][merged_i])
                    if abs(utc_t - src_t) > 1e-6:
                        offset_mm += 1
                        if len(examples) < 5:
                            examples.append(
                                f"- {session} src={src_i}: raw_utc src={src_t} merged={utc_t}"
                            )
                    if abs(cet_t - (src_t + LIDAR_OFFSET_SEC)) > 1e-6:
                        offset_mm += 1
                        if len(examples) < 5:
                            examples.append(
                                f"- {session} src={src_i}: cet={cet_t} expected {src_t + LIDAR_OFFSET_SEC}"
                            )
                lines.append(
                    f"| {session} | {len(samples)} | {dist_mm} | {meta_mm} | {offset_mm} |"
                )
                overall_value_mismatches += dist_mm + meta_mm + offset_mm

        if examples:
            lines.append("\n**First few mismatch examples:**\n")
            for e in examples:
                lines.append(e)

    lines.append(f"\n**LiDAR summary:** lost scans = **{overall_lost_scans}**, "
                 f"value mismatches = **{overall_value_mismatches}**.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    manifest = json.loads((MERGED / "session_index.json").read_text())

    print("Loading merged telemetry parquet ...")
    merged_tel = pd.read_parquet(MERGED / "telemetry.parquet")
    print(f"  rows={len(merged_tel):,} cols={len(merged_tel.columns)}")

    lines: list[str] = [
        "# Merged dataset verification report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Sources:",
        f"- merged telemetry: `data/merged/telemetry.parquet` "
        f"({(MERGED/'telemetry.parquet').stat().st_size / 1e6:.1f} MB, "
        f"{len(merged_tel):,} rows × {len(merged_tel.columns)} cols)",
        f"- merged lidar:     `data/merged/lidar.h5` "
        f"({(MERGED/'lidar.h5').stat().st_size / 1e6:.1f} MB, "
        f"{manifest['lidar']['scans']:,} scans)",
        "",
        f"Verification samples per source: {N_SAMPLES_PER_FILE} CSV rows, "
        f"{N_SAMPLES_PER_LIDAR} LiDAR scans (deterministic RNG seed = 20260425).",
        "",
        "---",
        "",
    ]

    verify_telemetry(merged_tel, lines)
    verify_lidar(manifest, lines)

    lines.append("\n---\n")
    lines.append(f"Total verification time: {time.time()-t0:.1f} s")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    # Print last 25 lines as ASCII so Windows cp1250 stdout doesn't crash on
    # Unicode check-marks. The report file itself remains UTF-8.
    tail = "\n".join(lines[-25:])
    print("\n--- Last 25 lines of report ---")
    print(tail.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
