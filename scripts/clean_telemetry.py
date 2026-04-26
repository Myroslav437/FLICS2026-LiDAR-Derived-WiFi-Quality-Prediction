"""Minimal cleanup pass on the merged telemetry parquet.

1. Verify that ``ts_str`` is the string form of ``ts`` and drop it if so.
2. Verify that ``ts_unix_cet`` is just ``ts`` re-encoded as Unix seconds and
   drop it if so.
3. Rename ``ts`` (the FH_ID_7000.TimeStamp value) to ``fh7000_timestamp``.
4. Drop columns that have a single unique value (or are all-NaN) anywhere
   in the merged dataset — they carry no information.
5. Save as ``data/merged/telemetry_cleaned.parquet`` and write a short
   cleanup report to ``analysis/summaries/TELEMETRY_CLEANUP.md``.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"
SUMMARY = ROOT / "scripts" / "summaries"
IN_PATH = MERGED / "telemetry.parquet"
OUT_PATH = MERGED / "telemetry_cleaned.parquet"
REPORT = SUMMARY / "TELEMETRY_CLEANUP.md"


def main():
    t0 = time.time()
    print(f"Loading {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)
    n_rows = len(df)
    n_cols_in = len(df.columns)
    print(f"  rows={n_rows:,} cols={n_cols_in}")

    notes: list[str] = []
    notes.append("# Telemetry cleanup report")
    notes.append("")
    notes.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    notes.append(f"Input:  `{IN_PATH.relative_to(ROOT)}` "
                 f"({IN_PATH.stat().st_size / 1e6:.1f} MB, "
                 f"{n_rows:,} rows × {n_cols_in} cols)")
    notes.append("")

    # ------------------------------------------------------------------
    # (1a) ts_str redundancy
    # ------------------------------------------------------------------
    notes.append("## 1. Redundant timestamp columns\n")
    ts_str_redundant = False
    if "ts_str" in df.columns and "ts" in df.columns:
        # Re-parse ts_str to datetime and compare element-wise with ts.
        reparsed = pd.to_datetime(df["ts_str"], errors="coerce")
        ts = df["ts"].astype("datetime64[ns]")
        rep = reparsed.astype("datetime64[ns]")
        diff = (rep - ts).abs()
        max_diff_ns = int(diff.max().value) if not diff.isna().all() else 0
        n_mismatch = int((diff > pd.Timedelta(0)).sum())
        notes.append(f"- `ts_str` re-parsed and compared with `ts` element-wise.")
        notes.append(f"- max abs(ts - parse(ts_str)) = **{max_diff_ns} ns**, mismatches = **{n_mismatch}**")
        ts_str_redundant = (max_diff_ns == 0 and n_mismatch == 0)
        notes.append(f"- ⇒ `ts_str` is {'redundant — DROPPED' if ts_str_redundant else 'NOT redundant — KEPT'}.")
    else:
        notes.append("- `ts_str` or `ts` missing; nothing to compare.")

    # ------------------------------------------------------------------
    # (1b) ts_unix_cet redundancy
    # ------------------------------------------------------------------
    ts_unix_redundant = False
    if "ts_unix_cet" in df.columns and "ts" in df.columns:
        # Use a unit-agnostic conversion: total seconds since the Unix epoch
        # works regardless of whether ts is stored at us or ns precision.
        ts_seconds = (df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
        diff_us = ((df["ts_unix_cet"] - ts_seconds) * 1e6).abs()
        max_diff_us = float(diff_us.max()) if not diff_us.isna().all() else 0.0
        n_mismatch = int((diff_us > 1).sum())  # tolerate <1 µs FP rounding
        notes.append(f"- `ts_unix_cet` reproduced from `ts` and compared element-wise.")
        notes.append(f"- max abs(ts_unix_cet - ts.timestamp) = **{max_diff_us:.3f} µs**, "
                     f"mismatches (>1 µs) = **{n_mismatch}**")
        ts_unix_redundant = (max_diff_us < 1.0 and n_mismatch == 0)
        notes.append(f"- ⇒ `ts_unix_cet` is {'redundant — DROPPED' if ts_unix_redundant else 'NOT redundant — KEPT'}.")

    # Apply drops
    cols_dropped_redundant: list[str] = []
    if ts_str_redundant:
        df = df.drop(columns=["ts_str"])
        cols_dropped_redundant.append("ts_str")
    if ts_unix_redundant:
        df = df.drop(columns=["ts_unix_cet"])
        cols_dropped_redundant.append("ts_unix_cet")

    # ------------------------------------------------------------------
    # (2) Rename ts -> fh7000_timestamp
    # ------------------------------------------------------------------
    notes.append("\n## 2. Rename `ts` to `fh7000_timestamp`\n")
    if "ts" in df.columns:
        df = df.rename(columns={"ts": "fh7000_timestamp"})
        notes.append("- `ts` (the FH_ID_7000.TimeStamp value, parsed as datetime64) "
                     "renamed to `fh7000_timestamp`.")
    else:
        notes.append("- `ts` not present; skipping rename.")

    # ------------------------------------------------------------------
    # (3) Drop columns whose value never changes
    # ------------------------------------------------------------------
    notes.append("\n## 3. Constant columns (no information)\n")
    META = {"fh7000_timestamp", "session_date", "run_file"}
    constant_cols: list[tuple[str, str, str]] = []  # (col, dtype, sole_value)
    for col in df.columns:
        if col in META:
            continue
        s = df[col]
        nun = s.nunique(dropna=True)
        n_nan = int(s.isna().sum())
        if nun == 0:
            constant_cols.append((col, str(s.dtype), f"all NaN ({n_nan:,} rows)"))
        elif nun == 1:
            sole = s.dropna().iloc[0]
            sole_repr = repr(sole)
            extra = f" + {n_nan:,} NaN" if n_nan else ""
            constant_cols.append((col, str(s.dtype), sole_repr + extra))

    if constant_cols:
        notes.append(f"Identified **{len(constant_cols)}** columns with a single unique "
                     "(or all-NaN) value across all 653,612 rows. These have zero "
                     "explanatory power and are dropped.\n")
        notes.append("| Column | Dtype | Sole value |")
        notes.append("|---|---|---|")
        for col, dt, sole in constant_cols:
            notes.append(f"| `{col}` | {dt} | {sole} |")
        df = df.drop(columns=[c for c, _, _ in constant_cols])
    else:
        notes.append("No constant columns found.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    n_cols_out = len(df.columns)
    print(f"  After cleanup: {n_cols_out} columns "
          f"(-{n_cols_in - n_cols_out} from input)")
    print(f"  Writing {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, compression="zstd", index=False)
    out_size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"  done. {out_size_mb:.1f} MB on disk.")

    # ------------------------------------------------------------------
    # Wrap report
    # ------------------------------------------------------------------
    notes.append("\n## 4. Output\n")
    notes.append(f"- Path: `{OUT_PATH.relative_to(ROOT)}`")
    notes.append(f"- Size: {out_size_mb:.1f} MB")
    notes.append(f"- Rows: {n_rows:,} (unchanged)")
    notes.append(f"- Columns: **{n_cols_out}** (was {n_cols_in}; "
                 f"-{len(cols_dropped_redundant)} redundant timestamps, "
                 f"-{len(constant_cols)} constants)")
    notes.append("")
    notes.append("### Final column list\n")
    for c in df.columns:
        notes.append(f"- `{c}` ({df[c].dtype})")
    notes.append("")
    notes.append(f"Total cleanup time: {time.time()-t0:.1f} s")

    REPORT.write_text("\n".join(notes), encoding="utf-8")
    print(f"  Wrote {REPORT}")


if __name__ == "__main__":
    main()
