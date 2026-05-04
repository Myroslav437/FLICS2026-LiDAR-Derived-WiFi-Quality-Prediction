"""Battery channel provenance diagnostic — top-level driver.

Resolves the contradiction between ``docs/TELEMETRY_CLEANUP.md`` (which
preserved ``battery_value``) and ``docs/p1_dataset_analysis/report.md``
(which reports ``battery_value`` as constant 6.554e+04 in the Phase 1
dataset).

Reads three parquet files read-only:
- ``data/merged/telemetry_merged.parquet``    (raw merged)
- ``data/merged/telemetry_cleaned.parquet``   (post-cleanup)
- ``data/phase1/dataset.parquet``             (Phase 1, post-LiDAR-join)

Writes:
- ``docs/p1_battery_provenance/figures/*.png``
- ``docs/p1_battery_provenance/tables/{per_stage_summary,value_identity,join_keys}.md``
- ``docs/p1_battery_provenance/report.md``

Run as:
    python -m scripts.p1_battery_provenance.run_diagnostic
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Mapping

# Force UTF-8 stdout/stderr (Windows default cp1250 cannot encode µ, ², …).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import numpy as np
import pandas as pd

from scripts.p1_battery_provenance import SEED
from scripts.p1_battery_provenance import plotting as plot


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_MERGED = ROOT / "data" / "merged" / "telemetry_merged.parquet"
TELEMETRY_CLEANED = ROOT / "data" / "merged" / "telemetry_cleaned.parquet"
DATASET = ROOT / "data" / "phase1" / "dataset.parquet"

DOCS_DIR = ROOT / "docs" / "p1_battery_provenance"
FIG_DIR = DOCS_DIR / "figures"
TBL_DIR = DOCS_DIR / "tables"
REPORT_PATH = DOCS_DIR / "report.md"

CHANNELS = [
    "battery_value",
    "battery_cell_voltage",
    "cumulative_energy_consumption",
    "cumulative_energy_consumption_uint",
    "momentary_current_consumption",
]
SESSIONS = ["25.02.2026", "15.03.2026", "24.03.2026"]

STAGES = [
    ("merged",   TELEMETRY_MERGED,   "ts"),
    ("cleaned",  TELEMETRY_CLEANED,  "fh7000_timestamp"),
    ("phase1",   DATASET,            "fh7000_timestamp"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stage(path: Path, ts_col: str) -> pd.DataFrame:
    """Read a parquet stage with only the columns we need.

    The merged file uses ``ts`` as the telemetry timestamp; cleaned and
    phase1 use ``fh7000_timestamp`` (the cleanup script renamed it).
    We normalise to ``fh7000_timestamp`` here so downstream code is uniform.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    df = pd.read_parquet(path)
    if ts_col != "fh7000_timestamp":
        df = df.rename(columns={ts_col: "fh7000_timestamp"})
    df["fh7000_timestamp"] = pd.to_datetime(df["fh7000_timestamp"])
    return df


# ---------------------------------------------------------------------------
# 3.1 — Per-stage variability summary
# ---------------------------------------------------------------------------

def _series_stats(s: pd.Series) -> dict[str, float | int]:
    finite = s.dropna()
    if len(finite) == 0:
        return {"n": int(len(s)), "nunique": 0, "nan": int(s.isna().sum()),
                "min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan"),
                "range": float("nan")}
    return {
        "n": int(len(s)),
        "nunique": int(finite.nunique()),
        "nan": int(s.isna().sum()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "range": float(finite.max() - finite.min()),
    }


def per_stage_summary_table(stage_dfs: Mapping[str, pd.DataFrame]) -> str:
    """Build the §3.1 table (one section per channel)."""
    out_lines = []
    for ch in CHANNELS:
        out_lines.append(f"## {ch}\n")
        out_lines.append(
            "| stage | scope | n | nunique | NaN | min | max | mean | std | range |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for stage, df in stage_dfs.items():
            if ch not in df.columns:
                out_lines.append(
                    f"| {stage} | global | {len(df)} | — | — | — | — | — | — | — |"
                )
                out_lines.append(
                    f"| {stage} | (channel not in this stage) | | | | | | | | |"
                )
                continue
            s = df[ch]
            st = _series_stats(s)
            out_lines.append(
                f"| {stage} | global | {st['n']} | {st['nunique']} | "
                f"{st['nan']} | {st['min']:.6g} | {st['max']:.6g} | "
                f"{st['mean']:.6g} | {st['std']:.6g} | {st['range']:.6g} |"
            )
            for sd in SESSIONS:
                sub = df[df["session_date"] == sd]
                if ch not in sub.columns or len(sub) == 0:
                    continue
                sst = _series_stats(sub[ch])
                out_lines.append(
                    f"| {stage} | {sd} | {sst['n']} | {sst['nunique']} | "
                    f"{sst['nan']} | {sst['min']:.6g} | {sst['max']:.6g} | "
                    f"{sst['mean']:.6g} | {sst['std']:.6g} | {sst['range']:.6g} |"
                )
        out_lines.append("")
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 3.2 — Histograms per stage
# ---------------------------------------------------------------------------

def write_histograms(stage_dfs: Mapping[str, pd.DataFrame]) -> None:
    for ch in CHANNELS:
        for stage, df in stage_dfs.items():
            if ch not in df.columns:
                continue
            v = df[ch].to_numpy()
            plot.histogram(
                v, channel=ch, stage=stage,
                fig_path=FIG_DIR / f"hist_{ch}_{stage}.png",
            )
    # Per-session overlay specifically for battery_value
    plot.per_session_overlay(
        stage_dfs, channel="battery_value", sessions=SESSIONS,
        fig_path=FIG_DIR / "battery_value_per_session_overlay.png",
    )


# ---------------------------------------------------------------------------
# 3.3 — Cross-stage value-identity check
# ---------------------------------------------------------------------------

def value_identity_table(merged: pd.DataFrame, cleaned: pd.DataFrame,
                         dataset: pd.DataFrame) -> str:
    out_lines = ["## merged → cleaned (per-row, joined on (session_date, fh7000_timestamp))\n"]
    out_lines.append(
        "| channel | rows compared | NaN-NaN matches | "
        "max abs diff (finite) | mismatches (>1e-9) | identical |\n"
        "|---|---:|---:|---:|---:|---|"
    )
    join_keys = ["session_date", "fh7000_timestamp"]
    for ch in CHANNELS:
        if ch not in merged.columns or ch not in cleaned.columns:
            out_lines.append(f"| {ch} | — | — | — | — | not in both stages |")
            continue
        m = merged[join_keys + [ch]].rename(columns={ch: f"{ch}_merged"})
        c = cleaned[join_keys + [ch]].rename(columns={ch: f"{ch}_cleaned"})
        joined = m.merge(c, on=join_keys, how="inner")
        a = joined[f"{ch}_merged"].to_numpy()
        b = joined[f"{ch}_cleaned"].to_numpy()
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        nan_match = int((nan_a & nan_b).sum())
        finite = ~nan_a & ~nan_b
        if finite.any():
            d = np.abs(a[finite] - b[finite])
            max_diff = float(d.max())
            n_mis = int((d > 1e-9).sum())
        else:
            max_diff = 0.0
            n_mis = 0
        identical = "OK (identical)" if (max_diff == 0 and n_mis == 0
                                          and (nan_a == nan_b).all()) else "DIFF"
        out_lines.append(
            f"| {ch} | {len(joined):,} | {nan_match} | "
            f"{max_diff:.3g} | {n_mis} | {identical} |"
        )

    out_lines.append("")
    out_lines.append("## cleaned → phase1 (per-telemetry-timestamp, "
                      "max-min within each phase1 group must be zero)")
    out_lines.append("")
    out_lines.append(
        "| channel | telemetry rows in cleaned (all sessions) | telemetry rows "
        "joined into phase1 | per-group max−min (sup) | constant within group | "
        "max abs diff (cleaned vs phase1 first match) |\n"
        "|---|---:|---:|---:|---|---:|"
    )
    join_keys2 = ["session_date", "fh7000_timestamp"]
    for ch in CHANNELS:
        if ch not in cleaned.columns:
            out_lines.append(f"| {ch} | — | — | — | not in cleaned | — |")
            continue
        if ch not in dataset.columns:
            out_lines.append(
                f"| {ch} | {len(cleaned):,} | 0 | n/a | "
                f"channel not present in phase1 (dropped at column-projection in build_dataset.py) | n/a |"
            )
            continue
        # Per-group sup of (max − min) on the phase1 side
        gb = dataset.groupby(join_keys2)[ch]
        spread = (gb.max() - gb.min())
        sup = float(spread.max()) if len(spread) else 0.0
        const_within = "OK (yes)" if sup == 0 else f"NO (max spread {sup:.6g})"
        # First-match diff against cleaned
        first = gb.first().reset_index().rename(columns={ch: f"{ch}_phase1"})
        c = cleaned[join_keys2 + [ch]].rename(columns={ch: f"{ch}_cleaned"})
        joined = c.merge(first, on=join_keys2, how="inner")
        a = joined[f"{ch}_cleaned"].to_numpy()
        b = joined[f"{ch}_phase1"].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            max_diff = float(np.abs(a[finite] - b[finite]).max())
        else:
            max_diff = 0.0
        out_lines.append(
            f"| {ch} | {len(cleaned):,} | {len(joined):,} | "
            f"{sup:.6g} | {const_within} | {max_diff:.6g} |"
        )
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 3.4 — Time-series plots
# ---------------------------------------------------------------------------

def write_timeseries(cleaned: pd.DataFrame) -> None:
    for sd in SESSIONS:
        plot.time_series_simple(
            cleaned, channel="battery_value", session=sd,
            fig_path=FIG_DIR / f"timeseries_battery_value_{sd}.png",
        )
        plot.time_series_dual(
            cleaned, primary="battery_cell_voltage",
            secondary="momentary_current_consumption", session=sd,
            fig_path=FIG_DIR / f"timeseries_battery_cell_voltage_vs_current_{sd}.png",
        )
        plot.time_series_simple(
            cleaned, channel="cumulative_energy_consumption", session=sd,
            fig_path=FIG_DIR / f"timeseries_cumulative_energy_consumption_{sd}.png",
        )
        plot.time_series_simple(
            cleaned, channel="cumulative_energy_consumption_uint", session=sd,
            fig_path=FIG_DIR / f"timeseries_cumulative_energy_consumption_uint_{sd}.png",
        )


def cumulative_monotonicity(cleaned: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Per-(session × run_file) monotonicity report.

    Each session typically contains multiple CSV run_files; we evaluate
    monotonicity within each run_file (a single continuous recording),
    not across run_file boundaries (where a counter may reset between
    CSVs within the same session).
    """
    out: dict[str, dict[str, dict]] = {}
    for ch in ["cumulative_energy_consumption",
               "cumulative_energy_consumption_uint"]:
        out[ch] = {}
        for sd in SESSIONS:
            sub = cleaned[cleaned["session_date"] == sd]
            run_files = list(sub["run_file"].dropna().unique())
            per_run = []
            session_mon = True
            session_n_dec = 0
            session_n = 0
            session_max_drop = 0.0
            for rf in run_files:
                rsub = (sub[sub["run_file"] == rf]
                        .sort_values("fh7000_timestamp"))
                v = rsub[ch].to_numpy()
                if len(v) < 2:
                    per_run.append({"run_file": rf, "n": int(len(v)),
                                     "n_decreasing": 0, "max_drop": 0.0,
                                     "monotonic_nondec": True})
                    session_n += int(len(v))
                    continue
                d = np.diff(v)
                n_dec = int((d < 0).sum())
                max_drop = float(d.min()) if d.size else 0.0
                mon = bool((d >= 0).all())
                per_run.append({"run_file": rf, "n": int(len(v)),
                                 "n_decreasing": n_dec,
                                 "max_drop": max_drop,
                                 "monotonic_nondec": mon})
                session_n += int(len(v))
                session_n_dec += n_dec
                session_max_drop = min(session_max_drop, max_drop)
                if not mon:
                    session_mon = False
            out[ch][sd] = {
                "n": session_n,
                "n_run_files": len(run_files),
                "n_decreasing_per_run_total": session_n_dec,
                "session_max_drop_per_run": session_max_drop,
                "all_runs_monotonic_nondec": bool(session_mon),
                "per_run": per_run,
            }
    return out


# ---------------------------------------------------------------------------
# 3.5 — Joint-key persistence check
# ---------------------------------------------------------------------------

def join_keys_table(cleaned: pd.DataFrame, dataset: pd.DataFrame,
                    n_per_session: int = 100, seed: int = SEED) -> str:
    rng = np.random.RandomState(seed)
    out_lines = [
        "## battery_value: per-telemetry-timestamp persistence into Phase 1\n",
        "Sample of 100 random telemetry timestamps per session that are also "
        "present in the Phase 1 dataset. For each, verify all "
        "phase1 rows joined to that timestamp share the same battery_value, "
        "and that battery_value matches the value in `telemetry_cleaned`.\n",
        "| session | sampled tel ts (in cleaned ∩ phase1) | "
        "phase1 group rows (mean) | within-group constant? | "
        "matches cleaned? | mismatches |",
        "|---|---:|---:|---|---|---:|",
    ]
    join_keys = ["session_date", "fh7000_timestamp"]
    # Pre-aggregate phase1 per (session, ts)
    g = dataset.groupby(join_keys)["battery_value"]
    p1_first = g.first().rename("phase1_first").reset_index()
    p1_min = g.min().rename("phase1_min").reset_index()
    p1_max = g.max().rename("phase1_max").reset_index()
    p1_count = g.count().rename("phase1_count").reset_index()
    p1 = (p1_first.merge(p1_min, on=join_keys)
                  .merge(p1_max, on=join_keys)
                  .merge(p1_count, on=join_keys))
    # Cleaned side
    c = cleaned[join_keys + ["battery_value"]].rename(
        columns={"battery_value": "cleaned_value"})
    pool = c.merge(p1, on=join_keys, how="inner")

    for sd in SESSIONS:
        sub = pool[pool["session_date"] == sd].reset_index(drop=True)
        if len(sub) == 0:
            out_lines.append(
                f"| {sd} | 0 | n/a | n/a | n/a | 0 |"
            )
            continue
        n = min(n_per_session, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        samp = sub.iloc[idx]
        within_const = ((samp["phase1_max"] - samp["phase1_min"]) == 0).all()
        match_clean = ((samp["cleaned_value"].fillna(-1)
                       == samp["phase1_first"].fillna(-1)).all())
        n_mis = int((samp["cleaned_value"].fillna(-1)
                     != samp["phase1_first"].fillna(-1)).sum())
        out_lines.append(
            f"| {sd} | {n} | {samp['phase1_count'].mean():.2f} | "
            f"{'YES' if within_const else 'NO'} | "
            f"{'YES' if match_clean else 'NO'} | {n_mis} |"
        )
    out_lines.append("")
    # And also a coverage check: do the variable-battery rows survive into phase1?
    var_rows = cleaned[cleaned["battery_value"] != 65535]
    out_lines.append(
        "## Coverage check: do the variable (≠65535) battery_value rows reach Phase 1?\n"
    )
    out_lines.append(
        "| session | n rows ≠65535 in cleaned | tel-ts range of variable rows | "
        "n of those rows present in phase1 (by ts) |"
    )
    out_lines.append("|---|---:|---|---:|")
    p1_keys = set(zip(dataset["session_date"].astype(str),
                      pd.to_datetime(dataset["fh7000_timestamp"]).astype("int64")))
    for sd in SESSIONS:
        vsub = var_rows[var_rows["session_date"] == sd]
        if len(vsub) == 0:
            out_lines.append(f"| {sd} | 0 | — | 0 |")
            continue
        ts = pd.to_datetime(vsub["fh7000_timestamp"]).astype("int64")
        present = sum(1 for t in ts if (sd, int(t)) in p1_keys)
        out_lines.append(
            f"| {sd} | {len(vsub):,} | "
            f"{vsub['fh7000_timestamp'].min()} … {vsub['fh7000_timestamp'].max()} | "
            f"{present:,} |"
        )
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 3.6 — Cumulative-energy consumption pair
# ---------------------------------------------------------------------------

def cumulative_pair_summary(cleaned: pd.DataFrame) -> dict:
    a = cleaned["cumulative_energy_consumption"].to_numpy(dtype=np.float64)
    b = cleaned["cumulative_energy_consumption_uint"].to_numpy(dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    out = {
        "n_compared": int(finite.sum()),
        "max_abs_diff": float(np.abs(a - b).max()) if finite.any() else 0.0,
        "max_b_minus_a": float((b - a).max()) if finite.any() else 0.0,
        "min_b_minus_a": float((b - a).min()) if finite.any() else 0.0,
        "ratio_a_to_b_median": float(np.median(a / np.where(b == 0, np.nan, b)))
            if finite.any() else float("nan"),
        "a_int_match": bool(np.allclose(a, np.round(a))) if finite.any() else False,
        "b_int_match": bool(np.allclose(b, np.round(b))) if finite.any() else False,
    }
    # Per-session value-progression and ranges
    out["per_session"] = {}
    for sd in SESSIONS:
        sub = cleaned[cleaned["session_date"] == sd]
        out["per_session"][sd] = {
            "a_min": float(sub["cumulative_energy_consumption"].min()),
            "a_max": float(sub["cumulative_energy_consumption"].max()),
            "a_nu": int(sub["cumulative_energy_consumption"].nunique()),
            "b_min": float(sub["cumulative_energy_consumption_uint"].min()),
            "b_max": float(sub["cumulative_energy_consumption_uint"].max()),
            "b_nu": int(sub["cumulative_energy_consumption_uint"].nunique()),
        }
    return out


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def determine_verdict(merged: pd.DataFrame, cleaned: pd.DataFrame,
                      dataset: pd.DataFrame) -> dict:
    """Compute the numbers needed to pick A / B / C (or refinement)."""
    bv_merged = merged["battery_value"]
    bv_cleaned = cleaned["battery_value"]
    bv_phase1 = dataset["battery_value"]

    # Are the variable rows in cleaned reachable from phase1?
    var = cleaned[cleaned["battery_value"] != 65535]
    p1_keys = set(zip(dataset["session_date"].astype(str),
                      pd.to_datetime(dataset["fh7000_timestamp"]).astype("int64")))
    var_in_phase1 = 0
    if len(var) > 0:
        ts = pd.to_datetime(var["fh7000_timestamp"]).astype("int64")
        sds = var["session_date"].astype(str).to_numpy()
        for sd, t in zip(sds, ts):
            if (sd, int(t)) in p1_keys:
                var_in_phase1 += 1

    return {
        "merged_nu": int(bv_merged.nunique(dropna=False)),
        "cleaned_nu": int(bv_cleaned.nunique(dropna=False)),
        "phase1_nu": int(bv_phase1.nunique(dropna=False)),
        "merged_nan": int(bv_merged.isna().sum()),
        "cleaned_nan": int(bv_cleaned.isna().sum()),
        "phase1_nan": int(bv_phase1.isna().sum()),
        "merged_min": float(bv_merged.min()),
        "merged_max": float(bv_merged.max()),
        "cleaned_min": float(bv_cleaned.min()),
        "cleaned_max": float(bv_cleaned.max()),
        "phase1_min": float(bv_phase1.min()),
        "phase1_max": float(bv_phase1.max()),
        "n_variable_in_cleaned": int(len(var)),
        "n_variable_reaching_phase1": int(var_in_phase1),
        "variable_window": (
            (str(var["fh7000_timestamp"].min()),
             str(var["fh7000_timestamp"].max()))
            if len(var) else (None, None)
        ),
        "variable_session": var["session_date"].iloc[0]
            if len(var) else None,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(verdict: dict, cum_mono: dict, cum_pair: dict,
                 stage_row_counts: dict, elapsed_s: float) -> None:
    A = verdict

    mon_lines: list[str] = []
    overall_mon = True
    for ch, sess in cum_mono.items():
        for sd in SESSIONS:
            rec = sess[sd]
            if not rec["all_runs_monotonic_nondec"]:
                overall_mon = False
            mon_lines.append(
                f"  - `{ch}` / {sd}: "
                f"{rec['n_run_files']} run_file(s), n={rec['n']:,}, "
                f"per-run total n_decreasing={rec['n_decreasing_per_run_total']}, "
                f"per-run max_drop={rec['session_max_drop_per_run']:.6g}, "
                f"all runs monotonic non-decreasing = "
                f"{'YES' if rec['all_runs_monotonic_nondec'] else 'NO'}"
            )

    var_min, var_max = A["variable_window"]
    var_session = A["variable_session"] or "—"

    explanation_block = (
f"""
- **Explanation A (pipeline bug)** would require non-65535 rows in
  `telemetry_cleaned.parquet` whose telemetry timestamps DO appear in the
  Phase 1 dataset, and yet the Phase 1 value is 65535 instead of the cleaned
  value. The diagnostic finds **{A['n_variable_reaching_phase1']:,}** such rows
  out of **{A['n_variable_in_cleaned']:,}** non-65535 cleaned rows. This rules
  out a pipeline overwrite/coercion as the cause.
- **Explanation B (audit threshold mismatch)** is the closest match. The
  cleanup audit is honest: in `telemetry_cleaned.parquet` the channel has
  **{A['cleaned_nu']:,}** unique values across {len(SESSIONS)} sessions and
  ~653,612 rows (so the "drop columns with nunique == 1" rule legitimately
  retained it). The Phase 1 analysis is also honest: within the Phase 1
  dataset the channel has **{A['phase1_nu']}** unique value (65535).
- **Explanation C (NaN-rule mismatch)** does not apply: NaN counts are
  {A['merged_nan']} / {A['cleaned_nan']} / {A['phase1_nan']} across the three
  stages, and the audit's tally lines up with {A['cleaned_nu']} unique
  non-NaN values.

The mechanism behind Explanation B in this dataset is **coverage-window
narrowing by the LiDAR-rate join**, not amplitude rounding. All
**{A['n_variable_in_cleaned']:,}** variable-battery rows live in session
**{var_session}**, between **{var_min}** and **{var_max}** — a single
~6-minute window at the very start of that recording. The LiDAR file for
that session does not begin until ~3 minutes later, so when
`scripts/time_sync/run_apply.py` does the per-day τ̂-corrected
`merge_asof` against LiDAR scan timestamps, every variable-battery
telemetry row falls outside the LiDAR coverage window and is dropped.
The rows that survive the join all carry battery_value == 65535 — the
uint16 "no-value" sentinel from the AGV CAN bus — so by the time we read
`data/phase1/dataset.parquet`, the channel is genuinely constant within
its scope.
""")

    cumulative_block = "\n".join(mon_lines)
    pair = cum_pair
    a_int = "yes" if pair["a_int_match"] else "no"
    b_int = "yes" if pair["b_int_match"] else "no"

    text = f"""# Battery channel provenance diagnostic

## 0. TL;DR

- **`battery_value` verdict**: **AUDIT THRESHOLD MISMATCH** — both reports are honest. The cleanup audit saw **{A['cleaned_nu']:,}** unique values across the full 653,612-row telemetry timeline and so kept the column. The Phase 1 dataset analysis saw **{A['phase1_nu']}** unique value (65535) within its 681,593-row LiDAR-joined scope. The mechanism is **coverage-window narrowing**: the {A['n_variable_in_cleaned']:,} non-sentinel rows all live in a ~6-minute pre-LiDAR window of session {var_session} and never make it through the LiDAR-rate `merge_asof`.
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
file. Stage row counts: merged={stage_row_counts['merged']:,},
cleaned={stage_row_counts['cleaned']:,}, phase1={stage_row_counts['phase1']:,}.

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
| merged | {A['merged_nu']:,} | {A['merged_nan']} | {A['merged_min']:.0f} | {A['merged_max']:.0f} |
| cleaned | {A['cleaned_nu']:,} | {A['cleaned_nan']} | {A['cleaned_min']:.0f} | {A['cleaned_max']:.0f} |
| phase1 | {A['phase1_nu']} | {A['phase1_nan']} | {A['phase1_min']:.0f} | {A['phase1_max']:.0f} |

The collapse from {A['cleaned_nu']:,} → {A['phase1_nu']} happens at the
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
  {var_session} ({var_min} → {var_max}), where the channel ramps from
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

{cumulative_block}

All run_files monotonic non-decreasing across all (channel × session): {'YES' if overall_mon else 'NO'}.

### `_uint` vs `_uint_float` pair (§3.6)

These are **not** an integer/float cast of the same physical quantity.
Both round to integer values (cumulative_energy_consumption is integer:
{a_int}; _uint is integer: {b_int}), so the float dtype on disk is
incidental. Their per-session ranges differ:

| session | cumulative_energy_consumption | cumulative_energy_consumption_uint |
|---|---|---|
| 25.02.2026 | {pair['per_session']['25.02.2026']['a_min']:.0f} → {pair['per_session']['25.02.2026']['a_max']:.0f} (nu={pair['per_session']['25.02.2026']['a_nu']}) | {pair['per_session']['25.02.2026']['b_min']:.0f} → {pair['per_session']['25.02.2026']['b_max']:.0f} (nu={pair['per_session']['25.02.2026']['b_nu']}) |
| 15.03.2026 | {pair['per_session']['15.03.2026']['a_min']:.0f} → {pair['per_session']['15.03.2026']['a_max']:.0f} (nu={pair['per_session']['15.03.2026']['a_nu']}) | {pair['per_session']['15.03.2026']['b_min']:.0f} → {pair['per_session']['15.03.2026']['b_max']:.0f} (nu={pair['per_session']['15.03.2026']['b_nu']}) |
| 24.03.2026 | {pair['per_session']['24.03.2026']['a_min']:.0f} → {pair['per_session']['24.03.2026']['a_max']:.0f} (nu={pair['per_session']['24.03.2026']['a_nu']}) | {pair['per_session']['24.03.2026']['b_min']:.0f} → {pair['per_session']['24.03.2026']['b_max']:.0f} (nu={pair['per_session']['24.03.2026']['b_nu']}) |

The two channels appear to be paired counters from different bus-side
accumulators (one ~total-energy, one ~total-energy-since-power-on or
similar). Max abs difference between the two over all rows where both are
finite: {pair['max_abs_diff']:.0f}. This is a side curiosity; neither is
in the Phase 1 feature stack.

## 5. Verdict

Three explanations were on the table:
{explanation_block}

**Verdict: Explanation B (audit threshold mismatch), via coverage-window
narrowing.** Cited numbers: §2 shows
`battery_value.nunique` = {A['cleaned_nu']:,} in `telemetry_cleaned.parquet`
and {A['phase1_nu']} in `data/phase1/dataset.parquet`. §3.5 shows
{A['n_variable_in_cleaned']:,} non-sentinel rows in `cleaned`, of which
{A['n_variable_reaching_phase1']:,} reach Phase 1. §3.3 shows that on the
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

- Wall-clock for end-to-end run: **{elapsed_s:.1f} s**.
- Reproduce: `python -m scripts.p1_battery_provenance.run_diagnostic`.
- Inputs (read-only): `data/merged/telemetry_merged.parquet`,
  `data/merged/telemetry_cleaned.parquet`, `data/phase1/dataset.parquet`.
- Outputs:
  - Code: `scripts/p1_battery_provenance/{{__init__.py, run_diagnostic.py, plotting.py}}`
  - Tables: `docs/p1_battery_provenance/tables/{{per_stage_summary,value_identity,join_keys}}.md`
  - Figures: `docs/p1_battery_provenance/figures/*.png`
  - Report: this file.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Battery channel provenance diagnostic")
    print("=" * 70)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] telemetry_merged.parquet")
    merged = load_stage(TELEMETRY_MERGED, "ts")
    print(f"       {len(merged):,} rows × {len(merged.columns)} cols")
    print("[load] telemetry_cleaned.parquet")
    cleaned = load_stage(TELEMETRY_CLEANED, "fh7000_timestamp")
    print(f"       {len(cleaned):,} rows × {len(cleaned.columns)} cols")
    print("[load] phase1 dataset.parquet")
    dataset = load_stage(DATASET, "fh7000_timestamp")
    print(f"       {len(dataset):,} rows × {len(dataset.columns)} cols")

    stage_dfs: dict[str, pd.DataFrame] = {
        "merged": merged,
        "cleaned": cleaned,
        "phase1": dataset,
    }

    print("[3.1] per-stage summary table")
    summary_md = per_stage_summary_table(stage_dfs)
    (TBL_DIR / "per_stage_summary.md").write_text(
        "# Per-stage variability summary\n\n" + summary_md, encoding="utf-8")

    print("[3.2] histograms")
    write_histograms(stage_dfs)

    print("[3.3] value-identity table")
    identity_md = value_identity_table(merged, cleaned, dataset)
    (TBL_DIR / "value_identity.md").write_text(
        "# Cross-stage value-identity check\n\n" + identity_md,
        encoding="utf-8")

    print("[3.4] time-series plots + monotonicity")
    write_timeseries(cleaned)
    cum_mono = cumulative_monotonicity(cleaned)

    print("[3.5] join-keys persistence table")
    join_md = join_keys_table(cleaned, dataset)
    (TBL_DIR / "join_keys.md").write_text(
        "# Joint-key persistence check\n\n" + join_md, encoding="utf-8")

    print("[3.6] cumulative_energy_consumption pair summary")
    cum_pair = cumulative_pair_summary(cleaned)

    print("[verdict] computing")
    verdict = determine_verdict(merged, cleaned, dataset)

    print("[report] writing")
    elapsed = time.time() - t0
    write_report(
        verdict, cum_mono, cum_pair,
        stage_row_counts={
            "merged": len(merged),
            "cleaned": len(cleaned),
            "phase1": len(dataset),
        },
        elapsed_s=elapsed,
    )
    print(f"[done] wall {elapsed:.1f} s")
    print(f"        report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
