"""P0.4 v3 — LiDAR <-> path-loss residual univariate correlation (v3 mask).

Same per-session x per-feature x per-stratum table as v2, but operating on:
    - residual_v3 from path_loss_residuals_v3.parquet (truth-AP fit on v3-clean)
    - anomaly_mask_v3.parquet (confidence-threshold-based)

Drives the Gate C re-evaluation for the v3 update. Applies the v2 honesty
demotion rule (down-tier if max |rho| > 0.25 in-FOV but path-loss R^2 <= 0
on the same session).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import lidar_features as LF
from .io_utils import load_joint, read_json, write_json


def main() -> None:
    print("=" * 72)
    print("P0.4 v3  LiDAR <-> residual_v3 correlation (v3 mask)")
    print("=" * 72)
    feats = LF.compute_and_cache(force=False)
    resid = pd.read_parquet(C.CACHE_DIR / "path_loss_residuals_v3.parquet")
    apr = pd.read_parquet(C.CACHE_DIR / "ap_relative_features_v3.parquet")
    jp = load_joint(["session_date"])
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask_v3.parquet")

    full = pd.DataFrame({
        "joint_idx": np.arange(len(jp), dtype=np.int64),
        "session_date": jp["session_date"].to_numpy(),
        "in_fov": apr["is_AP_in_FOV"].to_numpy(),
        "anomaly_flag": mask["anomaly_flag"].to_numpy(),
    })
    for f in LF.FIELDS:
        full[f] = feats[f].to_numpy()

    full = full.merge(resid[["joint_idx", "residual_v3"]],
                      on="joint_idx", how="left")
    full = full[~full["anomaly_flag"] & full["residual_v3"].notna()].reset_index(drop=True)
    print(f"  rows considered: {len(full):,}")

    table_rows = []
    for sd in C.SESSIONS:
        sub = full[full["session_date"] == sd]
        for stratum, mask_s in (("overall", np.ones(len(sub), dtype=bool)),
                                ("in_fov", sub["in_fov"].to_numpy()),
                                ("out_of_fov", ~sub["in_fov"].to_numpy())):
            sub_s = sub[mask_s]
            for f in LF.FIELDS:
                if len(sub_s) < 50:
                    rho, pval = float("nan"), float("nan")
                else:
                    res = spearmanr(sub_s[f].to_numpy(),
                                    sub_s["residual_v3"].to_numpy(),
                                    nan_policy="omit")
                    rho, pval = float(res.statistic), float(res.pvalue)
                table_rows.append({
                    "session": sd, "stratum": stratum, "feature": f,
                    "n": int(len(sub_s)), "rho": rho, "pvalue": pval,
                })
    df_table = pd.DataFrame(table_rows)

    # Headline: max |rho| in-FOV per session
    in_fov_table = df_table[df_table["stratum"] == "in_fov"]
    headline = {}
    for sd in C.SESSIONS:
        s = in_fov_table[in_fov_table["session"] == sd]
        if len(s) == 0 or s["rho"].abs().max() == 0:
            headline[sd] = {"max_abs_rho": None, "feature": None}
        else:
            i = int(s["rho"].abs().idxmax())
            headline[sd] = {
                "max_abs_rho": float(abs(df_table.loc[i, "rho"])),
                "rho_signed": float(df_table.loc[i, "rho"]),
                "feature": str(df_table.loc[i, "feature"]),
                "p_value": float(df_table.loc[i, "pvalue"]),
                "n": int(df_table.loc[i, "n"]),
            }

    # Honesty demotion: per-session R^2 from P0.2 v3
    p0_2_v3 = read_json(C.CACHE_DIR / "p0_2_v3.json")
    r2_by_session = {sd: p0_2_v3["ap_results_v3"][sd]["R2"]
                     for sd in C.SESSIONS}

    # Apply Gate C rules from §6:
    # GREEN: max |rho| > 0.25 in-FOV on >= 2 of 3 sessions, with R^2 > 0 on those
    # YELLOW: same but R^2 <= 0 on one or more of those (honesty demotion)
    # YELLOW: max |rho| in [0.15, 0.25] in-FOV on most sessions
    # RED: max |rho| < 0.15 everywhere
    abs_rhos = {sd: headline[sd]["max_abs_rho"] for sd in C.SESSIONS
                if headline[sd]["max_abs_rho"] is not None}
    n_above_25 = sum(1 for r in abs_rhos.values() if r > 0.25)
    n_above_15 = sum(1 for r in abs_rhos.values() if r > 0.15)
    sessions_above_25_with_r2_le_0 = [
        sd for sd, r in abs_rhos.items()
        if r > 0.25 and r2_by_session[sd] <= 0
    ]

    if n_above_25 >= 2 and not sessions_above_25_with_r2_le_0:
        gate_c = "GREEN"
        demotion_note = ""
    elif n_above_25 >= 2:
        gate_c = "YELLOW"
        demotion_note = (f"honesty demotion: sessions with |rho|>0.25 "
                         f"but R^2<=0: {sessions_above_25_with_r2_le_0}")
    elif n_above_15 >= 2:
        gate_c = "YELLOW"
        demotion_note = ""
    elif abs_rhos and max(abs_rhos.values()) > 0.15:
        gate_c = "YELLOW"
        demotion_note = ""
    else:
        gate_c = "RED"
        demotion_note = ""

    summary = {
        "table": table_rows,
        "headline_per_session_in_fov": headline,
        "r2_by_session": r2_by_session,
        "gate_C": gate_c,
        "honesty_demotion_note": demotion_note,
    }
    write_json(C.CACHE_DIR / "p0_4_v3.json", summary)

    # v2-vs-v3 deltas (in-FOV headline)
    p0_4_v2 = read_json(C.CACHE_DIR / "p0_4_v2.json")
    print()
    print("  Headline (max |rho| in-FOV per session, residual_v3):")
    for sd in C.SESSIONS:
        h = headline[sd]
        h2 = p0_4_v2["headline_per_session_in_fov"].get(sd, {})
        if h["max_abs_rho"] is None:
            print(f"    {sd}: insufficient data")
            continue
        v2_rho = h2.get("max_abs_rho")
        delta = (h["max_abs_rho"] - v2_rho) if v2_rho is not None else None
        delta_str = f"  delta={delta:+.3f}" if delta is not None else ""
        print(f"    {sd}: |rho|={h['max_abs_rho']:.3f}  "
              f"({h['rho_signed']:+.3f}, feature={h['feature']}, n={h['n']:,})  "
              f"R2={r2_by_session[sd]:+.3f}{delta_str}")
    print(f"  Gate C: {gate_c}  {demotion_note}")
    print("P0.4 v3 done.")


if __name__ == "__main__":
    main()
