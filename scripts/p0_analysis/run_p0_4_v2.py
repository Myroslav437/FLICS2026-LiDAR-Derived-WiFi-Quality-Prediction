"""P0.4 v2 — LiDAR <-> path-loss residual univariate correlation.

Same per-session x per-feature x per-stratum table as v1, but operating on:
    - residual_v2 from path_loss_residuals_v2.parquet (truth-AP path-loss fit)
    - is_AP_in_FOV from ap_relative_features_v2.parquet (truth-AP bearing)

Drives the Gate C re-evaluation with the cleaner inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import lidar_features as LF
from .io_utils import load_joint, write_json


def main() -> None:
    print("=" * 72)
    print("P0.4 v2  LiDAR <-> residual_v2 correlation (truth-AP fit)")
    print("=" * 72)
    feats = LF.compute_and_cache(force=False)
    resid = pd.read_parquet(C.CACHE_DIR / "path_loss_residuals_v2.parquet")
    apr = pd.read_parquet(C.CACHE_DIR / "ap_relative_features_v2.parquet")
    jp = load_joint(["session_date"])
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")

    full = pd.DataFrame({
        "joint_idx": np.arange(len(jp), dtype=np.int64),
        "session_date": jp["session_date"].to_numpy(),
        "in_fov": apr["is_AP_in_FOV"].to_numpy(),
        "anomaly_flag": mask["anomaly_flag"].to_numpy(),
    })
    for f in LF.FIELDS:
        full[f] = feats[f].to_numpy()

    full = full.merge(resid[["joint_idx", "residual_v2"]],
                      on="joint_idx", how="left")
    full = full[~full["anomaly_flag"] & full["residual_v2"].notna()].reset_index(drop=True)
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
                                    sub_s["residual_v2"].to_numpy(),
                                    nan_policy="omit")
                    rho, pval = float(res.statistic), float(res.pvalue)
                table_rows.append({
                    "session": sd, "stratum": stratum, "feature": f,
                    "n": int(len(sub_s)), "rho": rho, "pvalue": pval,
                })
    df_table = pd.DataFrame(table_rows)

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

    abs_rhos = [headline[sd]["max_abs_rho"] for sd in C.SESSIONS
                if headline[sd]["max_abs_rho"] is not None]
    n_above_25 = sum(1 for r in abs_rhos if r > 0.25)
    n_above_15 = sum(1 for r in abs_rhos if r > 0.15)
    if n_above_25 >= 2:
        gate_c = "GREEN"
    elif n_above_15 >= 2:
        gate_c = "YELLOW"
    elif (max(abs_rhos) > 0.15) if abs_rhos else False:
        gate_c = "YELLOW"
    else:
        gate_c = "RED"

    summary = {
        "table": table_rows,
        "headline_per_session_in_fov": headline,
        "gate_C": gate_c,
    }
    write_json(C.CACHE_DIR / "p0_4_v2.json", summary)

    print()
    print("  Headline (max |rho| in-FOV per session, residual_v2):")
    for sd in C.SESSIONS:
        h = headline[sd]
        if h["max_abs_rho"] is None:
            print(f"    {sd}: insufficient data")
            continue
        print(f"    {sd}: |rho|={h['max_abs_rho']:.3f}  "
              f"({h['rho_signed']:+.3f}, feature={h['feature']}, n={h['n']:,})")
    print(f"  Gate C: {gate_c}")
    print("P0.4 v2 done.")


if __name__ == "__main__":
    main()
