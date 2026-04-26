"""P0.4 — LiDAR <-> path-loss residual univariate correlation.

Computes Spearman rho between each headline LiDAR scalar feature and the
per-row path-loss residual (from P0.2), per session, stratified by
is_AP_in_FOV. Drives Gate C — the project go/no-go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import lidar_features as LF
from .io_utils import load_joint, read_json, write_json


def _angle_to_AP(jp: pd.DataFrame, ap_coords: dict) -> np.ndarray:
    """Per-row body-frame angle (rad) from the AGV to the AP, in [-pi, pi]."""
    out = np.full(len(jp), np.nan, dtype=np.float64)
    for sd in C.SESSIONS:
        m = (jp["session_date"] == sd).to_numpy()
        x_ap = ap_coords[sd]["x_AP"]
        y_ap = ap_coords[sd]["y_AP"]
        x = jp.loc[m, "x_m"].to_numpy()
        y = jp.loc[m, "y_m"].to_numpy()
        h = jp.loc[m, "heading_rad"].to_numpy()
        dx = x_ap - x
        dy = y_ap - y
        world_ang = np.arctan2(dy, dx)
        body_ang = world_ang - h
        body_ang = (body_ang + np.pi) % (2 * np.pi) - np.pi
        out[m] = body_ang
    return out


def _is_in_fov(body_ang_rad: np.ndarray, fov_min_deg: float,
               fov_max_deg: float) -> np.ndarray:
    deg = np.rad2deg(body_ang_rad)
    return (deg >= fov_min_deg) & (deg <= fov_max_deg)


def main() -> None:
    print("=" * 72)
    print("P0.4  LiDAR <-> residual univariate correlation")
    print("=" * 72)
    fov = read_json(C.ARTIFACTS_DIR / "lidar_fov.json")
    ap_coords = read_json(C.ARTIFACTS_DIR / "ap_coords.json")
    feats = LF.compute_and_cache(force=False)
    resid = pd.read_parquet(C.CACHE_DIR / "path_loss_residuals.parquet")
    jp = load_joint(["session_date", "x_m", "y_m", "heading_rad"])
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")

    body_ang = _angle_to_AP(jp, ap_coords)
    in_fov = _is_in_fov(body_ang, fov["theta_min_deg"], fov["theta_max_deg"])

    # Build a joint frame indexed by joint row (positional)
    full = pd.DataFrame({
        "joint_idx": np.arange(len(jp), dtype=np.int64),
        "session_date": jp["session_date"].to_numpy(),
        "in_fov": in_fov,
        "anomaly_flag": mask["anomaly_flag"].to_numpy(),
    })
    for f in LF.FIELDS:
        full[f] = feats[f].to_numpy()

    # Merge residuals
    full = full.merge(resid[["joint_idx", "residual"]],
                      on="joint_idx", how="left")
    full = full[~full["anomaly_flag"] & full["residual"].notna()].reset_index(drop=True)
    print(f"  rows considered: {len(full):,}")

    # Build the 3 x 5 x 3 table
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
                                    sub_s["residual"].to_numpy(),
                                    nan_policy="omit")
                    rho, pval = float(res.statistic), float(res.pvalue)
                table_rows.append({
                    "session": sd, "stratum": stratum, "feature": f,
                    "n": int(len(sub_s)), "rho": rho, "pvalue": pval,
                })
    df_table = pd.DataFrame(table_rows)

    # Headline: max |rho| in_fov per session
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

    # Gate C
    abs_rhos = [headline[sd]["max_abs_rho"] for sd in C.SESSIONS
                if headline[sd]["max_abs_rho"] is not None]
    n_above_25 = sum(1 for r in abs_rhos if r > 0.25)
    n_above_15 = sum(1 for r in abs_rhos if r > 0.15)
    if n_above_25 >= 2:
        gate_c = "GREEN"
    elif n_above_15 >= 2:
        gate_c = "YELLOW"
    elif max(abs_rhos) > 0.15 if abs_rhos else False:
        gate_c = "YELLOW"
    else:
        gate_c = "RED"

    summary = {
        "table": table_rows,
        "headline_per_session_in_fov": headline,
        "gate_C": gate_c,
    }
    write_json(C.CACHE_DIR / "p0_4.json", summary)
    print()
    print("  Headline (max |rho| in-FOV per session):")
    for sd in C.SESSIONS:
        h = headline[sd]
        if h["max_abs_rho"] is None:
            print(f"    {sd}: insufficient data")
            continue
        print(f"    {sd}: |rho|={h['max_abs_rho']:.3f}  "
              f"({h['rho_signed']:+.3f}, feature={h['feature']}, n={h['n']:,})")
    print(f"  Gate C: {gate_c}")
    print("P0.4 done.")


if __name__ == "__main__":
    main()
