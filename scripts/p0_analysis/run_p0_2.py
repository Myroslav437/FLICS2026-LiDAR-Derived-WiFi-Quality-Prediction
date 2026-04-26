"""P0.2 — Per-session log-distance path-loss fit (AP calibration).

Fits S(x,y) = P0 - 10*n*log10(||(x,y)-(x_AP,y_AP)||) + eps
per session with bounded least-squares around the user-provided AP prior.
Bootstraps (x_AP, y_AP, P0, n); outputs residuals for P0.4.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .io_utils import load_joint, write_json


SEARCH_BOX_M = 3.0
N_BOOT = 200
N_DOWNSAMPLE_PER_SESSION = 30000   # cap to keep fit time reasonable
EPS_DIST_M = 0.05                   # avoid log(0) when AGV is right at AP


def _residual_fn(params, x, y, s):
    x_ap, y_ap, p0, n = params
    d = np.maximum(np.hypot(x - x_ap, y - y_ap), EPS_DIST_M)
    pred = p0 - 10.0 * n * np.log10(d)
    return pred - s


def _fit_session(sub: pd.DataFrame, x_init: float, y_init: float,
                 fix_xy: bool = False) -> tuple[np.ndarray, float]:
    """Fit (x_AP, y_AP, P0, n).  If fix_xy, hold (x_AP, y_AP) at the prior."""
    x = sub["x_m"].to_numpy()
    y = sub["y_m"].to_numpy()
    s = sub["signal_power"].to_numpy()
    if fix_xy:
        # Two free params only; reduce dimensionality so fit is well-posed
        def f(p, _x=x, _y=y, _s=s, _xa=x_init, _ya=y_init):
            d = np.maximum(np.hypot(_x - _xa, _y - _ya), EPS_DIST_M)
            return p[0] - 10.0 * p[1] * np.log10(d) - _s
        bounds = (np.array([-120.0, 1.0]), np.array([0.0, 6.0]))
        res = least_squares(f, np.array([-30.0, 2.5]), bounds=bounds,
                            method="trf", max_nfev=200)
        params = np.array([x_init, y_init, res.x[0], res.x[1]])
        pred = res.fun + s
    else:
        bounds = (
            np.array([x_init - SEARCH_BOX_M, y_init - SEARCH_BOX_M, -120.0, 1.0]),
            np.array([x_init + SEARCH_BOX_M, y_init + SEARCH_BOX_M, 0.0, 6.0]),
        )
        x0 = np.array([x_init, y_init, -30.0, 2.5])
        res = least_squares(_residual_fn, x0, args=(x, y, s), bounds=bounds,
                            method="trf", max_nfev=200)
        params = res.x
        pred = res.fun + s
    rss = float(np.sum((pred - s) ** 2))
    tss = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return params, r2


def _bootstrap(sub: pd.DataFrame, x_init: float, y_init: float,
               n_boot: int, rng: np.random.Generator,
               fix_xy: bool = False) -> np.ndarray:
    """Bootstrap (x_AP, y_AP, P0, n). When fix_xy, only (P0, n) are resampled
    and (x_AP, y_AP) are held at the prior — matches the fit's prior-fallback
    so the CIs are coherent with the reported point estimate."""
    n = len(sub)
    out = np.full((n_boot, 4), np.nan)
    x = sub["x_m"].to_numpy()
    y = sub["y_m"].to_numpy()
    s = sub["signal_power"].to_numpy()
    if fix_xy:
        bounds = (np.array([-120.0, 1.0]), np.array([0.0, 6.0]))
        x0 = np.array([-30.0, 2.5])
        def f(p, _x, _y, _s):
            d = np.maximum(np.hypot(_x - x_init, _y - y_init), EPS_DIST_M)
            return p[0] - 10.0 * p[1] * np.log10(d) - _s
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                res = least_squares(f, x0, args=(x[idx], y[idx], s[idx]),
                                    bounds=bounds, method="trf", max_nfev=80)
                out[b] = [x_init, y_init, res.x[0], res.x[1]]
            except Exception:
                pass
    else:
        bounds = (
            np.array([x_init - SEARCH_BOX_M, y_init - SEARCH_BOX_M, -120.0, 1.0]),
            np.array([x_init + SEARCH_BOX_M, y_init + SEARCH_BOX_M, 0.0, 6.0]),
        )
        x0 = np.array([x_init, y_init, -30.0, 2.5])
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                res = least_squares(_residual_fn, x0,
                                    args=(x[idx], y[idx], s[idx]),
                                    bounds=bounds, method="trf", max_nfev=80)
                out[b] = res.x
            except Exception:
                pass
    return out


def _plot_residual_map(sub: pd.DataFrame, params: np.ndarray, sd: str, out):
    x = sub["x_m"].to_numpy()
    y = sub["y_m"].to_numpy()
    s = sub["signal_power"].to_numpy()
    x_ap, y_ap, p0, n = params
    d = np.maximum(np.hypot(x - x_ap, y - y_ap), EPS_DIST_M)
    pred = p0 - 10.0 * n * np.log10(d)
    resid = s - pred

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=resid, s=2, cmap="RdBu_r", vmin=-15, vmax=15)
    ax.plot([x_ap], [y_ap], marker="*", markersize=18, color="black",
            markeredgecolor="white", linewidth=2, label="fitted AP")
    px, py = C.AP_PRIORS[sd]
    ax.plot([px], [py], marker="P", markersize=12, color="orange",
            markeredgecolor="black", label="prior AP")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"P0.2 residual map  {sd}  (R^2 displayed in caption)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="signal_power - predicted [dB]")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("P0.2  Per-session log-distance path-loss fit")
    print("=" * 72)
    cols = ["session_date", "x_m", "y_m", "signal_power", "speed_mps",
            "fh7000_timestamp"]
    jp = load_joint(cols)
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")
    jp["anomaly_flag"] = mask["anomaly_flag"].to_numpy()

    # motion-active rows excluding anomalies and rows with NaN signal_power
    keep = (jp["speed_mps"].abs() > 0.05) & (~jp["anomaly_flag"]) & jp["signal_power"].notna()
    jp_clean = jp[keep].reset_index(drop=True)
    print(f"  motion-active clean rows: {len(jp_clean):,} of {len(jp):,}")

    rng = np.random.default_rng(C.SEED)
    ap_results: dict = {}
    residual_records: list[pd.DataFrame] = []

    # We also need the FULL per-row residual (motion-active + stopped, all surviving rows)
    # for P0.4. The fit itself uses only motion-active rows; residual is computed for all rows
    # passing anomaly mask + signal_power not NaN.

    for sd in C.SESSIONS:
        sub_full = jp[(jp["session_date"] == sd) & ~jp["anomaly_flag"] &
                      jp["signal_power"].notna()].reset_index(drop=True)
        sub_fit = sub_full[sub_full["speed_mps"].abs() > 0.05].reset_index(drop=True)
        x_init, y_init = C.AP_PRIORS[sd]
        if len(sub_fit) > N_DOWNSAMPLE_PER_SESSION:
            idx = rng.choice(len(sub_fit), N_DOWNSAMPLE_PER_SESSION, replace=False)
            sub_fit_ds = sub_fit.iloc[idx].reset_index(drop=True)
        else:
            sub_fit_ds = sub_fit
        params, r2 = _fit_session(sub_fit_ds, x_init, y_init)
        x_ap, y_ap, p0, n = params
        disp = float(np.hypot(x_ap - x_init, y_ap - y_init))
        used_prior = False
        if disp > 3.0 or r2 < 0.05:
            # Fallback per brief: use the prior (x,y) and refit P0,n.
            params_p, r2_p = _fit_session(sub_fit_ds, x_init, y_init, fix_xy=True)
            x_ap, y_ap, p0, n = params_p
            disp = 0.0
            r2 = r2_p
            params = params_p
            used_prior = True

        # Bootstrap CIs
        boot_ds = sub_fit_ds.iloc[rng.choice(
            len(sub_fit_ds), min(len(sub_fit_ds), 5000), replace=False)
            ].reset_index(drop=True) if len(sub_fit_ds) > 5000 else sub_fit_ds
        boot = _bootstrap(boot_ds, x_init, y_init, N_BOOT, rng,
                          fix_xy=used_prior)
        ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)

        # Per-row residual on the full (motion-active + stopped) clean set
        x = sub_full["x_m"].to_numpy()
        y = sub_full["y_m"].to_numpy()
        s = sub_full["signal_power"].to_numpy()
        d = np.maximum(np.hypot(x - x_ap, y - y_ap), EPS_DIST_M)
        pred = p0 - 10.0 * n * np.log10(d)
        resid = s - pred

        # Map back to global joint_idx for cache
        # Build joint_idx -> sub_full mapping by re-deriving original indices
        # (simpler: re-compute boolean mask over original jp)
        global_idx = jp.index[(jp["session_date"] == sd) &
                              ~jp["anomaly_flag"] &
                              jp["signal_power"].notna()].to_numpy()
        residual_records.append(pd.DataFrame({
            "joint_idx": global_idx,
            "session_date": sd,
            "residual": resid,
        }))

        ap_results[sd] = {
            "prior_x": float(x_init), "prior_y": float(y_init),
            "x_AP": float(x_ap), "y_AP": float(y_ap),
            "displacement_m": disp,
            "used_prior": bool(used_prior),
            "P0": float(p0), "n": float(n), "R2": float(r2),
            "x_AP_ci": [float(ci[0, 0]), float(ci[1, 0])],
            "y_AP_ci": [float(ci[0, 1]), float(ci[1, 1])],
            "P0_ci":   [float(ci[0, 2]), float(ci[1, 2])],
            "n_ci":    [float(ci[0, 3]), float(ci[1, 3])],
            "n_fit_rows": int(len(sub_fit_ds)),
        }
        print(f"  {sd}: AP fit ({x_ap:.2f}, {y_ap:.2f})  "
              f"disp={disp:.2f}m  P0={p0:.1f}  n={n:.2f}  R^2={r2:.3f}")
        # Residual map
        _plot_residual_map(sub_fit_ds, params, sd,
                           C.FIGURES_DIR / f"p0_2_residual_map_{sd.replace('.', '-')}.png")

    # Concatenate residuals and write
    pd.concat(residual_records, ignore_index=True).to_parquet(
        C.CACHE_DIR / "path_loss_residuals.parquet", index=False)

    write_json(C.ARTIFACTS_DIR / "ap_coords.json", ap_results)

    # Gate A
    r2s = [ap_results[sd]["R2"] for sd in C.SESSIONS]
    disps = [ap_results[sd]["displacement_m"] for sd in C.SESSIONS]
    ns = [ap_results[sd]["n"] for sd in C.SESSIONS]
    if all(r > 0.85 for r in r2s):
        gate_a = "YELLOW"  # distance dominates; small LiDAR headroom
    elif any(0.4 <= r <= 0.85 for r in r2s):
        gate_a = "GREEN"   # moderate fit, structured residuals
    elif any(r < 0.4 for r in r2s):
        gate_a = "GREEN"   # blockage; LiDAR has more to do
    else:
        gate_a = "GREEN"
    flags = []
    if any(d > 3.0 for d in disps):
        flags.append("displacement>3m")
    # n / P0 cross-session inconsistency
    n_widths = [(ap_results[sd]["n_ci"][1] - ap_results[sd]["n_ci"][0]) for sd in C.SESSIONS]
    if max(ns) - min(ns) > 2 * max(n_widths):
        flags.append("n cross-session > 2x bootstrap CI")

    summary = {
        "ap_results": ap_results,
        "gate_A": gate_a,
        "flags": flags,
    }
    write_json(C.CACHE_DIR / "p0_2.json", summary)
    print(f"  Gate A: {gate_a}  flags={flags}")
    print("P0.2 done.")


if __name__ == "__main__":
    main()
