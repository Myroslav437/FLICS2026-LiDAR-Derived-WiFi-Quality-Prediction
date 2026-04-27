"""P0.2 v2 — Per-session log-distance path-loss fit at fixed ground-truth AP.

Replaces v1's joint (x_AP, y_AP, P0, n) least-squares fit. The lab-measured
AP coordinates from `config.AP_TRUTH` are taken as authoritative; only
(P0_d, n_d) are fit, in closed form via numpy.polyfit, against
X = log10(distance_to_AP_truth).

Outputs:
- artifacts/ap_coords.json      — ground-truth + (P0, n, R^2, CIs) per session
- cache/path_loss_residuals_v2.parquet — per-row residual_v2
- cache/p0_2_v2.json            — Gate A re-evaluation + summary
- figures/p0_2_residual_map_v2_<sd>.png  — residual maps with truth-AP star
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, write_json


N_BOOT = 200
N_DOWNSAMPLE_PER_SESSION = 30000
EPS_DIST_M = 0.05  # avoid log10(0) when AGV is right at AP


def _fit_p0_n(x_log: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """Closed-form 2-param linear regression:
        s = P0 - 10*n*log10(d) = P0 + slope * log10(d), with slope = -10*n.
    Returns (P0, n, R^2).
    """
    coef = np.polyfit(x_log, s, deg=1)  # [slope, intercept]
    slope, intercept = float(coef[0]), float(coef[1])
    p0 = intercept
    n = -slope / 10.0
    pred = intercept + slope * x_log
    ss_res = float(np.sum((s - pred) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p0, n, r2


def _bootstrap_p0_n(x_log: np.ndarray, s: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Return shape (n_boot, 2) of (P0, n) bootstrap estimates."""
    n = len(x_log)
    out = np.empty((n_boot, 2), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        coef = np.polyfit(x_log[idx], s[idx], deg=1)
        out[b, 0] = coef[1]              # P0
        out[b, 1] = -coef[0] / 10.0      # n
    return out


def _plot_residual_map(x: np.ndarray, y: np.ndarray, resid: np.ndarray,
                       x_ap: float, y_ap: float, sd: str, r2: float,
                       n_rows: int, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=resid, s=2, cmap="RdBu_r", vmin=-15, vmax=15)
    ax.plot([x_ap], [y_ap], marker="*", markersize=20, color="black",
            markeredgecolor="white", linewidth=2,
            label=f"AP truth ({x_ap:.3f}, {y_ap:.3f})")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"P0.2 v2 residual map  {sd}  (R^2={r2:.3f}, n={n_rows:,})")
    # Anchor the legend to the upper-right corner outside the data region —
    # 'best' tends to land on top of the AP star on the long-corridor plots.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9,
              framealpha=0.9)
    fig.colorbar(sc, ax=ax, label="signal_power - predicted [dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("P0.2 v2  Per-session path-loss fit at fixed ground-truth AP")
    print("=" * 72)
    cols = ["session_date", "x_m", "y_m", "signal_power", "speed_mps"]
    jp = load_joint(cols)
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")
    jp["anomaly_flag"] = mask["anomaly_flag"].to_numpy()

    rng = np.random.default_rng(C.SEED)

    ap_results: dict = {}
    residual_records: list[pd.DataFrame] = []

    for sd in C.SESSIONS:
        x_ap, y_ap = C.AP_TRUTH[sd]
        # All clean rows (motion-active + stopped) used for the per-row residual.
        # The fit itself uses motion-active rows only.
        sub_full = jp[(jp["session_date"] == sd) &
                      ~jp["anomaly_flag"] &
                      jp["signal_power"].notna()].copy()
        sub_fit = sub_full[sub_full["speed_mps"].abs() > 0.05]
        if len(sub_fit) > N_DOWNSAMPLE_PER_SESSION:
            idx_ds = rng.choice(len(sub_fit), N_DOWNSAMPLE_PER_SESSION,
                                replace=False)
            sub_fit_ds = sub_fit.iloc[idx_ds]
        else:
            sub_fit_ds = sub_fit

        x_fit = sub_fit_ds["x_m"].to_numpy()
        y_fit = sub_fit_ds["y_m"].to_numpy()
        s_fit = sub_fit_ds["signal_power"].to_numpy()
        d_fit = np.maximum(np.hypot(x_fit - x_ap, y_fit - y_ap), EPS_DIST_M)
        x_log_fit = np.log10(d_fit)

        p0, n, r2 = _fit_p0_n(x_log_fit, s_fit)
        # Bootstrap on a 5k subsample for speed; with 30k rows the CI is already
        # tight, and the fit cost dominates if we keep the full sample.
        if len(sub_fit_ds) > 5000:
            boot_idx = rng.choice(len(sub_fit_ds), 5000, replace=False)
            x_log_boot = x_log_fit[boot_idx]
            s_boot = s_fit[boot_idx]
        else:
            x_log_boot = x_log_fit
            s_boot = s_fit
        boot = _bootstrap_p0_n(x_log_boot, s_boot, N_BOOT, rng)
        p0_lo, p0_hi = np.nanpercentile(boot[:, 0], [2.5, 97.5])
        n_lo, n_hi = np.nanpercentile(boot[:, 1], [2.5, 97.5])

        # Per-row residual on the FULL (motion-active + stopped) clean set
        x_all = sub_full["x_m"].to_numpy()
        y_all = sub_full["y_m"].to_numpy()
        s_all = sub_full["signal_power"].to_numpy()
        d_all = np.maximum(np.hypot(x_all - x_ap, y_all - y_ap), EPS_DIST_M)
        pred = p0 - 10.0 * n * np.log10(d_all)
        resid = s_all - pred

        global_idx = sub_full.index.to_numpy()
        residual_records.append(pd.DataFrame({
            "joint_idx": global_idx.astype(np.int64),
            "session_date": sd,
            "residual_v2": resid.astype(np.float32),
        }))

        ap_results[sd] = {
            "x_AP": float(x_ap),
            "y_AP": float(y_ap),
            "x_AP_source": "lab_measured",
            "P0": float(p0),
            "n": float(n),
            "R2": float(r2),
            "P0_ci": [float(p0_lo), float(p0_hi)],
            "n_ci": [float(n_lo), float(n_hi)],
            "n_rows_used": int(len(sub_fit_ds)),
            "ap_pos_uncertainty_cm": C.AP_TRUTH_UNCERTAINTY_CM,
        }

        # Residual map (use fit-rows for visual consistency with v1)
        d_plot = np.maximum(np.hypot(x_fit - x_ap, y_fit - y_ap), EPS_DIST_M)
        resid_plot = s_fit - (p0 - 10.0 * n * np.log10(d_plot))
        _plot_residual_map(
            x_fit, y_fit, resid_plot, x_ap, y_ap, sd, r2,
            len(sub_fit_ds),
            C.FIGURES_DIR / f"p0_2_residual_map_v2_{sd.replace('.', '-')}.png",
        )
        print(f"  {sd}: AP=({x_ap:.3f}, {y_ap:.3f}) "
              f"P0={p0:.2f} dB  n={n:.3f}  R^2={r2:.3f}  n_rows={len(sub_fit_ds):,}")
        print(f"    bootstrap 95% CIs: P0=[{p0_lo:.2f},{p0_hi:.2f}]  "
              f"n=[{n_lo:.3f},{n_hi:.3f}]")

    # Persist residuals (v2)
    pd.concat(residual_records, ignore_index=True).to_parquet(
        C.CACHE_DIR / "path_loss_residuals_v2.parquet", index=False)

    # Authoritative ap_coords.json (replaces v1; v1 is at _archive/)
    write_json(C.ARTIFACTS_DIR / "ap_coords.json", ap_results)

    # Gate A re-evaluation per the brief's updated rules.
    r2s = [ap_results[sd]["R2"] for sd in C.SESSIONS]
    if all(r > 0.85 for r in r2s):
        gate_a = "YELLOW"  # distance dominates; LiDAR has little to add
    elif any(r < 0.4 for r in r2s):
        # Includes R^2 < 0 — log-distance from truth AP is worse than mean,
        # i.e. structurally non-radial propagation; LiDAR has *more* to do.
        gate_a = "GREEN"
    else:
        gate_a = "GREEN"  # 0.4 ≤ R² ≤ 0.85: moderate fit, structured residuals

    # Cross-session consistency check
    ns = np.array([ap_results[sd]["n"] for sd in C.SESSIONS])
    n_widths = np.array([
        ap_results[sd]["n_ci"][1] - ap_results[sd]["n_ci"][0]
        for sd in C.SESSIONS])
    flags: list[str] = []
    # Disagreement test: do the per-session n's mutually overlap their CIs?
    for i, sd_i in enumerate(C.SESSIONS):
        for j, sd_j in enumerate(C.SESSIONS):
            if j <= i:
                continue
            lo_i, hi_i = ap_results[sd_i]["n_ci"]
            lo_j, hi_j = ap_results[sd_j]["n_ci"]
            if hi_i < lo_j or hi_j < lo_i:
                flags.append(f"n_{sd_i} vs n_{sd_j} CIs disjoint")
    if float(np.ptp(ns)) > 2 * float(n_widths.max()):
        flags.append("n cross-session > 2x bootstrap CI")

    summary = {
        "ap_results": ap_results,
        "gate_A": gate_a,
        "flags": flags,
        "n_boot": N_BOOT,
    }
    write_json(C.CACHE_DIR / "p0_2_v2.json", summary)
    print(f"  Gate A: {gate_a}  flags={flags}")
    print("P0.2 v2 done.")


if __name__ == "__main__":
    main()
