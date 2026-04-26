"""Phase 4(c): within-day drift via sliding-window cross-correlation on the
per-day signal.

For each day, slide a fixed-length window over the (z_t, z_l) signature
pair and compute tau_hat per window. Per-day:
    * tau_hat(t), rho_peak(t)
    * linear regression of tau vs window-center time, slope in ms/min
      with a t-test for non-zero slope.
    * spread / std

Outputs:
    cache/drift_per_day.json
    figures/drift_per_day.png         per-day window-tau time series
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import lib


WINDOW_S = 300.0
WINDOW_S_LOW_RATE = 600.0
MIN_WINDOWS_FOR_SLOPE = 4
PROMINENCE_FOR_DRIFT = 0.10


def _windowed_taus(z_t: np.ndarray, z_l: np.ndarray,
                   t_grid: np.ndarray, gap_mask: np.ndarray,
                   window_s: float) -> list[dict]:
    """Yield (t_center_unix, tau, rho_peak) per non-overlapping window.

    Windows where >25% of the samples are inside a telemetry gap are
    skipped (the cross-correlation peak there is not trustworthy).
    """
    n = min(len(z_t), len(z_l))
    samples = int(window_s * C.GRID_HZ)
    n_windows = n // samples
    out = []
    for k in range(n_windows):
        s0 = k * samples; s1 = s0 + samples
        zt = z_t[s0:s1]; zl = z_l[s0:s1]
        gm = gap_mask[s0:s1] if gap_mask is not None else np.zeros_like(zt, dtype=bool)
        if gm.mean() > 0.25:
            continue
        if zt.std() < 1e-3 or zl.std() < 1e-3:
            continue
        zt = (zt - zt.mean()) / zt.std()
        zl = (zl - zl.mean()) / zl.std()
        lags, rho = lib.xcorr_lags(zt, zl, max_lag_s=C.SEARCH_RANGE_S)
        tau, k_peak, peak_y = lib.parabolic_refine(lags, rho)
        try:
            prom = lib.peak_prominence(rho, k_peak)
        except Exception:
            prom = 0.0
        if peak_y < PROMINENCE_FOR_DRIFT:
            continue
        t_center = float(0.5 * (t_grid[s0] + t_grid[s1 - 1]))
        out.append({
            "t_center_unix": t_center,
            "t_center_min": (t_center - float(t_grid[0])) / 60.0,
            "tau_hat_s": float(tau),
            "rho_peak": float(peak_y),
            "prominence": float(prom),
            "n_samples": s1 - s0,
            "gap_fraction": float(gm.mean()),
        })
    return out


def _slope_test(centers_s: np.ndarray, taus: np.ndarray,
                weights: np.ndarray | None = None):
    """Weighted linear regression tau = a + b*t. Returns (slope_ms_per_min,
    p_two_sided, se_slope_ms_per_min, intercept). Uses ordinary OLS when
    weights is None; otherwise weighted by the rho_peak^2 of each window.
    """
    n = len(centers_s)
    if n < 2 or np.std(centers_s) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if weights is None:
        weights = np.ones(n)
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    x_bar = float(np.sum(w * centers_s) / w_sum)
    y_bar = float(np.sum(w * taus) / w_sum)
    x_c = centers_s - x_bar
    y_c = taus - y_bar
    sxx = float(np.sum(w * x_c * x_c))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope_per_s = float(np.sum(w * x_c * y_c) / sxx)
    intercept = y_bar - slope_per_s * x_bar
    if n < 3:
        return slope_per_s * 60_000.0, float("nan"), float("nan"), intercept
    yhat = slope_per_s * (centers_s - x_bar) + y_bar
    resid = taus - yhat
    sse = float(np.sum(w * resid ** 2))
    var_slope = sse / max(1, (n - 2)) / sxx
    se_slope = float(np.sqrt(var_slope))
    t_stat = slope_per_s / max(se_slope, 1e-12)
    from scipy.stats import t as student_t
    p = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=n - 2)))
    return (slope_per_s * 60_000.0, p, se_slope * 60_000.0, intercept)


def _color_for_session_date(sd: str) -> str:
    return {
        "25.02.2026": "#d95f02",
        "15.03.2026": "#1b9e77",
        "24.03.2026": "#7570b3",
    }.get(sd, "#666666")


def render_drift_figure(per_day_results: list[dict], out_path: Path) -> Path:
    """Render the per-day window-tau time series figure.

    Pulled out of ``main()`` so ``run_report.py`` can re-render the figure
    from the cached drift JSON without re-running the (slow) sliding-window
    computation. ``main()`` calls this after computing the JSON; the report
    pipeline calls this after loading ``cache/drift_per_day.json``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(per_day_results), 1,
                             figsize=(11, 2.2 * len(per_day_results)),
                             sharex=False, squeeze=False)
    for ax, r in zip(axes[:, 0], per_day_results):
        sd = r["session_date"]; col = _color_for_session_date(sd)
        if r["n_windows"] == 0:
            ax.text(0.5, 0.5, "no usable windows", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        wins = r["windows"]
        t_disp = pd.to_datetime([w["t_center_unix"] for w in wins], unit="s")
        taus = np.array([w["tau_hat_s"] for w in wins])
        rhos = np.array([w["rho_peak"] for w in wins])
        sizes = 25 + 100 * (rhos - rhos.min()) / max(rhos.max() - rhos.min(), 1e-9)
        ax.scatter(t_disp, taus, s=sizes, c=col, alpha=0.85,
                   edgecolor="black", lw=0.4)
        ax.axhline(r["whole_day_tau_hat_s"], color=col, ls="--", lw=1.2,
                   label=f"whole-day τ̂={r['whole_day_tau_hat_s']:+.3f} s "
                         f"(±{r['whole_day_sigma_s']:.3f})")
        if np.isfinite(r["slope_ms_per_min"]):
            slope_per_s = r["slope_ms_per_min"] / 60_000.0
            t_unix = np.array([w["t_center_unix"] for w in wins])
            tx = np.linspace(t_unix.min(), t_unix.max(), 80)
            tx_disp = pd.to_datetime(tx, unit="s")
            ty = r["whole_day_tau_hat_s"] + slope_per_s * (tx - t_unix.mean())
            ax.plot(tx_disp, ty, color="black", lw=1.0, alpha=0.6,
                    label=f"slope = {r['slope_ms_per_min']:+.2f} ms/min "
                          f"(p={r['slope_p_value']:.3f})")
        ax.set_title(f"{sd}: per-window τ̂ vs wall-clock time "
                     f"({r['n_windows']} × {int(r['window_seconds']/60)}-min windows; "
                     f"std={r['within_day_std_s']:.3f} s, spread={r['spread_s']:.3f} s)",
                     fontsize=10)
        ax.set_ylabel("τ̂ (s)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("wall-clock time (CET)")
    fig.suptitle("Within-day drift: sliding-window τ̂ on the per-day signal",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    days = json.loads((C.CACHE_DIR / "days.json").read_text())
    offsets = json.loads((C.CACHE_DIR / "offsets_per_day.json").read_text())
    by_day = {r["session_date"]: r for r in offsets}

    per_day_results: list[dict] = []
    for d in days:
        if not d["eligible_for_estimation"]:
            continue
        sd = d["session_date"]
        chosen = by_day.get(sd)
        if chosen is None:
            continue
        sig_path = C.CACHE_DIR / f"sig_day_{sd}.npz"
        if not sig_path.exists():
            continue
        sig = np.load(sig_path)
        z_t = sig["z_t"]; z_l = sig["z_l"]; t_grid = sig["t_grid"]
        gm = sig["gap_mask"] if "gap_mask" in sig.files else np.zeros_like(z_t, bool)
        win_s = WINDOW_S_LOW_RATE if sd == "25.02.2026" else WINDOW_S
        windows = _windowed_taus(z_t, z_l, t_grid, gm, win_s)
        n_w = len(windows)

        if n_w >= MIN_WINDOWS_FOR_SLOPE:
            centers = np.array([w["t_center_unix"] for w in windows])
            taus = np.array([w["tau_hat_s"] for w in windows])
            rhos = np.array([w["rho_peak"] for w in windows])
            slope_ms_per_min, p_slope, se_slope, intercept = \
                _slope_test(centers, taus, weights=rhos**2)
        else:
            slope_ms_per_min = p_slope = se_slope = intercept = float("nan")

        if n_w >= 2:
            taus_arr = np.array([w["tau_hat_s"] for w in windows])
            spread = float(taus_arr.max() - taus_arr.min())
            std = float(taus_arr.std())
        else:
            spread, std = float("nan"), float("nan")

        if n_w > 0:
            print(f"[{sd}] {n_w} windows × {win_s/60:.0f} min  "
                  f"tau range [{min(w['tau_hat_s'] for w in windows):+.3f}, "
                  f"{max(w['tau_hat_s'] for w in windows):+.3f}] s  "
                  f"spread={spread:.3f}s  std={std:.3f}s  "
                  f"slope={slope_ms_per_min:+.2f} ms/min (p={p_slope:.3f})")
        else:
            print(f"[{sd}] no usable windows")

        per_day_results.append({
            "session_date": sd,
            "tel_start_unix": pd.Timestamp(d["tel_start"]).timestamp(),
            "window_seconds": win_s,
            "n_windows": n_w,
            "windows": windows,
            "slope_ms_per_min": slope_ms_per_min,
            "slope_se_ms_per_min": se_slope,
            "slope_p_value": p_slope,
            "spread_s": spread,
            "within_day_std_s": std,
            "whole_day_tau_hat_s": chosen["tau_hat"],
            "whole_day_sigma_s": chosen["sigma"],
        })

    out_path = C.CACHE_DIR / "drift_per_day.json"
    out_path.write_text(json.dumps(per_day_results, indent=2))
    print(f"\nWrote {out_path}")

    # ---- Figure: per-day window-tau time series ----
    p_per_day = render_drift_figure(per_day_results, C.FIG_DIR / "drift_per_day.png")
    print(f"  fig: {p_per_day}")


if __name__ == "__main__":
    main()
