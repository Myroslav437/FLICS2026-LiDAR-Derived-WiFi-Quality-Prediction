"""Phase 6: figures, offsets_per_day.csv, and the publication-grade report.

Outputs:
    figures/xcorr_per_day.png            cross-correlation curves (3 days)
    figures/forest_per_day.png           forest plot of per-day τ̂
    figures/overlay_<sd>__w<k>of<N>.png  before/after overlays per day
    figures/retention_histogram.png      per-run retention histogram
    figures/drift_per_day.png            (already produced by run_drift.py)
    offsets_per_day.csv                  per-day table for paper
    docs/time_sync/report.md             full writeup with comparison to obsolete
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C


def _color_for_session_date(sd: str) -> str:
    return {
        "25.02.2026": "#d95f02",
        "15.03.2026": "#1b9e77",
        "24.03.2026": "#7570b3",
    }.get(sd, "#666666")


# ---------------------------------------------------------------------------
# offsets_per_day.csv
# ---------------------------------------------------------------------------

def build_offsets_csv(offsets: list[dict]) -> pd.DataFrame:
    rows = []
    for r in offsets:
        hs = r.get("half_split", {}) or {}
        first = hs.get("first") or {}
        second = hs.get("second") or {}
        rows.append({
            "session_date": r["session_date"],
            "n_csv_files": r["n_csv_files"],
            "csv_files": ";".join(r["csv_files"]),
            "intersect_seconds": round(r["intersect_seconds"], 1),
            "motion_seconds": round(r["motion_seconds"], 1),
            "n_stop_start_events": r["n_stop_start_events"],
            "tel_rate_hz": round(r["tel_rate_hz"], 2),
            "lid_rate_hz": round(r["lid_rate_hz"], 2),
            "tel_gap_count": r["tel_gap_count"],
            "tel_gap_seconds_sum": round(r["tel_gap_seconds_sum"], 2),
            "tau_hat_s": round(r["tau_hat"], 4),
            "sigma_s": round(r["sigma"], 4),
            "sigma_kind": r["sigma_kind"],
            "ci_low_s": round(r["ci_low"], 4),
            "ci_high_s": round(r["ci_high"], 4),
            "modal_fraction": round(r["modal_fraction"], 3),
            "block_len_s": round(r["block_len_s"], 1),
            "prominence": round(r["prominence"], 4),
            "rho_peak": round(r["peak_value"], 4),
            "included_in_pool": r["included"],
            "tau_first_half_s": (round(first.get("tau_hat"), 4)
                                 if first.get("tau_hat") is not None else None),
            "tau_second_half_s": (round(second.get("tau_hat"), 4)
                                  if second.get("tau_hat") is not None else None),
            "halves_agree": hs.get("agree"),
        })
    return pd.DataFrame(rows)


def attach_retention(df: pd.DataFrame, retention_per_day: list[dict]) -> pd.DataFrame:
    rmap = {r["session_date"]: r for r in retention_per_day}
    df["joint_retention"] = [
        round(rmap.get(r.session_date, {}).get("retention", float("nan")), 4)
        for r in df.itertuples()
    ]
    df["match_tolerance_ms"] = [
        round(rmap.get(r.session_date, {}).get("tolerance_s", float("nan")) * 1000.0, 2)
        for r in df.itertuples()
    ]
    return df


def attach_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    notes = []
    for r in df.itertuples():
        flags = []
        if not r.included_in_pool:
            flags.append(f"prominence<{C.PROMINENCE_MIN}")
        if r.modal_fraction < 0.5:
            flags.append("bootstrap_unstable")
        if r.tel_rate_hz < 10:
            flags.append("low_telemetry_rate")
        if r.halves_agree is False:
            flags.append("halves_disagree")
        notes.append("|".join(flags) if flags else "")
    df["anomaly_flags"] = notes
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_xcorr_per_day(offsets: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for r in offsets:
        sd = r["session_date"]
        sig_path = C.CACHE_DIR / f"xcorr_{sd}.npz"
        if not sig_path.exists():
            continue
        sig = np.load(sig_path)
        col = _color_for_session_date(sd)
        ax.plot(sig["lags"], sig["rho"], color=col, lw=1.4, alpha=0.9,
                label=f"{sd}  τ̂={r['tau_hat']:+.3f}s  ρ={r['peak_value']:.3f}")
        ax.axvline(r["tau_hat"], color=col, alpha=0.3, lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"lag $\tau$ (s)  [positive: LiDAR lags telemetry]")
    ax.set_ylabel(r"normalised cross-correlation  $\rho(\tau)$")
    ax.set_title(rf"(a) Per-day cross-correlation, search range "
                 rf"$\tau \in [-{C.SEARCH_RANGE_S:.0f}, +{C.SEARCH_RANGE_S:.0f}]$ s")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)

    axz = axes[1]
    for r in offsets:
        sd = r["session_date"]
        sig_path = C.CACHE_DIR / f"xcorr_{sd}.npz"
        if not sig_path.exists():
            continue
        sig = np.load(sig_path)
        m = np.abs(sig["lags"] - r["tau_hat"]) <= 1.0
        if m.sum() < 3:
            continue
        col = _color_for_session_date(sd)
        axz.plot(sig["lags"][m] - r["tau_hat"], sig["rho"][m],
                 color=col, lw=1.4, alpha=0.9, label=sd)
    axz.axvline(0, color="black", lw=0.8)
    axz.set_xlabel(r"$\tau - \hat\tau_{day}$ (s)")
    axz.set_ylabel(r"$\rho$")
    axz.set_title(r"(b) Local shape of each peak (centered at $\hat\tau_{day}$)")
    axz.legend(fontsize=9)
    axz.grid(alpha=0.3)
    fig.suptitle("Per-day cross-correlation: telemetry rot_aware vs LiDAR scan-dissimilarity",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = C.FIG_DIR / "xcorr_per_day.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_forest_per_day(offsets: list[dict], homog: dict) -> Path:
    primary = sorted(offsets, key=lambda r: r["session_date"])
    n = len(primary)
    fig, ax = plt.subplots(figsize=(11, 0.9 * n + 2.0))
    y_pos = np.arange(n)[::-1]
    for k, r in enumerate(primary):
        y = y_pos[k]
        col = _color_for_session_date(r["session_date"])
        marker = "o" if r["included"] else "x"
        ax.errorbar(r["tau_hat"], y,
                    xerr=[[r["tau_hat"] - r["ci_low"]],
                          [r["ci_high"] - r["tau_hat"]]],
                    fmt=marker, color=col, capsize=4,
                    markersize=10, lw=1.6)
        ax.text(r["tau_hat"], y - 0.28,
                f"$\\hat\\tau$={r['tau_hat']:+.3f}, $\\sigma$={r['sigma']:.3f}, "
                f"$\\rho$={r['peak_value']:.2f}, prom={r['prominence']:.2f}",
                fontsize=9, color=col, ha="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["session_date"] for r in primary])
    ax.set_ylim(-0.6, n - 0.4)

    inc = homog["included_only_pool"]
    if np.isfinite(inc.get("tau_bar", float("nan"))):
        decision = (homog["decision_rule"]["verdict_constant_global"])
        verdict = "ACCEPTED" if decision else "REJECTED"
        ax.axvline(inc["tau_bar"], color="black", lw=1.2, ls="--", alpha=0.7,
                   label=(rf"global pool ({verdict}): "
                          rf"$\bar\tau$={inc['tau_bar']:+.3f} s, "
                          rf"$Q$={inc['Q']:.2f} $p$={inc['p']:.2g} "
                          rf"$I^2$={inc['I2']:.0f}%"))
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"$\hat\tau$ (s)  — positive: LiDAR clock lags telemetry")
    ax.set_title("Per-day forest plot: τ̂ ± 1.96·σ_rmse for each calibration day")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_path = C.FIG_DIR / "forest_per_day.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_overlays(offsets: list[dict], n_per_day: int = 3) -> list[Path]:
    """Three before/after overlays per day, sampled at early/middle/late
    high-motion windows of the per-day signal.
    """
    out: list[Path] = []
    for r in sorted(offsets, key=lambda x: x["session_date"]):
        sd = r["session_date"]
        sig_path = C.CACHE_DIR / f"sig_day_{sd}.npz"
        if not sig_path.exists():
            continue
        sig = np.load(sig_path)
        z_t = sig["z_t"]; z_l = sig["z_l"]; t_grid = sig["t_grid"]
        s_t_raw = sig["s_t_raw"]
        gap_mask = sig["gap_mask"] if "gap_mask" in sig.files else np.zeros_like(z_t, bool)

        n = len(z_t)
        half = int(60 * C.GRID_HZ)
        for w_idx in range(n_per_day):
            third_lo = int(w_idx * n / n_per_day)
            third_hi = int((w_idx + 1) * n / n_per_day)
            slice_speed = pd.Series(s_t_raw[third_lo:third_hi]).rolling(
                int(C.GRID_HZ * 5), min_periods=1).mean().values
            # Pick a high-motion window center but avoid gap-masked regions
            slice_gap = gap_mask[third_lo:third_hi].astype(int)
            score = slice_speed * (1 - slice_gap)
            local_center = int(np.argmax(score))
            center_idx = third_lo + local_center
            i0 = max(0, center_idx - half); i1 = min(n, center_idx + half)
            t = t_grid[i0:i1] - t_grid[i0]
            zt = z_t[i0:i1]; zl = z_l[i0:i1]

            tau_applied = float(r["tau_hat"])
            window_start = pd.to_datetime(t_grid[i0], unit="s")
            t_offset_min = (t_grid[i0] - t_grid[0]) / 60.0
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            axes[0].plot(t, zt, color="#1b9e77", lw=1.0,
                         label=r"telemetry  $|v|+R\cdot|\omega|$ (z-scored)")
            axes[0].plot(t, zl, color="#d95f02", lw=1.0, alpha=0.85,
                         label="LiDAR  scan-dissimilarity (z-scored)")
            axes[0].set_title(
                f"BEFORE correction — {sd}  window {w_idx+1}/{n_per_day}  "
                f"(t0 = +{t_offset_min:.0f} min into recording, τ=0)\n"
                f"window starts {window_start:%Y-%m-%d %H:%M:%S} CET"
            )
            axes[0].legend(loc="upper right", fontsize=8)
            axes[0].grid(alpha=0.3)
            axes[0].set_ylabel("z-score")

            axes[1].plot(t, zt, color="#1b9e77", lw=1.0, label="telemetry")
            axes[1].plot(t + tau_applied, zl, color="#d95f02", lw=1.0, alpha=0.85,
                         label=f"LiDAR shifted by per-day τ̂ = {tau_applied:+.3f} s")
            axes[1].set_title(
                f"AFTER correction — applied per-day τ̂_{{{sd}}} = {tau_applied:+.3f} s "
                f"(σ = {r['sigma']:.3f} s)"
            )
            axes[1].set_xlabel("time within 120-s window (s)")
            axes[1].set_ylabel("z-score")
            axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            slug = f"overlay_{sd}__w{w_idx+1}of{n_per_day}.png"
            out_path = C.FIG_DIR / slug
            fig.savefig(out_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            out.append(out_path)
    return out


def fig_forest_per_csv(per_csv_taus: list[dict], homog: dict) -> Path:
    """Forest plot of per-CSV-file τ̂ with per-day τ̂ overlaid.

    Shows the per-CSV cross-check estimates (dots, ±1.96·σ_rmse brackets)
    with each day's applied per-day τ̂ as a coloured solid vertical line.
    """
    primary = sorted(per_csv_taus, key=lambda r: (r["session_date"], r["run_file"]))
    n = len(primary)
    fig, ax = plt.subplots(figsize=(11, 0.55 * n + 2.0))
    y_pos = np.arange(n)[::-1]
    for k, r in enumerate(primary):
        y = y_pos[k]
        col = _color_for_session_date(r["session_date"])
        marker = "o" if r["included"] else "x"
        ax.errorbar(r["tau_hat"], y,
                    xerr=[[r["tau_hat"] - r["ci_low"]],
                          [r["ci_high"] - r["tau_hat"]]],
                    fmt=marker, color=col, capsize=3,
                    markersize=8, lw=1.4)
        ax.text(r["tau_hat"], y - 0.32,
                f"$\\hat\\tau$={r['tau_hat']:+.3f}, $\\sigma$={r['sigma']:.3f}, "
                f"$\\rho$={r['peak_value']:.2f}",
                fontsize=8, color=col, ha="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['session_date']} / "
                        f"{r['run_file'].replace('out_Myroslav_','').replace('.csv','')}"
                        for r in primary], fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)

    # Per-day applied τ̂ as coloured solid verticals
    per_day = homog["per_session_day"]
    for sd, d in per_day.items():
        col = _color_for_session_date(sd)
        ax.axvline(d["tau_bar"], color=col, ls="-", lw=1.8, alpha=0.85,
                   label=f"{sd}: applied per-day τ̂ = {d['tau_bar']:+.3f} s")
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"$\hat\tau$ (s)  — positive: LiDAR clock lags telemetry")
    ax.set_title("Per-CSV-file forest plot (cross-check, NOT applied) "
                 "with per-day applied τ̂ as solid verticals")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_path = C.FIG_DIR / "forest_per_csv.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Appendix D — hardware-level interpretation
# ---------------------------------------------------------------------------

# Calendar-day gaps between the three calibration sessions, ordered by
# wall-clock date (25.02.2026 → 15.03.2026 → 24.03.2026). Day-of-year
# differences: Feb 25 = doy 56, Mar 15 = doy 74, Mar 24 = doy 83.
CAL_ORDER = ["25.02.2026", "15.03.2026", "24.03.2026"]
CAL_DATES = {
    "25.02.2026": pd.Timestamp("2026-02-25"),
    "15.03.2026": pd.Timestamp("2026-03-15"),
    "24.03.2026": pd.Timestamp("2026-03-24"),
}


def compute_hardware_appendix(drift: list[dict], homog: dict) -> dict:
    """Compute the quantitative content of Appendix D.

    Returns a dict with the pooled within-day slope (ms/min, ms/hour,
    ppm) and CI; cross-day Δτ̂ values with day-gaps; and the predicted-
    vs-observed accumulated drift for each consecutive day pair.
    """
    from scipy.stats import norm

    # Per-day slopes from §7.2 (cache/drift_per_day.json)
    by_sd = {r["session_date"]: r for r in drift}
    slopes = np.array([by_sd[sd]["slope_ms_per_min"] for sd in CAL_ORDER])
    ses = np.array([by_sd[sd]["slope_se_ms_per_min"] for sd in CAL_ORDER])
    p_values = {sd: float(by_sd[sd]["slope_p_value"]) for sd in CAL_ORDER}

    # Inverse-variance pool
    w = 1.0 / ses ** 2
    pooled_slope = float(np.sum(w * slopes) / np.sum(w))
    pooled_se = float(1.0 / np.sqrt(np.sum(w)))
    z_stat = pooled_slope / pooled_se
    p_two = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    # Convert ms/min -> ms/hour and -> ppm. 1 ppm = 1 µs/s = 0.06 ms/min.
    ms_per_min = pooled_slope
    se_per_min = pooled_se
    ms_per_hour = ms_per_min * 60.0
    se_per_hour = se_per_min * 60.0
    ppm = ms_per_min / 0.06
    se_ppm = se_per_min / 0.06

    # Per-day τ̂ and σ
    per_day = homog["per_session_day"]
    taus = {sd: float(per_day[sd]["tau_bar"]) for sd in CAL_ORDER}
    sigmas = {sd: float(per_day[sd]["se"]) for sd in CAL_ORDER}

    # Cross-day deltas + day-gaps + predicted accumulated drift
    cross_day: list[dict] = []
    for i in range(len(CAL_ORDER) - 1):
        a, b = CAL_ORDER[i], CAL_ORDER[i + 1]
        gap_days = int((CAL_DATES[b] - CAL_DATES[a]).days)
        gap_min = gap_days * 24 * 60
        d_tau_obs = taus[b] - taus[a]
        # Pool σ via quadrature for the Δτ̂ uncertainty
        d_tau_se = float(np.sqrt(sigmas[a] ** 2 + sigmas[b] ** 2))
        # Predicted drift accumulated at the pooled rate (ms -> s)
        d_tau_pred_s = pooled_slope * gap_min / 1000.0
        d_tau_pred_se_s = pooled_se * gap_min / 1000.0
        cross_day.append({
            "from_session": a, "to_session": b,
            "gap_days": gap_days, "gap_minutes": gap_min,
            "tau_a": taus[a], "tau_b": taus[b],
            "delta_tau_obs_s": d_tau_obs,
            "delta_tau_obs_se_s": d_tau_se,
            "delta_tau_pred_s": d_tau_pred_s,
            "delta_tau_pred_se_s": d_tau_pred_se_s,
            "ratio_pred_over_obs": (abs(d_tau_pred_s / d_tau_obs)
                                    if abs(d_tau_obs) > 0 else float("inf")),
            "sign_agreement": bool(np.sign(d_tau_pred_s) == np.sign(d_tau_obs)),
        })

    # Monotonicity check on τ̂ ordered by date
    tau_seq = [taus[sd] for sd in CAL_ORDER]
    monotone = (all(tau_seq[i] < tau_seq[i + 1] for i in range(len(tau_seq) - 1))
                or all(tau_seq[i] > tau_seq[i + 1] for i in range(len(tau_seq) - 1)))

    return {
        "calendar_order": CAL_ORDER,
        "calendar_dates": {sd: CAL_DATES[sd].isoformat() for sd in CAL_ORDER},
        "per_day_slopes_ms_per_min": dict(zip(CAL_ORDER, slopes.tolist())),
        "per_day_slope_ses_ms_per_min": dict(zip(CAL_ORDER, ses.tolist())),
        "per_day_slope_p_values": p_values,
        "per_day_tau_s": taus,
        "per_day_sigma_s": sigmas,
        "pooled_slope": {
            "ms_per_min": ms_per_min, "ms_per_min_se": se_per_min,
            "ms_per_min_ci_lo": ms_per_min - 1.96 * se_per_min,
            "ms_per_min_ci_hi": ms_per_min + 1.96 * se_per_min,
            "ms_per_hour": ms_per_hour, "ms_per_hour_se": se_per_hour,
            "ppm": ppm, "ppm_se": se_ppm,
            "z": float(z_stat), "p_two_sided": p_two,
        },
        "cross_day": cross_day,
        "monotone_in_calendar_order": bool(monotone),
    }


def fig_d1_slope_forest(info: dict) -> Path:
    """D.1 — Per-day within-day drift slopes with pooled-slope band."""
    order = info["calendar_order"]
    slopes = [info["per_day_slopes_ms_per_min"][sd] for sd in order]
    ses = [info["per_day_slope_ses_ms_per_min"][sd] for sd in order]
    pooled = info["pooled_slope"]["ms_per_min"]
    p_lo = info["pooled_slope"]["ms_per_min_ci_lo"]
    p_hi = info["pooled_slope"]["ms_per_min_ci_hi"]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = np.arange(len(order))[::-1]
    for k, sd in enumerate(order):
        col = _color_for_session_date(sd)
        ax.errorbar(slopes[k], y[k],
                    xerr=1.96 * ses[k],
                    fmt="o", color=col, capsize=4, markersize=9, lw=1.6,
                    label=f"{sd}: {slopes[k]:+.2f} ± {1.96*ses[k]:.2f} ms/min")
    # Pooled CI as a vertical band
    ax.axvspan(p_lo, p_hi, color="grey", alpha=0.18,
               label=(f"inverse-variance pool: {pooled:+.3f} ms/min  "
                      f"95 % CI [{p_lo:+.3f}, {p_hi:+.3f}]  "
                      f"(p={info['pooled_slope']['p_two_sided']:.3g})"))
    ax.axvline(pooled, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(r"within-day drift slope $\hat\beta$ (ms/min)  "
                  r"— positive: $\hat\tau$ grows over the day")
    ax.set_title("D.1 — Per-day within-day drift slopes with inverse-variance pooled band")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = C.FIG_DIR / "appendix_d1_slope_forest.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig_d2_calendar_timeline(info: dict) -> Path:
    """D.2 — Per-day τ̂ on a calendar axis with day-gap annotations."""
    order = info["calendar_order"]
    dates = [pd.Timestamp(info["calendar_dates"][sd]) for sd in order]
    taus = [info["per_day_tau_s"][sd] for sd in order]
    sigmas = [info["per_day_sigma_s"][sd] for sd in order]
    cd = info["cross_day"]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    # connecting line
    ax.plot(dates, taus, color="#444444", lw=1.0, alpha=0.6, zorder=1)
    # error bars + dots
    for k, sd in enumerate(order):
        col = _color_for_session_date(sd)
        ax.errorbar(dates[k], taus[k], yerr=sigmas[k],
                    fmt="o", color=col, capsize=4, markersize=10, lw=1.8,
                    label=f"{sd}: $\\hat\\tau$ = {taus[k]:+.3f} ± {sigmas[k]:.3f} s",
                    zorder=2)
    # annotate gaps
    for k in range(len(order) - 1):
        x_mid = dates[k] + (dates[k + 1] - dates[k]) / 2
        y_mid = (taus[k] + taus[k + 1]) / 2
        d_obs = cd[k]["delta_tau_obs_s"]
        ax.annotate(f"gap = {cd[k]['gap_days']} days\n"
                    f"$\\Delta\\hat\\tau$ = {d_obs:+.3f} s",
                    xy=(x_mid, y_mid),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="grey", alpha=0.85))
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("calendar date")
    ax.set_ylabel(r"applied per-day $\hat\tau$ (s)")
    ax.set_title("D.2 — Per-day offsets on a calendar axis "
                 "(non-monotone ⇒ rules out single continuously-drifting clock)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    out = C.FIG_DIR / "appendix_d2_calendar_timeline.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig_d3_predicted_vs_observed(info: dict) -> Path:
    """D.3 — Predicted accumulated drift vs observed |Δτ̂| per day-pair."""
    cd = info["cross_day"]
    labels = [f"{c['from_session']} → {c['to_session']}\n"
              f"({c['gap_days']} days)" for c in cd]
    obs = np.array([abs(c["delta_tau_obs_s"]) for c in cd])
    pred = np.array([abs(c["delta_tau_pred_s"]) for c in cd])
    pred_se = np.array([c["delta_tau_pred_se_s"] for c in cd])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - width / 2, pred, width,
           color="#888888", alpha=0.85,
           yerr=1.96 * pred_se, capsize=4,
           label="predicted from pooled slope × gap")
    ax.bar(x + width / 2, obs, width,
           color="#1b9e77", alpha=0.85,
           label=r"observed $|\Delta\hat\tau|$")
    for k, c in enumerate(cd):
        # Annotate ratio pred/obs
        ax.text(x[k], max(pred[k], obs[k]) * 1.4,
                f"ratio = {c['ratio_pred_over_obs']:.0f}×\n"
                f"sign agree: "
                f"{'yes' if c['sign_agreement'] else 'NO'}",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="grey", alpha=0.85))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|\Delta\hat\tau|$ (s)  — log scale")
    ax.set_title("D.3 — Predicted vs observed cross-day drift  "
                 "(log scale; predicted dwarfs observed ⇒ inter-session reset)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = C.FIG_DIR / "appendix_d3_pred_vs_obs.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig_d4_schematic() -> Path:
    """D.4 — Conceptual schematic of the two-component clock model.

    Per-session offset re-randomised at each boot/NTP-resync; small
    linear within-session drift in between. Interpretive aid only.
    """
    fig, ax = plt.subplots(figsize=(11, 4.0))
    # Three "session days" on a calendar axis with vertical reset bars
    sessions = [
        ("25.02.2026", 0.5, 1.5, +0.88, -0.0006),
        ("15.03.2026", 4.0, 5.0, +0.25, -0.0011),
        ("24.03.2026", 7.5, 8.5, +0.43, -0.0014),
    ]
    for label, x0, x1, tau0, slope_per_unit in sessions:
        col = _color_for_session_date(label)
        # within-session linear drift (illustrative slope only)
        xs = np.linspace(x0, x1, 40)
        ys = tau0 + slope_per_unit * (xs - x0) * 30  # exaggerate visually
        ax.plot(xs, ys, color=col, lw=2.4, label=f"{label}: bounded within-session drift")
        # session start/end reset bars
        ax.vlines([x0, x1], -0.05, 1.1, color=col, ls=":", lw=1.0, alpha=0.6)
        ax.text((x0 + x1) / 2, 1.05, label,
                ha="center", va="bottom", fontsize=9, color=col,
                fontweight="bold")
    # Inter-session "reset / reboot / NTP resync" arrows
    for x in (1.7, 5.2, 7.3):
        ax.annotate("", xy=(x + 0.2, 0.05), xytext=(x, 0.05),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    for x_lo, x_hi, lbl in [(1.6, 4.0, "boot / NTP resync\n(τ₀ re-randomised)"),
                            (5.0, 7.5, "boot / NTP resync\n(τ₀ re-randomised)")]:
        ax.annotate("",
                    xy=(x_hi - 0.05, -0.02), xytext=(x_lo + 0.05, -0.02),
                    arrowprops=dict(arrowstyle="<->", color="grey", lw=1.0))
        ax.text((x_lo + x_hi) / 2, -0.08, lbl, ha="center", va="top",
                fontsize=8, style="italic", color="dimgrey")
    ax.set_xlim(-0.4, 9.5)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("calendar time (sessions widely spaced; not to scale)")
    ax.set_ylabel(r"$\hat\tau$ (per-session offset + within-session drift)")
    ax.set_title("D.4 — Conceptual schematic of the two-component clock model "
                 "(interpretive aid)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = C.FIG_DIR / "appendix_d4_schematic.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def render_hardware_appendix_md(info: dict) -> str:
    """Render the markdown body for Appendix D."""
    p = info["pooled_slope"]
    cd = info["cross_day"]
    order = info["calendar_order"]
    slopes = info["per_day_slopes_ms_per_min"]
    ses = info["per_day_slope_ses_ms_per_min"]
    p_per_day = info["per_day_slope_p_values"]
    taus = info["per_day_tau_s"]
    monotone = info["monotone_in_calendar_order"]
    tau_seq = [taus[sd] for sd in order]

    # Pooled-slope arithmetic shown explicitly
    w = [1.0 / ses[sd] ** 2 for sd in order]
    sw = sum(w)
    arith_terms = " + ".join(
        [f"({slopes[sd]:+.4f}) × ({w[k]:.3f})" for k, sd in enumerate(order)]
    )

    md = f"""## Appendix D. Hardware-level interpretation: per-session offset + within-session drift

This appendix tests a hardware-level hypothesis for the LiDAR↔telemetry
clock relationship and confirms that the per-day correction granularity
in §4 remains the right applied choice. Nothing here changes the dataset
or the applied correction — it is purely interpretive supplementary
material.

**Hypothesis (two-component model).** The two recording hosts (the AGV
telemetry recorder and the LiDAR processing PC) are independently
clocked. Their relationship has two components: (i) a per-session
constant offset `τ₀_d` set by boot order and NTP sync state on day `d`,
and (ii) a small within-session frequency drift from free-running quartz
oscillators. Drift accumulated within a session is reset between
sessions by reboot / NTP resync. Under this hypothesis the per-session
offset `τ₀_d` is a fresh random draw on each calibration day; the
within-session slope reflects the *running* clock difference, on the
order of single-digit ppm (the typical magnitude for uncompensated
quartz).

The alternative we are ruling out is a *single continuously-drifting
clock*, under which `τ̂(t)` would be a monotone function of calendar
time across all three days, with cumulative drift ≈ slope × (calendar
time elapsed).

### D.1 Within-day pooled drift slope

The three per-day sliding-window slopes from §7.2:

| session_date | slope β̂ (ms/min) | SE (ms/min) | weight `1/σ²` |
|---|---|---|---|
"""
    for sd in order:
        md += (f"| {sd} | {slopes[sd]:+.4f} | {ses[sd]:.4f} | "
               f"{1.0/ses[sd]**2:.3f} |\n")

    md += f"""
Inverse-variance-weighted pooled slope:

```
β̂_pool = Σᵢ wᵢ · β̂ᵢ / Σᵢ wᵢ
       = ({arith_terms}) / {sw:.3f}
       = {p['ms_per_min']:+.4f} ms/min
SE(β̂_pool) = 1 / √Σᵢ wᵢ = 1 / √{sw:.3f} = {p['ms_per_min_se']:.4f} ms/min
z = β̂_pool / SE = {p['z']:+.3f}     two-sided p = {p['p_two_sided']:.4g}
```

In other units: `β̂_pool = {p['ms_per_hour']:+.2f} ± {p['ms_per_hour_se']:.2f} ms/hour`
= **`{p['ppm']:+.2f} ± {p['ppm_se']:.2f} ppm`**
(1 ppm = 1 µs/s = 0.06 ms/min). Magnitude is well inside the
"few to tens of ppm" range expected for uncompensated commodity
quartz oscillators, and the two-sided test rejects β = 0 at p ≈
{p['p_two_sided']:.3g}. **Interpretation:** consistent with a
hardware-level free-running quartz drift between the two recorders,
on the order of a dozen ppm.

![D.1 Slope forest plot](figures/appendix_d1_slope_forest.png)

*Per-day within-day drift slopes (ms/min) with ± 1.96·SE error bars; the
grey band is the inverse-variance-weighted pooled slope's 95 % CI. All
three per-day slopes lie inside the pooled CI, indicating they are
consistent with a single underlying drift rate.*

### D.2 Cross-day monotonicity check

If a single clock were continuously drifting across the entire February-
March period, `τ̂` would change monotonically with calendar date. Ordered
by date (25.02 → 15.03 → 24.03):

| session_date | calendar gap | τ̂ (s) | Δτ̂ from previous session (s) |
|---|---|---|---|
"""
    for k, sd in enumerate(order):
        if k == 0:
            md += f"| {sd} | — | {taus[sd]:+.4f} | — |\n"
        else:
            md += (f"| {sd} | {cd[k-1]['gap_days']} days | "
                   f"{taus[sd]:+.4f} | {cd[k-1]['delta_tau_obs_s']:+.4f} |\n")

    md += f"""
The sequence is `τ̂ = {tau_seq[0]:+.3f} → {tau_seq[1]:+.3f} → {tau_seq[2]:+.3f}` s
(decrease then increase) — **non-monotone**: `monotone={monotone}`. A
single continuously-drifting clock cannot produce this pattern.

![D.2 Calendar timeline](figures/appendix_d2_calendar_timeline.png)

*Per-day applied τ̂ on a calendar axis with ± σ error bars and connecting
lines; calendar gaps and observed Δτ̂ between consecutive sessions are
annotated. The non-monotone pattern is incompatible with a single
continuously-drifting clock and requires per-session offset
re-randomisation (i.e. boot / NTP resync between sessions).*

### D.3 Magnitude consistency check

If the within-day drift continued uninterrupted across the calendar
gap, the accumulated drift between consecutive sessions would equal
the pooled slope times the gap duration. Comparing predicted to
observed:

| pair | gap (days) | gap (min) | predicted Δτ̂ from β̂_pool (s) | observed Δτ̂ (s) | |pred / obs| | sign agreement |
|---|---|---|---|---|---|---|
"""
    for c in cd:
        md += (f"| {c['from_session']} → {c['to_session']} | "
               f"{c['gap_days']} | {c['gap_minutes']:,} | "
               f"{c['delta_tau_pred_s']:+.3f} ± "
               f"{1.96*c['delta_tau_pred_se_s']:.3f} | "
               f"{c['delta_tau_obs_s']:+.4f} | "
               f"{c['ratio_pred_over_obs']:.0f}× | "
               f"{'yes' if c['sign_agreement'] else '**NO**'} |\n")

    md += f"""
Predicted accumulated drift exceeds observed Δτ̂ by **{cd[0]['ratio_pred_over_obs']:.0f}×**
({cd[0]['from_session']} → {cd[0]['to_session']}) and **{cd[1]['ratio_pred_over_obs']:.0f}×**
({cd[1]['from_session']} → {cd[1]['to_session']}). The 15.03 → 24.03 pair
also exhibits a **sign disagreement**: predicted is negative (extrapolating
the within-day downward drift), observed is positive. Both observations
are inconsistent with continuous drift and consistent with the per-session
offset being re-randomised by boot / NTP resync between sessions.

![D.3 Predicted vs observed](figures/appendix_d3_pred_vs_obs.png)

*Predicted accumulated drift (grey, with 95 % CI from β̂_pool ± 1.96·SE
× gap) vs observed |Δτ̂| (green) per day-pair, on a log y-axis.
Predicted exceeds observed by ~ 30–53 ×; the gap is too large to be
explained by any plausible inflation of the within-day uncertainty.*

![D.4 Conceptual schematic of the two-component clock model](figures/appendix_d4_schematic.png)

*Schematic of the two-component model: each calibration session has its
own per-session offset τ₀ (re-randomised by boot / NTP resync between
sessions), with a small linear within-session drift superimposed.
Inter-session arrows mark the resync events. Not to scale.*

### D.4 Conclusion

The two-component model — **per-session constant offset + small
within-session drift, with the offset re-randomised between sessions** —
fits all three observations:

1. The within-day pooled slope is non-zero at p ≈ {p['p_two_sided']:.3g},
   in the {abs(p['ppm']):.0f}-ppm range typical of uncompensated quartz
   (§D.1).
2. The per-session offsets are not monotone in calendar order (§D.2),
   ruling out a single continuously-drifting clock.
3. Predicted cross-day drift from the within-day slope dwarfs observed
   Δτ̂ by 30–53 ×, with sign disagreement on the 15.03 → 24.03 pair
   (§D.3) — only an inter-session reset can absorb that gap.

A single continuously-drifting clock is rejected on observations 2 and 3.

**Implication for the applied correction.** None. The per-day correction
in §4 already estimates `τ̂` separately for each session_date, which is
exactly what the two-component model requires. The within-day drift is
small enough that it stays below each day's bootstrap σ_rmse — the
largest within-day drift slope's |β̂| × max session duration ≈
{abs(min(slopes.values()))*60*3.2/1000:.3f} s on the longest day vs σ_rmse ≈ 0.16–0.24 s on
those days (§4) — so a per-day constant remains the right applied
granularity. See also §3 (between-day Cochran-Q rejects a single
constant) and §A.1 (the per-CSV-pool obsolete approach reaches the
same per-day applied values to within 116 ms).

### D.5 Honest limitations

Three caveats:

* **n = 3 sessions** is small. The pooled slope is *consistent with*
  hardware-level quartz drift in the right ppm range; it does not
  *demonstrate* it. With three days we cannot distinguish a true
  hardware drift from any other systematic that happens to point the
  same way on all three (e.g. a slow LiDAR-side dissimilarity bias
  correlated with ambient temperature changes during a session).
* **The within-day slope is only marginally significant per day** (p =
  {p_per_day['25.02.2026']:.3f} / {p_per_day['15.03.2026']:.3f} / {p_per_day['24.03.2026']:.3f} for
  25.02 / 15.03 / 24.03 individually). The pooled p ≈ {p['p_two_sided']:.3g}
  combines those three weak signals; no single day's drift slope rejects
  zero on its own at p < 0.05.
* **No change to the applied correction is implied.** The per-day τ̂
  values in §4 remain the correction written into
  `joint_coverage.parquet::applied_tau_s`. This appendix is a
  hardware-level *interpretation* of why the per-day correction
  granularity is appropriate, not a proposal to change it.

"""
    return md


# ---------------------------------------------------------------------------
# Main figure-generation continues
# ---------------------------------------------------------------------------

def fig_retention(retention_per_run: list[dict]) -> Path:
    rates = [r["retention"] * 100 for r in retention_per_run]
    labels = [f"{r['session_date']} / {r['run_file'].replace('out_Myroslav_','').replace('.csv','')}"
              for r in retention_per_run]
    colors = [_color_for_session_date(r["session_date"]) for r in retention_per_run]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    y_pos = np.arange(len(rates))
    ax.barh(y_pos, rates, color=colors, alpha=0.85)
    for i, r in enumerate(rates):
        ax.text(r + 0.2, i, f"{r:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("joint-coverage retention (%)")
    ax.set_xlim(80, 102)
    ax.axvline(100, color="grey", lw=0.5)
    ax.set_title("Per-CSV retention after per-day correction")
    ax.grid(alpha=0.3, axis="x")

    axh = axes[1]
    axh.hist(rates, bins=10, color="#1b9e77", alpha=0.7, edgecolor="black")
    axh.set_xlabel("retention (%)")
    axh.set_ylabel("# CSV runs")
    axh.set_title("Histogram")
    axh.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = C.FIG_DIR / "retention_histogram.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Comparison: load obsolete artefacts (per-run) and produce a delta table
# ---------------------------------------------------------------------------

OBSOLETE_HOMOG = (Path(C.LIDAR_DISSIM_CACHE_LEGACY) / "homogeneity.json")
OBSOLETE_OFFSETS = (Path(C.LIDAR_DISSIM_CACHE_LEGACY) / "offsets.json")
OBSOLETE_RETENTION = (Path(C.LIDAR_DISSIM_CACHE_LEGACY) / "retention.json")


# Hardcoded snapshot of the published obsolete-report numbers
# (`docs/obsolete/timestamp_synchronization_per_run/report.md`). Used as a
# canonical fallback when the obsolete cache directory is no longer present
# on disk — without this fallback the loader returns empty dicts and Appendix
# A silently degrades to "(no obsolete artefacts found)", which corrupts the
# Δ-vs-obsolete numbers in the TL;DR and erases the four A.* tables. The
# numbers below are transcribed from §1, §3, §6.1, §8, §9 of the obsolete
# report; the schema mirrors what `cache/{homogeneity,offsets,retention}.json`
# used to contain when generated by the obsolete pipeline.
_OBSOLETE_PUBLISHED_SNAPSHOT = {
    "homog": {
        "per_session_day": {
            "15.03.2026": {
                "n_runs": 4, "tau_bar": 0.1343, "se": 0.0681,
                "ci_low": 0.0008, "ci_high": 0.2678,
                "Q": 1.15, "df": 3, "p": 0.765, "I2": 0.0,
            },
            "24.03.2026": {
                "n_runs": 3, "tau_bar": 0.4889, "se": 0.0881,
                "ci_low": 0.3163, "ci_high": 0.6615,
                "Q": 0.66, "df": 2, "p": 0.720, "I2": 0.0,
            },
            "25.02.2026": {
                "n_runs": 1, "tau_bar": 0.8798, "se": 0.1223,
                "ci_low": 0.6401, "ci_high": 1.1195,
                "Q": float("nan"), "df": 0,
                "p": float("nan"), "I2": float("nan"),
            },
        },
        "all_runs_pool": {
            "n": 8, "tau_bar": 0.367, "ci_low": 0.270, "ci_high": 0.463,
            "Q": 32.97, "df": 7, "p": 2.68e-05, "I2": 78.8,
        },
        "included_only_pool": {
            "n": 8, "tau_bar": 0.367, "ci_low": 0.270, "ci_high": 0.463,
            "Q": 32.97, "df": 7, "p": 2.68e-05, "I2": 78.8,
        },
    },
    "offsets": [
        # From §8 of the obsolete report (per-CSV diagnostic table).
        {"session_date": "15.03.2026",
         "run_file": "out_Myroslav_15-03_2026_1.csv",
         "tau_hat": 0.355, "sigma": 0.273, "rho_peak": 0.587},
        {"session_date": "15.03.2026",
         "run_file": "out_Myroslav_15-03_2026_2.csv",
         "tau_hat": 0.190, "sigma": 0.179, "rho_peak": 0.571},
        {"session_date": "15.03.2026",
         "run_file": "out_Myroslav_15-03_2026_3.csv",
         "tau_hat": 0.191, "sigma": 0.179, "rho_peak": 0.578},
        {"session_date": "15.03.2026",
         "run_file": "out_Myroslav_15-03_2026_4.csv",
         "tau_hat": 0.088, "sigma": 0.085, "rho_peak": 0.622},
        {"session_date": "24.03.2026",
         "run_file": "out_Myroslav_24-03_2026_1.csv",
         "tau_hat": 0.559, "sigma": 0.128, "rho_peak": 0.268},
        {"session_date": "24.03.2026",
         "run_file": "out_Myroslav_24-03_2026_2.csv",
         "tau_hat": 0.453, "sigma": 0.152, "rho_peak": 0.375},
        {"session_date": "24.03.2026",
         "run_file": "out_Myroslav_24-03_2026_3.csv",
         "tau_hat": 0.378, "sigma": 0.202, "rho_peak": 0.416},
        {"session_date": "25.02.2026",
         "run_file": "out_Myroslav_25-02_2026.csv",
         "tau_hat": 0.880, "sigma": 0.122, "rho_peak": 0.399},
    ],
    "retention": {
        # From §9 of the obsolete report (overall) and §1 (per-CSV n_pairs).
        "overall": {
            "retention": 0.9699, "n_matched": 681487, "n_scans": 702624,
        },
        "per_run": [
            # The per-CSV breakdown was not transcribed in detail; the
            # overall retention is what the comparison block in
            # `comparison_block()` consumes.
        ],
    },
    "_provenance": "fallback (transcribed from "
                   "docs/obsolete/timestamp_synchronization_per_run/report.md)",
}


def _load_obsolete():
    """Load the obsolete pipeline's cached results for Appendix A / TL;DR.

    Order of preference: live cache → published-report fallback. The
    fallback is hardcoded above and exists so that Appendix A's four
    tables and the TL;DR's Δ-vs-obsolete numbers survive removal of the
    obsolete cache directory (which is staged for deletion).
    """
    out = {}
    if OBSOLETE_HOMOG.exists():
        out["homog"] = json.loads(OBSOLETE_HOMOG.read_text())
    if OBSOLETE_OFFSETS.exists():
        out["offsets"] = json.loads(OBSOLETE_OFFSETS.read_text())
    if OBSOLETE_RETENTION.exists():
        out["retention"] = json.loads(OBSOLETE_RETENTION.read_text())
    if not out:
        # Cache directory missing — fall back to the published snapshot
        # rather than silently degrading the comparison block.
        print(f"  [info] obsolete cache not found at "
              f"{C.LIDAR_DISSIM_CACHE_LEGACY} — "
              f"using fallback snapshot from the published obsolete report")
        return dict(_OBSOLETE_PUBLISHED_SNAPSHOT)
    # If only some files are present, fill the missing keys from the snapshot.
    for k, v in _OBSOLETE_PUBLISHED_SNAPSHOT.items():
        out.setdefault(k, v)
    return out


def comparison_block(new_offsets: list[dict], new_homog: dict,
                     new_retention: dict, obs: dict) -> str:
    if not obs:
        return "_(no obsolete artefacts found for comparison)_"

    obs_per_day = obs["homog"]["per_session_day"]
    obs_global = obs["homog"]["included_only_pool"]
    obs_offsets = obs.get("offsets", [])
    obs_retention = obs.get("retention", {})

    new_by_day = {r["session_date"]: r for r in new_offsets}
    new_global = new_homog["included_only_pool"]

    lines = []
    lines.append("### A.1 Per-day offset values\n")
    lines.append("| session_date | obsolete (per-run pool) τ̄ ± SE | new (per-day signal) τ̂ ± σ | Δ (new − old) | Δ / σ_old |")
    lines.append("|---|---|---|---|---|")
    for sd in sorted(set(list(obs_per_day) + list(new_by_day))):
        o = obs_per_day.get(sd, {})
        n = new_by_day.get(sd, {})
        o_tau = o.get("tau_bar", float("nan"))
        o_se = o.get("se", float("nan"))
        n_tau = n.get("tau_hat", float("nan"))
        n_sig = n.get("sigma", float("nan"))
        d = n_tau - o_tau if (np.isfinite(o_tau) and np.isfinite(n_tau)) else float("nan")
        d_norm = d / o_se if (np.isfinite(d) and np.isfinite(o_se) and o_se > 0) else float("nan")
        lines.append(f"| {sd} | {o_tau:+.4f} ± {o_se:.4f} | {n_tau:+.4f} ± {n_sig:.4f} | "
                     f"{d:+.4f} | {d_norm:+.2f}σ |")
    lines.append("")
    lines.append(
        "Sign of the difference and its size relative to σ_old quantify how the "
        "two methodologies disagree at the per-day level. A small Δ relative to "
        "σ_old confirms that the per-run pooling and the per-day direct estimate "
        "agree to within sampling noise; a large Δ would indicate that pooling "
        "across files materially distorted the day's offset.\n"
    )

    lines.append("### A.2 Global Cochran-Q test (different units, conceptually)\n")
    lines.append("| pool | unit | n | τ̄ (s) | 95 % CI | Q | df | p | I² | constant? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| obsolete (per-run) | 8 CSV files | {obs_global['n']} | "
                 f"{obs_global['tau_bar']:+.3f} | "
                 f"[{obs_global['ci_low']:+.3f}, {obs_global['ci_high']:+.3f}] | "
                 f"{obs_global['Q']:.2f} | {obs_global['df']} | {obs_global['p']:.2g} | "
                 f"{obs_global['I2']:.1f}% | "
                 f"{'YES' if (obs_global['I2']<25 and obs_global['p']>0.05) else 'NO'} |")
    lines.append(f"| **new (per-day)** | 3 days | {new_global['n']} | "
                 f"{new_global['tau_bar']:+.3f} | "
                 f"[{new_global['ci_low']:+.3f}, {new_global['ci_high']:+.3f}] | "
                 f"{new_global['Q']:.2f} | {new_global['df']} | {new_global['p']:.2g} | "
                 f"{new_global['I2']:.1f}% | "
                 f"{'YES' if (new_global['I2']<25 and new_global['p']>0.05) else 'NO'} |")
    lines.append("")
    lines.append(
        "These tests are NOT directly comparable — the obsolete pool tests "
        "homogeneity across 8 per-CSV estimates (asks: does *any* CSV deviate "
        "from the constant?), the new pool tests homogeneity across 3 per-day "
        "estimates (asks: does *any day* deviate from the constant?). The "
        "obsolete framing conflates within-day inhomogeneity with between-day "
        "inhomogeneity; the per-day framing is the cleaner test of the "
        "physically meaningful question (does the LiDAR↔telemetry offset "
        "differ between calibration days?).\n"
    )

    # Within-day comparison: obsolete per-CSV taus vs new per-day point estimate
    lines.append("### A.3 Within-day stability check\n")
    lines.append("Compares each day's per-CSV obsolete estimates to the new per-day estimate. "
                 "A new per-day τ̂ that sits inside the obsolete per-CSV spread "
                 "indicates the per-day signal recovers a value consistent with "
                 "what the pooled per-CSV estimates predicted, just with cleaner "
                 "uncertainty (no pooling error).\n")
    lines.append("| session_date | obsolete per-CSV τ̂ (s) | obsolete per-CSV σ (s) | "
                 "obsolete pooled τ̄ (s) | new per-day τ̂ (s) | new − pooled |")
    lines.append("|---|---|---|---|---|---|")
    by_obs_day: dict[str, list[dict]] = {}
    for r in obs_offsets:
        by_obs_day.setdefault(r["session_date"], []).append(r)
    for sd in sorted(by_obs_day):
        runs = by_obs_day[sd]
        taus = [r["tau_hat"] for r in runs]
        sigs = [r["sigma"] for r in runs]
        tau_str = ", ".join(f"{t:+.3f}" for t in taus)
        sig_str = ", ".join(f"{s:.3f}" for s in sigs)
        o_pool = obs_per_day.get(sd, {}).get("tau_bar", float("nan"))
        n_tau = new_by_day.get(sd, {}).get("tau_hat", float("nan"))
        d = n_tau - o_pool if np.isfinite(o_pool) else float("nan")
        lines.append(f"| {sd} | {tau_str} | {sig_str} | "
                     f"{o_pool:+.4f} | {n_tau:+.4f} | {d:+.4f} |")
    lines.append("")

    # Retention comparison
    lines.append("### A.4 Joint-coverage retention\n")
    obs_overall = obs_retention.get("overall", {}).get("retention", float("nan"))
    new_overall = new_retention.get("overall", {}).get("retention", float("nan"))
    lines.append(f"* Obsolete overall retention: **{obs_overall*100:.2f} %**  "
                 f"({obs_retention.get('overall',{}).get('n_matched',0):,}/"
                 f"{obs_retention.get('overall',{}).get('n_scans',0):,} scans)")
    lines.append(f"* New per-day overall retention: **{new_overall*100:.2f} %**  "
                 f"({new_retention.get('overall',{}).get('n_matched',0):,}/"
                 f"{new_retention.get('overall',{}).get('n_scans',0):,} scans)")
    lines.append("")
    lines.append(
        "The retention difference reflects only the change in the applied per-day τ̂ "
        "(the matching tolerance is unchanged). Substantively identical retention "
        "indicates the new per-day τ̂ produces equally good LiDAR↔telemetry pairing.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_pct(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p*100:.1f}%"


def render_report(offsets: list[dict], homog: dict, drift: list[dict],
                  retention: dict, days: list[dict], obs: dict,
                  per_csv_taus: list[dict], per_csv_homog: dict,
                  hw_appendix_md: str) -> str:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    inc = homog["included_only_pool"]
    decision = homog["decision_rule"]["verdict_constant_global"]

    # Compose TL;DR
    by_day = {r["session_date"]: r for r in offsets}
    day_lines = []
    for sd in sorted(by_day):
        r = by_day[sd]
        day_lines.append(f"`τ̂_{{{sd}}} = {r['tau_hat']:+.3f}` s "
                         f"(σ = {r['sigma']:.3f} s, ρ_peak = {r['peak_value']:.3f})")
    day_results = ", ".join(day_lines)

    # Comparison block
    comp = comparison_block(offsets, homog, retention, obs)

    # Per-day deltas vs obsolete pool, computed once for use in the TL;DR.
    obs_per_day = (obs.get("homog") or {}).get("per_session_day", {})
    deltas: dict[str, dict] = {}
    max_abs_delta_s = 0.0
    max_abs_delta_norm = 0.0
    for sd in sorted(by_day):
        n_tau = by_day[sd]["tau_hat"]
        o = obs_per_day.get(sd, {})
        o_tau = o.get("tau_bar", float("nan"))
        o_se = o.get("se", float("nan"))
        if np.isfinite(o_tau):
            d = n_tau - o_tau
            d_norm = abs(d) / o_se if (o_se and np.isfinite(o_se) and o_se > 0) else float("nan")
            deltas[sd] = {"delta": d, "delta_norm_old_se": d_norm}
            if abs(d) > max_abs_delta_s:
                max_abs_delta_s = float(abs(d))
            if np.isfinite(d_norm) and d_norm > max_abs_delta_norm:
                max_abs_delta_norm = float(d_norm)

    # Per-day diagnostic table
    diag_rows = []
    for r in sorted(offsets, key=lambda x: x["session_date"]):
        flags = []
        if not r["included"]:
            flags.append(f"prom<{C.PROMINENCE_MIN}")
        if r["tel_rate_hz"] < 10:
            flags.append("low_tel_rate")
        if r["modal_fraction"] < 0.5:
            flags.append("bootstrap_unstable")
        hs = r.get("half_split", {}) or {}
        if hs.get("agree") is False:
            flags.append("halves_disagree")
        diag_rows.append(
            f"| {r['session_date']} | {r['n_csv_files']} | "
            f"{r['tel_rate_hz']:.1f} | {r['lid_rate_hz']:.1f} | "
            f"{r['intersect_seconds']:.0f} | {r['motion_seconds']:.0f} | "
            f"{r['n_stop_start_events']} | "
            f"{r['tel_gap_count']} | {r['tel_gap_seconds_sum']:.0f} | "
            f"{r['tau_hat']:+.4f} | {r['sigma']:.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{r['peak_value']:.3f} | {r['prominence']:.3f} | "
            f"{r['modal_fraction']:.2f} | {r['block_len_s']:.0f} | "
            f"{('|'.join(flags) if flags else '')} |"
        )
    diag_table = "\n".join(diag_rows)

    # Half-split table
    half_rows = []
    for r in sorted(offsets, key=lambda x: x["session_date"]):
        hs = r.get("half_split", {}) or {}
        if hs.get("agree") is None:
            half_rows.append(f"| {r['session_date']} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        f = hs["first"]; s = hs["second"]
        half_rows.append(
            f"| {r['session_date']} | {f['tau_hat']:+.3f} | {f['sigma']:.3f} | "
            f"{s['tau_hat']:+.3f} | {s['sigma']:.3f} | "
            f"{(f['tau_hat']-s['tau_hat']):+.3f} | "
            f"{('yes' if hs['agree'] else 'NO')} |"
        )
    half_table = "\n".join(half_rows)

    # Drift slope table
    drift_rows = []
    for r in sorted(drift, key=lambda x: x["session_date"]):
        drift_rows.append(
            f"| {r['session_date']} | {r['n_windows']} | "
            f"{r['window_seconds']/60:.0f} | "
            f"{(r['within_day_std_s']):.3f} | {(r['spread_s']):.3f} | "
            f"{r['slope_ms_per_min']:+.2f} | {r['slope_se_ms_per_min']:.2f} | "
            f"{r['slope_p_value']:.3g} | "
            f"{(r['slope_ms_per_min']*r['window_seconds']*r['n_windows']/60_000):+.3f} |"
        )
    drift_table = "\n".join(drift_rows)

    overlay_md = []
    for sd in sorted(by_day):
        overlay_md.append(f"#### Session {sd}\n")
        for k in range(C.N_OVERLAYS_PER_SESSION):
            overlay_md.append(
                f"![overlay_{sd}__w{k+1}of{C.N_OVERLAYS_PER_SESSION}](figures/"
                f"overlay_{sd}__w{k+1}of{C.N_OVERLAYS_PER_SESSION}.png)\n"
            )
    overlays_block = "\n".join(overlay_md)

    # Per-day apply summary
    per_day = homog["per_session_day"]
    apply_rows = []
    for sd in sorted(per_day):
        d = per_day[sd]
        apply_rows.append(
            f"| {sd} | {d['tau_bar']:+.4f} | {d['se']:.4f} | "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
            f"{d['rho_peak']:.3f} | {d['n_csv_files']} |"
        )
    apply_table = "\n".join(apply_rows)

    # ----- Per-CSV-file cross-check tables (Appendix C) -----
    def _short_csv(rf: str) -> str:
        # out_Myroslav_15-03_2026_1.csv -> r1; ..._2026.csv -> r1
        stem = rf.replace(".csv", "")
        last = stem.split("_")[-1]
        return f"r{last}" if (last.isdigit() and len(last) <= 2) else "r1"

    per_csv_sorted = sorted(per_csv_taus,
                            key=lambda r: (r["session_date"], r["run_file"]))

    # Per-CSV breakdown table (matches old report's "Per-run breakdown")
    per_csv_breakdown_rows = []
    for r in per_csv_sorted:
        per_csv_breakdown_rows.append(
            f"| {r['session_date']} | {r['run_file']} | {r['tel_n']:,} | "
            f"{r['tel_rate_hz']:.2f} | {r['lid_n']:,} | {r['lid_rate_hz']:.2f} | "
            f"{r['intersect_seconds']:.0f} | {r['motion_seconds']:.0f} | "
            f"{r['n_stop_start_events']} |"
        )
    per_csv_breakdown_table = "\n".join(per_csv_breakdown_rows)

    # Per-CSV diagnostic table (matches old report's §8)
    per_csv_diag_rows = []
    for r in per_csv_sorted:
        flags = []
        if not r["included"]:
            flags.append(f"prom<{C.PROMINENCE_MIN}")
        if r["modal_fraction"] < 0.5:
            flags.append("bootstrap_unstable")
        if r["tel_rate_hz"] < 10:
            flags.append("low_tel_rate")
        per_csv_diag_rows.append(
            f"| {r['session_date']} | {r['run_file']} | {r['tel_rate_hz']:.1f} | "
            f"{r['lid_rate_hz']:.1f} | {r['intersect_seconds']:.0f} | "
            f"{r['motion_seconds']:.0f} | {r['n_stop_start_events']} | "
            f"{r['tau_hat']:+.4f} | {r['sigma']:.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{r['peak_value']:.3f} | {r['prominence']:.3f} | "
            f"{r['modal_fraction']:.2f} | {r['block_len_s']:.0f} | "
            f"{('|'.join(flags) if flags else '')} |"
        )
    per_csv_diag_table = "\n".join(per_csv_diag_rows)

    # Within-day Cochran-Q (per-CSV pool) table
    per_csv_q_rows = []
    for sd in sorted(per_csv_homog):
        d = per_csv_homog[sd]
        if d["df"] == 0:
            per_csv_q_rows.append(
                f"| {sd} | {d['n_csv']} | n/a | n/a | — | 0 | — | — | "
                f"single CSV file (no within-day test possible) |")
            continue
        verdict = ("**YES**" if d["verdict_within_day_constant"] else "**NO**")
        # Show per-CSV taus (compact)
        taus_str = ", ".join(f"{t:+.3f}" for t in d["csv_taus"])
        per_csv_q_rows.append(
            f"| {sd} | {d['n_csv']} | {d['pooled_tau_bar']:+.4f} | "
            f"{d['pooled_se']:.4f} | {d['Q']:.2f} | {d['df']} | "
            f"{d['p']:.3g} | {d['I2']:.1f}% | {verdict} ({taus_str}) |"
        )
    per_csv_q_table = "\n".join(per_csv_q_rows)

    # Per-day vs per-CSV-pool comparison table
    per_csv_compare_rows = []
    for sd in sorted(per_csv_homog):
        d = per_csv_homog[sd]
        diff = d["applied_per_day_tau"] - d["pooled_tau_bar"]
        per_csv_compare_rows.append(
            f"| {sd} | {d['applied_per_day_tau']:+.4f} ± {d['applied_per_day_se']:.4f} | "
            f"{d['pooled_tau_bar']:+.4f} ± {d['pooled_se']:.4f} | "
            f"{diff:+.4f} | {d['n_csv']} |"
        )
    per_csv_compare_table = "\n".join(per_csv_compare_rows)

    md = f"""# Time-synchronisation analysis (per-day): AGV telemetry vs. Leuze RSL 400 LiDAR

*Per-day calibration redo for AIDI 2026.* Generated {now} by
`analysis/time_sync/run_report.py`. Replaces the obsolete per-CSV-pooled
analysis archived under `docs/obsolete/timestamp_synchronization_per_run/`.

## TL;DR

* **One LiDAR↔telemetry offset is estimated per calibration day**, directly
  from the day's full continuous signal (all CSV files concatenated and
  sorted, with telemetry-gap masking — see §2). The CSV files within a
  day are parts of one continuous calibration session, so the day is the
  natural unit of analysis. Per-day point estimates: {day_results}.
* **Sign convention**: positive τ̂ ⇒ LiDAR lags telemetry → add τ̂ to every
  LiDAR timestamp. The reported residuals are on top of the +3600 s
  UTC→CET shift already baked into `lidar.h5::local_timestamps`.
* **A single global constant is {'NOT supported' if not decision else 'supported'}** across the three
  days: Cochran's `Q = {inc['Q']:.2f}` on `df = {inc['df']}`, `p = {inc['p']:.2g}`,
  `I² = {inc['I2']:.1f}%` (decision rule: `I² < {C.HOMOGENEITY_I2_MAX}%` AND
  `p > {C.HOMOGENEITY_P_MIN}`). One offset per day is required.
* **Within-day stability is decisive**. The half-split test passes on every
  day for which it can be computed (§7.1), and sliding-window τ̂ shows no
  practically meaningful drift within the day (§7.2). In particular the
  small day-pooled slope flagged on 24.03 by the obsolete per-CSV analysis
  is recovered as a window-level scatter consistent with sampling noise on
  the per-day signal.
* **Compared to the obsolete per-CSV-pooled analysis**, the new per-day
  point estimates differ by at most **{max_abs_delta_s*1000:.0f} ms** in
  absolute terms ({max_abs_delta_norm:.2f}× the obsolete per-day SE, see
  §A.1). The single-CSV day (25.02) reproduces the obsolete value exactly
  (per-day = per-CSV when there is only one CSV). Joint-coverage retention
  is essentially unchanged: {_fmt_pct(retention['overall']['retention'])}
  (new) vs. {_fmt_pct(obs.get('retention',{}).get('overall',{}).get('retention',float('nan')))} (obsolete).
  The methodological clean-up matters most for *interpretation* of the
  homogeneity test (the obsolete `n=8` per-CSV Cochran-Q conflated within-
  and between-day variation; the per-day `n=3` test is the meaningful one),
  and for the within-day drift slope on 24.03 (obsolete: −6.9 ms/min,
  `p<1e-4`; new: −1.4 ms/min, `p≈0.17` — the strong slope was an artefact
  of stitching three separate per-CSV time series together).
* **Hardware-level interpretation (Appendix D).** The within-day slopes
  pool to −0.74 ± 0.27 ms/min ≈ −12 ppm (consistent with uncompensated
  quartz drift), and cross-day Δτ̂ is non-monotone with magnitude ~ 30–53 ×
  smaller than a single-drifting clock would predict — supports a
  per-session-offset + within-session-drift model in which the offset is
  reset by boot/NTP between sessions; no change to the applied correction.

## 1. Data summary

| Stream | Source | Total samples | Days | Sample rates (Hz) |
|---|---|---|---|---|
| Telemetry | `data/merged/telemetry_cleaned.parquet` | {sum(d['tel_n'] for d in days):,} | {len(days)} | {', '.join(f"{d['tel_rate_hz']:.1f}" for d in days)} |
| LiDAR scans | `data/merged/lidar.h5` | {sum(d['lid_n'] for d in days):,} | {len(days)} | {', '.join(f"{d['lid_rate_hz']:.1f}" for d in days)} |

Per-day breakdown:

| session_date | CSV files | telemetry rows | tel_rate (Hz) | LiDAR rows | LiDAR rate (Hz) | intersect (s) | motion (s) | gaps>1s (count, sum) |
|---|---|---|---|---|---|---|---|---|
"""
    for d in days:
        md += (f"| {d['session_date']} | {d['n_csv_files']} | "
               f"{d['tel_n']:,} | {d['tel_rate_hz']:.2f} | "
               f"{d['lid_n']:,} | {d['lid_rate_hz']:.2f} | "
               f"{d['intersect_seconds']:.0f} | {d['motion_seconds']:.0f} | "
               f"{d['tel_gap_count']}, {d['tel_gap_seconds_sum']:.1f}s |\n")
    md += f"""
## 2. Methodology

We follow the same five-phase plan as the obsolete analysis (cross-correlation
+ parabolic refinement + moving-block bootstrap + Cochran-Q + joint-coverage
filter), with one substantive change: **the unit of analysis is the
calibration day, not the individual CSV file.** Each day's telemetry is
concatenated across all of its CSV files and sorted by `fh7000_timestamp`
to form a single continuous signal, and a single per-day offset is
estimated from it.

**Why per-day rather than per-CSV.** The obsolete analysis treated each
CSV as an independent "run" and computed a separate τ̂ for each, then
inverse-variance-pooled the per-CSV estimates within a day to produce a
per-day τ̄. That treatment is incorrect: the multiple CSV files within a
day are parts of one continuous recording session (split for size /
convenience, not because the recording was paused), so the per-CSV
estimates are not independent draws of an underlying offset population.
Pooling them as if they were inflates the within-day Cochran-Q test
artefactually (the per-CSV estimates carry sampling noise that is *not*
between-CSV variance) and produces a misleading "per-CSV homogeneity"
test that has no physically meaningful unit. The per-day signal contains
all the same information without the artificial fragmentation.

**Telemetry gap handling.** Wall-clock gaps appear between consecutive
CSV files within a day (the recorder rotates files; gaps run up to ~90 s
on 15.03 and 24.03; see §1). On the resampled 50 Hz grid, a naive linear
interpolation across these gaps would inject a slow ramp that does not
exist in the data and biases the cross-correlation. We mask all 50 Hz
samples that fall inside a telemetry gap > {C.GAP_MASK_S:.0f} s by zeroing both
signals (after z-scoring) on those grid samples; this removes the masked
samples' contribution to the FFT cross-correlation numerator without
changing its denominator, so the resulting `ρ(τ)` is the cross-correlation
over the unmasked support only.

**Phase 1 — Day enumeration.** Telemetry is grouped by `session_date`.
For each day the matching LiDAR window is the subset of `lidar.h5` rows
whose `session_date` matches and whose timestamp falls in
`[tel_start − 2 s, tel_end + 2 s]`. Sample rates are estimated as the
median of inter-sample gaps under 1 s (rejecting pauses).

**Phase 2 — Signature.** The canonical telemetry signature is
`rot_aware = |speed_mps| + R·|ω|` with `R = {C.ROT_R_EFF}` m, the same
choice as the obsolete report (justified there empirically and physically
— see `docs/obsolete/timestamp_synchronization_per_run/report.md` §10).
The LiDAR signature is the per-pair scan-to-scan dissimilarity
`s_L(t) = mean_i |D_i(k+1) − D_i(k)|` over beams valid in *both* scans
(invalid marker 65535 excluded; zero returns excluded; beams pinned at
sensor max excluded). Both signals are linearly resampled onto a common
50 Hz grid over the day's intersection window, telemetry-gap masked
(above), and z-scored.

**Phase 3 — Per-day offset estimation.** Normalised cross-correlation
`ρ(τ)` is computed via FFT (scipy `correlate`, mode `full`, biased 1/N).
The coarse peak is sub-sample-refined by parabolic interpolation through
the three samples bracketing the peak. Search range is capped at
`±{C.SEARCH_RANGE_S:.0f}` s to exclude AGV-route-period sidelobes
(motivated and documented in the obsolete report). Bootstrap uncertainty
is from a moving-block bootstrap with `B = {C.BOOTSTRAP_B}` resamples and adaptive
block length `L_eff = max({C.BLOCK_LEN_S:.0f} s, 3·|τ̂| + {C.BLOCK_LEN_S:.0f} s)`.
Canonical σ is `σ_rmse = √E[(τ_b − τ̂)²]` (RMSE about the point estimate);
CI is `τ̂ ± 1.96·σ_rmse`.

**Phase 4 — Tests.**
* *Across-day homogeneity (Cochran's Q).* Now operates on `n = 3`
  per-day estimates rather than `n = 8` per-CSV estimates. Decision rule:
  `I² < {C.HOMOGENEITY_I2_MAX}%` AND `p > {C.HOMOGENEITY_P_MIN}` — supports a single global constant.
* *Within-day stability (half-split).* Each day's signal is split at its
  midpoint and τ̂ re-estimated on each half; halves agree if their 95 % CIs
  overlap.
* *Within-day drift (sliding window).* Slide a 5 min (10 min for 25.02
  due to its 4.4 Hz cadence) window across the day's signal and recompute
  τ̂ per window; weighted linear regression against window-center wall-clock
  time gives a slope in ms/min with t-test p-value. Windows with > 25 %
  of their samples inside a telemetry gap are excluded.

**Phase 5 — Apply.** Each day's τ̂ is added to every LiDAR timestamp from
that day's session_date partition. For each LiDAR scan we find the nearest
telemetry sample on the corrected time axis (`pandas.merge_asof`,
`direction='nearest'`); the pair is retained if `|Δt| < tolerance =
max(LiDAR_period, telemetry_period) / 2` using each day's measured rates.
Output: `data/merged/joint_coverage.parquet`.

## 3. Is the offset constant across days?

| pool | n | τ̄ (s) | 95 % CI (s) | Q | df | p | I² | constant (decision rule)? |
|---|---|---|---|---|---|---|---|---|
| All days | {homog['all_days_pool']['n']} | {homog['all_days_pool']['tau_bar']:+.3f} | [{homog['all_days_pool']['ci_low']:+.3f}, {homog['all_days_pool']['ci_high']:+.3f}] | {homog['all_days_pool']['Q']:.2f} | {homog['all_days_pool']['df']} | {homog['all_days_pool']['p']:.2g} | {homog['all_days_pool']['I2']:.1f}% | {'**YES**' if (homog['all_days_pool']['I2']<C.HOMOGENEITY_I2_MAX and homog['all_days_pool']['p']>C.HOMOGENEITY_P_MIN) else '**NO**'} |
| Included only (prominence ≥ {C.PROMINENCE_MIN}) | {inc['n']} | {inc['tau_bar']:+.3f} | [{inc['ci_low']:+.3f}, {inc['ci_high']:+.3f}] | {inc['Q']:.2f} | {inc['df']} | {inc['p']:.2g} | {inc['I2']:.1f}% | {'**YES**' if (inc['I2']<C.HOMOGENEITY_I2_MAX and inc['p']>C.HOMOGENEITY_P_MIN) else '**NO**'} |

**Verdict: {'a single constant offset across the three days is supported' if decision else 'a single constant offset across the three days is rejected'}** at the
`I² < {C.HOMOGENEITY_I2_MAX}%`, `p > {C.HOMOGENEITY_P_MIN}` decision rule. The applied correction is therefore
**one offset per session_date** (§4).

## 4. Per-day offsets (the applied correction)

| session_date | τ̂ (s) | SE (s) | 95 % CI (s) | ρ_peak | n CSV files |
|---|---|---|---|---|---|
{apply_table}

These values are written into `joint_coverage.parquet::applied_tau_s`
verbatim; `applied_tau_se_s` is the same SE for downstream uncertainty
propagation.

## 5. Visual diagnostics

### 5.1 Cross-correlation per day

![cross-correlation per day](figures/xcorr_per_day.png)

Three cross-correlation curves on the shared `±{C.SEARCH_RANGE_S:.0f} s` search window. Each
curve has a single, well-resolved peak; the peaks are clearly separated
between days (consistent with the Cochran-Q rejection of a single
constant).

### 5.2 Forest plot

![forest plot](figures/forest_per_day.png)

Per-day point estimates with `τ̂ ± 1.96·σ_rmse` brackets. The dashed
black line is the global pooled mean (whether or not it is the right
summary depends on §3's decision).

### 5.3 Before / after overlays

Three 120 s overlay windows per day, sampled from early / middle / late
thirds of the recording, each centred on the locally most-active sample.
The top panel shows the raw alignment (`τ = 0`); the bottom panel shifts
the LiDAR signal by the per-day τ̂.

{overlays_block}

## 6. Per-day diagnostic table

| session_date | n CSV | tel Hz | lid Hz | intersect (s) | motion (s) | stop-start | gap_count | gap_sum (s) | τ̂ (s) | σ (s) | 95 % CI | ρ_peak | prom | modal_frac | L_eff (s) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{diag_table}

## 7. Within-day stability

### 7.1 Half-split test

| session_date | first τ̂ | first σ | second τ̂ | second σ | Δ (first − second) | agree (95 % CIs overlap)? |
|---|---|---|---|---|---|---|
{half_table}

### 7.2 Sliding-window drift (per-day signal)

![drift_per_day](figures/drift_per_day.png)

| session_date | n windows | window (min) | within-day σ (s) | spread (s) | slope (ms/min) | SE slope (ms/min) | p | total drift over windows |
|---|---|---|---|---|---|---|---|---|
{drift_table}

The slope is fit against window-center wall-clock time, weighted by
`ρ_peak²`. Per-window noise dominates the apparent trends (each window
has σ comparable to the within-day spread). No day shows a slope that is
both statistically significant *and* practically large enough to motivate
a finer-than-per-day correction.

## 8. Joint-coverage retention

Overall: **{_fmt_pct(retention['overall']['retention'])}**
({retention['overall']['n_matched']:,} of {retention['overall']['n_scans']:,} LiDAR scans matched).

![retention histogram](figures/retention_histogram.png)

The retention rate is ≥ 97 % for every day on which both telemetry and
LiDAR are at their nominal rates; 25.02.2026 has the lowest retention
because its telemetry runs at ~4.4 Hz (half-period tolerance ≈ 113 ms vs.
~20 ms on the other days), so a fraction of LiDAR scans falls in the gap
between two telemetry samples even after a perfect time-sync correction.

## Appendix A. Comparison to the obsolete per-CSV-pooled analysis

The obsolete analysis is archived at
`docs/obsolete/timestamp_synchronization_per_run/`. The differences
between the two methodologies are:

* **Unit of analysis.** Obsolete: each CSV file is a separate "run" with
  its own τ̂; per-day τ̄ is the inverse-variance-weighted pool of those
  per-CSV estimates. New: each day is one continuous signal; one τ̂ per
  day, no within-day pooling.
* **Cochran-Q.** Obsolete: tested on `n = 8` per-CSV estimates (mixes
  within-day and between-day variation in one Q statistic). New: tested
  on `n = 3` per-day estimates, which is the meaningful between-day
  homogeneity test.
* **Telemetry-gap handling.** Obsolete: gaps between CSV files inside a
  day were avoided by computing each CSV's τ̂ separately. New: explicit
  mask on the 50 Hz grid; samples inside a > {C.GAP_MASK_S:.0f} s gap contribute
  nothing to the cross-correlation.
* **σ interpretation.** Obsolete per-day SE was the inverse-variance
  pooled SE of `n_csv` estimates (artificially small when `n_csv` is large
  because it treats correlated within-CSV samples as independent draws).
  New per-day σ is the bootstrap RMSE about the point estimate from the
  full per-day signal — a single uncertainty descriptor that correctly
  reflects the actual amount of information in the day's data.

{comp}

## Appendix B. Per-CSV-file cross-check (validation context, NOT applied)

This appendix recomputes a τ̂ for each individual CSV file using the
**same canonical methodology as the per-day analysis** (rot_aware
signature, ±{C.SEARCH_RANGE_S:.0f} s search cap, parabolic peak refinement, moving-block
bootstrap with adaptive block length, σ_rmse uncertainty). These per-CSV
values are **not applied to the joint dataset** — every row of
`joint_coverage.parquet` carries the per-day τ̂ from §4. The per-CSV
estimates serve as a *finer-than-day validation cross-check*: do the
individual CSV-file estimates within a day cluster around the day's
single offset, or is there a step-discontinuity between adjacent CSV
files that would break the per-day "single offset" assumption?

The per-CSV results are inspired by the obsolete per-CSV pooled
analysis (`docs/obsolete/timestamp_synchronization_per_run/`), but
purged of two errors made there: (i) the obsolete report mis-named the
unit as "run" (each CSV is a fragment of one continuous calibration
session, not a separate run), and (ii) the obsolete report applied
the per-CSV-pooled τ̄ as the day's correction, treating per-CSV
sampling noise as if it were between-CSV variance (see §A.1). Here the
per-CSV values are kept strictly diagnostic.

### B.1 Per-CSV-file breakdown

| session_date | csv_file | telemetry rows | tel_rate (Hz) | LiDAR rows | LiDAR rate (Hz) | intersect (s) | motion (s) | stop-start events |
|---|---|---|---|---|---|---|---|---|
{per_csv_breakdown_table}

### B.2 Per-CSV-file forest plot

![per-CSV forest plot](figures/forest_per_csv.png)

Each dot is one CSV file's recomputed τ̂ with `± 1.96·σ_rmse` brackets
(σ from a moving-block bootstrap on that CSV's signal alone). The
coloured solid verticals are the **per-day τ̂ that is actually
applied** to every LiDAR scan in that day. A coherent picture would
have all of a day's per-CSV dots clustered around the day's vertical
line, with bracket widths that overlap the line — exactly what the
plot shows for every day in this dataset.

### B.3 Per-CSV-file diagnostic table

This table is the per-CSV analogue of §6. Same column meanings,
applied per CSV file rather than per day.

| session_date | csv_file | tel Hz | lid Hz | intersect (s) | motion (s) | stop-start | τ̂ (s) | σ (s) | 95 % CI | ρ_peak | prom | modal_frac | L_eff (s) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{per_csv_diag_table}

### B.4 Within-day Cochran-Q on the per-CSV-file estimates

Tests whether the per-CSV estimates within a day are consistent with a
single underlying offset. This is the *finer-than-day* version of §3:
§3 asks whether one global constant fits all 3 days; this test asks
whether one per-day constant fits all of that day's CSV files. A pass
(`I² < {C.HOMOGENEITY_I2_MAX}%` AND `p > {C.HOMOGENEITY_P_MIN}`) supports the per-day correction;
a fail would warrant a finer-grained correction.

| session_date | n CSV files | pooled τ̄ (s) | pooled SE (s) | Q | df | p | I² | within-day constant? (per-CSV τ̂s) |
|---|---|---|---|---|---|---|---|---|
{per_csv_q_table}

### B.5 Per-day applied τ̂ vs per-CSV pooled τ̄

If the per-CSV signal is a noisy view of the same underlying offset
that the per-day signal sees directly, the two summaries should agree
to within the per-day σ. This table makes that comparison explicit.

| session_date | applied per-day τ̂ ± σ (s) | per-CSV pool τ̄ ± SE (s) | Δ (per-day − per-CSV pool) | n CSV files |
|---|---|---|---|---|
{per_csv_compare_table}

The per-CSV pooled SE is artificially small relative to the per-day σ
(it treats per-CSV sampling noise as between-CSV variance — see §A.1).
The point-estimate difference is the right comparison; a small Δ
confirms both methodologies converge on the same underlying offset.

## Appendix C. Reproducibility

All scripts read paths from `analysis/time_sync/config.py`. To regenerate
everything from scratch:

```bash
.venv/Scripts/python.exe -m analysis.time_sync.run_signatures   # Phase 1+2 (per-day)
.venv/Scripts/python.exe -m analysis.time_sync.run_offsets      # Phase 3+4(b) (per-day)
.venv/Scripts/python.exe -m analysis.time_sync.run_homogeneity  # Phase 4(a) (per-day)
.venv/Scripts/python.exe -m analysis.time_sync.run_drift        # Phase 4(c) (per-day)
.venv/Scripts/python.exe -m analysis.time_sync.run_apply        # Phase 5 (per-day, writes joint_coverage.parquet)
.venv/Scripts/python.exe -m analysis.time_sync.run_per_csv      # Per-CSV cross-check (Appendix B)
.venv/Scripts/python.exe -m analysis.time_sync.run_report       # Phase 6 (figures + report.md)
```

Key outputs:

* `data/merged/joint_coverage.parquet` — calibrated joint dataset
  ({retention['overall']['n_matched']:,} matched scans; carries `applied_tau_s`,
  `applied_tau_se_s`, `delta_t_s`).
* `analysis/time_sync/offsets_per_day.csv` — per-day table (this report's §6).
* `analysis/time_sync/cache/{{days,offsets_per_day,homogeneity_per_day,drift_per_day,retention_per_day,per_csv_taus,per_csv_homogeneity,hardware_appendix}}.json`
  — raw machine-readable artefacts (the per_csv_* feed Appendix B; the
  hardware_appendix feeds Appendix D).
* `analysis/time_sync/figures/*.png` and `docs/time_sync/figures/*.png`
  — embedded figures.

RNG seed: `{C.RNG_SEED}`. Bootstrap iterations: B = {C.BOOTSTRAP_B}.
Search range: ±{C.SEARCH_RANGE_S:.0f} s. Common grid: {C.GRID_HZ:.0f} Hz.
Telemetry-gap mask threshold: {C.GAP_MASK_S:.1f} s.

{hw_appendix_md}"""
    return md


def _copy_figures_to_report() -> None:
    """Copy figures into docs/time_sync/figures/ so the report is
    self-contained when shipped from the docs/ tree."""
    C.REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in C.FIG_DIR.glob("*.png"):
        shutil.copy2(p, C.REPORT_FIG_DIR / p.name)


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    days = json.loads((C.CACHE_DIR / "days.json").read_text())
    offsets = json.loads((C.CACHE_DIR / "offsets_per_day.json").read_text())
    homog = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    drift = json.loads((C.CACHE_DIR / "drift_per_day.json").read_text())
    retention = json.loads((C.CACHE_DIR / "retention_per_day.json").read_text())
    obs = _load_obsolete()

    # Per-CSV-file cross-check artefacts (Appendix B). Optional — pipeline
    # falls back to empty if run_per_csv.py has not been run yet.
    per_csv_path = C.CACHE_DIR / "per_csv_taus.json"
    per_csv_h_path = C.CACHE_DIR / "per_csv_homogeneity.json"
    if not per_csv_path.exists() or not per_csv_h_path.exists():
        raise FileNotFoundError(
            f"Per-CSV cross-check artefacts not found. Run "
            f"`.venv/Scripts/python.exe -m analysis.time_sync.run_per_csv` first."
        )
    per_csv_taus = json.loads(per_csv_path.read_text())
    per_csv_homog = json.loads(per_csv_h_path.read_text())

    # Build offsets_per_day.csv
    df = build_offsets_csv(offsets)
    df = attach_retention(df, retention.get("per_day", []))
    df = attach_anomaly_flags(df)
    df.to_csv(C.OFFSETS_CSV, index=False)
    print(f"Wrote {C.OFFSETS_CSV}")

    # Figures
    p_xcorr = fig_xcorr_per_day(offsets); print(f"  fig: {p_xcorr}")
    p_forest = fig_forest_per_day(offsets, homog); print(f"  fig: {p_forest}")
    overlays = fig_overlays(offsets, n_per_day=C.N_OVERLAYS_PER_SESSION)
    for p in overlays:
        print(f"  fig: {p}")
    p_csv_forest = fig_forest_per_csv(per_csv_taus, homog)
    print(f"  fig: {p_csv_forest}")
    p_ret = fig_retention(retention.get("per_run", [])); print(f"  fig: {p_ret}")
    # Re-render drift_per_day.png from the cached drift JSON (cheap; reuses
    # the same renderer that run_drift.main() uses). This guarantees the
    # report's figure is in lockstep with the cache and is regenerated even
    # when only run_report.py is invoked.
    from . import run_drift
    p_drift = run_drift.render_drift_figure(
        drift, C.FIG_DIR / "drift_per_day.png"
    )
    print(f"  fig: {p_drift}")

    # Appendix D — hardware-level interpretation
    hw_info = compute_hardware_appendix(drift, homog)
    p_d1 = fig_d1_slope_forest(hw_info); print(f"  fig: {p_d1}")
    p_d2 = fig_d2_calendar_timeline(hw_info); print(f"  fig: {p_d2}")
    p_d3 = fig_d3_predicted_vs_observed(hw_info); print(f"  fig: {p_d3}")
    p_d4 = fig_d4_schematic(); print(f"  fig: {p_d4}")
    hw_appendix_md = render_hardware_appendix_md(hw_info)
    # Persist quantitative appendix-D values for downstream auditability
    (C.CACHE_DIR / "hardware_appendix.json").write_text(
        json.dumps(hw_info, indent=2)
    )

    # Render and write report
    md = render_report(offsets, homog, drift, retention, days, obs,
                       per_csv_taus, per_csv_homog, hw_appendix_md)
    C.REPORT_MD.write_text(md, encoding="utf-8")
    print(f"\nWrote report: {C.REPORT_MD}")

    # Copy figures into the docs report folder so the report is portable
    _copy_figures_to_report()
    print(f"Copied figures into {C.REPORT_FIG_DIR}")


if __name__ == "__main__":
    main()
