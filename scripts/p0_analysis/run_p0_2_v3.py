"""P0.2 v3 — Per-session log-distance path-loss fit on v3-cleaned data.

Identical to P0.2 v2 (AP fixed at lab-measured ground truth, (P0_d, n_d)
free, OLS via numpy.polyfit, bootstrap on 5k subsample), but rows are
filtered with the v3 anomaly mask instead of v2's.

Outputs:
- artifacts/ap_coords.json     — augmented with a per-session "v3" sub-record
                                  alongside the existing v2 fields
- cache/path_loss_residuals_v3.parquet
- cache/p0_2_v3.json           — Gate A re-eval, v2-vs-v3 deltas
- figures/p0_2_residual_map_v3_<sd>.png
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, read_json, write_json


N_BOOT = 200
N_DOWNSAMPLE_PER_SESSION = 30000
EPS_DIST_M = 0.05


def _fit_p0_n(x_log: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    coef = np.polyfit(x_log, s, deg=1)
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
    n = len(x_log)
    out = np.empty((n_boot, 2), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        coef = np.polyfit(x_log[idx], s[idx], deg=1)
        out[b, 0] = coef[1]
        out[b, 1] = -coef[0] / 10.0
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
    ax.set_title(f"P0.2 v3 residual map  {sd}  (R^2={r2:.3f}, n={n_rows:,})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9,
              framealpha=0.9)
    fig.colorbar(sc, ax=ax, label="signal_power - predicted [dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("P0.2 v3  Per-session path-loss fit on v3-cleaned data")
    print("=" * 72)
    cols = ["session_date", "x_m", "y_m", "signal_power", "speed_mps"]
    jp = load_joint(cols)
    # The canonical anomaly_mask.parquet now points to the v3 mask
    # (P0.7 v3 overwrites it). Use the explicit v3 file to be unambiguous.
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask_v3.parquet")
    jp["anomaly_flag"] = mask["anomaly_flag"].to_numpy()

    rng = np.random.default_rng(C.SEED)

    # Load existing v2 ap_coords so we can augment in place rather than overwrite
    ap_coords_path = C.ARTIFACTS_DIR / "ap_coords.json"
    ap_coords = read_json(ap_coords_path)

    v3_per_session: dict = {}
    residual_records: list[pd.DataFrame] = []

    for sd in C.SESSIONS:
        x_ap, y_ap = C.AP_TRUTH[sd]
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
            "residual_v3": resid.astype(np.float32),
        }))

        v3_record = {
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
        v3_per_session[sd] = v3_record
        # Augment the ap_coords entry: keep all v2 fields, attach a `v3`
        # sub-dict so downstream consumers can read either generation.
        if sd in ap_coords:
            ap_coords[sd]["v3"] = v3_record
        else:
            ap_coords[sd] = v3_record

        d_plot = np.maximum(np.hypot(x_fit - x_ap, y_fit - y_ap), EPS_DIST_M)
        resid_plot = s_fit - (p0 - 10.0 * n * np.log10(d_plot))
        _plot_residual_map(
            x_fit, y_fit, resid_plot, x_ap, y_ap, sd, r2,
            len(sub_fit_ds),
            C.FIGURES_DIR / f"p0_2_residual_map_v3_{sd.replace('.', '-')}.png",
        )
        print(f"  {sd}: AP=({x_ap:.3f}, {y_ap:.3f}) "
              f"P0={p0:.2f} dB  n={n:.3f}  R^2={r2:.3f}  n_rows={len(sub_fit_ds):,}")
        print(f"    bootstrap 95% CIs: P0=[{p0_lo:.2f},{p0_hi:.2f}]  "
              f"n=[{n_lo:.3f},{n_hi:.3f}]")

    pd.concat(residual_records, ignore_index=True).to_parquet(
        C.CACHE_DIR / "path_loss_residuals_v3.parquet", index=False)
    write_json(ap_coords_path, ap_coords)

    # Gate A re-evaluation
    r2s = [v3_per_session[sd]["R2"] for sd in C.SESSIONS]
    if all(r > 0.85 for r in r2s):
        gate_a = "YELLOW"
    elif any(r < 0.4 for r in r2s):
        gate_a = "GREEN"
    else:
        gate_a = "GREEN"

    ns = np.array([v3_per_session[sd]["n"] for sd in C.SESSIONS])
    n_widths = np.array([
        v3_per_session[sd]["n_ci"][1] - v3_per_session[sd]["n_ci"][0]
        for sd in C.SESSIONS])
    flags: list[str] = []
    for i, sd_i in enumerate(C.SESSIONS):
        for j, sd_j in enumerate(C.SESSIONS):
            if j <= i:
                continue
            lo_i, hi_i = v3_per_session[sd_i]["n_ci"]
            lo_j, hi_j = v3_per_session[sd_j]["n_ci"]
            if hi_i < lo_j or hi_j < lo_i:
                flags.append(f"n_{sd_i} vs n_{sd_j} CIs disjoint")
    if float(np.ptp(ns)) > 2 * float(n_widths.max()):
        flags.append("n cross-session > 2x bootstrap CI")

    # v2-vs-v3 comparison
    v2_payload = read_json(C.CACHE_DIR / "p0_2_v2.json")
    deltas = {}
    for sd in C.SESSIONS:
        v2 = v2_payload["ap_results"][sd]
        v3 = v3_per_session[sd]
        deltas[sd] = {
            "P0_v2": v2["P0"], "P0_v3": v3["P0"], "P0_delta": v3["P0"] - v2["P0"],
            "n_v2": v2["n"], "n_v3": v3["n"], "n_delta": v3["n"] - v2["n"],
            "R2_v2": v2["R2"], "R2_v3": v3["R2"], "R2_delta": v3["R2"] - v2["R2"],
            "rows_v2": v2["n_rows_used"], "rows_v3": v3["n_rows_used"],
        }

    summary = {
        "ap_results_v3": v3_per_session,
        "gate_A_v3": gate_a,
        "flags": flags,
        "n_boot": N_BOOT,
        "deltas_v2_to_v3": deltas,
    }
    write_json(C.CACHE_DIR / "p0_2_v3.json", summary)
    print(f"  Gate A: {gate_a}  flags={flags}")
    for sd in C.SESSIONS:
        d = deltas[sd]
        print(f"  {sd}: R2 {d['R2_v2']:+.3f} -> {d['R2_v3']:+.3f} "
              f"(delta={d['R2_delta']:+.3f})   n {d['n_v2']:+.3f} -> {d['n_v3']:+.3f} "
              f"(delta={d['n_delta']:+.3f})")
    print("P0.2 v3 done.")


if __name__ == "__main__":
    main()
