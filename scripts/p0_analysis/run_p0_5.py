"""P0.5 — Spatial autocorrelation of signal_power (semivariogram)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, write_json


N_SAMPLE = 5000
LAGS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]


def _semivariogram(x: np.ndarray, y: np.ndarray, s: np.ndarray
                   ) -> tuple[list, list, list]:
    """Empirical semivariogram across LAGS with proportional bin width."""
    n = len(x)
    # Pairwise distance matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy)
    # Pairwise squared diff
    s2 = (s[:, None] - s[None, :]) ** 2
    # Upper triangle only
    iu = np.triu_indices(n, k=1)
    dist = dist[iu]
    s2 = s2[iu]
    gammas, ns, lags = [], [], []
    for h in LAGS:
        delta = 0.25 * h
        m = (dist >= h - delta) & (dist <= h + delta)
        if m.sum() < 30:
            gammas.append(float("nan")); ns.append(int(m.sum())); lags.append(h)
            continue
        gammas.append(0.5 * float(np.mean(s2[m])))
        ns.append(int(m.sum())); lags.append(h)
    return lags, gammas, ns


def main() -> None:
    print("=" * 72)
    print("P0.5  Spatial autocorrelation of signal_power")
    print("=" * 72)
    jp = load_joint(["session_date", "x_m", "y_m", "signal_power", "speed_mps"])
    mask = pd.read_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet")
    jp["anomaly_flag"] = mask["anomaly_flag"].to_numpy()

    rng = np.random.default_rng(C.SEED)
    out_per_session = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for sd, color in zip(C.SESSIONS, ("C0", "C3", "C2")):
        sub = jp[(jp["session_date"] == sd) & (~jp["anomaly_flag"]) &
                 jp["signal_power"].notna()]
        if len(sub) > N_SAMPLE:
            idx = rng.choice(len(sub), N_SAMPLE, replace=False)
            sub = sub.iloc[idx]
        x = sub["x_m"].to_numpy()
        y = sub["y_m"].to_numpy()
        s = sub["signal_power"].to_numpy().astype(np.float64)
        lags, gammas, ns = _semivariogram(x, y, s)
        # Range = smallest lag where gamma >= 0.95 * gamma(h_max)
        gamma_arr = np.array(gammas)
        valid = np.isfinite(gamma_arr)
        if not valid.any():
            range_m = None
        else:
            max_g = np.nanmax(gamma_arr)
            idx_above = np.where(np.isfinite(gamma_arr) &
                                 (gamma_arr >= 0.95 * max_g))[0]
            range_m = float(LAGS[idx_above[0]]) if len(idx_above) else None
        out_per_session[sd] = {
            "lags_m": LAGS,
            "gammas": [float(g) if np.isfinite(g) else None for g in gammas],
            "n_pairs": ns,
            "range_m": range_m,
            "gamma_max": float(max_g) if valid.any() else None,
            "n_sample": int(len(sub)),
        }
        ax.plot(lags, gammas, "-o", label=f"{sd} (range~{range_m}m)", color=color)
        print(f"  {sd}: n={len(sub):,}  gamma_max={out_per_session[sd]['gamma_max']:.2f}  "
              f"range~{range_m} m")
    ax.set_xlabel("lag h [m]")
    ax.set_ylabel("γ(h) [dB²]")
    ax.set_title("P0.5  empirical semivariogram of signal_power")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "p0_5_semivariograms.png", dpi=120)
    plt.close(fig)

    write_json(C.CACHE_DIR / "p0_5.json", {
        "n_sample_target": N_SAMPLE,
        "lags_m": LAGS,
        "per_session": out_per_session,
    })
    print("P0.5 done.")


if __name__ == "__main__":
    main()
