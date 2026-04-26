"""Per-CSV-file cross-check.

Computes a τ̂ for each individual CSV file using the same canonical
methodology as the per-day analysis (rot_aware signature, ±3 s search,
parabolic peak refinement, moving-block bootstrap with adaptive block
length, σ_rmse uncertainty). These values are NOT applied to the joint
dataset — they serve as a finer-than-day validation cross-check that the
per-day estimate is a coherent summary of the day's data.

Within-day Cochran's Q across the per-CSV estimates tests whether the
per-day "single offset" assumption is supported on a finer time scale.
A pass means the per-day τ̂ is a good summary; a fail would flag a
step-discontinuity between CSV files that would warrant a finer-grained
correction.

Outputs:
    cache/per_csv_taus.json
    cache/per_csv_homogeneity.json
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config as C
from . import lib


def _run_one(z_t: np.ndarray, z_l: np.ndarray, search_range_s: float,
             rng: np.random.Generator) -> dict:
    lags, rho = lib.xcorr_lags(z_t, z_l, max_lag_s=search_range_s)
    tau_hat, k_peak, peak_y = lib.parabolic_refine(lags, rho)

    boundary_frac = abs(lags[k_peak]) / max(search_range_s, 1e-9)
    widened = False
    if boundary_frac > 0.9 and search_range_s < C.WIDEN_RANGE_S:
        widened = True
        lags, rho = lib.xcorr_lags(z_t, z_l, max_lag_s=C.WIDEN_RANGE_S)
        tau_hat, k_peak, peak_y = lib.parabolic_refine(lags, rho)

    prom = lib.peak_prominence(rho, k_peak)
    block_len_s = max(C.BLOCK_LEN_S, 3.0 * abs(tau_hat) + C.BLOCK_LEN_S)
    bs = lib.moving_block_bootstrap(z_t, z_l, B=C.BOOTSTRAP_B,
                                    block_len_s=block_len_s,
                                    max_lag_s=lags.max(),
                                    rng=rng,
                                    tau_point=tau_hat,
                                    modal_window_s=2.0)
    sigma_eff = bs["sigma_rmse"]
    if np.isfinite(sigma_eff):
        ci_lo = tau_hat - 1.96 * sigma_eff
        ci_hi = tau_hat + 1.96 * sigma_eff
    else:
        ci_lo = ci_hi = float("nan")
    bias = float(np.nanmean(bs["taus"]) - tau_hat) \
        if np.isfinite(bs["sigma"]) else float("nan")
    return {
        "lags": lags, "rho": rho,
        "tau_hat": tau_hat, "peak_value": float(peak_y),
        "prominence": float(prom),
        "search_range_s": float(lags.max()),
        "widened_search": bool(widened),
        "sigma": float(sigma_eff),
        "ci_low": float(ci_lo), "ci_high": float(ci_hi),
        "sigma_kind": "rmse_about_point",
        "bootstrap_bias_s": bias,
        "modal_fraction": bs["modal_fraction"],
        "block_len_s": bs["block_len_s"],
    }


def main():
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[Per-CSV cross-check] Enumerating CSV files ...")
    csvs, telemetry, _ = lib.enumerate_csvs(verbose=True)

    # Reuse per-day LiDAR dissim cache (same artefact)
    lid_sigs: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(C.RNG_SEED)
    results: list[dict] = []

    for meta in csvs:
        if not meta.eligible_for_estimation:
            print(f"[{meta.session_date}/{meta.run_file}] SKIP "
                  f"({meta.eligibility_reason})")
            continue
        if meta.session_date not in lid_sigs:
            p = C.CACHE_DIR / f"lidar_sig_{meta.session_date}.parquet"
            lid_sigs[meta.session_date] = pd.read_parquet(p)
        sig = lib.make_csv_signature(meta, telemetry, lid_sigs[meta.session_date])
        if not sig:
            print(f"[{meta.session_date}/{meta.run_file}] empty signature")
            continue
        z_t = sig["z_t"]; z_l = sig["z_l"]
        res = _run_one(z_t, z_l, C.SEARCH_RANGE_S, rng)
        print(f"[{meta.session_date}/{meta.run_file}]  "
              f"tau={res['tau_hat']:+.4f}s  sigma={res['sigma']:.4f}  "
              f"rho={res['peak_value']:.3f}  prom={res['prominence']:.3f}  "
              f"L={res['block_len_s']:.0f}s")
        results.append({
            "session_date": meta.session_date,
            "run_file": meta.run_file,
            "tel_n": meta.tel_n,
            "tel_rate_hz": meta.tel_rate_hz,
            "lid_n": meta.lid_n,
            "lid_rate_hz": meta.lid_rate_hz,
            "intersect_seconds": meta.intersect_seconds,
            "motion_seconds": meta.motion_seconds,
            "n_stop_start_events": meta.n_stop_start_events,
            "tau_hat": res["tau_hat"],
            "sigma": res["sigma"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "peak_value": res["peak_value"],
            "prominence": res["prominence"],
            "modal_fraction": res["modal_fraction"],
            "block_len_s": res["block_len_s"],
            "bootstrap_bias_s": res["bootstrap_bias_s"],
            "sigma_kind": res["sigma_kind"],
            "included": bool(res["prominence"] >= C.PROMINENCE_MIN),
        })

    out_path = C.CACHE_DIR / "per_csv_taus.json"

    def _to_jsonable(o):
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    out_path.write_text(json.dumps(_to_jsonable(results), indent=2))
    print(f"\nWrote {out_path}")

    # ---- Within-day Cochran-Q on the per-CSV estimates ----
    h = json.loads((C.CACHE_DIR / "homogeneity_per_day.json").read_text())
    per_day = h["per_session_day"]

    per_day_pool = {}
    for sd in sorted({r["session_date"] for r in results}):
        day_csvs = [r for r in results if r["session_date"] == sd]
        taus = np.array([r["tau_hat"] for r in day_csvs])
        sigs = np.array([r["sigma"] for r in day_csvs])
        if len(day_csvs) >= 2:
            agg = lib.cochran_q(taus, sigs)
        else:
            r0 = day_csvs[0]
            agg = {"tau_bar": r0["tau_hat"], "se_tau_bar": r0["sigma"],
                   "ci_low": r0["ci_low"], "ci_high": r0["ci_high"],
                   "Q": float("nan"), "df": 0, "p": float("nan"),
                   "I2": float("nan"), "n": 1}
        per_day_pool[sd] = {
            "n_csv": len(day_csvs),
            "csv_taus": [r["tau_hat"] for r in day_csvs],
            "csv_sigmas": [r["sigma"] for r in day_csvs],
            "csv_files": [r["run_file"] for r in day_csvs],
            "pooled_tau_bar": agg["tau_bar"],
            "pooled_se": agg["se_tau_bar"],
            "Q": agg["Q"], "df": agg["df"], "p": agg["p"], "I2": agg["I2"],
            "applied_per_day_tau": per_day[sd]["tau_bar"],
            "applied_per_day_se":  per_day[sd]["se"],
            "verdict_within_day_constant": (
                bool(np.isfinite(agg.get("p", float("nan")))
                     and agg["p"] > C.HOMOGENEITY_P_MIN
                     and agg["I2"] < C.HOMOGENEITY_I2_MAX)
                if agg["df"] > 0 else None
            ),
        }

    out_h = C.CACHE_DIR / "per_csv_homogeneity.json"
    out_h.write_text(json.dumps(per_day_pool, indent=2))
    print(f"Wrote {out_h}")
    print("\n=== Within-day Cochran-Q (per-CSV pool) ===")
    for sd, d in per_day_pool.items():
        if d["df"] == 0:
            print(f"  {sd}: single CSV file, within-day Q n/a")
            continue
        tag = "PASS" if d["verdict_within_day_constant"] else "FAIL"
        print(f"  {sd}: n_csv={d['n_csv']}  pooled_tau={d['pooled_tau_bar']:+.4f} s  "
              f"Q={d['Q']:.3f} df={d['df']} p={d['p']:.3g} I2={d['I2']:.1f}%  [{tag}]")
        print(f"          applied_per_day_tau = {d['applied_per_day_tau']:+.4f} s  "
              f"(diff vs CSV-pool = "
              f"{d['applied_per_day_tau']-d['pooled_tau_bar']:+.4f} s)")


if __name__ == "__main__":
    main()
