"""Phase 3 + Phase 4(b): per-DAY offset estimation, bootstrap, half-split.

Outputs:
    cache/xcorr_<session_date>.npz   lags & rho for plotting
    cache/offsets_per_day.json       per-day offset estimates and tests

Run after run_signatures.py.
"""
from __future__ import annotations
import json

import numpy as np

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
        ci_lo_eff = tau_hat - 1.96 * sigma_eff
        ci_hi_eff = tau_hat + 1.96 * sigma_eff
    else:
        ci_lo_eff = float("nan"); ci_hi_eff = float("nan")
    bias = float(np.nanmean(bs["taus"]) - tau_hat) \
        if np.isfinite(bs["sigma"]) else float("nan")

    return {
        "lags": lags, "rho": rho,
        "tau_hat": tau_hat,
        "peak_idx": int(k_peak),
        "peak_value": float(peak_y),
        "prominence": float(prom),
        "search_range_s": float(lags.max()),
        "widened_search": bool(widened),
        "sigma": float(sigma_eff),
        "ci_low": float(ci_lo_eff),
        "ci_high": float(ci_hi_eff),
        "sigma_kind": "rmse_about_point",
        "sigma_full": bs["sigma"],
        "ci_low_pct": bs["ci_low"],
        "ci_high_pct": bs["ci_high"],
        "bootstrap_bias_s": bias,
        "modal_fraction": bs["modal_fraction"],
        "block_len_s": bs["block_len_s"],
        "bootstrap_taus": bs["taus"],
    }


def main():
    days_meta = json.loads((C.CACHE_DIR / "days.json").read_text())

    results: list[dict] = []
    rng = np.random.default_rng(C.RNG_SEED)
    for meta in days_meta:
        if not meta["eligible_for_estimation"]:
            print(f"[{meta['session_date']}] SKIP ({meta['eligibility_reason']})")
            continue
        sig_path = C.CACHE_DIR / f"sig_day_{meta['session_date']}.npz"
        if not sig_path.exists():
            print(f"[{meta['session_date']}] no signature, skipping")
            continue
        sig = np.load(sig_path)
        z_t = sig["z_t"]; z_l = sig["z_l"]; t_grid = sig["t_grid"]

        print(f"\n[{meta['session_date']}] computing per-day offset ...")
        res = _run_one(z_t, z_l, C.SEARCH_RANGE_S, rng)
        np.savez_compressed(
            C.CACHE_DIR / f"xcorr_{meta['session_date']}.npz",
            lags=res["lags"], rho=res["rho"],
            bootstrap_taus=res["bootstrap_taus"],
        )
        print(f"   tau_hat={res['tau_hat']:+.4f}s  sigma_rmse={res['sigma']:.4f}s  "
              f"bias={res['bootstrap_bias_s']:+.4f}s  modal_frac={res['modal_fraction']:.2f}  "
              f"L={res['block_len_s']:.0f}s  prom={res['prominence']:.3f}  "
              f"rho_peak={res['peak_value']:.3f}")

        # Half-split test on the day signal
        half_results = {"first": None, "second": None, "agree": None}
        duration_s = float(t_grid[-1] - t_grid[0])
        if duration_s >= 2.0 * C.DRIFT_HALF_MIN_MOTION_S:
            mid = len(z_t) // 2
            z_t_a, z_l_a = z_t[:mid], z_l[:mid]
            z_t_b, z_l_b = z_t[mid:], z_l[mid:]
            if z_t_a.std() > 1e-3 and z_t_b.std() > 1e-3:
                res_a = _run_one(z_t_a, z_l_a, C.SEARCH_RANGE_S, rng)
                res_b = _run_one(z_t_b, z_l_b, C.SEARCH_RANGE_S, rng)
                agree = lib.cis_overlap(res_a["tau_hat"], res_a["sigma"],
                                        res_b["tau_hat"], res_b["sigma"])
                half_results = {
                    "first": {"tau_hat": res_a["tau_hat"], "sigma": res_a["sigma"],
                              "ci_low": res_a["ci_low"], "ci_high": res_a["ci_high"],
                              "prominence": res_a["prominence"],
                              "modal_fraction": res_a["modal_fraction"]},
                    "second": {"tau_hat": res_b["tau_hat"], "sigma": res_b["sigma"],
                               "ci_low": res_b["ci_low"], "ci_high": res_b["ci_high"],
                               "prominence": res_b["prominence"],
                               "modal_fraction": res_b["modal_fraction"]},
                    "agree": bool(agree),
                    "diff": float(res_a["tau_hat"] - res_b["tau_hat"]),
                }
                print(f"   half-split: first={res_a['tau_hat']:+.3f}±{res_a['sigma']:.3f}  "
                      f"second={res_b['tau_hat']:+.3f}±{res_b['sigma']:.3f}  agree={agree}")

        results.append({
            "session_date": meta["session_date"],
            "n_csv_files": meta["n_csv_files"],
            "csv_files": meta["csv_files"],
            "intersect_seconds": meta["intersect_seconds"],
            "motion_seconds": meta["motion_seconds"],
            "n_stop_start_events": meta["n_stop_start_events"],
            "tel_rate_hz": meta["tel_rate_hz"],
            "lid_rate_hz": meta["lid_rate_hz"],
            "tel_gap_count": meta["tel_gap_count"],
            "tel_gap_seconds_sum": meta["tel_gap_seconds_sum"],
            "n_grid_samples": int(len(z_t)),
            "tau_hat": res["tau_hat"],
            "sigma": res["sigma"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "sigma_kind": res["sigma_kind"],
            "sigma_full": res["sigma_full"],
            "ci_low_pct": res["ci_low_pct"],
            "ci_high_pct": res["ci_high_pct"],
            "bootstrap_bias_s": res["bootstrap_bias_s"],
            "modal_fraction": res["modal_fraction"],
            "block_len_s": res["block_len_s"],
            "prominence": res["prominence"],
            "peak_value": res["peak_value"],
            "search_range_s": res["search_range_s"],
            "widened_search": res["widened_search"],
            "included": bool(res["prominence"] >= C.PROMINENCE_MIN),
            "half_split": half_results,
        })

    out_path = C.CACHE_DIR / "offsets_per_day.json"

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
    print(f"\nWrote {out_path}  ({len(results)} day estimates)")


if __name__ == "__main__":
    main()
