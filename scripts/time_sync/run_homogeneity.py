"""Phase 4(a): across-day homogeneity test.

Pools the per-day offset estimates and computes:
    - inverse-variance-weighted mean tau_bar with 95% CI
    - Cochran's Q on n_days - 1 degrees of freedom
    - I^2

Decision rule: I^2 < HOMOGENEITY_I2_MAX AND p > HOMOGENEITY_P_MIN
            -> a single global constant is supported across days.

Outputs:
    cache/homogeneity_per_day.json
"""
from __future__ import annotations
import json

import numpy as np

from . import config as C
from . import lib


def main():
    offsets = json.loads((C.CACHE_DIR / "offsets_per_day.json").read_text())
    primary = list(offsets)
    inc = [r for r in primary if r["included"]]

    all_taus = np.array([r["tau_hat"] for r in primary])
    all_sigs = np.array([r["sigma"] for r in primary])
    all_labels = [r["session_date"] for r in primary]

    inc_taus = np.array([r["tau_hat"] for r in inc])
    inc_sigs = np.array([r["sigma"] for r in inc])
    inc_labels = [r["session_date"] for r in inc]

    res_all = lib.cochran_q(all_taus, all_sigs)
    res_inc = lib.cochran_q(inc_taus, inc_sigs)

    decision = (res_inc["I2"] < C.HOMOGENEITY_I2_MAX
                and res_inc["p"] > C.HOMOGENEITY_P_MIN)

    # Per-day correction values (the actual values applied downstream).
    # Each day is its own estimate — no within-day pooling needed.
    per_day = {}
    for r in primary:
        per_day[r["session_date"]] = {
            "n_csv_files": r["n_csv_files"],
            "csv_files": r["csv_files"],
            "tau_bar": r["tau_hat"],
            "se": r["sigma"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "rho_peak": r["peak_value"],
            "prominence": r["prominence"],
            "included_in_global_pool": r["included"],
        }

    # Within-day half-split summary
    half_pass = 0; half_fail = 0; half_unavail = 0
    drift_table = []
    for r in primary:
        hs = r.get("half_split", {})
        if hs.get("agree") is None:
            half_unavail += 1
            drift_table.append({
                "session_date": r["session_date"],
                "tau_first": None, "sigma_first": None,
                "tau_second": None, "sigma_second": None,
                "agree": None, "diff": None, "slope_ms_per_min": None,
            })
            continue
        if hs["agree"]:
            half_pass += 1
        else:
            half_fail += 1
        slope_ms_per_min = None
        if hs.get("agree") is False:
            half_dur_s = (r.get("intersect_seconds", 0) or 0.0) / 2.0
            if half_dur_s > 0:
                d_tau = hs["second"]["tau_hat"] - hs["first"]["tau_hat"]
                d_t_min = half_dur_s / 60.0
                slope_ms_per_min = float(d_tau / d_t_min * 1000.0)
        drift_table.append({
            "session_date": r["session_date"],
            "tau_first": hs["first"]["tau_hat"],
            "sigma_first": hs["first"]["sigma"],
            "tau_second": hs["second"]["tau_hat"],
            "sigma_second": hs["second"]["sigma"],
            "agree": hs["agree"],
            "diff": hs["diff"],
            "slope_ms_per_min": slope_ms_per_min,
        })

    out = {
        "primary_variant": "rot_aware (per-day, gap-masked)",
        "decision_rule": {
            "I2_max_pct": C.HOMOGENEITY_I2_MAX,
            "p_min": C.HOMOGENEITY_P_MIN,
            "verdict_constant_global": bool(decision),
        },
        "all_days_pool": {**res_all, "labels": all_labels,
                          "taus": all_taus.tolist(),
                          "sigmas": all_sigs.tolist()},
        "included_only_pool": {**res_inc, "labels": inc_labels,
                               "taus": inc_taus.tolist(),
                               "sigmas": inc_sigs.tolist()},
        "per_session_day": per_day,
        "drift_summary": {
            "n_pass": half_pass, "n_fail": half_fail,
            "n_unavailable": half_unavail,
            "fraction_agree": (half_pass / max(1, half_pass + half_fail)),
        },
        "drift_table": drift_table,
    }

    out_path = C.CACHE_DIR / "homogeneity_per_day.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print("\n=== Across-day pool (included only) ===")
    print(f"  n={res_inc['n']}  tau_bar={res_inc['tau_bar']:+.4f}s  "
          f"95% CI=[{res_inc['ci_low']:+.4f},{res_inc['ci_high']:+.4f}]")
    print(f"  Q={res_inc['Q']:.3f}  df={res_inc['df']}  "
          f"p={res_inc['p']:.3g}  I^2={res_inc['I2']:.1f}%")
    print(f"  -> constant-offset (global) supported? {decision}")
    print("\n=== Per-day correction values (APPLIED) ===")
    for sd, d in per_day.items():
        print(f"  {sd}:  tau={d['tau_bar']:+.4f}s  SE={d['se']:.4f}  "
              f"95% CI=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]  "
              f"rho_peak={d['rho_peak']:.3f}  csv_files={d['n_csv_files']}")
    print("\n=== Within-day half-split ===")
    print(f"  agree={half_pass}  fail={half_fail}  unavailable={half_unavail}")


if __name__ == "__main__":
    main()
