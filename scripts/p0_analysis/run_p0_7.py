"""P0.7 — Operational anomaly detection.

Identifies stop episodes per session, classifies them as routine /
manual-reposition / motor-overheat, and produces a per-row anomaly mask
plus example-event timeline figures.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import load_joint, write_json


# Stop-detection parameters
SPEED_THRESHOLD = 0.05            # m/s
MIN_STOP_DURATION_S = 30.0        # seconds
REPOSITION_DIST_M = 0.30          # 30 cm pre/post displacement threshold
CONF_SIGMA_DROP = 3.0             # n-sigma drop in nns_position_confidence
POST_REPOS_KEEP_S = 5.0           # rows in 5s after a manual reposition are anomalous
POST_OVERHEAT_KEEP_S = 10.0       # window after motor-overheat resumption


def _abs_speed(df: pd.DataFrame) -> np.ndarray:
    """Robust |speed|: prefer speed_mps, fall back to (|left|+|right|)/2."""
    s = df["speed_mps"].abs().to_numpy()
    backup = (df["actual_speed_left"].abs() + df["actual_speed_right"].abs()).to_numpy() / 2.0
    nan_mask = ~np.isfinite(s)
    s = np.where(nan_mask, backup, s)
    return s


def _detect_stops(ts_ns: np.ndarray, speed: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx_inclusive), ...] for stops >= MIN_STOP_DURATION_S."""
    is_stop = speed < SPEED_THRESHOLD
    if not is_stop.any():
        return []
    diff = np.diff(is_stop.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    out: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        dur = (ts_ns[e] - ts_ns[s]) / 1e9
        if dur >= MIN_STOP_DURATION_S:
            out.append((int(s), int(e)))
    return out


def _classify(df: pd.DataFrame, ep: tuple[int, int],
              session_conf_med: float, session_conf_std: float
              ) -> tuple[str, dict]:
    """Classify a stop episode.

    Note on flag selection: the brief lists left/right_drive_stop_executed
    and nncf_3108_abort_result as candidate motor-fault flags, but inspection
    shows these fire on essentially every stop (>= 95% rate when stopped) —
    they are routine brake-engagement / command-result signals, not faults.
    nncf_3108_abort_result is constant (= 2.0) across the entire dataset.
    We instead use nns_error_status (raised on ~50% of stopped rows but
    near-zero when moving) as the discriminating fault indicator. This
    deviation is documented in the report.
    """
    s, e = ep
    n = len(df)
    pre_idx = max(s - 1, 0)
    post_idx = min(e + 1, n - 1)
    dx = df["x_m"].iat[post_idx] - df["x_m"].iat[pre_idx]
    dy = df["y_m"].iat[post_idx] - df["y_m"].iat[pre_idx]
    pre_post_dist = float(np.hypot(dx, dy))

    conf = df["nns_position_confidence"].iloc[s:e+1]
    conf_dropped = bool(
        (conf.isna().any()) or
        (session_conf_std > 0
         and (session_conf_med - conf.min()) >= CONF_SIGMA_DROP * session_conf_std)
    )

    err_raised = bool(df["nns_error_status"].iloc[s:e+1].any())

    info = dict(
        pre_post_dist_m=pre_post_dist,
        conf_dropped=conf_dropped,
        err_raised=err_raised,
        duration_s=float((df["fh7000_timestamp"].iat[e] -
                          df["fh7000_timestamp"].iat[s]).total_seconds()),
    )

    is_repos = (pre_post_dist > REPOSITION_DIST_M) or conf_dropped
    is_overheat = err_raised and not is_repos
    if is_repos:
        return "manual_reposition", info
    if is_overheat:
        return "motor_overheat", info
    return "routine", info


def _build_anomaly_mask(df: pd.DataFrame,
                        episodes: list[tuple[tuple[int, int], str, dict]],
                        ts_s: np.ndarray, speed: np.ndarray) -> np.ndarray:
    n = len(df)
    flag = np.zeros(n, dtype=bool)
    typ = np.array([""] * n, dtype=object)

    for (s, e), kind, _info in episodes:
        if kind == "manual_reposition":
            flag[s:e+1] = True
            typ[s:e+1] = "manual_reposition"
            t_end = ts_s[e]
            j = e + 1
            while j < n and (ts_s[j] - t_end) <= POST_REPOS_KEEP_S:
                flag[j] = True
                typ[j] = "manual_reposition_post"
                j += 1
        elif kind == "motor_overheat":
            # the stop itself stays in the dataset
            t_end = ts_s[e]
            j = e + 1
            prev_v = speed[e]
            while j < n and (ts_s[j] - t_end) <= POST_OVERHEAT_KEEP_S:
                dt = max(ts_s[j] - ts_s[j-1], 1e-3)
                accel = (speed[j] - prev_v) / dt
                if abs(accel) > 0.3:
                    flag[j] = True
                    typ[j] = "motor_overheat_post"
                prev_v = speed[j]
                j += 1
        # routine: nothing
    return flag, typ


def _plot_event(df: pd.DataFrame, ep: tuple[int, int], kind: str,
                out_path):
    s, e = ep
    n = len(df)
    pad = 200
    a = max(0, s - pad)
    b = min(n, e + pad + 1)
    sub = df.iloc[a:b].copy()
    t = (sub["fh7000_timestamp"] - df["fh7000_timestamp"].iat[s]).dt.total_seconds().to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, sub["speed_mps"].to_numpy(), color="C0")
    axes[0].axhspan(-SPEED_THRESHOLD, SPEED_THRESHOLD, color="grey", alpha=0.2)
    axes[0].set_ylabel("speed_mps")
    axes[1].plot(t, sub["nns_position_confidence"].to_numpy(), color="C1")
    axes[1].set_ylabel("nns_position_confidence")
    axes[2].plot(t, sub["momentary_current_consumption"].to_numpy(), color="C2")
    axes[2].set_ylabel("momentary_current\n[A?]")
    axes[3].plot(t, sub["x_m"].to_numpy(), label="x_m", color="C3")
    axes[3].plot(t, sub["y_m"].to_numpy(), label="y_m", color="C4")
    axes[3].set_ylabel("x_m, y_m")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_xlabel("time relative to stop start [s]")
    for ax in axes:
        ax.axvspan(0, (df["fh7000_timestamp"].iat[e] - df["fh7000_timestamp"].iat[s]).total_seconds(),
                   color="red", alpha=0.15, label="stop episode")
    axes[0].set_title(f"{kind} example — session {sub['session_date'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("P0.7  Operational anomaly detection")
    print("=" * 72)

    cols = ["session_date", "run_file", "fh7000_timestamp", "speed_mps",
            "actual_speed_left", "actual_speed_right",
            "x_m", "y_m", "nns_position_confidence",
            "left_drive_stop_executed", "right_drive_stop_executed",
            "nns_error_status", "nncf_3108_abort_result",
            "momentary_current_consumption"]
    jp = load_joint(cols)
    jp["fh7000_timestamp"] = pd.to_datetime(jp["fh7000_timestamp"])

    rows_total = len(jp)
    out_flag = np.zeros(rows_total, dtype=bool)
    out_type = np.array([""] * rows_total, dtype=object)
    out_session = jp["session_date"].to_numpy()

    summary = {"per_session": {}, "params": {
        "speed_threshold_mps": SPEED_THRESHOLD,
        "min_stop_duration_s": MIN_STOP_DURATION_S,
        "reposition_dist_m": REPOSITION_DIST_M,
        "conf_sigma_drop": CONF_SIGMA_DROP,
        "post_repos_keep_s": POST_REPOS_KEEP_S,
        "post_overheat_keep_s": POST_OVERHEAT_KEEP_S,
    }}
    chosen_examples: dict[str, tuple[str, tuple[int, int]]] = {}

    for sd in C.SESSIONS:
        m = jp["session_date"].to_numpy() == sd
        idx = np.where(m)[0]
        sub = jp.iloc[idx].reset_index(drop=True)
        ts_s = (sub["fh7000_timestamp"].astype("datetime64[ns]")
                .astype("int64").to_numpy() / 1e9)
        ts_ns = sub["fh7000_timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
        speed = _abs_speed(sub)

        conf_med = float(sub["nns_position_confidence"].median())
        conf_std = float(sub["nns_position_confidence"].std(ddof=0))

        stops = _detect_stops(ts_ns, speed)
        classified = []
        kind_counts = {"routine": 0, "manual_reposition": 0, "motor_overheat": 0}
        kind_durations = {"routine": 0.0, "manual_reposition": 0.0, "motor_overheat": 0.0}
        for ep in stops:
            kind, info = _classify(sub, ep, conf_med, conf_std)
            classified.append((ep, kind, info))
            kind_counts[kind] += 1
            kind_durations[kind] += info["duration_s"]

        flag_local, type_local = _build_anomaly_mask(sub, classified, ts_s, speed)
        out_flag[idx] = flag_local
        out_type[idx] = type_local

        n_anom = int(flag_local.sum())
        summary["per_session"][sd] = {
            "n_rows": int(len(sub)),
            "n_stops_total": len(stops),
            "n_routine": kind_counts["routine"],
            "n_manual_reposition": kind_counts["manual_reposition"],
            "n_motor_overheat": kind_counts["motor_overheat"],
            "duration_routine_s": round(kind_durations["routine"], 2),
            "duration_manual_reposition_s": round(kind_durations["manual_reposition"], 2),
            "duration_motor_overheat_s": round(kind_durations["motor_overheat"], 2),
            "n_anomaly_rows": n_anom,
            "frac_anomaly": round(n_anom / max(len(sub), 1), 6),
            "session_conf_median": conf_med,
            "session_conf_std": conf_std,
        }

        # Pick the longest example per kind for a figure
        for kind in ("manual_reposition", "motor_overheat"):
            cands = [(ep, info["duration_s"], info)
                     for ep, k, info in classified if k == kind]
            if not cands:
                continue
            cands.sort(key=lambda x: -x[1])
            (ep, _, _) = cands[0]
            # Translate local idx back to global
            gs = int(idx[ep[0]]); ge = int(idx[ep[1]])
            cur = chosen_examples.get(kind)
            if cur is None or cands[0][1] > cur[2]:
                chosen_examples[kind] = (sd, (gs, ge), cands[0][1])

        print(f"  {sd}: {len(stops)} stops "
              f"(routine={kind_counts['routine']}, "
              f"reposition={kind_counts['manual_reposition']}, "
              f"overheat={kind_counts['motor_overheat']}); "
              f"anomaly rows: {n_anom}/{len(sub)} "
              f"({100*n_anom/max(len(sub),1):.2f}%)")

    # Write per-row mask (joint_idx == positional row in the joint parquet)
    mask_df = pd.DataFrame({
        "joint_idx": np.arange(rows_total, dtype=np.int64),
        "session_date": out_session,
        "anomaly_flag": out_flag,
        "anomaly_type": out_type,
    })
    mask_df.to_parquet(C.ARTIFACTS_DIR / "anomaly_mask.parquet", index=False)
    print(f"  -> wrote {C.ARTIFACTS_DIR / 'anomaly_mask.parquet'}")

    # Generate example figures (need full rows from joint, so re-load richer cols)
    for kind, (sd, (gs, ge), _dur) in chosen_examples.items():
        # gs,ge are global row indices; reuse the already-loaded jp
        # but we need a window of extra rows for context
        n = len(jp)
        a = max(0, gs - 200)
        b = min(n, ge + 200 + 1)
        sub = jp.iloc[a:b].reset_index(drop=True)
        s_local = gs - a
        e_local = ge - a
        out = C.FIGURES_DIR / f"p0_7_{kind}_example.png"
        _plot_event(sub, (s_local, e_local), kind, out)
        print(f"  -> wrote {out}")

    write_json(C.CACHE_DIR / "p0_7.json", summary)
    # flag any session with > 10% anomaly rows
    flags = [sd for sd, s in summary["per_session"].items() if s["frac_anomaly"] > 0.10]
    if flags:
        print(f"  WARNING: sessions with >10% anomaly rows: {flags}")
    print("P0.7 done.")


if __name__ == "__main__":
    main()
