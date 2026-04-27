"""P0.7 v3 — Confidence-threshold-based anomaly mask.

Replaces the v1/v2 stop-classifier with a single per-row threshold on
`nns_position_confidence`. Rows with confidence below the threshold (or with
NaN/invalid confidence) are flagged as anomalous; everything else is kept.

The threshold itself is derived three ways and reconciled:

    Method A — saddle of the empirical confidence distribution.
    Method B — 2-component Gaussian mixture (EM, no sklearn).
    Method C — ROC of (confidence vs. position-discontinuity flag); pick the
               threshold that maximises Youden's J.

Outputs:
    artifacts/anomaly_mask_v3.parquet            — per-row [joint_idx,
                                                   session_date, anomaly_flag]
    artifacts/anomaly_mask.parquet               — overwritten with the v3 mask
    artifacts/anomaly_threshold.json             — chosen T*, all method values
    cache/p0_7_v3.json                           — per-session summary +
                                                   v2-vs-v3 validation
    figures/p0_7_v3_confidence_hist.png          — Method A histogram
    figures/p0_7_v3_gmm.png                      — Method B GMM
    figures/p0_7_v3_roc.png                      — Method C ROC
    v2_backup/anomaly_mask_v2.parquet            — backup of the v2 mask
                                                   (created by the driver)
"""
from __future__ import annotations

import os
import shutil
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
from .io_utils import load_joint, write_json


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Method C — discontinuity definition
DISC_ABS_M = 0.30          # ≥ 30 cm position step …
DISC_EXP_MULT = 5.0        # … and ≥ 5× the expected step from telemetry
DISC_MAX_DT_S = 1.0        # only consider neighbouring rows within 1 s
                           # (gaps larger than this are session/run boundaries
                           # or sensor pauses; not a true discontinuity)

# Method C subsampling — 60k rows is plenty for a stable ROC
ROC_SUBSAMPLE = 60000

# A→C agreement tolerance for the decision rule (confidence units)
AC_AGREE_UNITS = 5.0

# Method C is only treated as a reliable threshold source if its Youden's J
# is meaningful. Below this floor, "confidence < T" is essentially uninformative
# about the row-level position-jump flag, and Method C cannot produce a usable
# operating point. For this dataset, J turns out to be ~0.01 (see §4.2): manual
# repositions present mostly as *stuck-at-wrong-position* runs (small step,
# stable but stale (x,y)) rather than as row-level jumps, so disc-vs-conf is
# barely correlated even though confidence does drop. We document this in the
# rationale and fall through to Method A.
METHOD_C_J_FLOOR = 0.10

# GMM EM
GMM_MAX_ITER = 200
GMM_TOL = 1e-6


# ---------------------------------------------------------------------------
# Method A — saddle finder
# ---------------------------------------------------------------------------

def _saddle_from_hist(values: np.ndarray) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Return (saddle, bin_centers, counts).

    Saddle = the bin with the smallest count between the two largest peaks
    of a 5-bin moving-average smoothed histogram. If the distribution is
    effectively unimodal (no separate low-end peak), return None.
    """
    edges = np.arange(-0.5, 101.5, 1.0)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 5-bin moving average for stability
    k = 5
    pad = k // 2
    padded = np.pad(counts.astype(np.float64), pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(k) / k, mode="valid")

    # Find local maxima (peaks)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks.append((i, smoothed[i]))
    peaks.sort(key=lambda x: -x[1])

    if len(peaks) < 2:
        return None, centers, counts

    # Take the two largest peaks; saddle = min between them
    p1_idx = peaks[0][0]
    # Find the largest peak that is at least ~20 confidence units away
    p2_idx = None
    for idx, _ in peaks[1:]:
        if abs(centers[idx] - centers[p1_idx]) >= 20.0:
            p2_idx = idx
            break
    if p2_idx is None:
        return None, centers, counts

    lo, hi = sorted([p1_idx, p2_idx])
    saddle_offset = int(np.argmin(smoothed[lo + 1:hi]))
    saddle_idx = lo + 1 + saddle_offset
    return float(centers[saddle_idx]), centers, counts


# ---------------------------------------------------------------------------
# Method B — 2-component Gaussian mixture by EM
# ---------------------------------------------------------------------------

def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma)


def _gmm2_fit(x: np.ndarray, rng: np.random.Generator,
              max_iter: int = GMM_MAX_ITER, tol: float = GMM_TOL) -> dict:
    """Fit a 2-component univariate Gaussian mixture by EM.

    Initialised with the 25th and 75th percentiles as means.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=np.float64)
    sd = np.array([max(np.std(x) * 0.5, 1.0)] * 2, dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step
        p0 = pi[0] * _normal_pdf(x, mu[0], sd[0])
        p1 = pi[1] * _normal_pdf(x, mu[1], sd[1])
        denom = np.maximum(p0 + p1, 1e-300)
        r = np.empty((2, len(x)), dtype=np.float64)
        r[0] = p0 / denom
        r[1] = p1 / denom

        # M-step
        nk = r.sum(axis=1)
        nk = np.maximum(nk, 1e-12)
        mu_new = (r * x).sum(axis=1) / nk
        sd_new = np.sqrt(((r * (x - mu_new[:, None]) ** 2).sum(axis=1) / nk))
        sd_new = np.maximum(sd_new, 0.5)
        pi_new = nk / nk.sum()

        ll = float(np.sum(np.log(denom)))
        mu, sd, pi = mu_new, sd_new, pi_new
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Order so component 0 = degraded (lower mean), component 1 = normal (higher)
    if mu[0] > mu[1]:
        mu = mu[::-1]
        sd = sd[::-1]
        pi = pi[::-1]

    # Threshold = where posterior P(degraded | x) = 0.5
    grid = np.linspace(0.0, 100.0, 10001)
    p_d = pi[0] * _normal_pdf(grid, mu[0], sd[0])
    p_n = pi[1] * _normal_pdf(grid, mu[1], sd[1])
    post_d = p_d / np.maximum(p_d + p_n, 1e-300)
    # Look for the rightmost crossing of 0.5 from above-to-below
    idx = np.argmin(np.abs(post_d - 0.5))
    threshold = float(grid[idx])

    overlap = float(min(sd) / max(abs(mu[1] - mu[0]), 1e-9))

    return dict(
        mu=mu.tolist(),
        sd=sd.tolist(),
        pi=pi.tolist(),
        threshold=threshold,
        overlap_ratio=overlap,
        log_lik=ll,
    )


# ---------------------------------------------------------------------------
# Method C — position-discontinuity ROC
# ---------------------------------------------------------------------------

def _build_disc_flag(jp: pd.DataFrame) -> np.ndarray:
    """Per-row position-discontinuity flag.

    For each row i (sorted by fh7000_timestamp within (session, run_file)),
    the flag is True iff the actual position step to row i+1 is > DISC_ABS_M
    and > DISC_EXP_MULT × the telemetry-expected step. Rows at the end of a
    run, or with a Δt > DISC_MAX_DT_S to the next row, get flag = False
    (cannot be evaluated).

    Crucially, disc is computed within-run only. Across-run boundaries the
    AGV is restarted at an arbitrary location, so a between-run "step" is not
    a true discontinuity (the parquet is just concatenated).
    """
    flag = np.zeros(len(jp), dtype=bool)
    for (sd, rf), grp in jp.groupby(["session_date", "run_file"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        t = grp["fh7000_timestamp"].astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
        order = np.argsort(t, kind="stable")
        idx_sorted = idx[order]
        t_sorted = t[order]
        x = grp["x_m"].to_numpy()[order]
        y = grp["y_m"].to_numpy()[order]
        v = np.abs(grp["speed_mps"].to_numpy()[order])
        v = np.where(np.isfinite(v), v, 0.0)

        dt = np.diff(t_sorted)
        dx = np.diff(x)
        dy = np.diff(y)
        step = np.hypot(dx, dy)
        expected = v[:-1] * dt
        thresh = np.maximum(DISC_ABS_M, DISC_EXP_MULT * expected)
        disc_local = (step > thresh) & (dt <= DISC_MAX_DT_S) & (dt > 0)
        flag[idx_sorted[:-1]] = disc_local
    return flag


def _roc_youden(conf: np.ndarray, disc: np.ndarray) -> dict:
    """ROC for "confidence < T" predicting disc==True.

    Returns thresholds, TPR, FPR and the Youden's-J optimal threshold.
    """
    pos = disc.sum()
    neg = len(disc) - pos
    if pos == 0 or neg == 0:
        return dict(threshold=None, j=None, sens=None, spec=None,
                    fpr=np.array([]), tpr=np.array([]),
                    thresholds=np.array([]))

    # We want a fine threshold sweep. Confidence is on [0, 100] (integers in
    # the dataset). Sweep at 0.5-unit resolution for smoothness.
    thresholds = np.arange(0.0, 100.5, 0.5)
    fpr = np.empty_like(thresholds)
    tpr = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = conf < t
        tp = int((pred & disc).sum())
        fp = int((pred & ~disc).sum())
        tpr[i] = tp / pos
        fpr[i] = fp / neg
    j = tpr - fpr
    best = int(np.argmax(j))
    return dict(
        threshold=float(thresholds[best]),
        j=float(j[best]),
        sens=float(tpr[best]),
        spec=float(1.0 - fpr[best]),
        fpr=fpr, tpr=tpr,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def _choose_threshold(t_a: float | None, t_b: dict, t_c: dict) -> tuple[float, str]:
    """Apply the §4.2 decision rule.

    Returns (T*, rationale).
    """
    c_unavailable = (t_c["threshold"] is None)
    c_too_weak = (
        not c_unavailable
        and t_c.get("j") is not None
        and t_c["j"] < METHOD_C_J_FLOOR
    )

    if c_unavailable or c_too_weak:
        weakness = (
            "no position discontinuities detected"
            if c_unavailable
            else (f"Youden's J = {t_c['j']:.3f} is below the {METHOD_C_J_FLOOR:.2f} "
                  "floor, i.e. the row-level position-jump flag and confidence "
                  "are barely correlated for this dataset (manual repositions "
                  "appear here mostly as stuck-at-wrong-position runs, not "
                  "row-level jumps)")
        )
        if t_a is None:
            return t_b["threshold"], (
                f"Method C is unavailable ({weakness}) and Method A is "
                "effectively unimodal; falling back to the GMM crossover "
                f"(Method B) at {t_b['threshold']:.2f}."
            )
        return float(t_a), (
            f"Method C is unavailable ({weakness}); using the Method A "
            f"saddle T_A = {t_a:.1f} as the operational threshold. "
            f"Method B (GMM crossover = {t_b['threshold']:.2f}) is reported "
            "for sanity."
        )

    if t_a is None:
        return t_c["threshold"], (
            "Method A is effectively unimodal (no clear saddle between two "
            f"peaks); using Method C (Youden = {t_c['threshold']:.1f}, "
            f"J = {t_c['j']:.3f}) as the operational threshold."
        )

    if abs(t_a - t_c["threshold"]) <= AC_AGREE_UNITS:
        return 0.5 * (t_a + t_c["threshold"]), (
            f"Methods A (saddle = {t_a:.1f}) and C (Youden = "
            f"{t_c['threshold']:.1f}, J = {t_c['j']:.3f}) agree within "
            f"+/-{AC_AGREE_UNITS:.0f} units; T* = average of the two. "
            f"Method B (GMM crossover = {t_b['threshold']:.1f}) is reported "
            "for sanity."
        )

    return float(t_c["threshold"]), (
        f"Methods A (saddle = {t_a:.1f}) and C (Youden = "
        f"{t_c['threshold']:.1f}, J = {t_c['j']:.3f}) disagree by more than "
        f"+/-{AC_AGREE_UNITS:.0f} units; preferring Method C as the "
        "operationally grounded source."
    )


# ---------------------------------------------------------------------------
# Validation against v2 manual_reposition events
# ---------------------------------------------------------------------------

def _episode_runs(flag: np.ndarray, types: np.ndarray, target_type: str
                  ) -> list[tuple[int, int]]:
    """Return [(start, end_inclusive)] for contiguous runs where flag is True
    AND types == target_type. Mixed-type rows break the run."""
    n = len(flag)
    runs = []
    i = 0
    while i < n:
        if flag[i] and types[i] == target_type:
            j = i
            while j + 1 < n and flag[j + 1] and types[j + 1] == target_type:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _validate_v3_against_v2(v3_flag: np.ndarray, v2_mask: pd.DataFrame,
                            jp: pd.DataFrame) -> dict:
    """Per-session episode-level coverage of v2 manual_reposition rows by v3
    plus motor_overheat over-flagging rate."""
    out: dict = {"per_session": {}, "global": {}}
    v2_flag = v2_mask["anomaly_flag"].to_numpy().astype(bool)
    v2_type = v2_mask["anomaly_type"].to_numpy()

    rows_repos_v2_total = 0
    rows_repos_caught_total = 0
    rows_overheat_v2_total = 0
    rows_overheat_caught_total = 0
    extension_total = 0

    largest_repos_episodes: list[dict] = []

    for sd in C.SESSIONS:
        sd_idx = np.where((jp["session_date"] == sd).to_numpy())[0]
        if len(sd_idx) == 0:
            continue
        # Episodes (manual_reposition core, excluding the *_post extension which
        # is also captured by v2 itself but with a different type marker)
        repos_runs_main = _episode_runs(v2_flag, v2_type, "manual_reposition")
        repos_runs_post = _episode_runs(v2_flag, v2_type, "manual_reposition_post")
        # Only keep runs whose rows are in this session
        repos_runs_main_sd = [(s, e) for (s, e) in repos_runs_main
                              if jp["session_date"].iat[s] == sd]
        repos_runs_post_sd = [(s, e) for (s, e) in repos_runs_post
                              if jp["session_date"].iat[s] == sd]

        # Coverage of repos main runs
        n_main = sum(e - s + 1 for s, e in repos_runs_main_sd)
        n_main_caught = sum(int(v3_flag[s:e + 1].sum()) for s, e in repos_runs_main_sd)
        # Coverage of repos post runs (the 5-second tails v2 added)
        n_post = sum(e - s + 1 for s, e in repos_runs_post_sd)
        n_post_caught = sum(int(v3_flag[s:e + 1].sum()) for s, e in repos_runs_post_sd)

        # Combined repos coverage
        n_repos = n_main + n_post
        n_repos_caught = n_main_caught + n_post_caught

        # Motor overheat rows in v2 (no anomaly flag, but their *_post tails do
        # get flagged in v2). Compare to v3 by *_post run only.
        overheat_post_runs_sd = [(s, e) for (s, e) in
                                 _episode_runs(v2_flag, v2_type, "motor_overheat_post")
                                 if jp["session_date"].iat[s] == sd]
        n_overheat_post = sum(e - s + 1 for s, e in overheat_post_runs_sd)
        n_overheat_post_caught = sum(int(v3_flag[s:e + 1].sum())
                                     for s, e in overheat_post_runs_sd)

        # Extension count: v3-flagged rows in this session that are NOT v2-flagged.
        sd_v3 = v3_flag[sd_idx]
        sd_v2 = v2_flag[sd_idx]
        extension = int(((sd_v3) & (~sd_v2)).sum())
        n_v3_total = int(sd_v3.sum())
        n_v2_total = int(sd_v2.sum())

        # Largest repos episodes
        for s, e in repos_runs_main_sd:
            length = e - s + 1
            caught = int(v3_flag[s:e + 1].sum())
            largest_repos_episodes.append({
                "session": sd,
                "start_idx": int(s),
                "end_idx": int(e),
                "length": length,
                "v3_caught": caught,
                "v3_coverage": float(caught / length) if length else float("nan"),
            })

        out["per_session"][sd] = {
            "v2_repos_main_rows": int(n_main),
            "v2_repos_post_rows": int(n_post),
            "v2_repos_total_rows": int(n_repos),
            "v3_repos_main_caught": int(n_main_caught),
            "v3_repos_post_caught": int(n_post_caught),
            "v3_repos_total_caught": int(n_repos_caught),
            "v3_repos_total_coverage": float(n_repos_caught / n_repos) if n_repos else float("nan"),
            "v2_overheat_post_rows": int(n_overheat_post),
            "v3_overheat_post_caught": int(n_overheat_post_caught),
            "v3_extension_rows_beyond_v2": extension,
            "v3_total_rows_flagged": n_v3_total,
            "v2_total_rows_flagged": n_v2_total,
        }

        rows_repos_v2_total += n_repos
        rows_repos_caught_total += n_repos_caught
        rows_overheat_v2_total += n_overheat_post
        rows_overheat_caught_total += n_overheat_post_caught
        extension_total += extension

    largest_repos_episodes.sort(key=lambda d: -d["length"])
    out["top_repos_episodes"] = largest_repos_episodes[:10]
    out["global"] = {
        "v2_repos_rows_total": rows_repos_v2_total,
        "v3_repos_rows_caught_total": rows_repos_caught_total,
        "v3_repos_global_coverage": (rows_repos_caught_total / rows_repos_v2_total
                                     if rows_repos_v2_total else float("nan")),
        "v2_overheat_post_rows_total": rows_overheat_v2_total,
        "v3_overheat_post_rows_caught_total": rows_overheat_caught_total,
        "v3_overheat_post_global_rate": (rows_overheat_caught_total / rows_overheat_v2_total
                                         if rows_overheat_v2_total else float("nan")),
        "v3_extension_rows_total": extension_total,
    }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_hist(centers: np.ndarray, counts: np.ndarray, t_a: float | None,
               t_star: float, out_path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, ylog in zip(axes, [False, True]):
        ax.bar(centers, counts, width=1.0, color="C0", edgecolor="none")
        if ylog:
            ax.set_yscale("log")
        if t_a is not None:
            ax.axvline(t_a, color="C1", linestyle="--",
                       label=f"saddle T_A = {t_a:.1f}")
        ax.axvline(t_star, color="black", linestyle="-",
                   label=f"chosen T* = {t_star:.2f}")
        ax.set_ylabel("count" + (" (log)" if ylog else ""))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("nns_position_confidence")
    axes[0].set_title("P0.7 v3 — empirical confidence distribution (Method A)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_gmm(values: np.ndarray, gmm: dict, t_star: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(0, 100, 1001)
    ax.hist(values, bins=np.arange(-0.5, 101.5, 1.0), density=True,
            color="C0", alpha=0.5, label="observed")
    p_d = gmm["pi"][0] * _normal_pdf(grid, gmm["mu"][0], gmm["sd"][0])
    p_n = gmm["pi"][1] * _normal_pdf(grid, gmm["mu"][1], gmm["sd"][1])
    ax.plot(grid, p_d, color="C3",
            label=f"degraded N(μ={gmm['mu'][0]:.1f}, σ={gmm['sd'][0]:.1f}), π={gmm['pi'][0]:.2f}")
    ax.plot(grid, p_n, color="C2",
            label=f"normal   N(μ={gmm['mu'][1]:.1f}, σ={gmm['sd'][1]:.1f}), π={gmm['pi'][1]:.2f}")
    ax.plot(grid, p_d + p_n, color="black", linestyle=":", label="mixture")
    ax.axvline(gmm["threshold"], color="C1", linestyle="--",
               label=f"GMM crossover T_B = {gmm['threshold']:.2f}")
    ax.axvline(t_star, color="black", label=f"chosen T* = {t_star:.2f}")
    ax.set_xlabel("nns_position_confidence")
    ax.set_ylabel("density")
    ax.set_title("P0.7 v3 — GMM (Method B)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_roc(roc: dict, t_star: float, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    if len(roc["fpr"]) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "Method C: no discontinuities", ha="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    axes[0].plot(roc["fpr"], roc["tpr"], color="C0")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle=":")
    axes[0].set_xlabel("FPR (1 - specificity)")
    axes[0].set_ylabel("TPR (sensitivity)")
    axes[0].set_title(f"P0.7 v3 ROC (Method C) — Youden's J = {roc['j']:.3f}")
    axes[0].grid(alpha=0.3)
    # Mark Youden point
    j = roc["tpr"] - roc["fpr"]
    bi = int(np.argmax(j))
    axes[0].scatter([roc["fpr"][bi]], [roc["tpr"][bi]], color="C3", zorder=5,
                    label=f"T_C = {roc['threshold']:.2f} "
                          f"(sens={roc['sens']:.2f}, spec={roc['spec']:.2f})")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(roc["thresholds"], j, color="C0")
    axes[1].axvline(roc["threshold"], color="C3", linestyle="--",
                    label=f"argmax J: T_C = {roc['threshold']:.2f}")
    axes[1].axvline(t_star, color="black", label=f"chosen T* = {t_star:.2f}")
    axes[1].set_xlabel("threshold T (confidence < T → predicted anomalous)")
    axes[1].set_ylabel("Youden's J = TPR - FPR")
    axes[1].set_title("P0.7 v3 — Youden's J vs threshold (Method C)")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("P0.7 v3  Confidence-threshold-based anomaly mask")
    print("=" * 72)

    cols = ["session_date", "run_file", "fh7000_timestamp", "x_m", "y_m",
            "speed_mps", "nns_position_confidence"]
    jp = load_joint(cols)
    jp["fh7000_timestamp"] = pd.to_datetime(jp["fh7000_timestamp"])
    n_total = len(jp)
    rng = np.random.default_rng(C.SEED)

    conf = jp["nns_position_confidence"].to_numpy()
    nan_per_session = {sd: int(((jp["session_date"] == sd) &
                                jp["nns_position_confidence"].isna()).sum())
                       for sd in C.SESSIONS}

    # ---- Method A — saddle ------------------------------------------------
    finite_conf = conf[np.isfinite(conf)]
    t_a, centers, counts = _saddle_from_hist(finite_conf)
    print(f"  Method A: saddle = {t_a if t_a is not None else 'unimodal'}")

    # ---- Method B — GMM ---------------------------------------------------
    if len(finite_conf) > 200000:
        gmm_sample = rng.choice(finite_conf, 200000, replace=False)
    else:
        gmm_sample = finite_conf
    gmm = _gmm2_fit(gmm_sample, rng)
    print(f"  Method B: GMM mu={gmm['mu']}, sd={gmm['sd']}, "
          f"pi={gmm['pi']}, T_B={gmm['threshold']:.2f}")

    # ---- Method C — ROC ---------------------------------------------------
    print("  Method C: building disc flag...")
    disc = _build_disc_flag(jp)
    print(f"  Method C: total disc rows = {int(disc.sum())} / {n_total}")
    valid = np.isfinite(conf)
    if disc.sum() > 0:
        idx_valid = np.where(valid)[0]
        if len(idx_valid) > ROC_SUBSAMPLE:
            sub_idx = rng.choice(idx_valid, ROC_SUBSAMPLE, replace=False)
        else:
            sub_idx = idx_valid
        roc = _roc_youden(conf[sub_idx], disc[sub_idx])
    else:
        roc = _roc_youden(np.array([0.0]), np.array([False]))  # null
    print(f"  Method C: T_C = {roc['threshold']}, J = {roc['j']}, "
          f"sens = {roc['sens']}, spec = {roc['spec']}")

    # ---- Decision ---------------------------------------------------------
    t_star, rationale = _choose_threshold(t_a, gmm, roc)
    print(f"  Chosen T* = {t_star:.2f}")
    print(f"  Rationale: {rationale}")

    # ---- Build the mask ---------------------------------------------------
    anomaly = (~np.isfinite(conf)) | (conf < t_star)
    print(f"  v3 anomaly rows: {int(anomaly.sum())} / {n_total} "
          f"({100 * anomaly.mean():.2f}%)")

    # ---- Per-session summary ---------------------------------------------
    v2_path = C.ARTIFACTS_DIR / "anomaly_mask.parquet"
    backup_dir = C.P0_DIR / "v2_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "anomaly_mask_v2.parquet"
    # Prefer the backup, since the canonical path may have been overwritten by
    # an earlier v3 run. The driver also creates this backup before calling us.
    if backup_path.exists():
        v2_mask = pd.read_parquet(backup_path)
    else:
        v2_mask = pd.read_parquet(v2_path)
        shutil.copyfile(v2_path, backup_path)
    v2_flag = v2_mask["anomaly_flag"].to_numpy().astype(bool)

    per_session = {}
    for sd in C.SESSIONS:
        m = (jp["session_date"] == sd).to_numpy()
        n_rows = int(m.sum())
        n_v3 = int((m & anomaly).sum())
        n_v2 = int((m & v2_flag).sum())
        per_session[sd] = {
            "n_rows": n_rows,
            "n_anomaly_v3": n_v3,
            "frac_anomaly_v3": round(n_v3 / max(n_rows, 1), 6),
            "n_anomaly_v2": n_v2,
            "frac_anomaly_v2": round(n_v2 / max(n_rows, 1), 6),
            "delta_pct_pts": round(100 * (n_v3 - n_v2) / max(n_rows, 1), 4),
            "n_nan_confidence": int(nan_per_session[sd]),
        }

    # ---- Validation against v2 detector -----------------------------------
    valid_summary = _validate_v3_against_v2(anomaly, v2_mask, jp)

    # ---- Write artifacts --------------------------------------------------
    # v3 mask (new file with binary anomaly_flag, no anomaly_type)
    mask_v3 = pd.DataFrame({
        "joint_idx": np.arange(n_total, dtype=np.int64),
        "session_date": jp["session_date"].to_numpy(),
        "anomaly_flag": anomaly,
    })
    mask_v3_path = C.ARTIFACTS_DIR / "anomaly_mask_v3.parquet"
    mask_v3.to_parquet(mask_v3_path, index=False)
    print(f"  -> wrote {mask_v3_path}")

    # Overwrite the canonical anomaly_mask.parquet to point to v3
    mask_v3.to_parquet(v2_path, index=False)
    print(f"  -> overwrote {v2_path} with v3 mask")

    # Threshold record
    threshold_payload = {
        "threshold": float(t_star),
        "method_A_saddle": (None if t_a is None else float(t_a)),
        "method_B_gmm": {
            "mu": gmm["mu"], "sd": gmm["sd"], "pi": gmm["pi"],
            "threshold": float(gmm["threshold"]),
            "overlap_ratio": float(gmm["overlap_ratio"]),
        },
        "method_C_youden": {
            "threshold": (None if roc["threshold"] is None else float(roc["threshold"])),
            "j": (None if roc["j"] is None else float(roc["j"])),
            "sensitivity": (None if roc["sens"] is None else float(roc["sens"])),
            "specificity": (None if roc["spec"] is None else float(roc["spec"])),
            "subsample_size": int(min(ROC_SUBSAMPLE, int(np.isfinite(conf).sum()))),
            "disc_definition": {
                "abs_step_m": DISC_ABS_M,
                "expected_step_multiplier": DISC_EXP_MULT,
                "max_dt_s": DISC_MAX_DT_S,
            },
        },
        "decision_rationale": rationale,
        "nan_handling": ("NaN/non-finite confidence treated as below "
                         "threshold (worst-case)."),
    }
    write_json(C.ARTIFACTS_DIR / "anomaly_threshold.json", threshold_payload)

    # Summary cache
    write_json(C.CACHE_DIR / "p0_7_v3.json", {
        "threshold": threshold_payload,
        "per_session": per_session,
        "validation_vs_v2": valid_summary,
    })

    # ---- Figures ----------------------------------------------------------
    _plot_hist(centers, counts, t_a, t_star,
               C.FIGURES_DIR / "p0_7_v3_confidence_hist.png")
    _plot_gmm(finite_conf, gmm, t_star,
              C.FIGURES_DIR / "p0_7_v3_gmm.png")
    _plot_roc(roc, t_star,
              C.FIGURES_DIR / "p0_7_v3_roc.png")

    # Console summary
    print()
    print("  Per-session summary (v3 vs v2):")
    for sd in C.SESSIONS:
        s = per_session[sd]
        print(f"    {sd}: n={s['n_rows']:,}  v3={s['n_anomaly_v3']:,} "
              f"({100*s['frac_anomaly_v3']:.2f}%)  "
              f"v2={s['n_anomaly_v2']:,} ({100*s['frac_anomaly_v2']:.2f}%)  "
              f"delta={s['delta_pct_pts']:+.2f}pp")
    g = valid_summary["global"]
    print(f"  v2 manual_reposition coverage by v3: "
          f"{100 * g['v3_repos_global_coverage']:.2f}% "
          f"({g['v3_repos_rows_caught_total']}/{g['v2_repos_rows_total']})")
    print(f"  v3 extension beyond v2: {g['v3_extension_rows_total']:,} rows")
    print(f"  v3 over-flagging on v2 motor_overheat_post tails: "
          f"{100 * g['v3_overheat_post_global_rate']:.2f}% "
          f"({g['v3_overheat_post_rows_caught_total']}/"
          f"{g['v2_overheat_post_rows_total']})")
    print("P0.7 v3 done.")


if __name__ == "__main__":
    main()
