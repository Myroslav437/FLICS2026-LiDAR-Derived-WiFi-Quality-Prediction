"""Plotting helpers for the battery-channel provenance diagnostic.

All plots write to ``docs/p1_battery_provenance/figures/``. matplotlib is
the only plotting dependency; no seaborn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SESSION_COLOURS = {
    "25.02.2026": "tab:blue",
    "15.03.2026": "tab:orange",
    "24.03.2026": "tab:green",
}


def _kde_curve(values: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian KDE on bare numpy. Returns (x, y) or None if degenerate."""
    v = values[np.isfinite(values)]
    if len(v) < 5:
        return None
    if v.min() == v.max():
        return None
    std = float(v.std(ddof=1))
    if std == 0:
        return None
    bw = 1.06 * std * len(v) ** (-1 / 5)
    if bw == 0:
        return None
    xs = np.linspace(float(v.min()), float(v.max()), n)
    diffs = (xs[:, None] - v[None, :]) / bw
    weights = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
    ys = weights.sum(axis=1) / (len(v) * bw)
    return xs, ys


def histogram(values: np.ndarray, *, channel: str, stage: str,
              fig_path: Path) -> None:
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    n_total = len(v)
    n_nan = int(np.isnan(v).sum())
    nu = int(pd.Series(v).nunique(dropna=True))

    fig, ax = plt.subplots(figsize=(7, 4))
    if nu == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                transform=ax.transAxes)
    else:
        bins = min(60, max(10, nu)) if nu < 200 else 60
        ax.hist(finite, bins=bins, color="tab:blue", alpha=0.7,
                edgecolor="black", linewidth=0.3)
        if nu < 50:
            ax.set_yscale("log")
        kde = _kde_curve(finite)
        if kde is not None:
            xs, ys = kde
            ax2 = ax.twinx()
            ax2.plot(xs, ys, color="tab:red", linewidth=1.2)
            ax2.set_ylabel("density (KDE)", color="tab:red")
            ax2.tick_params(axis="y", colors="tab:red")
    rng = (
        f"[{finite.min():.6g}, {finite.max():.6g}]"
        if finite.size else "n/a"
    )
    ax.set_title(
        f"{channel} — {stage}\n"
        f"nunique={nu}, range={rng}, n={n_total} (NaN={n_nan})"
    )
    ax.set_xlabel(channel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def per_session_overlay(stage_to_df: Mapping[str, pd.DataFrame],
                        *, channel: str, sessions: Iterable[str],
                        fig_path: Path) -> None:
    """Overlay a per-session strip of values across pipeline stages.

    Each stage gets one row of subplots; each row shows one box per session.
    """
    sessions = list(sessions)
    fig, axes = plt.subplots(len(stage_to_df), 1, figsize=(8, 3.0 * len(stage_to_df)),
                              sharex=True)
    if len(stage_to_df) == 1:
        axes = [axes]
    for ax, (stage, df) in zip(axes, stage_to_df.items()):
        for sd in sessions:
            sub = df[df["session_date"] == sd][channel].dropna().to_numpy()
            if sub.size == 0:
                continue
            x = np.full_like(sub, list(sessions).index(sd), dtype=np.float64)
            x = x + (np.random.RandomState(0).rand(len(sub)) - 0.5) * 0.3
            ax.scatter(x, sub, s=2, alpha=0.25,
                       color=SESSION_COLOURS.get(sd, "gray"), label=sd)
        ax.set_title(f"{channel} — {stage}")
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(sessions)
        ax.set_ylabel(channel)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def time_series_simple(df: pd.DataFrame, *, channel: str, session: str,
                       fig_path: Path, ts_col: str = "fh7000_timestamp") -> None:
    sub = df[df["session_date"] == session].sort_values(ts_col)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"no rows for session {session}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(sub[ts_col].to_numpy(), sub[channel].to_numpy(),
                color="tab:blue", linewidth=0.6)
    ax.set_title(f"{channel} — session {session}")
    ax.set_xlabel("fh7000_timestamp")
    ax.set_ylabel(channel)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def time_series_dual(df: pd.DataFrame, *, primary: str, secondary: str,
                     session: str, fig_path: Path,
                     ts_col: str = "fh7000_timestamp") -> None:
    sub = df[df["session_date"] == session].sort_values(ts_col)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"no rows for session {session}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(sub[ts_col].to_numpy(), sub[primary].to_numpy(),
                color="tab:blue", linewidth=0.6, label=primary)
        ax.set_ylabel(primary, color="tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(sub[ts_col].to_numpy(), sub[secondary].to_numpy(),
                 color="tab:red", linewidth=0.6, label=secondary)
        ax2.set_ylabel(secondary, color="tab:red")
        ax2.tick_params(axis="y", colors="tab:red")
    ax.set_title(f"{primary} vs {secondary} — session {session}")
    ax.set_xlabel("fh7000_timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
