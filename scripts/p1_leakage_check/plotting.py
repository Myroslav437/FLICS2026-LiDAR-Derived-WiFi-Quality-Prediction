"""Plotting helpers for the leakage investigation.

All plots use matplotlib only (seaborn is not a dependency in this env).
KDE overlays use scipy.stats.gaussian_kde.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


SESSION_COLOURS = {
    "25.02.2026": "#1f77b4",
    "15.03.2026": "#ff7f0e",
    "24.03.2026": "#2ca02c",
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _kde_or_none(values: np.ndarray) -> gaussian_kde | None:
    if len(values) < 2 or np.allclose(values.std(), 0.0):
        return None
    try:
        return gaussian_kde(values)
    except Exception:  # noqa: BLE001
        return None


def plot_distribution_per_session(
    values: np.ndarray,
    feature: str,
    session: str,
    dataset_min: float,
    dataset_max: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    n_unique = int(np.unique(values).shape[0])
    cur_min, cur_max = float(values.min()), float(values.max())
    cur_med = float(np.median(values))
    bins = min(80, max(20, n_unique))
    ax.hist(values, bins=bins, density=True, alpha=0.55, color="#4c72b0", edgecolor="white")
    kde = _kde_or_none(values)
    if kde is not None:
        xs = np.linspace(cur_min, cur_max, 400)
        ax.plot(xs, kde(xs), color="#cc4f4f", lw=1.6, label="KDE")
    for v, lbl in [(cur_min, "min"), (cur_med, "median"), (cur_max, "max")]:
        ax.axvline(v, color="#444", lw=1.0, ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.95, f" {lbl}", rotation=90, va="top", fontsize=8)
    annotation = (
        f"unique = {n_unique:,}\n"
        f"dataset min/max = {dataset_min:.4g} / {dataset_max:.4g}\n"
        f"session min/max = {cur_min:.4g} / {cur_max:.4g}\n"
        f"n = {len(values):,}"
    )
    ax.text(
        0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
    )
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.set_title(f"{feature} distribution — {session}")
    if kde is not None:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_distribution_all_sessions(
    per_session_values: Mapping[str, np.ndarray],
    feature: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_min = min(float(v.min()) for v in per_session_values.values())
    all_max = max(float(v.max()) for v in per_session_values.values())
    xs = np.linspace(all_min, all_max, 500)
    for sess, vals in per_session_values.items():
        colour = SESSION_COLOURS.get(sess, "k")
        ax.hist(
            vals, bins=60, density=True, alpha=0.30, color=colour,
            label=f"{sess} (n={len(vals):,})", range=(all_min, all_max),
        )
        kde = _kde_or_none(vals)
        if kde is not None:
            ax.plot(xs, kde(xs), color=colour, lw=1.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.set_title(f"{feature} distribution — all sessions overlay")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _subsample(values: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if n <= target:
        return np.arange(n), values
    idx = np.linspace(0, n - 1, target).astype(int)
    return idx, values[idx]


def plot_timeseries(
    feature_values: np.ndarray,
    speed_values: np.ndarray,
    feature: str,
    session: str,
    out_path: Path,
    target: int = 10000,
) -> None:
    idx, fv = _subsample(feature_values, target)
    sv = speed_values[idx]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(idx, fv, color="#1f77b4", lw=0.7)
    ax.set_xlabel("row index within session")
    ax.set_ylabel(feature, color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    s_min, s_max = float(speed_values.min()), float(speed_values.max())
    if s_max > s_min:
        sv_norm = (sv - s_min) / (s_max - s_min)
    else:
        sv_norm = np.zeros_like(sv)
    ax2.plot(idx, sv_norm, color="#bbbbbb", lw=0.5, alpha=0.7)
    ax2.set_ylabel("speed_mps (rescaled to [0, 1])", color="#888")
    ax2.tick_params(axis="y", labelcolor="#888")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title(f"{feature} time-series — {session}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_acf(
    per_session_curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    feature: str,
    out_path: Path,
) -> None:
    """per_session_curves: {session: (lags_array, acf_values_array)}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for sess, (lags, vals) in per_session_curves.items():
        colour = SESSION_COLOURS.get(sess, "k")
        ax.plot(lags, vals, marker="o", lw=1.4, color=colour, label=sess)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--")
    ax.axhline(0.1, color="grey", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("lag (rows)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"ACF — {feature}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_correlation_heatmap(
    matrix: pd.DataFrame, title: str, out_path: Path, cmap: str = "RdBu_r"
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    masked = matrix.copy()
    for i in range(min(masked.shape)):
        masked.iloc[i, i] = np.nan
    im = ax.imshow(masked.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(masked.shape[1]))
    ax.set_yticks(np.arange(masked.shape[0]))
    ax.set_xticklabels(masked.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(masked.index, fontsize=9)
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            v = masked.iloc[i, j]
            if pd.isna(v):
                continue
            colour = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=colour)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spatial(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    feature: str,
    session: str,
    out_path: Path,
    discrete: bool = False,
    target: int = 20000,
) -> None:
    n = len(x)
    if n > target:
        idx = np.linspace(0, n - 1, target).astype(int)
        x, y, c = x[idx], y[idx], c[idx]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    if discrete:
        unique = np.unique(c)
        cmap = plt.get_cmap("tab10")
        for i, val in enumerate(unique):
            m = c == val
            ax.scatter(
                x[m], y[m], c=[cmap(i % 10)], s=2, alpha=0.6,
                label=f"{feature}={val:g}",
            )
        ax.legend(markerscale=3, fontsize=8)
    else:
        sc = ax.scatter(x, y, c=c, s=2, cmap="viridis", alpha=0.7)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(feature)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    ax.set_title(f"{feature} spatial map — {session}  (n_plot={len(x):,})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
