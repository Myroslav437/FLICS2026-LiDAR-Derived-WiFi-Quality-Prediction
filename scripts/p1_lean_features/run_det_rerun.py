"""Deterministic-split rerun for Project B's lean-vs-locked comparison.

Fixes the non-deterministic-`hash()` validation-split seed used in
`scripts.p1_project_b.folds._random_train_val_split` (Python's hash() of
strings is randomised across invocations unless PYTHONHASHSEED=0). Runs 30
fits — 15 locked-full-8 + 15 lean-B (3 telemetry) — using
`scripts.p1_lean_features.det_folds.build_fold` so both feature stacks share
identical train/val/test rows for every fold × hyperparameter combination.

The 15 configurations per stack:
  - R-{1..5} × W2 (locked hyperparams)
  - R-{1..5} × W4 (locked hyperparams)
  - R-{1..5} × W4 (H1 hyperparams)

Outputs
  models      → scripts/p1_lean_features/models/det/{descriptor}.json
  predictions → scripts/p1_lean_features/cache/det/predictions_{descriptor}.parquet
  metrics     → scripts/p1_lean_features/results/{lean_b_metrics_det,locked_det_metrics,comparison_det}.parquet
  fit log     → scripts/p1_lean_features/results/fit_inventory_det.parquet

Naming: `R-{i}_W{2,4}_{lockedFull,leanB}_{locked,H1}`.

Top-level: `python -m scripts.p1_lean_features.run_det_rerun`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p1_project_a import metrics as a_metrics
from scripts.p1_project_a import training as a_training
from scripts.p1_project_b import config as b_config
from scripts.p1_project_b.run_modeling import prepare_regions

from . import det_folds
from . import feature_lists as fl


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

DET_MODELS_DIR = fl.MODELS_DIR / "det"
DET_CACHE_DIR = fl.CACHE_DIR / "det"
RESULTS_DIR = fl.RESULTS_DIR
LOG_PATH = fl.ANALYSIS_DIR / "det_rerun_log.md"

for _d in (DET_MODELS_DIR, DET_CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Locked-artifact snapshot — verifies that we don't perturb scripts/p1_project_b/
# ---------------------------------------------------------------------------

LOCKED_DIR = fl.PROJECT_ROOT / "scripts" / "p1_project_b"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_locked() -> dict[str, str]:
    out: dict[str, str] = {}
    if not LOCKED_DIR.exists():
        return out
    for p in sorted(LOCKED_DIR.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rel = p.relative_to(fl.PROJECT_ROOT).as_posix()
            out[rel] = _hash_file(p)
    return out


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class FitRow:
    fold: str
    variant: str               # "W2", "W4"
    stack: str                 # "lockedFull" or "leanB"
    hyperparams: str           # "locked" or "H1"
    descriptor: str            # filename token, e.g. "R-1_W2_lockedFull_locked"
    best_iteration: int
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    test_region: int
    wallclock_s: float
    model_sha256: str
    model_path: str
    predictions_path: str


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------


def _fit_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    xgb_params: dict,
) -> tuple[xgb.Booster, int]:
    dtrain = a_training.make_dmatrix(train_df, features)
    dval = a_training.make_dmatrix(val_df, features)
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=b_config.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=b_config.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return booster, best_iter


def _evaluate_to_rows(
    preds: pd.DataFrame,
    *,
    fold: str,
    variant: str,
    stack: str,
    hyperparams: str,
    descriptor: str,
) -> list[dict]:
    rows: list[dict] = []
    for sm in a_metrics.evaluate_predictions(preds):
        rows.append({
            "fold": fold,
            "variant": variant,
            "stack": stack,
            "hyperparams": hyperparams,
            "descriptor": descriptor,
            **sm,
        })
    return rows


def _features_for(stack: str, variant: str) -> list[str]:
    """Locked-full uses scripts.p1_project_b.config.VARIANTS; lean-B uses fl.project_b_variant_features('lean_b')."""
    if stack == "lockedFull":
        return list(b_config.VARIANTS[variant])
    if stack == "leanB":
        return list(fl.project_b_variant_features("lean_b")[variant])
    raise ValueError(f"unknown stack {stack!r}")


def _expected_feat_count(stack: str, variant: str) -> int:
    if stack == "lockedFull":
        return {"W2": 15, "W4": 34}[variant]
    return {"W2": 10, "W4": 29}[variant]


def _run_one_fit(
    fold: det_folds.WithinFoldSplit,
    *,
    variant: str,
    stack: str,
    hyperparams: str,
) -> tuple[FitRow, list[dict]]:
    features = _features_for(stack, variant)
    expected = _expected_feat_count(stack, variant)
    if len(features) != expected:
        raise ValueError(
            f"feature-count mismatch {stack}/{variant}: expected {expected}, got {len(features)}: {features}"
        )

    descriptor = f"{fold.name}_{variant}_{stack}_{hyperparams}"
    xgb_params = b_config.HYPERPARAM_LABELS[hyperparams]
    model_path = DET_MODELS_DIR / f"{descriptor}.json"
    preds_path = DET_CACHE_DIR / f"predictions_{descriptor}.parquet"

    if model_path.exists() and preds_path.exists():
        print(f"  reuse {descriptor}  ({len(features)} feat, {hyperparams})")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
        preds = pd.read_parquet(preds_path)
        wall = 0.0
    else:
        print(f"  fit {descriptor}  ({len(features)} feat, {hyperparams})")
        t0 = time.time()
        booster, best_iter = _fit_xgb(fold.train, fold.val, features, xgb_params)
        y_pred = a_training.predict(booster, fold.test, features, best_iter)
        preds = a_training.build_predictions_frame(fold.test, features, y_pred)
        if "region_id" in fold.test.columns:
            preds["region_id"] = fold.test["region_id"].to_numpy()
        booster.save_model(str(model_path))
        preds.to_parquet(preds_path, index=False)
        wall = float(time.time() - t0)
        print(f"    best_iter={best_iter} wall={wall:.1f}s")

    fit_row = FitRow(
        fold=fold.name,
        variant=variant,
        stack=stack,
        hyperparams=hyperparams,
        descriptor=descriptor,
        best_iteration=best_iter,
        n_features=len(features),
        n_train=int(len(fold.train)),
        n_val=int(len(fold.val)),
        n_test=int(len(fold.test)),
        test_region=int(fold.test_region),
        wallclock_s=wall,
        model_sha256=a_training.model_sha256(model_path),
        model_path=str(model_path.relative_to(fl.PROJECT_ROOT)),
        predictions_path=str(preds_path.relative_to(fl.PROJECT_ROOT)),
    )
    metric_rows = _evaluate_to_rows(
        preds,
        fold=fold.name,
        variant=variant,
        stack=stack,
        hyperparams=hyperparams,
        descriptor=descriptor,
    )
    return fit_row, metric_rows


# ---------------------------------------------------------------------------
# Comparison builder (locked-full vs lean-B on identical det splits)
# ---------------------------------------------------------------------------


def build_comparison(metrics_df: pd.DataFrame, locked_orig_df: pd.DataFrame) -> pd.DataFrame:
    """Three-way table: RMSE(locked, orig), RMSE(locked-full, det), RMSE(lean-B, det)."""

    keys = [
        ("R-1", "W2", "locked"),
        ("R-2", "W2", "locked"),
        ("R-3", "W2", "locked"),
        ("R-4", "W2", "locked"),
        ("R-5", "W2", "locked"),
        ("R-1", "W4", "locked"),
        ("R-2", "W4", "locked"),
        ("R-3", "W4", "locked"),
        ("R-4", "W4", "locked"),
        ("R-5", "W4", "locked"),
        ("R-1", "W4", "H1"),
        ("R-2", "W4", "H1"),
        ("R-3", "W4", "H1"),
        ("R-4", "W4", "H1"),
        ("R-5", "W4", "H1"),
    ]
    strata = ["overall", "in_fov", "out_of_fov"]

    rows: list[dict] = []
    for fold, variant, hp in keys:
        for stratum in strata:
            sub_lf = metrics_df[
                (metrics_df["fold"] == fold)
                & (metrics_df["variant"] == variant)
                & (metrics_df["hyperparams"] == hp)
                & (metrics_df["stack"] == "lockedFull")
                & (metrics_df["stratum"] == stratum)
            ]
            sub_lb = metrics_df[
                (metrics_df["fold"] == fold)
                & (metrics_df["variant"] == variant)
                & (metrics_df["hyperparams"] == hp)
                & (metrics_df["stack"] == "leanB")
                & (metrics_df["stratum"] == stratum)
            ]
            if hp == "locked":
                orig = locked_orig_df[
                    (locked_orig_df["fold"] == fold)
                    & (locked_orig_df["variant"] == variant)
                    & (locked_orig_df["stratum"] == stratum)
                ]
                rmse_orig = float(orig.iloc[0]["rmse"]) if not orig.empty else float("nan")
                n_orig = int(orig.iloc[0]["n"]) if not orig.empty else -1
            else:
                # H1: read from robustness_metrics.parquet (W4 H1 only)
                rob = pd.read_parquet(b_config.RESULTS_DIR / "robustness_metrics.parquet")
                sub = rob[
                    (rob["fold"] == fold)
                    & (rob["variant"] == variant)
                    & (rob["check"] == "hyperparams")
                    & (rob["hyperparams"] == "H1")
                    & (rob["stratum"] == stratum)
                ]
                rmse_orig = float(sub.iloc[0]["rmse"]) if not sub.empty else float("nan")
                n_orig = int(sub.iloc[0]["n"]) if not sub.empty else -1

            rmse_lf = float(sub_lf.iloc[0]["rmse"]) if not sub_lf.empty else float("nan")
            rmse_lb = float(sub_lb.iloc[0]["rmse"]) if not sub_lb.empty else float("nan")
            n_det = int(sub_lf.iloc[0]["n"]) if not sub_lf.empty else -1

            rows.append({
                "fold": fold,
                "variant": variant,
                "hyperparams": hp,
                "stratum": stratum,
                "n": n_det,
                "rmse_locked_orig": rmse_orig,
                "n_locked_orig": n_orig,
                "rmse_lockedFull_det": rmse_lf,
                "rmse_leanB_det": rmse_lb,
                "delta_leanB_minus_lockedFull": rmse_lb - rmse_lf,
                "delta_lockedFull_det_minus_locked_orig": rmse_lf - rmse_orig,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

ROBUST_THRESHOLD = 0.3
MOSTLY_ROBUST_THRESHOLD = 0.5

# Headline rows for verdict assessment (Project B paper-cited).
# F-B is project-A-only (not in this rerun); R-4_W2_H1 lives in the R-4 robustness
# section which uses the same hash-driven split as the H1-no-buffer R-4 W2 fold,
# so its analogue here would be R-4 W4 H1 (same fold, since W2 H1 uses build_fold(R-4)
# = identical val rows). The within-section headline is R-4 W4 locked / R-4 W4 H1.
HEADLINE_KEYS: list[tuple[str, str, str, str]] = [
    ("R-4", "W4", "locked", "in_fov"),
    ("R-4", "W4", "H1", "in_fov"),
]


def assess_verdict(comp_df: pd.DataFrame) -> dict:
    """Apply the §3 thresholds: ROBUST ≤ 0.3 dB, MOSTLY ROBUST ≤ 0.5 dB if headlines ≤ 0.3 dB, else SHIFTED."""
    deltas = comp_df["delta_leanB_minus_lockedFull"].to_numpy()
    deltas = deltas[~np.isnan(deltas)]
    max_abs = float(np.max(np.abs(deltas))) if deltas.size else float("nan")

    headline_mask = pd.Series(False, index=comp_df.index)
    for fold, variant, hp, stratum in HEADLINE_KEYS:
        m = (
            (comp_df["fold"] == fold)
            & (comp_df["variant"] == variant)
            & (comp_df["hyperparams"] == hp)
            & (comp_df["stratum"] == stratum)
        )
        headline_mask = headline_mask | m
    headline_deltas = comp_df.loc[headline_mask, "delta_leanB_minus_lockedFull"].to_numpy()
    headline_deltas = headline_deltas[~np.isnan(headline_deltas)]
    max_abs_headline = float(np.max(np.abs(headline_deltas))) if headline_deltas.size else float("nan")

    if np.isnan(max_abs):
        verdict = "INDETERMINATE"
    elif max_abs <= ROBUST_THRESHOLD:
        verdict = "ROBUST"
    elif max_abs <= MOSTLY_ROBUST_THRESHOLD and (
        np.isnan(max_abs_headline) or max_abs_headline <= ROBUST_THRESHOLD
    ):
        verdict = "MOSTLY ROBUST"
    else:
        verdict = "SHIFTED"

    return {
        "verdict": verdict,
        "max_abs_delta": max_abs,
        "max_abs_delta_headline": max_abs_headline,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    # --- Smoke test: deterministic hash returns identical values ---
    print("[det.rerun] determinism smoke test")
    expected_offset_R1 = det_folds.deterministic_fold_offset("R-1")
    for _ in range(5):
        assert det_folds.deterministic_fold_offset("R-1") == expected_offset_R1
    print(f"  deterministic_fold_offset('R-1') = {expected_offset_R1} (5 calls, identical)")

    # --- Pre-snapshot locked artifacts ---
    print("[det.rerun] pre-snapshotting scripts/p1_project_b/")
    pre_snap = _snapshot_locked()
    print(f"  snapshotted {len(pre_snap)} files")

    # --- Load data + regions ---
    print("[det.rerun] loading regions on 15.03 non-anomaly rows")
    non_anom_with_region, _assn = prepare_regions()
    print(f"  rows: {len(non_anom_with_region):,}")

    # --- 30 fits: 5 folds × {W2 locked, W4 locked, W4 H1} × {lockedFull, leanB} ---
    fits: list[FitRow] = []
    metric_rows: list[dict] = []

    for fold_name in b_config.FOLD_NAMES:
        print(f"[det.rerun] === {fold_name} ===")
        fold = det_folds.build_fold(non_anom_with_region, fold_name)
        print(
            f"  test region={fold.test_region}: train n={len(fold.train):,}, "
            f"val n={len(fold.val):,}, test n={len(fold.test):,}"
        )

        for variant, hp in [("W2", "locked"), ("W4", "locked"), ("W4", "H1")]:
            for stack in ["lockedFull", "leanB"]:
                fit_row, rows = _run_one_fit(
                    fold,
                    variant=variant,
                    stack=stack,
                    hyperparams=hp,
                )
                fits.append(fit_row)
                metric_rows.extend(rows)

    metrics_df = pd.DataFrame(metric_rows)

    # --- Persist metric splits ---
    locked_full_metrics = metrics_df[metrics_df["stack"] == "lockedFull"].copy()
    lean_b_metrics = metrics_df[metrics_df["stack"] == "leanB"].copy()
    locked_full_path = RESULTS_DIR / "locked_det_metrics.parquet"
    lean_b_path = RESULTS_DIR / "lean_b_metrics_det.parquet"
    locked_full_metrics.to_parquet(locked_full_path, index=False)
    lean_b_metrics.to_parquet(lean_b_path, index=False)
    print(f"[det.rerun] wrote {locked_full_path} ({len(locked_full_metrics)} rows)")
    print(f"[det.rerun] wrote {lean_b_path} ({len(lean_b_metrics)} rows)")

    # --- Comparison: read original locked WLRO + robustness for orig-RMSE column ---
    locked_orig_df = pd.read_parquet(b_config.RESULTS_DIR / "wlro_metrics.parquet")
    comp_df = build_comparison(metrics_df, locked_orig_df)
    comp_path = RESULTS_DIR / "comparison_det.parquet"
    comp_df.to_parquet(comp_path, index=False)
    print(f"[det.rerun] wrote {comp_path} ({len(comp_df)} rows)")

    asmt = assess_verdict(comp_df)
    print(
        f"[det.rerun] verdict={asmt['verdict']}  "
        f"max|delta|={asmt['max_abs_delta']:.3f} dB  "
        f"max|delta|_headline={asmt['max_abs_delta_headline']:.3f} dB"
    )

    # --- Fit inventory ---
    fits_df = pd.DataFrame([asdict(f) for f in fits])
    fits_path = RESULTS_DIR / "fit_inventory_det.parquet"
    fits_df.to_parquet(fits_path, index=False)

    # --- Post-snapshot + audit log ---
    print("[det.rerun] post-snapshotting scripts/p1_project_b/")
    post_snap = _snapshot_locked()
    keys_pre = set(pre_snap.keys())
    keys_post = set(post_snap.keys())
    added = sorted(keys_post - keys_pre)
    removed = sorted(keys_pre - keys_post)
    changed = sorted(k for k in keys_pre & keys_post if pre_snap[k] != post_snap[k])

    write_audit_log(
        pre_snap=pre_snap,
        post_snap=post_snap,
        added=added,
        removed=removed,
        changed=changed,
        smoke_offset=expected_offset_R1,
        n_fits=len(fits),
        verdict=asmt,
    )
    print(
        f"[det.rerun] locked artifacts: added={len(added)}, removed={len(removed)}, "
        f"changed={len(changed)}"
    )
    print(f"[det.rerun] all done in {time.time() - t0:.1f}s — {len(fits)} fits")


def write_audit_log(
    *,
    pre_snap: dict[str, str],
    post_snap: dict[str, str],
    added: list[str],
    removed: list[str],
    changed: list[str],
    smoke_offset: int,
    n_fits: int,
    verdict: dict,
) -> None:
    md: list[str] = []
    md.append("# Deterministic-split rerun — audit log\n")
    md.append(
        "Snapshots `scripts/p1_project_b/` before and after the deterministic rerun "
        "(`python -m scripts.p1_lean_features.run_det_rerun`). The locked artifacts "
        "are byte-frozen audit-trail files; this rerun must not modify any of them.\n"
    )
    md.append("## 1. Determinism smoke test\n")
    md.append(
        "Five sequential calls to `det_folds.deterministic_fold_offset('R-1')` "
        f"return `{smoke_offset}` every time. The wrapper uses `hashlib.sha256` "
        "instead of Python's built-in `hash()`, so the offset is reproducible across "
        "Python invocations, machines, and PYTHONHASHSEED settings.\n"
    )
    md.append("## 2. Locked-artifact non-modification\n")
    md.append(f"- Files snapshotted before: {len(pre_snap)}")
    md.append(f"- Files snapshotted after:  {len(post_snap)}")
    md.append(f"- Files added:              {len(added)}")
    md.append(f"- Files removed:            {len(removed)}")
    md.append(f"- Files with changed SHA-256: {len(changed)}")
    md.append("")
    if not added and not removed and not changed:
        md.append(
            "**Result: PASS** — every file under `scripts/p1_project_b/` is byte-identical "
            "before and after the rerun.\n"
        )
    else:
        md.append("**Result: FAIL** — see lists below.\n")
        if changed:
            md.append("### Changed files\n")
            for k in changed:
                md.append(f"- `{k}`")
                md.append(f"  - before: `{pre_snap[k][:16]}…`")
                md.append(f"  - after:  `{post_snap[k][:16]}…`")
            md.append("")
        if added:
            md.append("### Added files\n")
            for k in added:
                md.append(f"- `{k}` (new SHA-256: `{post_snap[k][:16]}…`)")
            md.append("")
        if removed:
            md.append("### Removed files\n")
            for k in removed:
                md.append(f"- `{k}` (was: `{pre_snap[k][:16]}…`)")
            md.append("")
    md.append("## 3. Rerun summary\n")
    md.append(f"- Total fits: {n_fits} (15 lockedFull + 15 leanB)")
    md.append(f"- Verdict: **{verdict['verdict']}**")
    md.append(f"- max |Δ_lean-B − lockedFull| (deterministic): **{verdict['max_abs_delta']:.3f} dB**")
    md.append(
        f"- max |Δ| on headline rows (R-4 W4 in_fov, locked + H1): "
        f"**{verdict['max_abs_delta_headline']:.3f} dB**"
    )
    md.append("")
    md.append("## 4. Snapshot scope\n")
    md.append(f"- `{LOCKED_DIR.relative_to(fl.PROJECT_ROOT).as_posix()}/` (recursive, excluding `__pycache__`)")
    md.append("")
    LOG_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"[det.rerun] audit log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
