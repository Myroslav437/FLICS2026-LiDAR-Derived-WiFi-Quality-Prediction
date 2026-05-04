# F-C focused diagnostic — audit log

This log accompanies `docs/p1_project_a/fc_diagnostic.md`. It records the
end-to-end run, the locked-artefact integrity check, and the per-fit
inventory in a form that makes the diagnostic reproducible from this
directory alone.

## 1. Operational summary

- **Entry point**: `python -m scripts.p1_project_a.run_fc_diagnostic`
- **Seed**: `20260427` (matches the leakage-fixed Project A rerun).
- **Dataset**: `data/phase1/dataset.parquet`
  (SHA-256 prefix `c164c53b5dd27238…`, unmodified).
- **Total new fits**: 13 (9 XGBoost + 4 LightGBM).
- **Wall-clock end-to-end**: 42.3 s on a single CPU.
- **All fits succeeded**: yes.

## 2. New artefacts (under `fc_diagnostic/` subdirectories)

```
scripts/p1_project_a/
├── models/fc_diagnostic/
│   ├── F-C_B1_H1.json
│   ├── F-C_B5_H1.json
│   ├── F-C_B1_H2.json
│   ├── F-C_B5_H2.json
│   ├── F-C_B1_H3.json
│   ├── F-C_B5_H3.json
│   ├── F-C_B5p_H1.json
│   ├── F-C_B5_placebo_locked.json
│   ├── F-C_B5_placebo_H1.json
│   ├── F-C_B1_lgb_default.txt
│   ├── F-C_B5_lgb_default.txt
│   ├── F-C_B1_lgb_H1_equiv.txt
│   └── F-C_B5_lgb_H1_equiv.txt
├── cache/fc_diagnostic/
│   └── … 13 corresponding predictions_F-C_*.parquet …
└── results/
    ├── fc_diagnostic_metrics.parquet     (39 rows: 13 fits × 3 strata)
    └── fc_diagnostic_fit_inventory.parquet
```

The metrics parquet additionally carries 12 reference rows from the locked
F-C run (B1, B5, B5', B5'' × 3 strata) tagged `diagnostic = "ref"` for
cross-check convenience.

## 3. Per-fit SHA-256 inventory

| # | Diag | Variant | Descriptor | Framework | Config | best_iter | wall (s) | model SHA-256 |
|---:|---|---|---|---|---|---:|---:|---|
| 1 | A | B1 | B1_H1 | xgboost | H1 | 90 | 0.29 | `abd080efc57d9b09fb20e267f00b0bbc0f6b43bf0151617bff7e034d23e16b7a` |
| 2 | A | B5 | B5_H1 | xgboost | H1 | 93 | 0.57 | `b46886df5b4bd998c6365cf2b5adbf057a0d45b704252872108abafb5c3f7c8d` |
| 3 | A | B1 | B1_H2 | xgboost | H2 | 229 | 0.57 | `d9b472e0960d83662751b6ec834f5841e1d8f4ed1631685372ce5ea562c7962a` |
| 4 | A | B5 | B5_H2 | xgboost | H2 | 184 | 1.14 | `ef41dfe8ceb80924225b38e0ef33a1973b10e00b083624043001d3d0c36518b3` |
| 5 | A | B1 | B1_H3 | xgboost | H3 | 46 | 0.40 | `34779ffffeef379c21c015489931949dbabb67b3207f3c36a0dfe455676e341c` |
| 6 | A | B5 | B5_H3 | xgboost | H3 | 35 | 0.97 | `de5c89b415c07470eaa3f4131393e53359e777d0e0bb0527cd3c90c720615114` |
| 7 | A | B5p | B5p_H1 | xgboost | H1 | 162 | 0.48 | `db5cc3b44b040b3d8543f6f3cf40c8eaab060e6f5f80e42f9ed8e9c0c36159b3` |
| 8 | B | B5 | B5_placebo_locked | xgboost | locked | 57 | 0.74 | `cd60a3b251f9fc3b775a893b958a5377e5f25677a30fc26dfb996d3b06a63f88` |
| 9 | B | B5 | B5_placebo_H1 | xgboost | H1 | 86 | 0.59 | `9d7aae78e8003fa7a63e0f0e66e2c814c32805b3cfbfd82d199d2ac189039f98` |
| 10 | C | B1 | B1_lgb_default | lightgbm | default | 43 | 0.33 | `67d005d58d8bdc284278a2a0aa110cb281e7a97404e6019e0c3caf52ef3894a2` |
| 11 | C | B5 | B5_lgb_default | lightgbm | default | 47 | 0.52 | `8096edefc732e109cf91d14a1b6b2f5a90ab8f1bf1f413303bee7fe21ad181b3` |
| 12 | C | B1 | B1_lgb_H1_equiv | lightgbm | H1_equiv | 68 | 0.29 | `b32073c269bcb1578d9121028830baf43a3911885a2c6020ed9f010eb401edf4` |
| 13 | C | B5 | B5_lgb_H1_equiv | lightgbm | H1_equiv | 60 | 0.44 | `f03de380f9283fcd1d32584b2e82e2475bcfa981a21cf1d6f7133c78a367e218` |

## 4. F-C fold structure (sanity)

- **Train sessions**: 15.03.2026 (29 865 rows after F-C cadence-decimation by 7),
  24.03.2026 (20 367 rows). Total train pool = 50 232 rows.
- **Validation**: chronological-last-10% per training session = 5 581 rows.
- **Test session**: 25.02.2026 = 216 472 rows (in-FOV n = 121 831).

## 5. Diagnostic B integrity (LiDAR placebo)

- ρ(mean_dist_mm, signal_power) before per-session block-shuffle: **0.0451**
- ρ(mean_dist_mm, signal_power) after  per-session block-shuffle: **−0.0016**
- max |Δ mean| across the 19 LiDAR columns post-shuffle: **0** (block shuffle preserves marginals exactly)

The shuffle behaves as designed: marginal distributions are preserved
identically while row-aligned correlation with the target is broken.

## 6. Locked-artefact integrity check

A pre/post SHA-256 comparison was performed over:
- `scripts/p1_project_a/models/`           (excluding `fc_diagnostic/`)
- `scripts/p1_project_a/cache/`            (excluding `fc_diagnostic/`)
- `scripts/p1_project_a/results/`          (excluding the new `fc_diagnostic_*.parquet`)
- `scripts/p1_project_a/diagnostic/`       (Hardening artefacts)
- `scripts/p1_project_b/`                  (entire Project B tree)

Procedure:
```
python scripts/p1_project_a/fc_baseline_snapshot.py        # before run
python -m scripts.p1_project_a.run_fc_diagnostic            # run
python -m scripts.p1_project_a.fc_integrity_check          # after run
```

Result:
```
baseline files: 263
current files:  263
added:    0
removed:  0
modified: 0

[OK] locked artefact integrity check passed — 0 changes
```

The baseline manifest is preserved at
`scripts/p1_project_a/fc_diagnostic_baseline.json` for forensic re-checking.
The check helper is `scripts/p1_project_a/fc_integrity_check.py`.

## 7. Combined verdict

| Diagnostic | Result |
|---|---|
| A — Hyperparameter robustness | **GREEN** (all 4 H-configs clear +0.5 dB Δ_LiDAR) |
| B — Cross-session LiDAR placebo | **MARGINAL_DISTRIBUTION_DRIVEN** (locked Δ_real − Δ_placebo = +0.46 dB; Δ_placebo = +0.74 dB) |
| C — LightGBM cross-check | **FRAMEWORK_ROBUST** (both LGB configs Δ_LiDAR ≥ +0.5 dB) |
| **Combined** | **PARTIALLY_ROBUST** |

See `docs/p1_project_a/fc_diagnostic.md` §0 (TL;DR), §3 (Diagnostic B
narrative and F-B sanity-check comparison) and §6 (downstream-work
implications) for the diagnostic interpretation.
