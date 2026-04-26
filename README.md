# FLICS-2026 — LiDAR-Derived Wi-Fi Quality Prediction

Data preparation and analysis pipeline for the FLICS-2026 study on predicting
Wi-Fi link quality from LiDAR-derived environmental features collected on a
mobile AGV platform.

## Repository layout

```
data/                Raw + merged datasets (tracked via Git LFS).
  15.03.2026/        Per-session raw captures (LiDAR .h5 + telemetry CSVs).
  24.03.2026/
  25.02.2026/
  merged/            Time-aligned, cleaned outputs of the merge pipeline.

scripts/             Analysis and data-preparation code.
  merge_dataset.py             Build data/merged/{lidar.h5,telemetry.parquet}.
  clean_telemetry.py           Post-merge telemetry cleanup.
  verify_merge.py              Lightweight merge sanity checks.
  verify_merged_dataset.py     End-to-end verification against raw sources.
  verify_session_endpoints.py  Per-session start/end checks.
  extract_docs.py              Extract .docx/.pdf docs to plain text.
  initial_analysis/            Descriptive analysis of joint_coverage.parquet.
  time_sync/                   Per-day clock-offset calibration pipeline.

docs/                Generated reports and the paper draft.
  initial_dataset_analysis/    report.md / report.pdf + figures.
  time_sync/                   report.md / report.pdf + figures.
  paper/                       LaTeX manuscript.
  obsolete/                    Historical reference docs.

artifacts/           Local-only intermediate outputs (gitignored).
```

## Setup

The project targets **Python 3.11+** (developed and tested on 3.14).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. (Optional) editable install via pyproject.toml
pip install -e .
```

## Git LFS

Large binary dataset files (`.h5`, `.csv`, `.parquet`, `.npz`, `.ntm`, `.ntt`
under `data/` and generated `.pdf` under `docs/`) are tracked through Git LFS.
Before cloning or pulling, make sure LFS is installed:

```bash
git lfs install
git clone <repo-url>
# or, in an existing checkout:
git lfs pull
```

## Running the pipeline

The analysis pipeline runs in this order:

```bash
# 1. Merge raw captures into data/merged/
python -m scripts.merge_dataset

# 2. Verify the merge
python -m scripts.verify_merge
python -m scripts.verify_merged_dataset
python -m scripts.verify_session_endpoints

# 3. Clean telemetry
python -m scripts.clean_telemetry

# 4. Time-sync calibration (per-day clock offsets)
python -m scripts.time_sync.run_signatures
python -m scripts.time_sync.run_offsets
python -m scripts.time_sync.run_homogeneity
python -m scripts.time_sync.run_drift
python -m scripts.time_sync.run_apply
python -m scripts.time_sync.validate_joint
python -m scripts.time_sync.run_report

# 5. Initial descriptive analysis on the corrected joint dataset
python -m scripts.initial_analysis.run_initial_analysis
```

Generated figures and reports are written to `docs/` and committed to the
repo. Cache files written under `scripts/**/cache/` and `artifacts/` are
local-only.
