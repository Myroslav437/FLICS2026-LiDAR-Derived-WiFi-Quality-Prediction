"""DEPRECATED. The p1_hardening package was merged into p1_project_a and p1_project_b
on 2026-04-28.

Stage 1 of the merge moved hardening artifacts to their natural homes:
- Experiment A (cross-session placebo on F-B):     scripts/p1_project_a/
- Experiment B (LightGBM cross-check on F-B):      scripts/p1_project_a/
- Experiment C (combined H1+buffer on R-4):        scripts/p1_project_b/
- Experiment D (dataset noise floor):              docs/p1_dataset_analysis/
- Experiment E (within-session placebo on R-4):    scripts/p1_project_b/

For audit trail, see scripts/maintenance/merge_hardening_log.md.

This file kept solely so any old proposal references to scripts/p1_hardening/
resolve to a discoverable explanation, not an ImportError.
"""

raise ImportError(
    "scripts.p1_hardening was merged into p1_project_a and p1_project_b. "
    "See the deprecation notice in scripts/p1_hardening/__init__.py for the new locations."
)
