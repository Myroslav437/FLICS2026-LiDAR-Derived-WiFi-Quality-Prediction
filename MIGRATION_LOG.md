# Migration log — leakage-fixed feature stack rerun

Generated: 2026-05-02
Trigger: post-hoc identification of MikroTik-router-side telemetry features
in the locked Phase 1 feature stack (target leakage), plus the `battery_value`
sentinel and the `nns_state` within-session-only signal. The locked Phase 1
artefacts (Project A LORO ablation, hyperparameter diagnostic, Project B WLRO,
R-4 robustness diagnostic, Hardening A–E) were therefore invalidated and the
full Phase 1 pipeline re-run with a 3-feature telemetry stack.

This file is the audit trail for the migration. It records what was deleted,
what survived, the rationale, the pre-deletion SHA-256 hashes of every locked
artefact destroyed, and the post-rerun SHA-256 hashes of the new artefacts.

---

## 1. Feature-stack diff

### 1.1 Removed

The post-hoc audit identified five features in the locked Phase 1 telemetry
block that should not have been used as model inputs:

| Removed feature | Reason |
|---|---|
| `load_long`   | MikroTik router CPU 1-minute load (`FH.7000.[mikrotik].load_long`); target leakage from the AP side of the same WiFi link the model is predicting. |
| `load_mid`    | MikroTik router CPU 5-minute load (`FH.7000.[mikrotik].load_mid`); same router-side target leakage. |
| `load_short`  | MikroTik router CPU 15-second load (`FH.7000.[mikrotik].load_short`); same router-side target leakage. |
| `battery_value` | `0xFFFF` CAN-bus sentinel within the LiDAR-coverage window; provably no-op (`docs/p1_battery_provenance/report.md`; lean-features comparison §6 showed 32/32 byte-identical model SHA-256 across the full ablation when removed). |
| `nns_state`   | Discrete 2/3 navigation-system state. Produces a material within-session shift but no cross-session shift; plausibly a within-session spatial-mode fingerprint that does not generalise across deployment scenarios. |

The locked telemetry stack had 8 features; the leakage-fixed stack has 3:

```python
LEAKAGE_FIXED_TELEMETRY = [
    "speed_mps",
    "turn_rate",
    "momentary_current_consumption",
]
```

### 1.2 Variant feature counts after the fix

The variant *definitions* (B0–B5, B5', B5'', W0–W4, W4'') are unchanged in
meaning — they still partition the same feature universe into the same
nested groups. Only the telemetry block shrank, so the variants that include
telemetry shrink correspondingly.

| Variant | Description (existing locked-code structure) | Old count | New count |
|---|---|---:|---:|
| B0  | `dist_to_AP` only                                                          |  1 |  1 |
| B1  | B0 + `sin/cos_angle_to_AP`                                                 |  3 |  3 |
| B2  | B1 + LiDAR scalar                                                          |  8 |  8 |
| B3  | B2 + LiDAR sectoral                                                        | 22 | 22 |
| B4  | B3 + telemetry                                                             | 30 | **25** |
| B5  | B4 + `clutter_frac_toward_AP` + `is_AP_in_FOV`                             | 32 | **27** |
| B5' | telemetry + AP-relative                                                    | 13 | **8**  |
| B5''| telemetry + LiDAR scalar + LiDAR sectoral                                  | 27 | **22** |
| W0  | position (`x_m`, `y_m`)                                                    |  2 |  2 |
| W1  | W0 + telemetry                                                             | 10 | **5**  |
| W2  | W1 + AP-relative                                                           | 15 | **10** |
| W3  | W2 + LiDAR scalar                                                          | 20 | **15** |
| W4  | W3 + LiDAR sectoral                                                        | 34 | **29** |
| W4''| position + telemetry + LiDAR (no AP-relative)                              | 29 | **24** |
| W4' | feature-identical to W2                                                    | 15 | **10** |

Note on the original prompt's table: the prompt-supplied `Old count` for B2/B3
(11, 16) reflected an alternative variant ordering in which telemetry enters
at B2 rather than at B4. The locked code orders variants as B2 = B1 + LiDAR
scalar, B3 = B2 + LiDAR sectoral, B4 = B3 + telemetry. We kept that ordering
(per the prompt's "definitions are unchanged in *meaning*" requirement); the
B4/B5/B5'/B5''/W* rows in the new-count column above match the prompt's
expected new counts exactly.

### 1.3 Deterministic fold construction

Project B's `_random_train_val_split` previously seeded with
`SEED + (hash(fold_name) & 0x7FFFFFFF)`. Python's built-in `hash()` on strings
is randomised across invocations unless `PYTHONHASHSEED=0`, so the locked
val rows were not reproducible across re-runs. As part of the leakage-fixed
rerun the fold-stable seed offset is now derived from `sha256(fold_name)[:4]`
(the same logic that lived in `scripts/p1_lean_features/det_folds.py`). That
det_folds shim is therefore retired and the canonical
`scripts/p1_project_b.folds` is now byte-stable across machines and Python
invocations. The `hash()` use in `scripts/p1_project_a/run_xai.py` (SHAP
subsample seed) has been similarly switched to a SHA-256-based offset.

---

## 2. Rationale

The four MikroTik load channels (`load_long`/`mid`/`short`, named
`FH.7000.[mikrotik].load_*` in the source CSV) are router CPU load averages
read from the receiving end of the same WiFi link whose AGV-side
`signal_power` the model is predicting. They constitute target leakage —
their predictive value comes from being a function of the same downstream
infrastructure as the target, not from a physically deployable signal.

`battery_value` reads `0xFFFF` (the CAN-bus "no value" sentinel) for every
row inside the LiDAR-coverage window in the dataset; the provenance audit in
`docs/p1_battery_provenance/report.md` and the empirical comparison in the
prior lean-features run showed that removing it produces byte-identical model
files on all 32 fits of the locked ablation, confirming it carries zero
information for these models.

`nns_state` is a discrete navigation-system state with two values (2 and 3)
across the dataset. The lean-features comparison report's §3 showed that
removing it produces a small shift in the cross-session Project A ablation
but a material (+3.9 dB) shift on the Project B R-4 W4 in_fov headline
under a clean deterministic-split comparison. Consistent with a
within-session spatial-mode fingerprint — useful when train and test see the
same map/mode mix but not transferable to a new deployment — it was removed
for honest external generalisation.

The new feature stack is therefore the leakage-fixed minimum: position +
3 telemetry features + AP-relative + LiDAR.

---

## 3. What was removed

The pre-deletion SHA-256 hashes of every artefact below are recorded in
`.pre_deletion_shas.txt` at the project root (485 file rows total). The
hashes are reproduced in §6 of this log so the audit trail survives even if
that scratch file is later cleaned up.

### 3.1 Locked Phase 1 model + cache + results trees

```
scripts/p1_project_a/models/*           (34 files, XGBoost JSON + LightGBM TXT)
scripts/p1_project_a/cache/*            (38 prediction parquets)
scripts/p1_project_a/results/*          (loro_metrics, disambig_metrics, rq4_metrics, fit_inventory, hardening_a/b_*, shap_*)
scripts/p1_project_a/diagnostic/*       (8 robustness model files + 8 prediction parquets + 2 inventory parquets)

scripts/p1_project_b/models/*           (61 files)
scripts/p1_project_b/cache/*            (61 prediction parquets)
scripts/p1_project_b/results/*          (wlro_metrics, disambig_metrics, robustness_metrics, fit_inventory, r4_diagnostic_*, hardening_c/e_*, shap_*)
scripts/p1_project_b/artifacts/*        (spatial_regions.parquet — REGENERATED, not deleted, since region assignment depends only on x_m/y_m, not on telemetry)
```

### 3.2 Lean-features comparison artefacts

```
scripts/p1_lean_features/models/*       (output of the locked-vs-lean comparison)
scripts/p1_lean_features/cache/*
scripts/p1_lean_features/results/*
docs/p1_lean_features/comparison_report.md
docs/p1_lean_features/tables/*
```

The diagnostic-infrastructure code under `scripts/p1_lean_features/`
(`__init__.py`, `det_folds.py`, `feature_lists.py`, `run_*.py`) is retained
as the methodological precursor to this rerun — see §4 below.

### 3.3 Stale documentation

```
docs/p1_project_a/results_report.md
docs/p1_project_a/figures/*
docs/p1_project_a/tables/*
docs/p1_project_b/results_report.md
docs/p1_project_b/r4_diagnostic.md
docs/p1_project_b/figures/*
docs/p1_project_b/tables/*
docs/proposal_rev9.md
docs/unified_report.md
docs/unified_report.pdf            (if previously rendered)
docs/unified_report_inventory.md
docs/unified_report_verification.md
```

### 3.4 Hardening shim

`scripts/p1_hardening/` was already a deprecation shim (the experiments had
been merged into Projects A and B on 2026-04-28). The shim package and its
empty `__init__.py`-only `__pycache__` are removed.

---

## 4. What was kept

### 4.1 Phase 0 (locked, untouched)

```
scripts/p0_analysis/                              code + locked artefacts
scripts/p0_analysis/artifacts/*                   anomaly_mask.parquet, agv_body_mask.npz,
                                                  lidar_fov.json, ap_coords.json, feature_extractor.py,
                                                  anomaly_threshold.json, anomaly_mask_v3.parquet
docs/p0_analysis/                                 Phase 0 documentation
docs/initial_dataset_analysis/                    locked exploratory analysis
docs/time_sync/                                   locked time-sync analysis
```

Pre-rerun and post-rerun SHA-256 hashes are in §5 below; both match.

### 4.2 Phase 1 dataset (locked, untouched)

```
data/phase1/dataset.parquet      SHA-256 = c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270
data/merged/*.parquet            upstream merged data
```

The dataset still contains *all* telemetry columns including the leaked ones
as raw data — the leakage fix is at the *feature selection* layer, not the
data layer.

### 4.3 Pre-rerun investigations

```
docs/p1_dataset_analysis/                  Phase 1 dataset construction
docs/p1_battery_provenance/                source document for §1.1's battery_value rationale
docs/p1_leakage_check/                     leakage-investigation report
docs/MERGE_VERIFICATION.md
docs/TELEMETRY_CLEANUP.md
```

### 4.4 lean_features as historical context

```
scripts/p1_lean_features/__init__.py
scripts/p1_lean_features/det_folds.py        retained as historical context;
                                             the SHA-256-based offset was promoted
                                             to scripts/p1_project_b/folds.py
scripts/p1_lean_features/feature_lists.py    docstring updated to note the leakage-fixed
                                             rerun made LEAN_B_TELEMETRY canonical
scripts/p1_lean_features/run_*.py            retained as historical context
scripts/p1_lean_features/det_rerun_log.md
scripts/p1_lean_features/locked_artifacts_unchanged.md
```

These are no longer wired into the active pipeline (`run_leakage_fixed_pipeline`
does not invoke them) but remain importable.

### 4.5 The Phase 0 pipeline scripts

```
scripts/clean_telemetry.py
scripts/merge_dataset.py
scripts/verify_*.py
scripts/initial_analysis/
scripts/time_sync/
scripts/extract_docs.py
scripts/maintenance/
```

---

## 5. Phase 0 + dataset SHA-256 verification

Pre-deletion (recorded 2026-05-02 before any deletions):

```
24c79bcb9767a959d97356c7d9131b2fa94263e53b4aef0a2ee49b7a1812f541  scripts/p0_analysis/artifacts/anomaly_mask.parquet
93c27efc6a4e4c0e37f456a87f87c58a32719841cf5f44bac37af4f5c5d7175f  scripts/p0_analysis/artifacts/agv_body_mask.npz
22a8211da6734dcd8218dd9138007468e4fccfe537d56226e96c8e99afbe004a  scripts/p0_analysis/artifacts/lidar_fov.json
f574f1c69bb468e3eb6eed2dd89b9c85de496db526793f9f3573f330ffdde616  scripts/p0_analysis/artifacts/ap_coords.json
e139ae2e8444d6ab05079b883405b160ca47dbff3b62a7042c5b23604bb2c168  scripts/p0_analysis/artifacts/feature_extractor.py
24c79bcb9767a959d97356c7d9131b2fa94263e53b4aef0a2ee49b7a1812f541  scripts/p0_analysis/artifacts/anomaly_mask_v3.parquet
8fa7874e41ee900ac111accfa571c1a130b8fb974cfc2d81cf0ffc2a0a8e3e83  scripts/p0_analysis/artifacts/anomaly_threshold.json
c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270  data/phase1/dataset.parquet
```

Post-rerun: see §7. The Phase 0 + dataset rows must be byte-identical.

---

## 6. Pre-deletion SHA-256 hashes of the destroyed artefacts

The full per-file table (485 rows) is in `.pre_deletion_shas.txt` at the
project root. The first 32 rows of the Project A models tree are reproduced
here as a representative sample for the audit trail.

```
c4f1e7f7c0f8c2c27e5e198e6ca8a1a06a0e898943b8df6d6c38eafc08175ec2  scripts/p1_project_a/models/A_B5_placebo_H1.json
dade19df5f1e23d40e1ae9f36b80814f98b82c620dfa5b43150166a4f837adad  scripts/p1_project_a/models/A_B5_placebo_locked.json
39f72d43fed9e3bf55036147ed3c59653fd949f5e616e1fe9c4e0ddf7fd96288  scripts/p1_project_a/models/B_B1_lgb_default.txt
526f093c05832450d1291a2df714d65d573e0ea02c48206cec560b64fd597a9a  scripts/p1_project_a/models/B_B1_lgb_H1_equiv.txt
5d7bff046d58de3265762502a92281cea161baff7ee3ca7cb50d55e786382549  scripts/p1_project_a/models/B_B5_lgb_default.txt
28276de3967af2b91401ba05c2af3e4c837f30108f2c1f1127866da4c7805414  scripts/p1_project_a/models/B_B5_lgb_H1_equiv.txt
a78fcbfc87d16beeda64c9f75cb200b54f856513ac04ab4682c23d9053bd2cac  scripts/p1_project_a/models/F-A_B0.json
bc54e940f042cc2dcaffa0fb5f02a2892a765374e90d08002428babd0a44d3be  scripts/p1_project_a/models/F-A_B1.json
5a8a727554c864d3e2212ab58f86b06c5834eee87e97c9579f1f80a925e5cc3a  scripts/p1_project_a/models/F-A_B2.json
2a8f049377f44c06feff21483f4c18faa6030da806fd25e5e801512955a46d08  scripts/p1_project_a/models/F-A_B3.json
edcbb9953aae7dc917b215edfc9214a455b1cbdc17241f4eff0ce85611fdf944  scripts/p1_project_a/models/F-A_B4.json
614f0a3d3b0a21e103274eb6da7fb2e4538fa1e80ac24bc57da64b2c9dd36134  scripts/p1_project_a/models/F-A_B5.json
fda044e2c0a0cf736331934c332f9d06d63f029d611ce49e226682e432980446  scripts/p1_project_a/models/F-A_B5p.json
953af418eae21277dd9659b131d8c1ae86fb78988cac594d5115613eb9d8d684  scripts/p1_project_a/models/F-A_B5pp.json
b013350674bad64dcdece0e8b7a6591e43293576c697f6adbc051f9d3cfdb9fa  scripts/p1_project_a/models/F-B_B0.json
5ab4c5e025db7ccd355588f1bad2663beed3b1266a0dd34ec98424cd4ea50efb  scripts/p1_project_a/models/F-B_B1.json
d5b037aa1eb82051265ea104025ef65b0a138cd41c426312261afc6ee5d66d5a  scripts/p1_project_a/models/F-B_B2.json
ce7743836b6e51277568fb6374cfce318253fbf50246fa58a79878d7df587883  scripts/p1_project_a/models/F-B_B3.json
6b03a726ac6b9790ff740e69c138a72d6670b9396bcb644f85bf7b79c3dd3123  scripts/p1_project_a/models/F-B_B4.json
040cddab5b6db10cce6951afb2582503d376e370a49d7d5c75d8028d3f59358a  scripts/p1_project_a/models/F-B_B5.json
1145034744370a3a69e27688836b14fa673709bf9a24fa1b547401384e227903  scripts/p1_project_a/models/F-B_B5p.json
8df3dd377f6b38157e0f9710b5e9a07d809e1e6d9b3c7d89d0731299590f7470  scripts/p1_project_a/models/F-B_B5pp.json
b05a1af46b5ebfbd48159496878b53b8a878747904955b89d2f4001742936e93  scripts/p1_project_a/models/F-C_B0.json
c2be61c4d51f9da2dfa76b19a77cc17ea21fa0a518732e58af67c43fc51b8100  scripts/p1_project_a/models/F-C_B1.json
44137f12bb67aeb0d2c578a322ae255a4936096a1e850f5cc740347c047a513f  scripts/p1_project_a/models/F-C_B2.json
d0ee20f1e5e72e4b861df45b648a114074d9a0f836402f88b802314914f33de7  scripts/p1_project_a/models/F-C_B3.json
83868dab6364911e9f63e745e3838fffdc593f2fcb92861c14041fc6d70dd8b5  scripts/p1_project_a/models/F-C_B4.json
4db05809c3a29f5314d119b7b7557665bafa2ad3ea5eb853e1e7782d109a2340  scripts/p1_project_a/models/F-C_B5.json
422e5117a19d172bf3070a6e1aa8dc650df22a3d641cedc86d37b7d35b34f63e  scripts/p1_project_a/models/F-C_B5p.json
8030290d08df3927dee81af535f68e41292bc17b51457228ae035526d44f1080  scripts/p1_project_a/models/F-C_B5pp.json
958a5f6527db9db21dcd05cff313df24e00a7dfe4a6fdfabdbf55b4f13b8796d  scripts/p1_project_a/models/rq4_rq4a_full.json
a77d003dfeed8e237ac8808d15f55a16ea06cbe4d98f3ee593f5408c4ea3b2a0  scripts/p1_project_a/models/rq4_rq4a_same_map.json
72226e56b03e2025b22c4452b14bd0d1466944522f3b49f1331936198a8ee08e  scripts/p1_project_a/models/rq4_rq4b_full.json
3a8d0e7e416a5a2f9c5d6f0bfdff8eaf9911bae122f22f702fcf073a820f89db  scripts/p1_project_a/models/rq4_rq4b_same_map.json
```

For the full 485-row pre-deletion table, see `.pre_deletion_shas.txt`.

---

## 7. Post-rerun SHA-256 hashes


Post-rerun inventory: 238 files across the Project A and Project B models, caches, results, diagnostic, and artifacts directories.

Per-source-group fit counts:

- `project_a_modeling`: 28 fits
- `project_a_diagnostic`: 8 fits
- `project_a_hardening`: 6 fits
- `project_b_modeling`: 41 fits
- `project_b_r4_diagnostic`: 11 fits
- `project_b_hardening`: 9 fits

First 32 rows of the post-rerun model SHA-256 inventory:

```
f2196cb5a7d1d313448d786afa226c970fa28eebb38ce463ae648812e9822f37  scripts/p1_project_a/models/A_B5_placebo_H1.json
8b3d9b063c2585d207c09df826d2b8fb07858c84a4e7f7408a19962d9f70fb6a  scripts/p1_project_a/models/A_B5_placebo_locked.json
39f72d43fed9e3bf55036147ed3c59653fd949f5e616e1fe9c4e0ddf7fd96288  scripts/p1_project_a/models/B_B1_lgb_default.txt
526f093c05832450d1291a2df714d65d573e0ea02c48206cec560b64fd597a9a  scripts/p1_project_a/models/B_B1_lgb_H1_equiv.txt
dd91ddd358db7a29c339dbb4602c2f18a3aec5862cb2fcdfb3867bf181d6166f  scripts/p1_project_a/models/B_B5_lgb_default.txt
8fb43c8a1d61e5b6ff2f87485f804a4e9450c400bc037e2f5c206937114617ca  scripts/p1_project_a/models/B_B5_lgb_H1_equiv.txt
a78fcbfc87d16beeda64c9f75cb200b54f856513ac04ab4682c23d9053bd2cac  scripts/p1_project_a/models/F-A_B0.json
bc54e940f042cc2dcaffa0fb5f02a2892a765374e90d08002428babd0a44d3be  scripts/p1_project_a/models/F-A_B1.json
5a8a727554c864d3e2212ab58f86b06c5834eee87e97c9579f1f80a925e5cc3a  scripts/p1_project_a/models/F-A_B2.json
2a8f049377f44c06feff21483f4c18faa6030da806fd25e5e801512955a46d08  scripts/p1_project_a/models/F-A_B3.json
1a9875c49c63a91718bef8d65a30534e63818dcf513737ba93862b5722dd981e  scripts/p1_project_a/models/F-A_B4.json
4b99cae6cb28c00c1d931d328ba9c6b3306ccbd791366139c1793a3894ae4759  scripts/p1_project_a/models/F-A_B5.json
01c5307d7127f068599df9accc33e321d1f0d7a4f249a944b890e20142e32e93  scripts/p1_project_a/models/F-A_B5p.json
aab9a542f63265682aaca9b00e79ad1487449306ae8435ecaff44aa29f016d22  scripts/p1_project_a/models/F-A_B5pp.json
b013350674bad64dcdece0e8b7a6591e43293576c697f6adbc051f9d3cfdb9fa  scripts/p1_project_a/models/F-B_B0.json
5ab4c5e025db7ccd355588f1bad2663beed3b1266a0dd34ec98424cd4ea50efb  scripts/p1_project_a/models/F-B_B1.json
d5b037aa1eb82051265ea104025ef65b0a138cd41c426312261afc6ee5d66d5a  scripts/p1_project_a/models/F-B_B2.json
ce7743836b6e51277568fb6374cfce318253fbf50246fa58a79878d7df587883  scripts/p1_project_a/models/F-B_B3.json
0802fcf537746bd8a3934e51cc436a181478e1bf80e408655d71d0994f1b06a2  scripts/p1_project_a/models/F-B_B4.json
fe1d4aab00e3d70ca566714afcd8d68e3e4910b78e32398cab637c127d3e32d5  scripts/p1_project_a/models/F-B_B5.json
0b70a024afeec9fb424986c537f7451ce8c95107d9db1e049b4d24f2de3102fd  scripts/p1_project_a/models/F-B_B5p.json
274037f520bdb85fe24107b39cc4ee273d8057ff8443905199f7a4a42d8ff97e  scripts/p1_project_a/models/F-B_B5pp.json
b05a1af46b5ebfbd48159496878b53b8a878747904955b89d2f4001742936e93  scripts/p1_project_a/models/F-C_B0.json
c2be61c4d51f9da2dfa76b19a77cc17ea21fa0a518732e58af67c43fc51b8100  scripts/p1_project_a/models/F-C_B1.json
44137f12bb67aeb0d2c578a322ae255a4936096a1e850f5cc740347c047a513f  scripts/p1_project_a/models/F-C_B2.json
d0ee20f1e5e72e4b861df45b648a114074d9a0f836402f88b802314914f33de7  scripts/p1_project_a/models/F-C_B3.json
5809ba6369cc2b2bf95eeb39486e5650fda725081c0fb7e912bc7acff33b939c  scripts/p1_project_a/models/F-C_B4.json
94e726cbf698433e2da8470dbbf66f9e2001278f9c2a09bc08c92600c44c1d95  scripts/p1_project_a/models/F-C_B5.json
0eb45881e38630cf61f6abc9ca75d8779adec3dcd2c856ec85415a1411369084  scripts/p1_project_a/models/F-C_B5p.json
c454e01461dc757113ae468e80cba93054d92f804e4b3bcf47aceca0d26dc2bb  scripts/p1_project_a/models/F-C_B5pp.json
32132e5139801ccc3f90d2afcef5aeacb5264ef92029c9f69465b0f00ac9b284  scripts/p1_project_a/models/rq4_rq4a_full.json
8e14d93c8ae3be1f5197c2a9963c2502a43c8ce6f66f5b7fbe02aaf33d972e58  scripts/p1_project_a/models/rq4_rq4a_same_map.json
```

Total post-rerun model files: 103.
Full per-file inventory persisted at `scripts/run_leakage_fixed_pipeline_inventory.parquet`.

## 8. Post-rerun verification


Stage-5 verification log per the leakage-fixed rerun prompt §6.

- **(1) Phase 0 + dataset SHAs unchanged**: PASS.
- **(2) Dataset SHA still `c164c53b5dd27238...`**: PASS (got `c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270`).
- **(3) All fits tagged `feature_stack_version = "leakage_fixed_v1"`**: PASS (103 fits; counts = {'leakage_fixed_v1': 103}).
- **(4) No legacy feature names (`load_long`/`load_mid`/`load_short`/`battery_value`/`nns_state`) in active code paths**: PASS.
- **(5) All new reports mention 'leakage-fixed' in TL;DR**: PASS.

**Overall: PASS**.

