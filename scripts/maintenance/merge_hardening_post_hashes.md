# Post-move SHA-256 verification

Compares pre-move hash (from `merge_hardening_pre_hashes.md`) against post-move hash at the destination.

Source files (.py) are EXPECTED to differ in SHA after Stage 1.3 import-path updates.

Model / cache / results / docs files MUST match exactly; mismatches there are critical failures.

| Pre-path | Post-path | Pre SHA | Post SHA | Status |
|---|---|---|---|---|
| `docs/p1_hardening/figures/dataset_noise_floor.png` | `docs/p1_dataset_analysis/figures/dataset_noise_floor.png` | `cc86b41de7f4cfd718ca15d632b110a52c6b978c310c4db3b5a6c68fc9099ae8` | `cc86b41de7f4cfd718ca15d632b110a52c6b978c310c4db3b5a6c68fc9099ae8` | MATCH |
| `docs/p1_hardening/results_report.md` | (dissolved) | `b793db695c574aefa8515287a178f78c18c037e88364593dc06f51bb5e82d97d` | — | DISSOLVED |
| `docs/p1_hardening/tables/dataset_noise_floor.md` | `docs/p1_dataset_analysis/tables/dataset_noise_floor.md` | `0007617551a6f1c3bbd4ca61cd85f3f540251fdce9e2f7043016da7c92bb89ba` | `0007617551a6f1c3bbd4ca61cd85f3f540251fdce9e2f7043016da7c92bb89ba` | MATCH |
| `scripts/p1_hardening/__init__.py` | (dissolved) | `69f0395b630b8b59a98aaa5c0a785537b5b3531b50436cb5291beb48c4809fdd` | — | DISSOLVED |
| `scripts/p1_hardening/build_results_report.py` | (dissolved) | `28cc27464f319478d93a40593ea27ede701c89cdf4c58bc3419570da4cf3dc30` | — | DISSOLVED |
| `scripts/p1_hardening/cache/predictions_A_B5_placebo_H1.parquet` | `scripts/p1_project_a/cache/predictions_A_B5_placebo_H1.parquet` | `af688a7e685923ba3716e4efc6ab6510c9b521e361dfae4f7a000a25914ed465` | `af688a7e685923ba3716e4efc6ab6510c9b521e361dfae4f7a000a25914ed465` | MATCH |
| `scripts/p1_hardening/cache/predictions_A_B5_placebo_locked.parquet` | `scripts/p1_project_a/cache/predictions_A_B5_placebo_locked.parquet` | `fcadeb8576da76d7b8ebf423017022ff6a7d36c2ba6476856836937b5bcad909` | `fcadeb8576da76d7b8ebf423017022ff6a7d36c2ba6476856836937b5bcad909` | MATCH |
| `scripts/p1_hardening/cache/predictions_B_B1_lgb_H1_equiv.parquet` | `scripts/p1_project_a/cache/predictions_B_B1_lgb_H1_equiv.parquet` | `ccd8f717b2e5627c2e041ed15688b6291dc8008f05936d35be5762619e7506b1` | `ccd8f717b2e5627c2e041ed15688b6291dc8008f05936d35be5762619e7506b1` | MATCH |
| `scripts/p1_hardening/cache/predictions_B_B1_lgb_default.parquet` | `scripts/p1_project_a/cache/predictions_B_B1_lgb_default.parquet` | `e1dd8b8dd434be86f7723bffab018fda0e0f9ea3ab80a7e972f0c156e700a983` | `e1dd8b8dd434be86f7723bffab018fda0e0f9ea3ab80a7e972f0c156e700a983` | MATCH |
| `scripts/p1_hardening/cache/predictions_B_B5_lgb_H1_equiv.parquet` | `scripts/p1_project_a/cache/predictions_B_B5_lgb_H1_equiv.parquet` | `3d7d82066ba90ec14087447d1cd933a0a577a8329f72d36d2abd3e62a483d96b` | `3d7d82066ba90ec14087447d1cd933a0a577a8329f72d36d2abd3e62a483d96b` | MATCH |
| `scripts/p1_hardening/cache/predictions_B_B5_lgb_default.parquet` | `scripts/p1_project_a/cache/predictions_B_B5_lgb_default.parquet` | `ab49521b0d72163b8e55324b429c52f5cc895432edc9b41170f4ec79d14e0bb9` | `ab49521b0d72163b8e55324b429c52f5cc895432edc9b41170f4ec79d14e0bb9` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W0.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W0.parquet` | `0548297f3773714f9da50efaaf20bca9fd3ade6a3778c08de1ed9faa511f794b` | `0548297f3773714f9da50efaaf20bca9fd3ade6a3778c08de1ed9faa511f794b` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W1.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W1.parquet` | `1ad4574a33517b1b2fe93f53edf99f9de19f22d4e6a84425e8c3fcd7c9e83e60` | `1ad4574a33517b1b2fe93f53edf99f9de19f22d4e6a84425e8c3fcd7c9e83e60` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W2.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W2.parquet` | `90097b9e1f862ab114451d887797bfdff3fa1a71c3f100f69a5fcd975180fb86` | `90097b9e1f862ab114451d887797bfdff3fa1a71c3f100f69a5fcd975180fb86` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W3.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W3.parquet` | `942aa7ef702ae1b5e110af0def08c1465b4ca40603907363d989a5e5e6c52d1f` | `942aa7ef702ae1b5e110af0def08c1465b4ca40603907363d989a5e5e6c52d1f` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W4.parquet` | `5bfff562988a53664251c1afb1e2ad457c631ccb5261f81d766f4983a5b2e6d9` | `5bfff562988a53664251c1afb1e2ad457c631ccb5261f81d766f4983a5b2e6d9` | MATCH |
| `scripts/p1_hardening/cache/predictions_C_R-4_buffer_H1_W4pp.parquet` | `scripts/p1_project_b/cache/predictions_C_R-4_buffer_H1_W4pp.parquet` | `3375a716360c0c534e92110521966788b39c3588ec5c83a04efb7e8061142541` | `3375a716360c0c534e92110521966788b39c3588ec5c83a04efb7e8061142541` | MATCH |
| `scripts/p1_hardening/cache/predictions_E_R-4_W2_real_locked.parquet` | `scripts/p1_project_b/cache/predictions_E_R-4_W2_real_locked.parquet` | `59cd2af170571ffa1b5350de39b7810763836e29f0faf484ffc918dfc8c40be8` | `59cd2af170571ffa1b5350de39b7810763836e29f0faf484ffc918dfc8c40be8` | MATCH |
| `scripts/p1_hardening/cache/predictions_E_R-4_W4_placebo_locked.parquet` | `scripts/p1_project_b/cache/predictions_E_R-4_W4_placebo_locked.parquet` | `09d2a3f0593f80b059628da97fc64e92a875a0801cd8a7a9ebf0af0ebe974c86` | `09d2a3f0593f80b059628da97fc64e92a875a0801cd8a7a9ebf0af0ebe974c86` | MATCH |
| `scripts/p1_hardening/cache/predictions_E_R-4_W4_real_locked.parquet` | `scripts/p1_project_b/cache/predictions_E_R-4_W4_real_locked.parquet` | `719622cddc64a7822d6641ac07ac81a006b15cfe1a8bc862725c041938f4abce` | `719622cddc64a7822d6641ac07ac81a006b15cfe1a8bc862725c041938f4abce` | MATCH |
| `scripts/p1_hardening/config.py` | (dissolved) | `85c236cc022a22d2d44687b0aa69d4eeca3f8bb6916c59d58d67eebb08799fce` | — | DISSOLVED |
| `scripts/p1_hardening/lightgbm_training.py` | `scripts/p1_project_a/lightgbm_training.py` | `c70e5be6816df15a11d627818fcd57127dcab14de70ee7777c85f84a755c743a` | `6a92729c026f580840109986eebef0ae4c999191b4be3f1b7d35035bdbb7138a` | EDITED-EXPECTED |
| `scripts/p1_hardening/models/A_B5_placebo_H1.json` | `scripts/p1_project_a/models/A_B5_placebo_H1.json` | `c4f1e7f7c0f8c2c27e5e198e6ca8a1a06a0e898943b8df6d6c38eafc08175ec2` | `c4f1e7f7c0f8c2c27e5e198e6ca8a1a06a0e898943b8df6d6c38eafc08175ec2` | MATCH |
| `scripts/p1_hardening/models/A_B5_placebo_locked.json` | `scripts/p1_project_a/models/A_B5_placebo_locked.json` | `dade19df5f1e23d40e1ae9f36b80814f98b82c620dfa5b43150166a4f837adad` | `dade19df5f1e23d40e1ae9f36b80814f98b82c620dfa5b43150166a4f837adad` | MATCH |
| `scripts/p1_hardening/models/B_B1_lgb_H1_equiv.txt` | `scripts/p1_project_a/models/B_B1_lgb_H1_equiv.txt` | `526f093c05832450d1291a2df714d65d573e0ea02c48206cec560b64fd597a9a` | `526f093c05832450d1291a2df714d65d573e0ea02c48206cec560b64fd597a9a` | MATCH |
| `scripts/p1_hardening/models/B_B1_lgb_default.txt` | `scripts/p1_project_a/models/B_B1_lgb_default.txt` | `39f72d43fed9e3bf55036147ed3c59653fd949f5e616e1fe9c4e0ddf7fd96288` | `39f72d43fed9e3bf55036147ed3c59653fd949f5e616e1fe9c4e0ddf7fd96288` | MATCH |
| `scripts/p1_hardening/models/B_B5_lgb_H1_equiv.txt` | `scripts/p1_project_a/models/B_B5_lgb_H1_equiv.txt` | `28276de3967af2b91401ba05c2af3e4c837f30108f2c1f1127866da4c7805414` | `28276de3967af2b91401ba05c2af3e4c837f30108f2c1f1127866da4c7805414` | MATCH |
| `scripts/p1_hardening/models/B_B5_lgb_default.txt` | `scripts/p1_project_a/models/B_B5_lgb_default.txt` | `5d7bff046d58de3265762502a92281cea161baff7ee3ca7cb50d55e786382549` | `5d7bff046d58de3265762502a92281cea161baff7ee3ca7cb50d55e786382549` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W0.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W0.json` | `ea239acbdcfc3adcc40f90f88e53b0643c490b4cb02fa93e7e0eb0a31398f542` | `ea239acbdcfc3adcc40f90f88e53b0643c490b4cb02fa93e7e0eb0a31398f542` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W1.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W1.json` | `7497344ea17c54ecc9e09363e7349063977b8372bbebd4530f0791f1b7211f10` | `7497344ea17c54ecc9e09363e7349063977b8372bbebd4530f0791f1b7211f10` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W2.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W2.json` | `ecf292ad7c4f9659dbe57fc5fd5abdbafdb8db0da230ef345cba47dc9bf09d0c` | `ecf292ad7c4f9659dbe57fc5fd5abdbafdb8db0da230ef345cba47dc9bf09d0c` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W3.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W3.json` | `3827f34bccc398b1e65f0438835acd1bb73d37bea40a20a8795a149844769ce8` | `3827f34bccc398b1e65f0438835acd1bb73d37bea40a20a8795a149844769ce8` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W4.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W4.json` | `d91c0479a54ca1a1fa51cb79e6fda68558ae40b728558c3facc54900a5d49098` | `d91c0479a54ca1a1fa51cb79e6fda68558ae40b728558c3facc54900a5d49098` | MATCH |
| `scripts/p1_hardening/models/C_R-4_buffer_H1_W4pp.json` | `scripts/p1_project_b/models/C_R-4_buffer_H1_W4pp.json` | `dec53224a148f563cb7f643cc4f4a9d233f103cd71979617f8d32bb82ae83b33` | `dec53224a148f563cb7f643cc4f4a9d233f103cd71979617f8d32bb82ae83b33` | MATCH |
| `scripts/p1_hardening/models/E_R-4_W2_real_locked.json` | `scripts/p1_project_b/models/E_R-4_W2_real_locked.json` | `c32c5db1afa12bd819d33fd224f81ac6759133fe3aa1a0f19547008f45c1f9e0` | `c32c5db1afa12bd819d33fd224f81ac6759133fe3aa1a0f19547008f45c1f9e0` | MATCH |
| `scripts/p1_hardening/models/E_R-4_W4_placebo_locked.json` | `scripts/p1_project_b/models/E_R-4_W4_placebo_locked.json` | `670b02dfed6d8277f5dbe7b3dba9ccefcab58ac1f2c2d38c17f27edea3177a27` | `670b02dfed6d8277f5dbe7b3dba9ccefcab58ac1f2c2d38c17f27edea3177a27` | MATCH |
| `scripts/p1_hardening/models/E_R-4_W4_real_locked.json` | `scripts/p1_project_b/models/E_R-4_W4_real_locked.json` | `89148526e3737b5383bae9881f865e7d864246aa8250209b53ac875435a72c68` | `89148526e3737b5383bae9881f865e7d864246aa8250209b53ac875435a72c68` | MATCH |
| `scripts/p1_hardening/placebo.py` | `scripts/p1_project_a/placebo.py` | `1db8b7e221569506f2ead0c0e6dfdfbb6560ba075f6efb09557ef2c625f46c45` | `1600937958cbf52659d94da33bd0da7bdee78aa5c164167942241cefa01dd6cf` | EDITED-EXPECTED |
| `scripts/p1_hardening/results/experiment_a_integrity.parquet` | `scripts/p1_project_a/results/hardening_a_integrity.parquet` | `3f9efeab7c270e4d3eb342a3c5fd5b319bbc03a966aa52145b19d96da79a0526` | `3f9efeab7c270e4d3eb342a3c5fd5b319bbc03a966aa52145b19d96da79a0526` | MATCH |
| `scripts/p1_hardening/results/experiment_a_metrics.parquet` | `scripts/p1_project_a/results/hardening_a_metrics.parquet` | `d8c0182d0e13fae3be4fbea470c3a486721483172520bc5c3b3567a24130c26b` | `d8c0182d0e13fae3be4fbea470c3a486721483172520bc5c3b3567a24130c26b` | MATCH |
| `scripts/p1_hardening/results/experiment_b_metrics.parquet` | `scripts/p1_project_a/results/hardening_b_metrics.parquet` | `46503d993bb2c6641a69fe78b420ad0333f9f397ee1c0d31f45e566d6d2d0b27` | `46503d993bb2c6641a69fe78b420ad0333f9f397ee1c0d31f45e566d6d2d0b27` | MATCH |
| `scripts/p1_hardening/results/experiment_c_metrics.parquet` | `scripts/p1_project_b/results/hardening_c_metrics.parquet` | `1fdc3fca01b2a95b7a49acfc13d8a8c9ba70e2c478cb9b518303b5c54402cf3f` | `1fdc3fca01b2a95b7a49acfc13d8a8c9ba70e2c478cb9b518303b5c54402cf3f` | MATCH |
| `scripts/p1_hardening/results/experiment_d_per_cell_sigma.parquet` | `scripts/p1_dataset_analysis/results/noise_floor_per_cell_sigma.parquet` | `7b483ba3978e157e266147264e548ccfa926bb112e14a63ad737c5defacfe67a` | `7b483ba3978e157e266147264e548ccfa926bb112e14a63ad737c5defacfe67a` | MATCH |
| `scripts/p1_hardening/results/experiment_d_same_cell_deltas.parquet` | `scripts/p1_dataset_analysis/results/noise_floor_same_cell_deltas.parquet` | `9c9595d7e8b3746cd3badd594223d9b529d47da98145e0a9b8f736e39b12225b` | `9c9595d7e8b3746cd3badd594223d9b529d47da98145e0a9b8f736e39b12225b` | MATCH |
| `scripts/p1_hardening/results/experiment_d_summary.json` | `scripts/p1_dataset_analysis/results/noise_floor_summary.json` | `1a8f8124c26260caf0fa626f74922eb3c915f3b70e7515670fb7e4044392c35e` | `1a8f8124c26260caf0fa626f74922eb3c915f3b70e7515670fb7e4044392c35e` | MATCH |
| `scripts/p1_hardening/results/experiment_e_integrity.parquet` | `scripts/p1_project_b/results/hardening_e_integrity.parquet` | `1df44a259ee3d8ca2b12239e3c8264bf183fc4f26621ab3a0c55168d2a91c67f` | `1df44a259ee3d8ca2b12239e3c8264bf183fc4f26621ab3a0c55168d2a91c67f` | MATCH |
| `scripts/p1_hardening/results/experiment_e_metrics.parquet` | `scripts/p1_project_b/results/hardening_e_metrics.parquet` | `adc12fc1fff092cd5720139b2cebe1266be9454114860c0e7e1a876256061ff7` | `adc12fc1fff092cd5720139b2cebe1266be9454114860c0e7e1a876256061ff7` | MATCH |
| `scripts/p1_hardening/results/fit_inventory.parquet` | (split) | `c498a7711d10161d6fef411c2c0954608209922cdf58b39a0be596be1d4b4ec9` | — | SPLIT |
| `scripts/p1_hardening/results/hardening_metrics.parquet` | (split) | `8b2a3bb5259ef69ef0d1507db90b760890442d6db62b01c4bf2f72709f186b4d` | — | SPLIT |
| `scripts/p1_hardening/run_a_cross_session_placebo.py` | `scripts/p1_project_a/run_hardening_placebo.py` | `e95cbefd230f44ba3abe84c07067b0b96731637242aada4eff0d27c82f195335` | `92996617cbaea23dbe45f7b8de9d8dd26134f98fb79e7f72868ae15cfcbac382` | EDITED-EXPECTED |
| `scripts/p1_hardening/run_all.py` | (dissolved) | `df9374cafc85a687cabc34850bd81f508845795dacab231df361edb41d7446ee` | — | DISSOLVED |
| `scripts/p1_hardening/run_b_lightgbm.py` | `scripts/p1_project_a/run_hardening_lightgbm.py` | `54fa8d6ea1d54005c80bf1758afccc31928b1d488dc1de5623827bd53eb04d0a` | `b35fdab5815ce301783ec1f732883f88351bc1999cb8d39239e05119ef6930ae` | EDITED-EXPECTED |
| `scripts/p1_hardening/run_c_r4_combined.py` | `scripts/p1_project_b/run_hardening_combined.py` | `fc8684e5f3597c18d5cbaa29c774ac44a6e7fcfd18a1c6a751dc2eb474782d12` | `9cc4914680ee7511a3b4e681d1b23d84888d063d98e104c87ff7993f97117363` | EDITED-EXPECTED |
| `scripts/p1_hardening/run_d_dataset_noise_floor.py` | `scripts/p1_dataset_analysis/run_noise_floor.py` | `43b2d851bcdab1b7ec0041ee6fe5630d6a821a0088992c8ec9211bfb4e58b68e` | `71cf4c169f831553ffefdb52147635f7573b964970b7dc917a9576052df7e476` | EDITED-EXPECTED |
| `scripts/p1_hardening/run_e_within_session_placebo.py` | `scripts/p1_project_b/run_hardening_placebo.py` | `ccd461e0d082dd94b30d5551c26691146f1035ee2a997ea2cca619ffcaaee64c` | `d1bc9524c2d4b6d7f45c50f59fd11c23badf4acd4ecc0670514a993868ae8f43` | EDITED-EXPECTED |

## Summary

- MATCH (byte-stable file unchanged): 41
- EDITED-EXPECTED (.py source updated in Stage 1.3): 7
- MISMATCH (byte-stable file changed — CRITICAL): 0
- SPLIT: 2 (expected for hardening_metrics.parquet, fit_inventory.parquet)
- DISSOLVED: 5 (expected for __init__.py, config.py, build_results_report.py, run_all.py, docs results_report.md)
- MISSING/UNMAPPED: 0
