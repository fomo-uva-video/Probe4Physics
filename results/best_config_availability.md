# Best Config Availability Audit

Generated: 2026-08-05 on the current server, after validating `/gpfs/home2/spunzo1/recovered_best_configs.json`.

Updated: 2026-08-20 to add the recovered local MVP LTX-2B attentive LR-matrix artifact.

Updated: 2026-08-21 to add the recovered local MVP main V-JEPA 2 attentive LR-matrix artifact from job 25836554.

Updated: 2026-08-21 to add the recovered local MVP main V-JEPA 2.1 attentive LR-matrix artifact from job 25836555.

Updated: 2026-08-21 to add the recovered local MVP main V-JEPA attentive LR-matrix artifact from job 25836556.

Scope: current primary result sheets in `results/intphys_primary_db.xlsx` and `results/mvp_primary_db.xlsx`. Control artifacts and random-label-control runs were excluded from primary availability, except where explicitly noted as excluded evidence.

Definitions:

- `YES_RECOVERED`: full best-row rerun config is in a validated local source (`recovered_best_configs.json` or an explicit `train_eval_summary.json` artifact), matches the current Excel row on all six metrics when metrics are available, and passes current search-space sanity checks.
- `YES_LR_ONLY`: the current Excel has the attentive LR-grid winner (`Selected LR`) and selected layer, but the recovered file does not contain full run hparams/provenance for that row.
- `NO`: the current best row exists in Excel, but no recoverable search-winning config was found on this server.

## Validation Of Recovered File

| Workbook | Recovered SHA256 | Current SHA256 | Same |
| --- | --- | --- | --- |
| intphys_primary_db.xlsx | bd32fe29c3daaf8213c629e2e29c4b5d83c48aad5422e46ca17b7c55163c026a | bd32fe29c3daaf8213c629e2e29c4b5d83c48aad5422e46ca17b7c55163c026a | yes |
| mvp_primary_db.xlsx | 00bce22046099bdf9f46597f78cda79d368806090de27be96525927a83316faa | 33bb1d05abd27c44c4d1f5ce4e4b7e35750be292b9481d9a2ce8496f0da65bee | no |

Recovered configurations checked: 205. Current row/metric/search-space matches: 205. Failed checks: 0.
Recovered execution entries: 410; configs with exactly two seed executions: 205/205; seed counts: {0: 205, 1: 205}.
Referenced git blobs available locally: 29/29.

Additional local artifact recovered on 2026-08-20: `artifacts/probes/mvp/mvp_probe_temporal_attn_ltx_video_ltx_2b_0_9_8_distilled_lr_matrix/train_eval_summary.json` for MVP LTX-2B attentive.

Additional local artifact recovered on 2026-08-21: `artifacts/probes/mvp/mvp_probe_temporal_attn_jepa_v2_vitg_384_config_recovery_seed42_lr_matrix` / `artifacts/results/mvp_probe_temporal_attn_jepa_v2_vitg_384_config_recovery_seed42_lr_matrix` for MVP main V-JEPA 2 attentive.

Additional local artifact recovered on 2026-08-21: `artifacts/probes/mvp/mvp_probe_temporal_attn_jepa_v2_1_vitG_384_config_recovery_seed42_lr_matrix` / `artifacts/results/mvp_probe_temporal_attn_jepa_v2_1_vitG_384_config_recovery_seed42_lr_matrix` for MVP main V-JEPA 2.1 attentive.

Additional local artifact recovered on 2026-08-21: `artifacts/probes/mvp/mvp_probe_temporal_attn_jepa_v1_vith16_384_config_recovery_seed42_lr_matrix` / `artifacts/results/mvp_probe_temporal_attn_jepa_v1_vith16_384_config_recovery_seed42_lr_matrix` for MVP main V-JEPA attentive.

Interpretation: the IntPhys workbook hash matches exactly. The MVP workbook hash differs, but every recovered MVP config still matches a current row on all six metrics, so I counted those rows as current. The legacy `recovered_best_configs.json` attentive configs are only MVP `VideoMAE`/`VideoMAE-v2`; additional local LR-matrix artifacts listed above now provide MVP main V-JEPA, V-JEPA 2, V-JEPA 2.1, and MVP LTX-2B attentive configs. There are no IntPhys attentive configs in the recovered JSON file.

## Best-Row Availability Summary

| Dataset | Probe | YES_RECOVERED | YES_LR_ONLY | NO | Total |
| --- | --- | --- | --- | --- | --- |
| IntPhys2 | Linear | 6 | 0 | 9 | 15 |
| IntPhys2 | MLP | 6 | 0 | 9 | 15 |
| IntPhys2 | Attentive | 0 | 15 | 0 | 15 |
| MVP | Linear | 5 | 0 | 10 | 15 |
| MVP | MLP | 5 | 0 | 10 | 15 |
| MVP | Attentive | 6 | 0 | 9 | 15 |

Total current experiment/model/backbone/probe combinations: 90. Full recovered configs: 28. LR-only attentive winners: 15. Missing: 47.

Main conclusions:

- The recovered file is useful and passes the checks above. It upgrades many previously missing linear configs.
- Full recovered best configs are now available for all main linear and MLP rows on both IntPhys2 and MVP.
- Full recovered best configs are also available for IntPhys2 LTX-13B linear and MLP best rows.
- Full recovered attentive configs are available for MVP main `V-JEPA`, `V-JEPA 2`, `V-JEPA 2.1`, `VideoMAE`/`VideoMAE-v2`, and MVP LTX-2B; the V-JEPA, V-JEPA 2, V-JEPA 2.1, and LTX-2B rows were recovered from local LR-matrix artifacts.
- IntPhys2 attentive remains LR-only from the Excel `Selected LR` column; the recovered file does not prove the fixed training recipe for those rows. Do not use the recovered file to infer 30-vs-90 epochs for IntPhys2 attentive.
- Still missing: same-L and backbone-sweep linear/MLP configs; IntPhys2 LTX-2B linear/MLP; MVP LTX-2B linear/MLP best rows; MVP LTX-13B attentive; and most non-main MVP attentive rows.

## Full Recovered Best Configs

| Dataset | Experiment | Model | Backbone | Probe | Best Row | Recoverable Config | Config ID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | ltx | LTX-Video | LTX-13B | Linear | noise_0.2_block_24 | lr=0.0082968; weight_decay=1.70746e-08; epochs=100; batch_size=64 | intphys2__ltx_video__linear__slot_34 |
| IntPhys2 | ltx | LTX-Video | LTX-13B | MLP | noise_0.2_block_24 | lr=0.00205973; weight_decay=2.78143e-08; dropout=0.385484; hidden_dims=[256]; epochs=1000; batch_size=128 | intphys2__ltx_video__mlp__slot_34 |
| MVP | ltx | LTX-Video | LTX-13B | Linear | noise_0.6_block_24 | lr=0.000763711; weight_decay=0.00178306; epochs=2000; batch_size=64 | mvp__ltx__ltxv_13b_0_9_8_distilled__linear__slot_18 |
| MVP | ltx | LTX-Video | LTX-13B | MLP | noise_0.6_block_24 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | mvp__ltx__ltxv_13b_0_9_8_distilled__mlp__slot_18 |
| IntPhys2 | main | V-JEPA | ViT-H/16 | Linear | Layer 0.75 | lr=0.00221648; weight_decay=5.99176e-05; epochs=100; batch_size=64 | intphys2__jepa_v1__linear__layer_24 |
| IntPhys2 | main | V-JEPA | ViT-H/16 | MLP | Layer 0.75 | lr=0.000461635; weight_decay=0.000555638; dropout=0.155019; hidden_dims=[1024]; epochs=100; batch_size=32 | intphys2__jepa_v1__mlp__layer_24 |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | Linear | Final layer | lr=0.0088989; weight_decay=9.30939e-06; epochs=100; batch_size=64 | intphys2__jepa_v2__linear__layer_40 |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | MLP | Final layer | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | intphys2__jepa_v2__mlp__layer_40 |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | Layer 0.75 | lr=0.0085899; weight_decay=6.91774e-06; epochs=100; batch_size=256 | intphys2__jepa_v2_1__linear__layer_38 |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | Layer 0.75 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | intphys2__jepa_v2_1__mlp__layer_38 |
| IntPhys2 | main | VideoMAE | ViT-H/16 | Linear | Layer 0.5 | lr=0.000132929; weight_decay=0.00506158; epochs=2000; batch_size=32 | intphys2__videomae__linear__layer_16 |
| IntPhys2 | main | VideoMAE | ViT-H/16 | MLP | Layer 0.75 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | intphys2__videomae__mlp__layer_24 |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | Linear | Layer 0.5 | lr=0.0093763; weight_decay=3.02674e-05; epochs=2000; batch_size=64 | intphys2__videomae_v2__linear__layer_20 |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | MLP | Layer 0.5 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | intphys2__videomae_v2__mlp__layer_20 |
| MVP | ltx | LTX-Video | LTX-2B | Attentive | 29 | layer=29; lr=1e-5; epochs=30; patience=5; batch_size=1; eval_batch_size=1; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2; feature_view=tokens | mvp__ltx__ltx_video__ltx_2b__temporal_attn__layer_29 |
| MVP | main | V-JEPA | ViT-H/16 | Attentive | Layer 0.75 | layer=24; lr=1e-5; epochs=30; patience=5; batch_size=2; eval_batch_size=2; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2; feature_view=tokens | mvp__jepa_v1__temporal_attn__layer_24 |
| MVP | main | V-JEPA | ViT-H/16 | Linear | Final layer | lr=0.00801335; weight_decay=1.08638e-08; epochs=100; batch_size=64 | mvp__jepa_v1__linear__layer_32 |
| MVP | main | V-JEPA | ViT-H/16 | MLP | Final layer | lr=0.000650625; weight_decay=5.82706e-07; dropout=0.175017; hidden_dims=[1024, 512, 256]; epochs=2000; batch_size=64 | mvp__jepa_v1__mlp__layer_32 |
| MVP | main | V-JEPA 2 | ViT-G/16 | Attentive | Final layer | layer=40; lr=1e-5; epochs=30; patience=5; batch_size=1; eval_batch_size=1; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2; feature_view=tokens | mvp__jepa_v2__temporal_attn__layer_40 |
| MVP | main | V-JEPA 2 | ViT-G/16 | Linear | Final layer | lr=0.00253859; weight_decay=0.000330633; epochs=100; batch_size=64 | mvp__jepa_v2__linear__layer_40 |
| MVP | main | V-JEPA 2 | ViT-G/16 | MLP | Final layer | lr=0.000591918; weight_decay=0.000166609; dropout=0.155019; hidden_dims=[1024, 512, 256]; epochs=2000; batch_size=64 | mvp__jepa_v2__mlp__layer_40 |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | Attentive | Layer 0.75 | layer=38; lr=5e-5; epochs=30; patience=5; batch_size=2; eval_batch_size=2; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2; feature_view=tokens | mvp__jepa_v2_1__temporal_attn__layer_38 |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | Final layer | lr=0.00500366; weight_decay=2.26317e-06; epochs=100; batch_size=64 | mvp__jepa_v2_1__linear__layer_48 |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | Layer 0.75 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | mvp__jepa_v2_1__mlp__layer_38 |
| MVP | main | VideoMAE | ViT-H/16 | Attentive | Layer 0.75 | layer=24; lr=1e-05; epochs=30; patience=5; batch_size=2; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2 | mvp__videomae__temporal_attn__layer_24 |
| MVP | main | VideoMAE | ViT-H/16 | Linear | Layer 0.75 | lr=0.00462327; weight_decay=6.0743e-05; epochs=100; batch_size=32 | mvp__videomae__linear__layer_24 |
| MVP | main | VideoMAE | ViT-H/16 | MLP | Layer 0.75 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | mvp__videomae__mlp__layer_24 |
| MVP | main | VideoMAE-v2 | ViT-G/16 | Attentive | Layer 0.75 | layer=30; lr=1e-05; epochs=30; patience=5; batch_size=2; weight_decay=0.01; heads=16; blocks=1; mlp_ratio=2.0; dropout=0.2 | mvp__videomae_v2__temporal_attn__layer_30 |
| MVP | main | VideoMAE-v2 | ViT-G/16 | Linear | Layer 0.5 | lr=0.00409963; weight_decay=2.97888e-06; epochs=100; batch_size=32 | mvp__videomae_v2__linear__layer_20 |
| MVP | main | VideoMAE-v2 | ViT-G/16 | MLP | Layer 0.5 | lr=0.000132929; weight_decay=0.00506158; dropout=0.215973; hidden_dims=[256]; epochs=2000; batch_size=32 | mvp__videomae_v2__mlp__layer_20 |

## LR-Only Attentive Winners

| Dataset | Experiment | Model | Backbone | Probe | Best Row | Layer/LR | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Attentive | 12 | layer=12; lr=1e-4 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Attentive | 40 | layer=40; lr=1e-4 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Attentive | 9 | layer=9; lr=5e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | ltx | LTX-Video | LTX-13B | Attentive | 38 | layer=38; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | ltx | LTX-Video | LTX-2B | Attentive | 34 | layer=34; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | main | V-JEPA | ViT-H/16 | Attentive | 16 | layer=16; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | Attentive | 40 | layer=40; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | Attentive | 48 | layer=48; lr=1e-4 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | main | VideoMAE | ViT-H/16 | Attentive | 24 | layer=24; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | Attentive | 20 | layer=20; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | Attentive | 18 | layer=18; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | Attentive | 18 | layer=18; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | Attentive | 24 | layer=24; lr=5e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | Attentive | 24 | layer=24; lr=5e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | Attentive | 18 | layer=18; lr=1e-5 | LR-grid winner is in current Excel / recovered file has no full run hparams/provenance for this row / do not infer 30 vs 90 epochs from recovered file |

## Full Availability Matrix

| Dataset | Experiment | Model | Backbone | Probe | Availability | Best Row | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Attentive | YES_LR_ONLY | 12 | current Excel Selected LR |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Linear | NO | 9 | missing |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | MLP | NO | 9 | missing |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Attentive | YES_LR_ONLY | 40 | current Excel Selected LR |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Linear | NO | 20 | missing |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | MLP | NO | 30 | missing |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Attentive | YES_LR_ONLY | 9 | current Excel Selected LR |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Linear | NO | 9 | missing |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | MLP | NO | 9 | missing |
| IntPhys2 | ltx | LTX-Video | LTX-13B | Attentive | YES_LR_ONLY | 38 | current Excel Selected LR |
| IntPhys2 | ltx | LTX-Video | LTX-13B | Linear | YES_RECOVERED | noise_0.2_block_24 | recovered_best_configs.json |
| IntPhys2 | ltx | LTX-Video | LTX-13B | MLP | YES_RECOVERED | noise_0.2_block_24 | recovered_best_configs.json |
| IntPhys2 | ltx | LTX-Video | LTX-2B | Attentive | YES_LR_ONLY | 34 | current Excel Selected LR |
| IntPhys2 | ltx | LTX-Video | LTX-2B | Linear | NO | noise_0.7_block_14 | missing |
| IntPhys2 | ltx | LTX-Video | LTX-2B | MLP | NO | noise_0.4_block_14 | missing |
| IntPhys2 | main | V-JEPA | ViT-H/16 | Attentive | YES_LR_ONLY | 16 | current Excel Selected LR |
| IntPhys2 | main | V-JEPA | ViT-H/16 | Linear | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| IntPhys2 | main | V-JEPA | ViT-H/16 | MLP | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | Attentive | YES_LR_ONLY | 40 | current Excel Selected LR |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | Linear | YES_RECOVERED | Final layer | recovered_best_configs.json |
| IntPhys2 | main | V-JEPA 2 | ViT-G/16 | MLP | YES_RECOVERED | Final layer | recovered_best_configs.json |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | Attentive | YES_LR_ONLY | 48 | current Excel Selected LR |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| IntPhys2 | main | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| IntPhys2 | main | VideoMAE | ViT-H/16 | Attentive | YES_LR_ONLY | 24 | current Excel Selected LR |
| IntPhys2 | main | VideoMAE | ViT-H/16 | Linear | YES_RECOVERED | Layer 0.5 | recovered_best_configs.json |
| IntPhys2 | main | VideoMAE | ViT-H/16 | MLP | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | Attentive | YES_LR_ONLY | 20 | current Excel Selected LR |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | Linear | YES_RECOVERED | Layer 0.5 | recovered_best_configs.json |
| IntPhys2 | main | VideoMAE-v2 | ViT-G/16 | MLP | YES_RECOVERED | Layer 0.5 | recovered_best_configs.json |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | Attentive | YES_LR_ONLY | 18 | current Excel Selected LR |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | Linear | NO | 12 | missing |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | MLP | NO | 18 | missing |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | Attentive | YES_LR_ONLY | 18 | current Excel Selected LR |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | Linear | NO | 18 | missing |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | MLP | NO | 18 | missing |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | Attentive | YES_LR_ONLY | 24 | current Excel Selected LR |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | Linear | NO | 12 | missing |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | MLP | NO | 12 | missing |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | Attentive | YES_LR_ONLY | 24 | current Excel Selected LR |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | Linear | NO | 18 | missing |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | MLP | NO | 18 | missing |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | Attentive | YES_LR_ONLY | 18 | current Excel Selected LR |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | Linear | NO | 18 | missing |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | MLP | NO | 18 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Attentive | NO | 12 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Linear | NO | 12 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | MLP | NO | 12 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Attentive | NO | 40 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Linear | NO | 40 | missing |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | MLP | NO | 30 | missing |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Attentive | NO | 9 | missing |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Linear | NO | 12 | missing |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | MLP | NO | 12 | missing |
| MVP | ltx | LTX-Video | LTX-13B | Attentive | NO | noise_0.2_block_24 | missing |
| MVP | ltx | LTX-Video | LTX-13B | Linear | YES_RECOVERED | noise_0.6_block_24 | local recovery optuna_summary.json |
| MVP | ltx | LTX-Video | LTX-13B | MLP | YES_RECOVERED | noise_0.6_block_24 | local recovery optuna_summary.json |
| MVP | ltx | LTX-Video | LTX-2B | Attentive | YES_RECOVERED | 29 | local train_eval_summary.json |
| MVP | ltx | LTX-Video | LTX-2B | Linear | NO | noise_0.1_block_7 | missing |
| MVP | ltx | LTX-Video | LTX-2B | MLP | NO | noise_0.2_block_7 | missing |
| MVP | main | V-JEPA | ViT-H/16 | Attentive | YES_RECOVERED | Layer 0.75 | local recovery LR matrix |
| MVP | main | V-JEPA | ViT-H/16 | Linear | YES_RECOVERED | Final layer | recovered_best_configs.json |
| MVP | main | V-JEPA | ViT-H/16 | MLP | YES_RECOVERED | Final layer | recovered_best_configs.json |
| MVP | main | V-JEPA 2 | ViT-G/16 | Attentive | YES_RECOVERED | Final layer | local recovery LR matrix |
| MVP | main | V-JEPA 2 | ViT-G/16 | Linear | YES_RECOVERED | Final layer | recovered_best_configs.json |
| MVP | main | V-JEPA 2 | ViT-G/16 | MLP | YES_RECOVERED | Final layer | recovered_best_configs.json |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | Attentive | YES_RECOVERED | Layer 0.75 | local recovery LR matrix |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | YES_RECOVERED | Final layer | recovered_best_configs.json |
| MVP | main | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| MVP | main | VideoMAE | ViT-H/16 | Attentive | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| MVP | main | VideoMAE | ViT-H/16 | Linear | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| MVP | main | VideoMAE | ViT-H/16 | MLP | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| MVP | main | VideoMAE-v2 | ViT-G/16 | Attentive | YES_RECOVERED | Layer 0.75 | recovered_best_configs.json |
| MVP | main | VideoMAE-v2 | ViT-G/16 | Linear | YES_RECOVERED | Layer 0.5 | recovered_best_configs.json |
| MVP | main | VideoMAE-v2 | ViT-G/16 | MLP | YES_RECOVERED | Layer 0.5 | recovered_best_configs.json |
| MVP | same_L | V-JEPA | ViT-L/16 | Attentive | NO | 18 | missing |
| MVP | same_L | V-JEPA | ViT-L/16 | Linear | NO | 18 | missing |
| MVP | same_L | V-JEPA | ViT-L/16 | MLP | NO | 18 | missing |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | Attentive | NO | 24 | missing |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | Linear | NO | 24 | missing |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | MLP | NO | 24 | missing |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | Attentive | NO | 24 | missing |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | Linear | NO | 18 | missing |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | MLP | NO | 24 | missing |
| MVP | same_L | VideoMAE | ViT-L/16 | Attentive | NO | 24 | missing |
| MVP | same_L | VideoMAE | ViT-L/16 | Linear | NO | 24 | missing |
| MVP | same_L | VideoMAE | ViT-L/16 | MLP | NO | 24 | missing |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | Attentive | NO | 18 | missing |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | Linear | NO | 18 | missing |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | MLP | NO | 18 | missing |

## Missing Best Rows

| Dataset | Experiment | Model | Backbone | Probe | Best Row | Why Missing |
| --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Linear | 9 | no recovered config for the current best row |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | MLP | 9 | no recovered config for the current best row |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Linear | 20 | no recovered config for the current best row |
| IntPhys2 | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | MLP | 30 | no recovered config for the current best row |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Linear | 9 | no recovered config for the current best row |
| IntPhys2 | backbone_sweep | VideoMAE-v2 | ViT-B/16 | MLP | 9 | no recovered config for the current best row |
| IntPhys2 | ltx | LTX-Video | LTX-2B | Linear | noise_0.7_block_14 | no recovered config for the current best row |
| IntPhys2 | ltx | LTX-Video | LTX-2B | MLP | noise_0.4_block_14 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | Linear | 12 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA | ViT-L/16 | MLP | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA 2 | ViT-L/16 | MLP | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | Linear | 12 | no recovered config for the current best row |
| IntPhys2 | same_L | V-JEPA 2.1 | ViT-L/16 | MLP | 12 | no recovered config for the current best row |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | VideoMAE | ViT-L/16 | MLP | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| IntPhys2 | same_L | VideoMAE-v2 | ViT-L/16 | MLP | 18 | no recovered config for the current best row |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Attentive | 12 | no recovered config and no Selected LR in current Excel |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | Linear | 12 | no recovered config for the current best row |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-B/16 | MLP | 12 | no recovered config for the current best row |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Attentive | 40 | no recovered config and no Selected LR in current Excel |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | Linear | 40 | no recovered config for the current best row |
| MVP | backbone_sweep | V-JEPA 2.1 | ViT-G/16 | MLP | 30 | no recovered config for the current best row |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Attentive | 9 | no recovered config and no Selected LR in current Excel |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | Linear | 12 | no recovered config for the current best row |
| MVP | backbone_sweep | VideoMAE-v2 | ViT-B/16 | MLP | 12 | no recovered config for the current best row |
| MVP | ltx | LTX-Video | LTX-13B | Attentive | noise_0.2_block_24 | no recovered config and no Selected LR in current Excel |


| MVP | ltx | LTX-Video | LTX-2B | Linear | noise_0.1_block_7 | no recovered config for the current best row |
| MVP | ltx | LTX-Video | LTX-2B | MLP | noise_0.2_block_7 | no recovered config for the current best row |
| MVP | same_L | V-JEPA | ViT-L/16 | Attentive | 18 | no recovered config and no Selected LR in current Excel |
| MVP | same_L | V-JEPA | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| MVP | same_L | V-JEPA | ViT-L/16 | MLP | 18 | no recovered config for the current best row |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | Attentive | 24 | no recovered config and no Selected LR in current Excel |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | Linear | 24 | no recovered config for the current best row |
| MVP | same_L | V-JEPA 2 | ViT-L/16 | MLP | 24 | no recovered config for the current best row |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | Attentive | 24 | no recovered config and no Selected LR in current Excel |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| MVP | same_L | V-JEPA 2.1 | ViT-L/16 | MLP | 24 | no recovered config for the current best row |
| MVP | same_L | VideoMAE | ViT-L/16 | Attentive | 24 | no recovered config and no Selected LR in current Excel |
| MVP | same_L | VideoMAE | ViT-L/16 | Linear | 24 | no recovered config for the current best row |
| MVP | same_L | VideoMAE | ViT-L/16 | MLP | 24 | no recovered config for the current best row |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | Attentive | 18 | no recovered config and no Selected LR in current Excel |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | Linear | 18 | no recovered config for the current best row |
| MVP | same_L | VideoMAE-v2 | ViT-L/16 | MLP | 18 | no recovered config for the current best row |

Machine-readable best-row table: `results/best_config_availability.csv`.
