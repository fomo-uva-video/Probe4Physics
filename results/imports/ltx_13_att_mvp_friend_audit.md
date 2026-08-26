# LTX-13B MVP Attentive Friend Import Audit

Generated: 2026-08-26T09:37:04.168815+00:00

Raw artifact folder: `/gpfs/home2/spunzo1/ltx_13_att_mvp_friend`
Workspace symlink: `artifacts/external_imports/ltx_13_att_mvp_friend`

## Verdict

Accepted as an external confirmed import and promoted to the normal result CSV/Excel records.

## Training Provenance Confirmed Externally

Friend reported the successful seed-42 launch used:

- `DATASET_NAME=mvp`
- `BACKBONE_NAME=ltx_video`
- `BACKBONE_VARIANT=ltxv_13b_0_9_8_distilled`
- `PROBE_NAME=temporal_attn`
- `PROBE_FEATURE_VIEW=tokens`
- `PROBE_EPOCHS=30`
- `PROBE_BATCH_SIZE=1`
- `PROBE_EVAL_BATCH_SIZE=1`
- `PROBE_WEIGHT_DECAY=0.01`
- `probe.early_stopping.enabled=true`
- `probe.early_stopping.patience=5`
- `RUN_SEED=42`, `SPLIT_SEED=42`
- `TEMPORAL_NUM_HEADS=16`
- `TEMPORAL_NUM_SELF_ATTN_BLOCKS=1`
- `TEMPORAL_MLP_RATIO=2.0`
- `TEMPORAL_DROPOUT=0.2`
- `FEATURE_LAYER_IDS=[18,22,26,10]`
- `feature_cache.include_tokens=true`, `feature_cache.include_pooled=true`

Seed 101/102 are imported from the fixed-`lr=1e-5` eval folders and recorded in `seed_manifest_mvp_ltx13b_attentive_friend_external.csv`; they follow the project seeded policy of full 30 epochs without early stopping.

## What Is Present

- Recovery seed 42 LR matrix: 4 layers x 4 learning rates = 16 aggregate eval summaries.
- Seeded fixed-LR evals: 4 layers x seeds 101/102 = 8 aggregate eval summaries.
- Per-split train/val/test metrics and predictions are present.

## What Is Missing Locally

- No `train_summary.json` copied into this server folder.
- No copied probe checkpoints; summaries point to the friend's scratch path.

The eval `run_config.snapshot.yaml` files contain eval-time/default probe fields and are not used as training hyperparameter provenance. In this codebase, temporal-attention checkpoints save and reload architecture fields from the checkpoint payload.

## Layer/LR Check

The layers `10,18,22,26` match the top four MVP LTX-13B MLP layers by test pair-consistency, with validation pair-consistency as tie-break. Recovery selection uses best validation pair-consistency per layer. Winning LR is `1e-5` for all four layers.

## Files Written

- `results/imports/ltx_13_att_mvp_friend_metrics_long.csv`
- `results/imports/ltx_13_att_mvp_friend_recovery_winners.csv`
- `results/imports/ltx_13_att_mvp_friend_seed_summary.csv`
- `results/seed_runs/seed_manifest_mvp_ltx13b_attentive_friend_external.csv`
- `results/mvp_primary_db.xlsx`
- `results/verified_layerwise_probe_configs.csv`
- `results/verified_best_probe_configs.csv`
