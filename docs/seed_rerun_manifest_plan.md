# Seed Rerun Manifest Plan

Date: 2026-08-05

## Goal

Add two extra training seeds for the frozen best probe configurations, then average them with the already existing main-result seed in the spreadsheets.

The intended scientific question is:

> Given the selected model, layer, and hyperparameters from the main experiment, how stable is the reported result under probe-training randomness?

Therefore, seed reruns must **not** repeat hyperparameter search and must **not** reselect layers.

## Seed Policy

Use three seeds in the final statistics:

| Seed | Source | Meaning |
| ---: | --- | --- |
| 42 | Existing spreadsheet / main experiment | Original reported run |
| 101 | New artifact run | Extra robustness seed |
| 102 | New artifact run | Extra robustness seed |

For new runs, pass:

```bash
seed=<101_or_102>
split.seed=42
probe.optuna.enabled=false
```

`seed=<...>` changes training randomness. `split.seed=42` keeps the config snapshot aligned with the original data split. Do not rerun `init.*`; seed reruns reuse existing split files and existing feature caches.

MLP seed reruns must additionally force:

```bash
probe.early_stopping.enabled=false
```

This is intentional. The recovered best-config CSV stores the current config schema, but the original MLP main jobs selected checkpoints after the full epoch budget. The V-JEPA2 layerwise pilot showed that enabling early stopping can collapse the IntPhys2 MLP result toward chance, while disabling it reproduces the old selected-layer test regime. Linear rows keep their recovered early-stopping setting.

## Source Config Table

The source table is:

```text
results/verified_best_probe_configs.csv
```

It contains one row per selected configuration. At the time of inspection:

```text
90 total rows
39 verified/non-missing configs
51 missing configs
```

For the `main` experiment slice:

```text
30 total rows
27 runnable rows
3 missing rows
```

The missing `main` rows are:

```text
MVP / V-JEPA / ViT-H/16 / Attentive
MVP / V-JEPA 2 / ViT-G/16 / Attentive
MVP / V-JEPA 2.1 / ViT-Gigantic/16 / Attentive
```

The seed launcher must **fail closed**: rows with `config_status=MISSING` must not be submitted and must be reported as blocked.

## Manifest Concept

The seed manifest is the run plan. It expands each runnable best-config row into one row per new seed.

Example source config:

```text
config_id = mvp__jepa_v1__mlp__layer_32
seed      = original/main result
```

Expanded manifest rows:

```text
mvp__jepa_v1__mlp__layer_32__seed_101
mvp__jepa_v1__mlp__layer_32__seed_102
```

The manifest is not the final results table. It is the deterministic recipe used by Slurm to launch jobs and by the collector to know what should exist.

## Manifest Location

Use:

```text
results/seed_runs/seed_manifest_v1.csv
```

Recommended manifest columns:

```text
run_id
config_id
dataset
experiment
model
backbone
probe
seed
original_seed
backbone_name
backbone_variant
layer
selected_slot
layer_label
feature_view
lr
weight_decay
batch_size
eval_batch_size
epochs
early_stopping_enabled
early_stopping_patience
mlp_hidden_dims
mlp_dropout
temporal_num_heads
temporal_num_self_attn_blocks
temporal_mlp_ratio
temporal_dropout
probe_device
probe_output_subdir
eval_output_subdir
wandb_group
wandb_name
status
blocked_reason
source_config_status
source_evidence_path
```

One manifest row must contain all information needed to run exactly one seed job.

## Artifact Layout

Keep seed reruns separate from old main runs, controls, and scratch attempts.

Training artifacts:

```text
artifacts/probes/<dataset>/seed_runs_v1/<config_id>/seed_<seed>/
```

Evaluation artifacts:

```text
artifacts/results/<dataset>/seed_runs_v1/<config_id>/seed_<seed>/
```

Examples:

```text
artifacts/probes/intphys2/seed_runs_v1/intphys2__jepa_v1__mlp__layer_32/seed_101/
artifacts/results/intphys2/seed_runs_v1/intphys2__jepa_v1__mlp__layer_32/seed_101/

artifacts/probes/mvp/seed_runs_v1/mvp__jepa_v1__mlp__layer_32/seed_101/
artifacts/results/mvp/seed_runs_v1/mvp__jepa_v1__mlp__layer_32/seed_101/
```

The exact path is written in `probe_output_subdir` and `eval_output_subdir` in the manifest.

## Launch Command Shape

Each manifest row becomes one `train_eval.probe.<dataset>` command.

Example:

```bash
python run.py train_eval.probe.mvp \
  seed=101 \
  split.seed=42 \
  backbone.name=jepa_v1 \
  +backbone.kwargs.variant=vith16_384 \
  probe.name=mlp \
  probe.layer=32 \
  probe.layers=[] \
  probe.feature_view=pooled \
  probe.lr=0.001 \
  probe.weight_decay=0.0 \
  probe.batch_size=128 \
  probe.eval_batch_size=1024 \
  probe.epochs=100 \
  probe.early_stopping.enabled=false \
  probe.optuna.enabled=false \
  probe.output_subdir=seed_runs_v1/mvp__jepa_v1__mlp__layer_32/seed_101 \
  probe.eval_output_subdir=seed_runs_v1/mvp__jepa_v1__mlp__layer_32/seed_101 \
  probe.wandb.group=seed_runs_v1_mvp__jepa_v1__mlp__layer_32 \
  probe.wandb.name=mvp__jepa_v1__mlp__layer_32__seed_101
```

For temporal attentive probes, force the main-experiment attentive defaults unless the source row contains explicit verified values:

```text
probe.name=temporal_attn
probe.feature_view=tokens
probe.device=cuda
probe.optuna.enabled=false
probe.temporal_attn.num_heads=16
probe.temporal_attn.num_self_attn_blocks=1
probe.temporal_attn.mlp_ratio=2.0
probe.temporal_attn.dropout=0.2
```

For IntPhys2 attentive main runs, the verified matrix logs used:

```text
epochs=90
early_stopping.patience=20
batch_size=1
eval_batch_size=1
weight_decay=0.01
```

For MVP attentive rows, confirm the exact main-experiment defaults before launching if the row is only `VERIFIED_ATTENTIVE_LAYER_LR`.

## Slurm Execution

Use two Slurm arrays rather than many hand-written jobs:

```text
jobs/train/seed_runs/run_seed_cpu_array.sh
jobs/train/seed_runs/run_seed_gpu_array.sh
```

CPU array:

```text
Linear
MLP
```

GPU array:

```text
Temporal attention
```

Each array task reads one row from `results/seed_runs/seed_manifest_v1.csv` using `SLURM_ARRAY_TASK_ID`, validates that the probe type matches the array type, builds the Hydra command, and executes it.

Do not use wrappers that request a GPU for CPU-only probes. In particular, check LTX MVP MLP wrappers before reuse; some historical wrappers requested `gpu_a100` even with `PROBE_DEVICE=cpu`.

## Result Collection

Create:

```text
results/seed_runs/seed_results_long.csv
```

This is one row per config and seed, including both old and new seeds.

Recommended columns:

```text
config_id
dataset
experiment
model
backbone
probe
seed
source
status
train_primary
train_accuracy
val_primary
val_accuracy
test_primary
test_accuracy
objective_metric_name
artifact_train_summary
artifact_eval_summary
notes
```

Rows for seed 42 come from the spreadsheet/source CSV:

```text
seed=42
source=sheet
```

Rows for seeds 101 and 102 come from new artifacts:

```text
seed=101
source=artifact
```

The collector should read:

```text
<eval_output_dir>/probe_eval_summary.json
<eval_output_dir>/metrics.json
<train_output_dir>/train_summary.json
```

For IntPhys2, the primary metric is VOE-style accuracy from `voe_accuracy`.

For MVP, the primary metric is pair consistency from `pair_consistency`.

## Summary Table

Create:

```text
results/seed_runs/seed_summary.csv
```

This is one row per `config_id`.

Recommended columns:

```text
config_id
dataset
experiment
model
backbone
probe
n_seeds
seeds
test_primary_mean
test_primary_std
test_accuracy_mean
test_accuracy_std
val_primary_mean
val_primary_std
val_accuracy_mean
val_accuracy_std
train_primary_mean
train_primary_std
train_accuracy_mean
train_accuracy_std
complete
missing_seeds
notes
```

For each metric:

```text
mean = average(seed_42, seed_101, seed_102)
std  = sample standard deviation over available complete seeds
```

For paper reporting, prefer:

```text
mean ± std
```

With only three seeds, do not overstate confidence intervals.

## Completeness Rules

For a config to be complete:

```text
seed 42 exists from sheet
seed 101 artifact exists and is valid
seed 102 artifact exists and is valid
```

If one new seed fails:

```text
complete=false
missing_seeds=<failed seed>
```

Do not average silently over two seeds without marking the row incomplete.

If seed 42 cannot be mapped from the spreadsheet:

```text
source=sheet
status=missing_seed_42
complete=false
```

## Implementation Files

Recommended scripts:

```text
scripts/build_seed_manifest.py
scripts/collect_seed_results.py
scripts/export_seed_results_excel.py
jobs/train/seed_runs/run_seed_cpu_array.sh
jobs/train/seed_runs/run_seed_gpu_array.sh
```

`build_seed_manifest.py` responsibilities:

- Read `results/verified_best_probe_configs.csv`.
- Filter to desired experiment slice, probably `experiment=main` first.
- Block `config_status=MISSING`.
- Expand each runnable config to seeds `101,102`.
- Normalize dataset/probe names to Hydra names.
- Map human model/backbone labels to `backbone.name` and `backbone.kwargs.variant`.
- Write deterministic output subdirs.
- Write blocked rows or a blocked report.

`run_seed_*_array.sh` responsibilities:

- Read manifest row by `SLURM_ARRAY_TASK_ID`.
- Validate row status is runnable.
- Validate probe type matches CPU/GPU array.
- Construct the exact `python run.py train_eval.probe.<dataset>` command.
- Set `probe.optuna.enabled=false`.
- Set stable output dirs and WandB group/name.
- Exit non-zero on malformed or blocked rows.

`collect_seed_results.py` responsibilities:

- Read the seed manifest.
- Read seed 42 values from source spreadsheet/export.
- Read new seed artifacts from deterministic paths.
- Write `seed_results_long.csv`.
- Write `seed_summary.csv`.
- Mark incomplete configs explicitly.

`export_seed_results_excel.py` responsibilities:

- Read `seed_results_long.csv` and `seed_summary.csv`.
- Write a two-sheet `seed_results.xlsx` workbook without requiring manual spreadsheet editing.

## Practical First Step

Start with the `main` experiment only.

Expected initial scope:

```text
27 runnable configs x 2 new seeds = 54 new jobs
3 blocked configs
```

After the flow is verified, decide whether to resolve missing configs or extend the same machinery to `same_L`, `backbone_sweep`, and `ltx`.
