# Seed Rerun Manifest Plan

Date: 2026-08-09

## Goal

Run two extra probe-training seeds for fixed, already selected configurations, then aggregate them with the existing seed-42 result. The scientific question is seed robustness under probe-training randomness, not whether hyperparameter search or layer selection is stable.

Seed reruns must therefore use:

```bash
seed=<101_or_102>
split.seed=42
probe.optuna.enabled=false
```

`seed` changes the probe-training randomness. `split.seed=42` keeps the original data split. The runs reuse existing feature caches and do not rerun extraction.

## Source Tables

There are now two config sources, and they should not be mixed.

| Source CSV | Intended use | Output namespace |
| --- | --- | --- |
| `results/verified_best_probe_configs.csv` | Final selected-layer robustness runs | `seed_runs_v1` |
| `results/verified_layerwise_probe_configs.csv` | Layerwise/diagnostic robustness runs over all available layers | `seed_runs_layerwise_v1` |

The selected-layer CSV keeps the paper-facing best-layer flow. The layerwise CSV is broader and is the right source for all-layer experiments.

Current layerwise CSV inspection:

| Slice | Status |
| --- | --- |
| All rows | 648 configs |
| `VERIFIED_FULL` | 581 configs |
| `MISSING` | 63 configs |
| `VERIFIED_ATTENTIVE_LAYER_LR` | 4 configs |
| `main` Linear/MLP | 80 configs, all `VERIFIED_FULL` |
| `main` Attentive | 28 `VERIFIED_FULL`, 12 `MISSING` |

The missing `main` attentive rows are MVP V-JEPA, MVP V-JEPA 2, and MVP V-JEPA 2.1 across their four reported layers. LTX remains partly incomplete, especially MVP LTX-13B Linear/MLP and attentive configs.

Rows with `config_status=MISSING` must stay blocked. Missing rows have `config_id=NULL`, so blocked manifests generate synthetic IDs from row content and source line number.

## MLP Early-Stopping Policy

The default manifest policy is:

```bash
--mlp-early-stopping-policy force_disabled
```

This forces:

```bash
probe.early_stopping.enabled=false
```

for MLP reruns. This is intentional for the current seed-recovery flow: the V-JEPA2 layerwise pilot showed that enabling early stopping can collapse the IntPhys2 MLP result toward chance, while disabling it reproduces the historical full-budget selected-layer result.

Use this only when the goal is to match historical main-runtime behavior. If a future source CSV is known to contain the exact desired runtime semantics, the builder supports:

```bash
--mlp-early-stopping-policy source
```

Linear rows keep their source early-stopping settings.

## Manifest Builder

Use `scripts/build_seed_manifest.py`. It supports both selected-layer and layerwise CSV schemas.

The builder reads either of these layer fields:

```text
selected_layer_id
probe_layer
excel_layer
best_config_json.layer
```

It reads explicit Hydra backbone fields when available:

```text
backbone_name
backbone_variant
probe_name
```

and falls back to the legacy human-label mapping only when needed.

## Selected-Layer Manifest

For final paper-facing selected-layer Linear/MLP seed runs:

```bash
python scripts/build_seed_manifest.py \
  --source results/verified_best_probe_configs.csv \
  --output results/seed_runs/seed_manifest_main_linear_mlp_v1.csv \
  --blocked-output results/seed_runs/seed_manifest_main_linear_mlp_blocked_v1.csv \
  --experiment main \
  --probes Linear,MLP \
  --run-group seed_runs_v1 \
  --mlp-early-stopping-policy force_disabled
```

Artifacts go to:

```text
artifacts/probes/<dataset>/seed_runs_v1/<config_id>/seed_<seed>/
artifacts/results/<dataset>/seed_runs_v1/<config_id>/seed_<seed>/
```

## Layerwise Manifest

For all-layer main Linear/MLP seed runs from the updated layerwise source:

```bash
python scripts/build_seed_manifest.py \
  --source results/verified_layerwise_probe_configs.csv \
  --output results/seed_runs/seed_manifest_layerwise_main_linear_mlp_v1.csv \
  --blocked-output results/seed_runs/seed_manifest_layerwise_main_linear_mlp_blocked_v1.csv \
  --experiment main \
  --probes Linear,MLP \
  --run-group seed_runs_layerwise_v1 \
  --mlp-early-stopping-policy force_disabled
```

Expected current shape:

```text
80 runnable configs x 2 new seeds = 160 manifest rows
0 blocked rows for main Linear/MLP
```

Artifacts go to:

```text
artifacts/probes/<dataset>/seed_runs_layerwise_v1/<config_id>/seed_<seed>/
artifacts/results/<dataset>/seed_runs_layerwise_v1/<config_id>/seed_<seed>/
```

## Slurm Runners

The shared launcher is:

```text
jobs/train/seed_runs/run_manifest_task.sh
```

It reads one manifest row by `SLURM_ARRAY_TASK_ID` after filtering by probe type. The selected-layer wrappers keep the small selected-layer manifest:

```text
jobs/train/seed_runs/main_linear_cpu_array.sh
jobs/train/seed_runs/main_mlp_cpu_array.sh
```

Layerwise wrappers use the larger layerwise manifest:

```text
jobs/train/seed_runs/layerwise_main_linear_cpu_array.sh
jobs/train/seed_runs/layerwise_main_mlp_cpu_array.sh
```

The current main layerwise Linear/MLP manifest has 80 rows per probe after filtering, so the wrappers use array range `0-79`.

## Result Collection

Use:

```bash
python scripts/collect_seed_results.py \
  --manifest results/seed_runs/seed_manifest_layerwise_main_linear_mlp_v1.csv \
  --long-output results/seed_runs/layerwise_seed_results_long.csv \
  --summary-output results/seed_runs/layerwise_seed_summary.csv
```

The collector now infers the source CSV from the manifest. It reads seed-42 metrics directly from these CSV columns when available:

```text
train_accuracy
val_accuracy
test_accuracy
train_primary_metric
val_primary_metric
test_primary_metric
```

If those six metrics are incomplete, it falls back to the Excel workbook/sheet/range fields.

For each config, the summary aggregates complete seeds over:

```text
42, 101, 102
```

and marks incomplete rows explicitly through `complete=false` and `missing_seeds`.

## Excel Export

After collection:

```bash
python scripts/export_seed_results_excel.py \
  --long results/seed_runs/layerwise_seed_results_long.csv \
  --summary results/seed_runs/layerwise_seed_summary.csv \
  --output results/seed_runs/layerwise_seed_results.xlsx
```

The exporter is source-agnostic and does not need changes for selected-layer versus layerwise runs.
