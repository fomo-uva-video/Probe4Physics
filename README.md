# Probe4Physics (MVP Pipeline)

This repository provides a benchmark-centric pipeline for probing frozen video backbones on MVP.

Current v1 scope:
- deterministic MVP split initialization
- frozen feature extraction
- linear probe training and evaluation
- experiment recipes through `run.py`

`run.py` is the single launcher for all workflows.

## Project Layout
- `benchmarks/`: benchmark logic (MVP loading/selection/scoring + shared splitting)
- `models/`: frozen backbone adapters (currently V-JEPA v1)
- `probes/`: probe contracts and implementations (currently `linear` runnable)
- `training/`: feature extraction + probe train/eval orchestration
- `experiments/`: experiment registry recipes
- `configs/mvp.yaml`: main runtime config

## Environment Setup
```bash
conda env create -f environment.yml
conda activate probe4physics
# if environment already exists:
conda env update -n probe4physics -f environment.yml --prune
```

## Submodules and Checkpoint
Initialize submodules (MVP official code + JEPA adapter dependencies):
```bash
git submodule update --init --recursive
git -C third_party/jepa_v1 checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
```

Download an official V-JEPA v1 checkpoint from:
- https://github.com/facebookresearch/jepa

Use it via config override:
- `backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar`

## End-to-End Pipeline
Run from repository root:

```bash
python run.py init.mvp
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train.linear.mvp
python run.py eval.linear.mvp
```

### Stage 1: `init.mvp`
Builds deterministic train/val/test splits and selection artifacts.

Main outputs in `split.dir`:
- `split_pairs.parquet`
- `manifest.json`
- `selection_kept.csv`
- `selection_dropped.csv`
- `selection_report.json`

### Stage 2: `extract.mvp`
Runs frozen backbone forward pass and writes deterministic feature cache.

Main outputs in `feature_cache.dir/<backbone>/<split_key>/<signature>/`:
- `index.parquet`
- `features_pooled.pt`
- `features_tokens.pt`
- `manifest.json`

If videos are missing or `videos_root` is invalid, extraction writes warnings to:
- `feature_cache.dir/extract_warnings.log`

### Stage 3: `train.linear.mvp`
Trains the linear probe from cached features (no backbone forward).

Main outputs in `linear_probe.output_dir/...`:
- `linear_probe.pt`
- `train_summary.json`

### Stage 4: `eval.linear.mvp`
Loads linear checkpoint, verifies feature signature, writes predictions, and runs official MVP scoring.

Main outputs in `linear_probe.eval_output_dir/...`:
- `linear_predictions.json`
- `linear_eval_summary.json`
- official MVP eval artifacts (`metrics.json`, `predictions.csv`, `summary.md`, etc.)

## Experiment Recipes
List available experiments:
```bash
python run.py exp.list
```

Run a full recipe (default: extract -> train.linear -> eval.linear):
```bash
python run.py exp.run name=mvp.jepa_v1.linear backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

If a valid feature cache already exists, extraction is skipped automatically.
Force re-extraction with:
```bash
python run.py exp.run name=mvp.jepa_v1.linear feature_cache.force_reextract=true backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

## Command Reference
- `python run.py init.mvp`
- `python run.py extract.mvp`
- `python run.py train.linear.mvp`
- `python run.py eval.linear.mvp`
- `python run.py eval.mvp`
- `python run.py exp.list`
- `python run.py exp.run name=<experiment_id>`

`eval.mvp` is a direct MVP evaluation command using built-in predictor modes (`oracle`, `random`, `from_file`).

## Common Overrides
Use Hydra-style overrides on any command.

Examples:
```bash
# extract only train split for a quick smoke run
python run.py extract.mvp feature_cache.split_names=[train] backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar

# train with token-mean features instead of pooled
python run.py train.linear.mvp linear_probe.feature_view=tokens_mean

# evaluate a specific split and checkpoint
python run.py eval.linear.mvp split_name=val linear_probe.checkpoint_path=/absolute/path/to/linear_probe.pt
```

## Main Config Keys (`configs/mvp.yaml`)
- `split.*`: split location and ratios
- `backbone.*`: adapter name and kwargs
- `feature_cache.*`: cache location/content/re-extraction behavior
- `linear_probe.*`: probe training/eval settings and output locations
- `decode.*`: frame sampling/resizing settings
- `predictor.*`: predictor mode for `eval.mvp`
