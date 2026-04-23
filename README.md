# Probe4Physics

This repository provides a benchmark-centric pipeline for probing frozen video backbones on intuitive physics and action recognition benchmarks.

Current v1 scope:
- deterministic split initialization for MVP, IntPhys2, and SSv2
- frozen feature extraction
- linear probe training and evaluation
- experiment recipes through `run.py`

`run.py` is the single launcher for all workflows.

## Project Layout
- `benchmarks/`: benchmark logic (MVP, IntPhys2, SSv2 — loading/scoring/splitting)
- `models/`: frozen backbone adapters (`jepa_v1`, `jepa_v2`, `jepa_v2_1`, `videomae`, `videomae_v2`, `ltx_video`)
- `probes/`: probe contracts and implementations (currently `linear` runnable)
- `training/`: feature extraction + probe train/eval orchestration
- `experiments/`: experiment registry recipes
- `configs/mvp.yaml`: MVP runtime config
- `configs/intphys2.yaml`: IntPhys2 runtime config
- `configs/ssv2.yaml`: SSv2 runtime config

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

## IntPhys2 Pipeline
IntPhys2 is a video benchmark for intuitive physics understanding (4 principles: permanence, immutability, continuity, solidity). Evaluation uses Violation-of-Expectation (VOE): a scene is correctly recognised only if all possible clips score higher than all impossible clips.

The dataset is downloaded directly from HuggingFace (`facebook/IntPhys2`, CC BY-NC 4.0).

```bash
python run.py download.intphys2
python run.py init.intphys2
python run.py extract.intphys2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train.linear.intphys2
python run.py eval.linear.intphys2
```

### Stage 0: `download.intphys2`
Downloads the Debug and Main splits from HuggingFace and writes a normalised metadata CSV.

Main outputs:
- `data/videos/intphys2/` — video files organised by HF split (`Debug/Videos/`, `Main/Videos/`)
- `data/annotations/intphys2_metadata.csv` — normalised metadata (video_path, scene_id, condition, plausibility, split)

Requires `pip install huggingface_hub`. To download only specific splits:
```bash
python run.py download.intphys2 download.splits=[Debug]
```

### Stage 1: `init.intphys2`
Validates metadata, groups clips into scene quadruplets, and writes split artifacts.

Main outputs in `split.dir`:
- `split_scenes.parquet`
- `manifest.json`

### Stage 2: `extract.intphys2`
Runs frozen backbone forward pass and writes feature cache.

Main outputs mirror the MVP cache layout under `feature_cache.dir`.

### Stage 3: `train.linear.intphys2`
Trains a binary linear probe (impossible=0, possible=1) from cached features.

Main outputs in `linear_probe.output_dir/...`:
- `linear_probe.pt`
- `train_summary.json`

### Stage 4: `eval.linear.intphys2`
Loads checkpoint, computes per-video P(possible) scores via softmax, and runs IntPhys2 scoring.

Main outputs in `linear_probe.eval_output_dir/...`:
- `linear_predictions.json`
- `linear_eval_summary.json`
- `metrics.json` (accuracy + VOE accuracy, per-condition breakdown)
- `predictions.csv`, `summary.md`, `provenance.json`

## SSv2 Pipeline
Something-Something v2 (SSv2) is a 174-class action recognition benchmark used as a temporal reasoning control task. Evaluation reports Top-1 and Top-5 accuracy.

Requires the official SSv2 annotation files (`train.json`, `validation.json`, `labels.json`) from the [official dataset page](https://developer.qualcomm.com/software/ai-datasets/something-something). Place them at the paths configured in `configs/ssv2.yaml`.

```bash
python run.py init.ssv2
python run.py extract.ssv2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train.linear.ssv2
python run.py eval.linear.ssv2
```

### Stage 1: `init.ssv2`
Validates annotation files, subsets to at most `split.max_samples_per_class` clips per class (default 200, ~29k total), and writes split artifacts.

Main outputs in `split.dir`:
- `split_clips.parquet`
- `manifest.json`

### Stage 2: `extract.ssv2`
Runs frozen backbone forward pass and writes feature cache.

Main outputs mirror the MVP cache layout under `feature_cache.dir`.

### Stage 3: `train.linear.ssv2`
Trains a 174-class linear probe from cached features.

Main outputs in `linear_probe.output_dir/...`:
- `linear_probe.pt`
- `train_summary.json`

### Stage 4: `eval.linear.ssv2`
Loads checkpoint, computes per-class softmax scores (enables Top-5), and runs SSv2 scoring.

Main outputs in `linear_probe.eval_output_dir/...`:
- `linear_predictions.json`
- `linear_eval_summary.json`
- `metrics.json` (top1_accuracy, top5_accuracy, per-template breakdown)
- `predictions.csv`, `summary.md`, `provenance.json`

## Experiment Recipes
List available experiments:
```bash
python run.py exp.list
```

Run a full recipe (default: extract -> train.linear -> eval.linear):
```bash
python run.py exp.run name=mvp.jepa_v1.linear backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=intphys2.jepa_v1.linear backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=ssv2.jepa_v1.linear backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=mvp.ltx_video.linear backbone.kwargs.device=cuda
python run.py exp.run name=intphys2.ltx_video.linear backbone.kwargs.device=cuda
python run.py exp.run name=ssv2.ltx_video.linear backbone.kwargs.device=cuda
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
- `python run.py download.intphys2`
- `python run.py init.intphys2`
- `python run.py extract.intphys2`
- `python run.py train.linear.intphys2`
- `python run.py eval.linear.intphys2`
- `python run.py eval.intphys2`
- `python run.py init.ssv2`
- `python run.py extract.ssv2`
- `python run.py train.linear.ssv2`
- `python run.py eval.linear.ssv2`
- `python run.py eval.ssv2`
- `python run.py exp.list`
- `python run.py exp.run name=<experiment_id>`

`eval.mvp`, `eval.intphys2`, and `eval.ssv2` are direct evaluation commands using built-in predictor modes (`oracle`, `random`, `from_file`).

## Common Overrides
Use Hydra-style overrides on any command.

Examples:
```bash
# extract only train split for a quick smoke run
python run.py extract.mvp feature_cache.split_names=[train] backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar

# run LTX-Video feature extraction with the default LTX checkpoint
python run.py extract.mvp backbone.name=ltx_video backbone.kwargs.device=cuda

# train with token-mean features instead of pooled
python run.py train.linear.mvp linear_probe.feature_view=tokens_mean

# evaluate a specific split and checkpoint
python run.py eval.linear.mvp split_name=val linear_probe.checkpoint_path=/absolute/path/to/linear_probe.pt

# download only the Debug split of IntPhys2
python run.py download.intphys2 download.splits=[Debug]

# limit SSv2 subset size
python run.py init.ssv2 split.max_samples_per_class=50
```

## LTX-Video Notes
- The `ltx_video` adapter extracts deterministic features from LTX VAE encoder stages (not denoising trajectories).
- Default Hydra variant: `ltxv_13b_0_9_8_dev` (`Lightricks/LTX-Video-0.9.8-dev`).
- First pull/smoke command:

```bash
python experiments/smoke_ltx_video.py --device cuda
```

- On Snellius, use:

```bash
sbatch jobs/setup/ltx_smoke.sh
```

## Main Config Keys (`configs/mvp.yaml`)
- `split.*`: split location and ratios
- `backbone.*`: adapter name and kwargs
- `feature_cache.*`: cache location/content/re-extraction behavior
- `linear_probe.*`: probe training/eval settings and output locations
- `decode.*`: frame sampling/resizing settings
- `predictor.*`: predictor mode for `eval.mvp`

## Main Config Keys (`configs/intphys2.yaml`)
- `download.*`: HuggingFace repo, splits to download, local storage directory
- `split.*`: split artifact location
- `backbone.*`: adapter name and kwargs
- `feature_cache.*`: cache location/content/re-extraction behavior
- `linear_probe.*`: probe training/eval settings and output locations
- `decode.*`: frame sampling/resizing settings
- `predictor.*`: predictor mode for `eval.intphys2`

## Main Config Keys (`configs/ssv2.yaml`)
- `split.*`: split artifact location and `max_samples_per_class` subsetting
- `backbone.*`: adapter name and kwargs
- `feature_cache.*`: cache location/content/re-extraction behavior
- `linear_probe.*`: probe training/eval settings and output locations
- `decode.*`: frame sampling/resizing settings
- `predictor.*`: predictor mode for `eval.ssv2`
