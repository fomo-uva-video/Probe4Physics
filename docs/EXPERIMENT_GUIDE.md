# Probe4Physics Experiment Guide

This guide is a practical, end-to-end playbook for running experiments in this repository.

It covers:
- environment setup
- dataset checks and downloads
- model/backbone and checkpoint selection
- split initialization and feature extraction (cache creation)
- probe training and evaluation
- where every artifact is stored
- common failure modes and quick fixes

This guide assumes you run commands from the repository root.

---

## 1. Big Picture Pipeline

All experiments follow this sequence:

1. Prepare data and metadata
2. Initialize deterministic split artifacts (`init.*`)
3. Extract frozen backbone features (`extract.*`)
4. Train probe (`train.probe.*`)
5. Evaluate probe (`eval.probe.*`)

You can run each stage manually, or run full recipes with `exp.run`.

Main launcher:

```bash
python run.py <command> [hydra_overrides]
```

List available experiment recipes:

```bash
python run.py exp.list
```

---

## 2. Environment Setup

### Local or generic cluster

```bash
conda env create -f environment.yml
conda activate probe4physics
```

If the env already exists:

```bash
conda env update -n probe4physics -f environment.yml --prune
```

### Snellius (recommended project flow)

```bash
sbatch jobs/setup/setup_env.sh
```

Notes:
- The "mixing Conda and modules" banner is informational on Snellius.
- Keep the module setup exactly as in the job scripts.

---

## 3. One-Time Repository Setup

Initialize submodules:

```bash
git submodule update --init --recursive
```

Pin JEPA v1 submodule commit used by this project:

```bash
git -C third_party/jepa_v1 checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
```

---

## 4. Data Presence Checks and Downloads

## 4.1 MVP

### What must exist

- Annotation file (JSONL)
- Video root with actual video files
- Split manifest (created by `init.mvp`)

Default config keys are in `configs/mvp.yaml`.

### Quick presence checks

```bash
ls -l /scratch-shared/$USER/probe4physics/data/annotations/mvp_full.jsonl
ls -ld third_party/minimal_video_pairs/videos
find third_party/minimal_video_pairs/videos -type f | wc -l
```

### Download MVP videos

Official downloader (writes into `third_party/minimal_video_pairs/videos` by default):

```bash
cd third_party/minimal_video_pairs
make download_videos
```

If specific targets failed and you need to re-run one target:

```bash
cd third_party/minimal_video_pairs
rm -f .download.intphys
make .download.intphys
```

### Build deterministic MVP splits

```bash
python run.py init.mvp \
  annotation_file=/scratch-shared/$USER/probe4physics/data/annotations/mvp_full.jsonl \
  official_repo_root=$PWD/third_party/minimal_video_pairs \
  videos_root=$PWD/third_party/minimal_video_pairs/videos \
  split.dir=/scratch-shared/$USER/probe4physics/data/splits/mvp/full_60_20_20
```

---

## 4.2 IntPhys2

### Download

```bash
python run.py download.intphys2
```

Optional split subset:

```bash
python run.py download.intphys2 download.splits=[Debug]
```

### Initialize splits

```bash
python run.py init.intphys2
```

---

## 4.3 SSv2

Provide official annotation files first:
- `train.json`
- `validation.json`
- `labels.json`

Then initialize:

```bash
python run.py init.ssv2
```

---

## 5. Model Selection and Checkpoint Readiness

Backbone variants are defined in `configs/backbones.yaml`.

Supported adapters:
- `jepa_v1`
- `jepa_v2`
- `jepa_v2_1`
- `videomae`
- `videomae_v2`
- `ltx_video`

### JEPA v1 default

- Variant: `vitl16_224`
- Expected filename: `vitl16.pth.tar`
- Default checkpoint directory: `data/checkpoints/jepa_v1`

### Check checkpoint exists

```bash
ls -lh data/checkpoints/jepa_v1/vitl16.pth.tar
```

If missing, download official JEPA v1 checkpoint from the model zoo in:
- `third_party/jepa_v1/README.md`

### Override checkpoint and variant

Important Hydra note: when adding keys not explicitly present under `backbone.kwargs`, use `+` prefix.

Example:

```bash
python run.py extract.mvp \
  +backbone.kwargs.variant=vitl16_224 \
  +backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

### LTX-Video default

- Default variant: `ltxv_13b_0_9_8_distilled`
- Default HF model id: `Lightricks/LTX-Video-0.9.8-13B-distilled`

Quick smoke/pull:

```bash
python experiments/smoke_ltx_video.py --device cuda
```

Extraction example:

```bash
python run.py extract.mvp \
  backbone.name=ltx_video \
  backbone.kwargs.device=cuda
```

---

## 6. Split + Forward Pass + Cache Storage

Extraction command (MVP example):

```bash
python run.py extract.mvp \
  annotation_file=/scratch-shared/$USER/probe4physics/data/annotations/mvp_full.jsonl \
  official_repo_root=$PWD/third_party/minimal_video_pairs \
  videos_root=$PWD/third_party/minimal_video_pairs/videos \
  split.dir=/scratch-shared/$USER/probe4physics/data/splits/mvp/full_60_20_20 \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  feature_cache.split_names=[train,val,test] \
  feature_cache.include_pooled=true \
  feature_cache.include_tokens=true \
  +backbone.kwargs.checkpoint_path=$PWD/data/checkpoints/jepa_v1/vitl16.pth.tar
```

### Where extracted features are stored

Layout:

```text
feature_cache.dir/
  <backbone_or_backbone_variant>/
    <split_key>/
      <signature>/
        index.parquet
        features_pooled.pt
        features_tokens.pt
        manifest.json
```

### Cache reuse (important for speed)

If a valid cache already exists for the exact config signature, extraction is skipped automatically during `exp.run`.

To force re-extraction:

```bash
python run.py exp.run name=mvp.jepa_v1.probe feature_cache.force_reextract=true
```

---

## 7. Probe Selection, Training, and Evaluation

Current pipeline commands are wired for probes:
- `train.probe.<benchmark>`
- `eval.probe.<benchmark>`

These commands dispatch into `training/run_probe.py`, which owns probe
selection, checkpoint loading, train/eval orchestration, and optional Optuna
sweeps.

### Train (MVP)

```bash
python run.py train.probe.mvp \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  probe.output_dir=/scratch-shared/$USER/probe4physics/artifacts/probes \
  probe.epochs=30 \
  probe.batch_size=128 \
  probe.device=cuda \
  probe.wandb.enabled=true \
  probe.wandb.project=probe4physics
```

### Evaluate (MVP)

```bash
python run.py eval.probe.mvp \
  split_name=val \
  probe.device=cuda \
  probe.checkpoint_path=/scratch-shared/$USER/probe4physics/artifacts/probes/<run_subdir>/probe_best.pt \
  probe.eval_output_dir=/scratch-shared/$USER/probe4physics/artifacts/results
```

### Output locations

Train outputs:

```text
probe.output_dir/<timestamp_or_output_subdir>/
  probe_best.pt
  probe_last.pt
  train_summary.json
```

When `probe.wandb.enabled=true`, `train.probe.*` also logs per-epoch loss/accuracy and final run metadata to Weights & Biases. The run name defaults to `<experiment>/<benchmark>/<feature_view>/<output_subdir>`, and can be overridden with `probe.wandb.name=...`.

Optuna and W&B can be combined by enabling both:

```bash
python run.py train.probe.mvp \
  probe.optuna.enabled=true \
  probe.optuna.n_trials=10 \
  probe.wandb.enabled=true
```

Eval outputs:

```text
probe.eval_output_dir/<timestamp_or_eval_output_subdir>/
  probe_predictions.json
  probe_eval_summary.json
  metrics.json
  predictions.csv
  summary.md
  provenance.json
  run_config.snapshot.yaml
```

---

## 8. Full Experiment Recipes (`exp.run`)

### MVP full recipe

```bash
python run.py exp.run \
  name=mvp.jepa_v1.probe \
  annotation_file=/scratch-shared/$USER/probe4physics/data/annotations/mvp_full.jsonl \
  official_repo_root=$PWD/third_party/minimal_video_pairs \
  videos_root=$PWD/third_party/minimal_video_pairs/videos \
  split.dir=/scratch-shared/$USER/probe4physics/data/splits/mvp/full_60_20_20 \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  probe.output_dir=/scratch-shared/$USER/probe4physics/artifacts/probes \
  probe.eval_output_dir=/scratch-shared/$USER/probe4physics/artifacts/results \
  +backbone.kwargs.variant=vitl16_224 \
  +backbone.kwargs.checkpoint_path=$PWD/data/checkpoints/jepa_v1/vitl16.pth.tar
```

### MVP LTX full recipe

```bash
python run.py exp.run \
  name=mvp.ltx_video.probe \
  backbone.kwargs.device=cuda
```

### IntPhys2 full recipe

```bash
python run.py exp.run name=intphys2.jepa_v1.probe +backbone.kwargs.checkpoint_path=$PWD/data/checkpoints/jepa_v1/vitl16.pth.tar
```

### IntPhys2 LTX full recipe

```bash
python run.py exp.run name=intphys2.ltx_video.probe backbone.kwargs.device=cuda
```

### SSv2 full recipe

```bash
python run.py exp.run name=ssv2.jepa_v1.probe +backbone.kwargs.checkpoint_path=$PWD/data/checkpoints/jepa_v1/vitl16.pth.tar
```

### SSv2 LTX full recipe

```bash
python run.py exp.run name=ssv2.ltx_video.probe backbone.kwargs.device=cuda
```

---

## 9. Running on Snellius (SLURM)

Recommended workflow scripts in `jobs/setup`:
- `setup_env.sh`
- `download_mvp_2.sh`
- `setup_mvp_data.sh`
- `mvp_linear_test.sh` (smoke)
- `ltx_smoke.sh` (LTX smoke + targeted tests)

Submit smoke test:

```bash
PROJECT_ROOT=/gpfs/home3/$USER/Probe4Physics sbatch /gpfs/home3/$USER/Probe4Physics/jobs/setup/mvp_linear_test.sh
```

Submit LTX smoke test:

```bash
PROJECT_ROOT=/gpfs/home3/$USER/Probe4Physics sbatch /gpfs/home3/$USER/Probe4Physics/jobs/setup/ltx_smoke.sh
```

---

## 10. Troubleshooting Cookbook

## 10.1 "run.py not found in REPO_ROOT"

Cause: job submitted from non-root folder without `PROJECT_ROOT`.

Fix:

```bash
PROJECT_ROOT=/gpfs/home3/$USER/Probe4Physics sbatch /gpfs/home3/$USER/Probe4Physics/jobs/setup/mvp_linear_test.sh
```

## 10.2 Missing video file during extraction

Error pattern: `Resolved video path does not exist for extraction`.

Fix:
- verify `videos_root`
- verify specific missing file exists under that root
- rerun missing download target if needed (`make .download.intphys`, etc.)

## 10.3 "No probe checkpoint found automatically"

Cause: eval auto-discovery searches timestamped folders, but your run used a fixed output subdir.

Fix: pass checkpoint explicitly:

```bash
probe.checkpoint_path=/path/to/probe_best.pt
```

## 10.4 FutureWarning: `torch.load(... weights_only=False)`

Current status: warning only; run still succeeds.

Preferred code update for future compatibility:
- use `weights_only=True` when loading trusted tensors/checkpoints
- keep backward-compatible fallback for older torch versions

## 10.5 `set -u` + conda activation failures (MKL variable)

If shell scripts use `set -u`, wrap activation:

```bash
set +u
conda activate <env>
set -u
```

---

## 11. Fast Iteration Strategy (Recommended)

For repeated experiments:

1. Keep one stable split dir per benchmark
2. Extract features once for your chosen backbone/frames/split_names
3. Reuse cache while sweeping probe hyperparameters
4. Only force re-extract if you changed anything affecting signature:
   - backbone or variant
   - checkpoint path/adapter kwargs
   - decode settings
   - split_names
   - include_pooled/include_tokens

This saves the most GPU time.

---

## 12. Minimal End-to-End MVP Example

```bash
# 1) Init splits
python run.py init.mvp \
  annotation_file=/scratch-shared/$USER/probe4physics/data/annotations/mvp_full.jsonl \
  official_repo_root=$PWD/third_party/minimal_video_pairs \
  videos_root=$PWD/third_party/minimal_video_pairs/videos \
  split.dir=/scratch-shared/$USER/probe4physics/data/splits/mvp/full_60_20_20

# 2) Extract (first run only)
python run.py extract.mvp \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  feature_cache.split_names=[train,val] \
  feature_cache.include_pooled=true \
  feature_cache.include_tokens=false \
  +backbone.kwargs.checkpoint_path=$PWD/data/checkpoints/jepa_v1/vitl16.pth.tar

# 3) Train probe
python run.py train.probe.mvp \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  feature_cache.split_names=[train,val] \
  probe.output_dir=/scratch-shared/$USER/probe4physics/artifacts/probes \
  probe.output_subdir=my_mvp_run \
  probe.epochs=5 \
  probe.device=cuda

# 4) Eval probe
python run.py eval.probe.mvp \
  split_name=val \
  feature_cache.dir=/scratch-shared/$USER/probe4physics/artifacts/features/mvp \
  feature_cache.split_names=[train,val] \
  probe.checkpoint_path=/scratch-shared/$USER/probe4physics/artifacts/probes/my_mvp_run/probe_best.pt \
  probe.eval_output_dir=/scratch-shared/$USER/probe4physics/artifacts/results \
  probe.eval_output_subdir=my_mvp_run \
  probe.device=cuda
```

---

If you want, this guide can be extended with a dedicated section per benchmark that includes copy-paste command blocks customized for your exact Snellius paths.
