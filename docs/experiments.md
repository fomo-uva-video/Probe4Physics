# Experiments

This document is the runtime command reference for local or generic-cluster
execution. The public command surface remains stable under `run.py`.

## Core Commands

```bash
python run.py help
python run.py exp.list
python run.py exp.run name=<experiment_id>
```

Dataset command families:

- `init.mvp`, `init.intphys2`, `init.ssv2`
- `extract.mvp`, `extract.intphys2`, `extract.ssv2`
- `train.probe.mvp`, `train.probe.intphys2`, `train.probe.ssv2`
- `train_eval.probe.mvp`, `train_eval.probe.intphys2`, `train_eval.probe.ssv2`
- `eval.probe.mvp`, `eval.probe.intphys2`, `eval.probe.ssv2`
- `eval.mvp`, `eval.intphys2`, `eval.ssv2`

Additional commands:

- `download.intphys2`
- `health`
- `health.layers`
- `health.features`

## Standard Pipeline

The standard flow is:

1. `init.<dataset>`
2. `extract.<dataset>`
3. `train.probe.<dataset>` or `train_eval.probe.<dataset>`
4. `eval.probe.<dataset>` if training and eval are split

Examples:

```bash
python run.py init.mvp
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train.probe.mvp
python run.py eval.probe.mvp
```

```bash
python run.py download.intphys2
python run.py init.intphys2
python run.py extract.intphys2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train_eval.probe.intphys2
```

## Experiment Recipes

List recipes:

```bash
python run.py exp.list
```

Run a recipe:

```bash
python run.py exp.run name=mvp.jepa_v1.probe backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=intphys2.ltx_video.probe backbone.kwargs.device=cuda
python run.py exp.run name=ssv2.videomae.probe backbone.kwargs.device=cuda
```

Extraction is skipped automatically when a compatible cache already exists. To
force a refresh:

```bash
python run.py exp.run name=mvp.jepa_v1.probe feature_cache.force_reextract=true backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

## Common Overrides

Hydra-style overrides apply to every command.

```bash
python run.py extract.mvp feature_cache.split_names=[train] backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train.probe.mvp probe.feature_view=tokens_mean
python run.py train.probe.mvp probe.wandb.enabled=true probe.wandb.project=probe4physics
python run.py train_eval.probe.mvp probe.layers=[8,16,24,last]
python run.py eval.probe.intphys2 probe.checkpoint_path=/absolute/path/to/probe_best.pt
```

Portable tracked defaults should be overridden in one of two ways:

- local/manual runs: pass CLI overrides directly
- cluster runs: set environment variables consumed by the `ops/hpc/` wrappers

## Control Baselines

The control-baseline command families remain available:

- `extract.mvp.single_frame`
- `eval.probe.mvp.single_frame`
- `extract.mvp.displacement`
- `eval.probe.mvp.displacement`
- `extract.intphys2.single_frame`
- `eval.probe.intphys2.single_frame`
- `extract.intphys2.displacement`
- `eval.probe.intphys2.displacement`

These commands reuse the same train/eval contract while namespacing caches by
baseline tag.
