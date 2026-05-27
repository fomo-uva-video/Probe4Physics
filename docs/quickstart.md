# Quickstart

This is the shortest path from a fresh clone to a first successful local run.

## 1. Create The Environment

```bash
conda env create -f environment.yml
conda activate probe4physics
git submodule update --init --recursive
git -C third_party/jepa_v1 checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
python run.py help
```

## 2. Prepare One Dataset

MVP expects:

- `data/annotations/mvp_full.jsonl`
- `data/videos/mvp/`

IntPhys2 can be downloaded by the repo:

```bash
python run.py download.intphys2
```

SSv2 expects:

- `data/annotations/ssv2/train.json`
- `data/annotations/ssv2/validation.json`
- `data/annotations/ssv2/labels.json`
- `data/videos/ssv2/`

## 3. Run The MVP Pipeline

```bash
python run.py init.mvp
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train_eval.probe.mvp
```

This writes:

- splits under `data/splits/mvp/`
- features under `artifacts/features/mvp/`
- probe artifacts under `artifacts/probes/mvp/`
- eval outputs under `artifacts/results/`

## 4. Run Sanity Checks

```bash
python run.py exp.list
python run.py health.features
python -m unittest tests.test_run_commands -q
```

## 5. Next Documents

- `docs/datasets.md` for benchmark-specific preparation
- `docs/experiments.md` for full command coverage
- `docs/artifacts.md` for output semantics
