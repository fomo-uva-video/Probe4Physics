# Repository Guidelines

## Project Structure & Module Organization
`run.py` is the main launcher; keep new top-level workflows reachable from it. Core benchmark logic lives in `benchmarks/` (`mvp`, `intphys2`, `ssv2`), frozen backbone adapters in `models/`, probe implementations in `probes/`, and extraction/train orchestration in `training/` with `training/run_probe.py` as the probe entrypoint. Shared defaults belong in `configs/*.yaml`; `experiments/` holds recipe-style command bundles. Cluster wrappers stay under `jobs/extract/`, `jobs/train/`, and `jobs/init/`. Tests live in `tests/`, and runtime outputs should stay in `artifacts/`, `logs/`, `output/`, or scratch storage, not in tracked source trees.

## Build, Test, and Development Commands
Create the environment with `conda env create -f environment.yml` and activate `probe4physics`. Use `python run.py help` to inspect commands and `python run.py exp.list` to list experiment recipes. Common local checks:

```bash
python run.py health.features
python run.py train_eval.probe.mvp
python -m unittest discover -s tests -q
python -m unittest tests.test_run_commands -q
```

For HPC runs, prefer the existing wrappers such as `jobs/extract/mvp/run_extract.sh` or `jobs/train/intphys2/linear/run_train.sh` instead of inventing ad hoc submission scripts.

## Coding Style & Naming Conventions
Target Python 3.11, use 4-space indentation, and follow the existing typed style (`from __future__ import annotations`, `Path`, dataclasses, explicit `dict[str, Any]` shapes). Use `snake_case` for modules/functions/CLI keys, `PascalCase` for classes, and keep dataset-specific logic inside the matching benchmark package. Put shared training defaults in Hydra config, not in Slurm wrappers; wrappers should only add cluster/runtime overrides.

## Testing Guidelines
Tests use the standard `unittest` runner. Name new files `tests/test_<area>.py` and keep them CPU-safe and fixture-driven when possible. Add targeted coverage for new commands, config validation, and artifact semantics before relying on long extraction or training jobs.

## Commit & Pull Request Guidelines
Recent history uses short, direct subjects such as `updated jobs` or `csv generation for each run`. Keep commit titles imperative and specific to the touched area, for example `add ssv2 probe cache validation`. PRs should state the affected dataset/backbone/probe path, list the exact commands run, and note any artifact or log paths used for verification. Do not commit datasets, checkpoints, feature caches, `wandb/`, or changes inside `third_party/` unless the task explicitly requires it.
