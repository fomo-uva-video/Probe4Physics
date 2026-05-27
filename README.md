# Probe4Physics

Probe4Physics is a benchmark-centric reproducibility package for probing frozen
video backbones on intuitive-physics and temporal-reasoning tasks.

The repository keeps one stable command surface through `run.py`:

- `python run.py help`
- `python run.py exp.list`
- `python run.py init.<dataset>`
- `python run.py extract.<dataset>`
- `python run.py train.probe.<dataset>`
- `python run.py train_eval.probe.<dataset>`
- `python run.py eval.probe.<dataset>`
- `python run.py eval.<dataset>`

The primary workflow is local or generic multi-GPU execution from the repository
root. Cluster wrappers are preserved under `ops/hpc/`, but they are secondary to
the documented runtime path.

## Runtime Map

- `run.py`: top-level launcher
- `benchmarks/`: dataset logic for MVP, IntPhys2, and SSv2
- `models/`: frozen backbone adapters
- `probes/`: linear, MLP, and temporal attentive probes
- `training/`: extraction plus probe train/eval orchestration
- `configs/`: tracked runtime defaults
- `experiments/`: recipe registry for `exp.run`
- `tests/`: CPU-safe regression coverage
- `docs/`: runtime documentation only
- `ops/hpc/`: optional cluster wrappers and notes
- `paper/archive/`: manuscript, planning, and historical records

Tracked defaults are portable:

- datasets and annotations live under `data/`
- caches, probes, metrics, and reports live under `artifacts/`
- machine-specific paths should come from CLI overrides or `ops/hpc/` wrappers

## Environment Setup

```bash
conda env create -f environment.yml
conda activate probe4physics
```

If the environment already exists:

```bash
conda env update -n probe4physics -f environment.yml --prune
```

Initialize submodules:

```bash
git submodule update --init --recursive
git -C third_party/jepa_v1 checkout 51c59d518fc63c08464af6de585f78ac0c7ed4d5
```

Inspect the command surface:

```bash
python run.py help
python run.py exp.list
```

## Data And Checkpoints

Tracked configs assume this local layout:

```text
data/
  annotations/
  videos/
  splits/
artifacts/
  features/
  probes/
  results/
```

Checkpoint paths are intentionally not committed into tracked YAMLs. Pass them
on the CLI when needed, for example:

```bash
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

Dataset-specific preparation details live in [docs/datasets.md](docs/datasets.md).

## End-To-End Reproduction Flow

Every experiment family follows the same stages:

1. Prepare datasets and checkpoints.
2. Initialize deterministic split artifacts with `init.*`.
3. Extract frozen features with `extract.*`.
4. Train probes with `train.probe.*` or `train_eval.probe.*`.
5. Evaluate predictions with `eval.probe.*` or the built-in `eval.*` modes.

The shortest MVP path is:

```bash
python run.py init.mvp
python run.py extract.mvp backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train_eval.probe.mvp
```

The shortest IntPhys2 path is:

```bash
python run.py download.intphys2
python run.py init.intphys2
python run.py extract.intphys2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train_eval.probe.intphys2
```

The shortest SSv2 path is:

```bash
python run.py init.ssv2
python run.py extract.ssv2 backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py train_eval.probe.ssv2
```

More examples, including control baselines and recipes, live in
[docs/experiments.md](docs/experiments.md).

## Experiment Recipes

List registered recipes:

```bash
python run.py exp.list
```

Run a full recipe:

```bash
python run.py exp.run name=mvp.jepa_v1.probe backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=intphys2.jepa_v1.probe backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
python run.py exp.run name=ssv2.jepa_v1.probe backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

If a compatible feature cache already exists, `exp.run` skips extraction unless
you force a refresh:

```bash
python run.py exp.run name=mvp.jepa_v1.probe feature_cache.force_reextract=true backbone.kwargs.checkpoint_path=/absolute/path/to/vitl16.pth.tar
```

## Outputs

The main runtime outputs are:

- `data/splits/<dataset>/...`: initialized split artifacts
- `artifacts/features/<dataset>/...`: frozen feature caches
- `artifacts/probes/<dataset>/...`: train outputs and checkpoints
- `artifacts/results/<dataset>/...`: evaluation predictions and metrics

Artifact semantics and naming are documented in [docs/artifacts.md](docs/artifacts.md).

## Verification

Minimum repo-level checks:

```bash
python run.py help
python run.py exp.list
python run.py health.features
python -m unittest discover -s tests -q
```

Reproducibility notes, cache semantics, and deterministic settings are described
in [docs/reproducibility.md](docs/reproducibility.md).

## Documentation Map

- [docs/quickstart.md](docs/quickstart.md): shortest successful run from clone
- [docs/datasets.md](docs/datasets.md): dataset preparation and prerequisites
- [docs/experiments.md](docs/experiments.md): command families and overrides
- [docs/artifacts.md](docs/artifacts.md): output layout and expected files
- [docs/reproducibility.md](docs/reproducibility.md): seeds, caches, and limits
- [ops/hpc/README.md](ops/hpc/README.md): cluster wrappers and submission notes

## HPC And Archive Material

Cluster wrappers are intentionally outside the primary runtime path. Use
`ops/hpc/` only when you need scheduler-backed execution of the same workflows
documented above.

Paper planning notes, PDFs, figure assets, and exported result tables are kept
under `paper/archive/` so they remain tracked without cluttering runtime docs.
