# Probe4Physics MVP Eval

MVP-only evaluation with:
- strict official MVP scoring integration (no local fallback)
- deterministic sample selection for `intuitive_physics` plausibility yes/no
- always-on selection audit artifacts by default
- automatic annotation download on first run when file is missing

## Run

```bash
python run.py
```

Hydra overrides example:

```bash
python run.py annotation_file=tests/fixtures/mvp_selection_fixture.jsonl videos_root=/tmp output_subdir=manual_strict_selection predictor.mode=oracle
```

You can also call the explicit command (future-proof style):

```bash
python run.py eval.mvp annotation_file=tests/fixtures/mvp_selection_fixture.jsonl predictor.mode=oracle
```

On first run, if `annotation_file` does not exist, the runner auto-downloads from
`facebook/minimal_video_pairs` using `annotations.*` config values.

## Outputs per run
- `metrics.json`
- `predictions.csv`
- `summary.md`
- `run_config.snapshot.yaml`
- `provenance.json`
- `selection_kept.csv` (default on)
- `selection_dropped.csv` (default on)
- `selection_report.json` (default on)

## Selection defaults
In `configs/mvp.yaml`:
- `selection.enabled: true`
- `selection.subset: intuitive_physics`
- `selection.plausibility_only: true`
- `selection.require_binary_yes_no: true`
- `selection.drop_incomplete_pairs: true`
- `selection.artifacts.enabled: true`

## Environment

```bash
conda env create -f environment.yml
conda activate probe4physics
```
