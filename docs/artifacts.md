# Artifacts

This repository separates tracked source from generated runtime outputs.

## Runtime Roots

Tracked defaults write to:

- `data/splits/<dataset>/...`
- `artifacts/features/<dataset>/...`
- `artifacts/probes/<dataset>/...`
- `artifacts/results/<dataset>/...`

These trees are ignored by git and are safe to regenerate.

## Split Artifacts

Initialization writes deterministic split metadata:

- MVP: `split_pairs.parquet`, `manifest.json`, selection CSV/JSON reports
- IntPhys2: `split_scenes.parquet`, `manifest.json`
- SSv2: `split_clips.parquet`, `manifest.json`

## Feature Caches

Feature caches live under:

```text
artifacts/features/<dataset>/<backbone>/<split-key>/<signature>/
```

Typical contents:

- `manifest.json`
- `index.parquet`
- `features_pooled.pt`
- `features_tokens.pt`
- `.resume/` metadata when resume mode is enabled

Extraction warnings are written next to the dataset feature root, for example:

- `artifacts/features/mvp/extract_warnings.log`
- `artifacts/features/intphys2/extract_warnings.log`

## Probe Training Artifacts

Probe training writes under `probe.output_dir`, which defaults to
`artifacts/probes/<dataset>/...`.

Typical contents:

- `probe_best.pt`
- `probe_last.pt`
- `train_summary.json`
- `train_eval_summary.json` for combined sweeps

## Evaluation Artifacts

Probe evaluation writes under `probe.eval_output_dir`, which defaults to
`artifacts/results/<dataset>/...`.

Typical contents:

- `probe_predictions.json`
- `probe_eval_summary.json`
- benchmark-specific metrics and export files such as `metrics.json`,
  `predictions.csv`, `summary.md`, and `provenance.json`

## Non-Runtime Tracked Material

Paper exports, manuscripts, and planning records live under `paper/archive/`.
They are tracked, but they are not part of the main runtime artifact contract.
