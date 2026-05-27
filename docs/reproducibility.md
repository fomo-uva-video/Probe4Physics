# Reproducibility

This repository is organized around reproducible split generation, deterministic
cache identities, and explicit runtime artifacts.

## Path Policy

Tracked configs use portable repo-relative defaults:

- `data/annotations/...`
- `data/videos/...`
- `data/splits/...`
- `artifacts/features/...`
- `artifacts/probes/...`
- `artifacts/results/...`

If you need different locations, override them on the CLI or use the
environment-aware wrappers under `ops/hpc/`.

## Determinism

- dataset split configs carry explicit seeds
- feature caches are keyed by a config-derived signature
- probe training supports stricter deterministic behavior with
  `probe.deterministic=true`
- `train_eval.probe.*` sweeps reuse the same artifact naming contract as manual
  `train.probe.*` plus `eval.probe.*`

Deterministic training can be slower and may disable some backend optimizations.

## Cache Semantics

- `exp.run` skips extraction when a valid cache already exists
- set `feature_cache.force_reextract=true` to ignore a compatible cache
- extraction resumes are controlled by `feature_cache.resume_enabled` and
  `feature_cache.resume_strict`
- baseline commands namespace their caches automatically through `baseline_tag`

## Health Checks

Use:

```bash
python run.py health
python run.py health.layers
python run.py health.features
```

`health.features` validates the same portable feature roots declared in tracked
configs. It does not rely on hidden personal scratch defaults.

## Known Limits

- datasets and pretrained checkpoints are not vendored into the repository
- SSv2 still depends on the official annotation release
- some backbones require substantial GPU memory even though the command surface
  is the same
- HPC wrappers remain optional operational tooling, not the main reproducibility
  story
