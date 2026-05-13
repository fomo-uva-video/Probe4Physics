# Temporal Baseline Snellius Jobs

This folder contains Slurm jobs for the temporal control baselines on `mvp` and
`intphys2`.

The supported baselines are:

- `single_frame`: repeat one deterministic frame from the canonical 16-frame clip.
- `displacement`: apply a deterministic non-zero temporal roll inside the same 16-frame clip.

Both baseline extractors are enforced as test-only in the Python code. Even if a
job passes a wider split config, the command rewrites it to `feature_cache.split_names=[test]`.

## Submit Everything

Smoke extraction for all datasets, baselines, and backbones:

```bash
MODE=smoke ./jobs/baseline/submit_extract_all.sh
```

Full test-split extraction:

```bash
MODE=full ./jobs/baseline/submit_extract_all.sh
```

Baseline evaluation:

```bash
./jobs/baseline/submit_eval_all.sh
```

The eval jobs use normal probe checkpoints. Train the normal probes first; do
not train on baseline features.

If your normal training was a layer sweep, set `PROBE_LAYER` to the layer you
want to evaluate, for example:

```bash
PROBE_LAYER=32 ./jobs/baseline/submit_eval_all.sh
```

## Useful Filters

```bash
ONLY_DATASET=intphys2 MODE=smoke ./jobs/baseline/submit_extract_all.sh
ONLY_BASELINE=single_frame ./jobs/baseline/submit_eval_all.sh
ONLY_BACKBONE=jepa_v1 DRY_RUN=true ./jobs/baseline/submit_extract_all.sh
```

## Single Job Examples

```bash
DATASET_NAME=mvp BASELINE_NAME=single_frame BACKBONE_NAME=jepa_v1 BACKBONE_VARIANT=vith16_384 MODE=smoke sbatch jobs/baseline/extract.sh
DATASET_NAME=intphys2 BASELINE_NAME=displacement BACKBONE_NAME=videomae BACKBONE_VARIANT=vit_huge_16_224 sbatch jobs/baseline/eval.sh
```

If automatic checkpoint discovery is ambiguous, pass the checkpoint explicitly:

```bash
PROBE_CHECKPOINT_PATH=/path/to/probe_best.pt DATASET_NAME=mvp BASELINE_NAME=single_frame BACKBONE_NAME=jepa_v1 BACKBONE_VARIANT=vith16_384 sbatch jobs/baseline/eval.sh
```

The all-backbone submitters use the same backbone variants as the existing
linear probe jobs:

- `jepa_v1`: `vith16_384`
- `jepa_v2`: `vitg_384`
- `jepa_v2_1`: `vitG_384`
- `videomae`: `vit_huge_16_224`
- `videomae_v2`: `vit_giant_16_224`
- `ltx_video`: `ltxv_13b_0_9_8_dev`
