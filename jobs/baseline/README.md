# Temporal Baseline Snellius Jobs

This folder contains explicit per-backbone Slurm jobs for temporal baseline
evaluation on `mvp` and `intphys2`.

The layout is intentionally the same style as `jobs/train`: choose the baseline,
then the dataset, then submit the backbone script you want.

```text
jobs/baseline/
  single_frame/
    mvp/
      extract/
      eval/
    intphys2/
      extract/
      eval/
  frame_shuffling/
    mvp/
      extract/
      eval/
    intphys2/
      extract/
      eval/
```

Supported baselines:

- `single_frame`: repeats one deterministic random frame from the canonical
  16-frame extracted clip.
- `frame_shuffling`: runs the displacement baseline, a deterministic non-zero
  temporal roll inside the same 16-frame extracted clip.

Both extractors are test-only in the Python code. Even if someone passes a wider
split config, the baseline command rewrites it to `feature_cache.split_names=[test]`.
These jobs do not train probes on baseline features: eval uses normal probe
checkpoints trained on the original, non-baseline features.

## Submitting Jobs

Extraction jobs run on `gpu_a100` because they execute the frozen video
backbone. Eval jobs run on `rome` because they only load cached features,
normal probe checkpoints, and compute metrics.

Extraction keeps `feature_cache.layer_ids` empty, matching the original
extraction style: the adapter extracts its canonical layers automatically.
Eval wrappers expose the same canonical layers through `PROBE_LAYERS`.

```bash
sbatch jobs/baseline/single_frame/mvp/extract/jepa_v1.sh
sbatch jobs/baseline/single_frame/mvp/eval/jepa_v1.sh
```

For a quick extraction smoke test:

```bash
MODE=smoke sbatch jobs/baseline/frame_shuffling/intphys2/extract/videomae.sh
```

If automatic checkpoint discovery is ambiguous, pass the normal probe checkpoint
explicitly and evaluate a single layer:

```bash
PROBE_LAYER=32 PROBE_LAYERS=32 PROBE_CHECKPOINT_PATH=/path/to/probe_best.pt sbatch jobs/baseline/single_frame/mvp/eval/jepa_v1.sh
```

By default, eval jobs loop over the canonical layer set for the backbone and
resolve one normal `probe_best.pt` checkpoint per layer:

- `jepa_v1`: `8,16,24,32`
- `jepa_v2`: `10,20,30,40`
- `jepa_v2_1`: `12,24,38,48`
- `videomae`: `8,16,24,32`
- `videomae_v2`: `10,20,30,40`
- `ltx_video`: `1..40`

## Runtime Controls

- `MODE=smoke`: use `feature_cache.max_samples=2` unless `MAX_SAMPLES` is set.
- `MODE=full`: extract the full test split.
- `FEATURE_DIR=/path/to/cache`: override the feature cache location.
- `PROBE_LAYERS=8,16,24,32`: override the default eval layer loop.
- `PROBE_CHECKPOINT_PATH=/path/to/probe_best.pt`: skip checkpoint discovery for a single-layer eval.

The per-model scripts use these backbone variants:

- `jepa_v1`: `vith16_384`
- `jepa_v2`: `vitg_384`
- `jepa_v2_1`: `vitG_384`
- `videomae`: `vit_huge_16_224`
- `videomae_v2`: `vit_giant_16_224`
- `ltx_video`: `ltxv_13b_0_9_8_dev`
