# LTX Video Probe Slots

This note describes the default LTX probe-slot layout used by the repo when `backbone.name=ltx_video` and `feature_cache.layer_ids=[]`.

## What One LTX Slot Is

For LTX, a "layer" in probe training is not a plain transformer depth. The adapter flattens a 2D grid:

- noise levels: `1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1`
- relative depths: `0.25, 0.5, 0.75, 1.0`
- resolved transformer depths for the 48-block LTX transformer: `12, 24, 36, 48`

Flattening order is `noise-major, depth-minor`:

- slots `1..4` are noise `1.0` at depths `12,24,36,48`
- slots `5..8` are noise `0.9` at depths `12,24,36,48`
- ...
- slots `37..40` are noise `0.1` at depths `12,24,36,48`

The adapter does one transformer forward per noise level and reads the requested block activations from hooks inside that forward. So the default 40-slot cache means:

- `10` transformer forwards per video clip
- `4` depth readouts from each forward
- `40` saved slot ids per sample

## What Is Saved Per Slot

Each slot id is stored under the integer key in the feature payloads.

- pooled view: one tensor per slot with shape `[N, D]`
  - for the current IntPhys2 distilled extraction, `D = 4096`
- token view: one tensor per slot with shape `[N, T, D]`
  - for the current IntPhys2 distilled extraction chunks, `T = 147` and `D = 4096`

So for the current IntPhys2 LTX distilled run, a single slot contains either:

- pooled features: `[N, 4096]`
- token features: `[N, 147, 4096]`

`linear` and `mlp` probes consume the pooled view by default.
`temporal_attn` consumes the token view by default.

## Slot Mapping

| slot_id | noise_fraction | depth_layer_id | meaning |
| ---: | ---: | ---: | --- |
| 1 | 1.0 | 12 | transformer block 12 at noise fraction 1.0 |
| 2 | 1.0 | 24 | transformer block 24 at noise fraction 1.0 |
| 3 | 1.0 | 36 | transformer block 36 at noise fraction 1.0 |
| 4 | 1.0 | 48 | transformer block 48 at noise fraction 1.0 |
| 5 | 0.9 | 12 | transformer block 12 at noise fraction 0.9 |
| 6 | 0.9 | 24 | transformer block 24 at noise fraction 0.9 |
| 7 | 0.9 | 36 | transformer block 36 at noise fraction 0.9 |
| 8 | 0.9 | 48 | transformer block 48 at noise fraction 0.9 |
| 9 | 0.8 | 12 | transformer block 12 at noise fraction 0.8 |
| 10 | 0.8 | 24 | transformer block 24 at noise fraction 0.8 |
| 11 | 0.8 | 36 | transformer block 36 at noise fraction 0.8 |
| 12 | 0.8 | 48 | transformer block 48 at noise fraction 0.8 |
| 13 | 0.7 | 12 | transformer block 12 at noise fraction 0.7 |
| 14 | 0.7 | 24 | transformer block 24 at noise fraction 0.7 |
| 15 | 0.7 | 36 | transformer block 36 at noise fraction 0.7 |
| 16 | 0.7 | 48 | transformer block 48 at noise fraction 0.7 |
| 17 | 0.6 | 12 | transformer block 12 at noise fraction 0.6 |
| 18 | 0.6 | 24 | transformer block 24 at noise fraction 0.6 |
| 19 | 0.6 | 36 | transformer block 36 at noise fraction 0.6 |
| 20 | 0.6 | 48 | transformer block 48 at noise fraction 0.6 |
| 21 | 0.5 | 12 | transformer block 12 at noise fraction 0.5 |
| 22 | 0.5 | 24 | transformer block 24 at noise fraction 0.5 |
| 23 | 0.5 | 36 | transformer block 36 at noise fraction 0.5 |
| 24 | 0.5 | 48 | transformer block 48 at noise fraction 0.5 |
| 25 | 0.4 | 12 | transformer block 12 at noise fraction 0.4 |
| 26 | 0.4 | 24 | transformer block 24 at noise fraction 0.4 |
| 27 | 0.4 | 36 | transformer block 36 at noise fraction 0.4 |
| 28 | 0.4 | 48 | transformer block 48 at noise fraction 0.4 |
| 29 | 0.3 | 12 | transformer block 12 at noise fraction 0.3 |
| 30 | 0.3 | 24 | transformer block 24 at noise fraction 0.3 |
| 31 | 0.3 | 36 | transformer block 36 at noise fraction 0.3 |
| 32 | 0.3 | 48 | transformer block 48 at noise fraction 0.3 |
| 33 | 0.2 | 12 | transformer block 12 at noise fraction 0.2 |
| 34 | 0.2 | 24 | transformer block 24 at noise fraction 0.2 |
| 35 | 0.2 | 36 | transformer block 36 at noise fraction 0.2 |
| 36 | 0.2 | 48 | transformer block 48 at noise fraction 0.2 |
| 37 | 0.1 | 12 | transformer block 12 at noise fraction 0.1 |
| 38 | 0.1 | 24 | transformer block 24 at noise fraction 0.1 |
| 39 | 0.1 | 36 | transformer block 36 at noise fraction 0.1 |
| 40 | 0.1 | 48 | transformer block 48 at noise fraction 0.1 |

## How Probe Training Selects One Pair

`train_eval.probe.*` uses this rule:

- if `probe.layers` is non-empty, it runs one train+eval subrun per listed slot id
- otherwise it falls back to `probe.layer`

For a single slot smoke run, pass both:

- `probe.layer=<slot_id>`
- `probe.layers=[<slot_id>]`

That keeps the run summary, W&B group suffix, and Optuna study name aligned to the same slot.

When `train_eval.probe.intphys2` runs a sweep, it creates:

- `layer_<slot_id>/train`
- `layer_<slot_id>/eval`

under the run root, and appends `_layer_<slot_id>` to the W&B group and Optuna study name.

## Recommended Single-Slot Commands

Fastest smoke check, pooled features only:

```bash
cd /gpfs/home3/scur0511/Probe4Physics/jobs/train/intphys2/linear
sbatch ltx_video.sh   probe.layer=17   'probe.layers=[17]'   probe.optuna.enabled=false   probe.epochs=20   probe.wandb.enabled=true   probe.wandb.mode=online   probe.output_subdir=ltx_slot_17_linear_smoke
```

This trains on slot `17`, which means `(noise_fraction=0.6, depth_layer_id=12)`.

Token-path smoke check for the attentive probe:

```bash
cd /gpfs/home3/scur0511/Probe4Physics/jobs/train/intphys2/temporal_attn
sbatch ltx_video.sh   probe.layer=17   'probe.layers=[17]'   probe.optuna.enabled=false   probe.epochs=5   probe.batch_size=1   probe.eval_batch_size=1   probe.wandb.enabled=true   probe.wandb.mode=online   probe.output_subdir=ltx_slot_17_attn_smoke
```

## Important Constraint

Probe training loads only finalized caches. For IntPhys2 this means the cache root must contain:

- `manifest.json`
- `index.parquet`
- `features_pooled.pt`
- optionally `features_tokens.pt` when token views are requested

Chunk-only `.resume/chunks/...` artifacts are not enough for `train_eval.probe.intphys2`.
