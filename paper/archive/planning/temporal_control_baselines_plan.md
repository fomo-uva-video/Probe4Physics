# Temporal Control Baselines Plan

## Summary

Implement proposal-style temporal control baselines at feature extraction time for `MVP` and `IntPhys2`, without changing probe training or evaluation code.

The first version will add one new config field:

- `decode.temporal_control=none`
- `decode.temporal_control=single_frame_repeat`
- `decode.temporal_control=time_shuffled`

This will produce separate feature caches for:

- normal clips
- single-frame control clips
- time-shuffled control clips

The controls will be applied after the current normal frame sampling and before backbone feature extraction.

## Locked Decisions

- Scope for v1: `MVP` and `IntPhys2` only.
- `SSv2` is out of scope for the first patch.
- `single_frame_repeat` means:
  - sample the normal clip first
  - choose the middle sampled frame
  - repeat that frame across all `T` positions
- `time_shuffled` means:
  - sample the normal clip first
  - apply a deterministic per-video permutation
  - the same video gets the same permutation across all backbones
- Shuffle identity key: `video_ref`
- Shuffle policy: deterministic derangement
- Do not vary shuffle by backbone.
- Do not implement the controls from existing cached features.
- Do not change `training/run_probe.py`, probe classes, or eval code.
- First surface is Hydra-only, not wrapper-polished.

## Why This Needs Extraction-Side Code

The current repo only supports one real temporal decode behavior: uniform frame sampling.

The config already records decode settings in the cache signature, but the actual decode path still always:

- decodes the video
- samples frames uniformly
- keeps those sampled frames in order

That means proposal controls must be added in the extraction path, not in probe training, and not by relabeling caches.

Applying the controls before `adapter.extract(...)` is the correct seam because:

- pooled features have already lost temporal order
- `tokens_mean` has already averaged over token positions
- token caches contain backbone outputs, not raw frame sequences
- the proposal baselines are supposed to change temporal input, not post-hoc feature tensors

## Public Interface Changes

Add one new config field to both benchmark extraction configs:

- `decode.temporal_control`

Allowed values:

- `none`
- `single_frame_repeat`
- `time_shuffled`

Default:

- `none`

Example intended usage:

```bash
python run.py extract.mvp decode.temporal_control=none
python run.py extract.mvp decode.temporal_control=single_frame_repeat
python run.py extract.mvp decode.temporal_control=time_shuffled

python run.py extract.intphys2 decode.temporal_control=none
python run.py extract.intphys2 decode.temporal_control=single_frame_repeat
python run.py extract.intphys2 decode.temporal_control=time_shuffled
```

## Behavioral Specification

### Normal mode

`decode.temporal_control=none`

- Keep current behavior exactly unchanged.
- Decode frames as today.
- Uniformly sample `num_frames`.
- Preserve sampled order.
- Forward to the backbone.

### Single-frame mode

`decode.temporal_control=single_frame_repeat`

- Decode the normal clip exactly as today.
- Build the sampled frame list with the existing uniform sampler.
- Select the middle sampled frame using:
  - `middle_index = len(sampled_frames) // 2`
- Repeat that one sampled frame across all `T` positions.
- Preserve output tensor shape exactly as in normal mode.

For a standard 16-frame clip:

- sampled index `8` is used
- the output still has `T=16`

### Time-shuffled mode

`decode.temporal_control=time_shuffled`

- Decode the normal clip exactly as today.
- Build the sampled frame list with the existing uniform sampler.
- Compute one deterministic permutation from:
  - `video_ref`
  - `num_frames`
- Apply that permutation to the sampled frame order.
- Preserve output tensor shape exactly as in normal mode.

Required properties:

- the permutation must be reproducible
- the same `video_ref` and `num_frames` must always give the same permutation
- different `video_ref` values should usually give different permutations
- the permutation must be a derangement
- the permutation must not depend on backbone name, variant, or feature view

## Deterministic Shuffle Design

Implement deterministic per-video shuffling using `video_ref` as the stable key.

Recommended design:

- hash `video_ref` plus `num_frames`
- seed a local RNG from that hash
- generate a permutation of `[0, ..., T-1]`
- reject identity permutations
- reject permutations with any fixed point
- retry until a derangement is produced

Behavioral rules:

- the same video gets the same shuffle across all backbones
- different videos get different shuffles
- the same video on another machine still gets the same shuffle if `video_ref` is unchanged
- local absolute `video_path` must not affect the permutation

Validation rule:

- `time_shuffled` requires `num_frames >= 2`
- otherwise raise `FeatureConfigError`

## Implementation Shape

Add one shared helper for temporal controls rather than duplicating control logic separately per benchmark.

Recommended new internal utility:

- `benchmarks/temporal_controls.py`

This helper should own:

- control-mode validation
- middle-frame repeat logic
- deterministic per-video derangement generation
- transformation of sampled frame lists before tensor stacking

This helper should not own:

- video decoding backend
- resizing
- backbone forwarding
- cache writing
- manifest writing

## Extraction-Side Changes

### MVP extraction

Update the `MVP` feature extraction path so that:

- decode config reads `temporal_control`
- cache signature includes `temporal_control`
- the actual clip decode path applies temporal control after frame sampling
- manifest decode metadata records `temporal_control`

The actual extraction loop remains unchanged in structure:

- resolve ordered records
- decode clip
- forward through adapter
- accumulate pooled and tokens
- write index, tensors, and manifest

Only the clip-construction step changes.

### IntPhys2 extraction

Make the same change in `IntPhys2`:

- decode config reads `temporal_control`
- cache signature includes `temporal_control`
- real decode path applies temporal control after sampling
- manifest decode metadata records `temporal_control`

The `MVP` and `IntPhys2` control semantics must be identical.

## Cache and Manifest Policy

The temporal control mode must be part of cache identity, so normal and control caches can never be mixed accidentally.

The cache signature payload must include:

- `num_frames`
- `sampling`
- `crop_size`
- `temporal_control`

Manifest decode metadata must also include:

- `num_frames`
- `sampling`
- `crop_size`
- `temporal_control`

Consequences:

- `none`, `single_frame_repeat`, and `time_shuffled` produce different cache signatures
- existing normal caches remain valid
- users can train probes on any control cache with no changes to training code

## Backward Compatibility

Backward compatibility requirements:

- existing configs without `decode.temporal_control` must behave exactly as before
- existing cache signatures for default runs must remain unchanged if default `temporal_control=none` is inserted in a compatibility-preserving way
- training and eval commands must continue to work unchanged against old normal caches
- no migration is required for old caches

Implementation note:

- if old manifests do not contain `temporal_control`, treat them semantically as `none` in compatibility checks where needed

## Validation and Failure Modes

Raise explicit errors for invalid or unsupported cases.

Required checks:

- reject unknown `decode.temporal_control` values
- reject non-string control-mode values
- reject `time_shuffled` when `num_frames < 2`
- fail clearly if future code requests unsupported interactions with non-uniform temporal sampling
- do not silently fall back from invalid control modes to `none`

Expected failure style:

- use existing `FeatureConfigError`
- error messages should say exactly which field is invalid and why

## Test Plan

### Shared helper tests

Add focused unit tests for the new helper:

- `none` leaves sampled frames unchanged
- `single_frame_repeat` repeats exactly the middle sampled frame
- `single_frame_repeat` preserves sequence length
- `time_shuffled` produces a non-identity permutation
- `time_shuffled` produces a derangement
- `time_shuffled` is deterministic for the same `video_ref`
- `time_shuffled` changes across different `video_ref`
- `time_shuffled` does not depend on backbone metadata
- `time_shuffled` rejects `num_frames < 2`

### MVP cache signature tests

Extend existing `MVP` signature tests to verify:

- changing `decode.temporal_control` changes cache signature
- `temporal_control=none` is stable
- `single_frame_repeat` and `time_shuffled` produce different signatures

### IntPhys2 cache signature tests

Extend existing `IntPhys2` signature tests to verify:

- changing `decode.temporal_control` changes cache signature
- `temporal_control=none` is stable
- `single_frame_repeat` and `time_shuffled` produce different signatures

### End-to-end extraction tests

Add small CPU-safe extraction tests with fake or minimal fixtures that verify:

- extraction succeeds in `none`
- extraction succeeds in `single_frame_repeat`
- extraction succeeds in `time_shuffled`
- manifests include `decode.temporal_control`
- output tensor ranks remain unchanged
- sample ordering and index semantics remain unchanged

### Regression tests

Verify no regressions in the current normal path:

- old default extraction behavior still works
- old signature comparisons still behave correctly for unchanged configs
- probe loading still works on caches extracted with `temporal_control=none`

## Acceptance Criteria

The implementation is complete when all of the following are true:

- `MVP` and `IntPhys2` accept `decode.temporal_control`
- extraction actually changes temporal content for the two new modes
- cache signatures differ across control modes
- manifests record the control mode
- no probe or eval code has to be changed
- default extraction behavior is unchanged
- tests cover deterministic shuffling, middle-frame repeat, and signature changes

## Non-Goals for v1

These are explicitly out of scope:

- `SSv2` implementation
- wrapper env var polish for the new control mode
- experiment recipe additions
- training-time temporal controls
- post-hoc control generation from existing feature caches
- reverse-order baseline as a primary mode
- per-backbone shuffle variation
- broad refactor of all decode logic across benchmarks

## Recommended Follow-Up After v1

After this lands, the next likely steps are:

- extend the same shared helper to `SSv2`
- add wrapper convenience flags for extraction jobs
- add experiment and job recipes for normal vs single-frame vs shuffled comparisons
- optionally add temporal reversal as a separate ablation
- optionally add control-mode reporting into result summaries and CSV exports

## Assumptions

- `video_ref` is stable enough across runs to define per-video shuffle deterministically
- all intended control experiments in v1 still use the current uniform temporal sampler
- preserving clip length is more important than changing `num_frames` for proposal comparability
- the correct proposal interpretation is "modify temporal content of the clip," not "change the backbone input shape"
- first-pass users are comfortable passing Hydra overrides directly to extraction commands
