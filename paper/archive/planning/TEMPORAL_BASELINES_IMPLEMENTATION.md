# Temporal Baselines Implementation

This note captures the current agreed implementation for the temporal control
baselines on `MVP` and `IntPhys2`.

It is intentionally practical and not deeply technical. The goal is to state
clearly what we need to build and how to keep the baselines consistent across
backbones.

This note reflects the latest supervisor decision. In particular, for the
shuffle-style control we will use a `displacement` within the already sampled
16-frame clip.

## Scope

We want two baselines for:

- `MVP`
- `IntPhys2`

The two baselines are:

1. `single-frame repeated`
2. `displacement`

These are evaluation controls. They are not meant to become a new dataset with
newly curated videos on disk.

## Shared Rules

These rules apply to both baselines:

1. Start from the same 16-frame clip used in the original extraction.
2. Do not sample new frames from the full raw video.
3. The random frame choice for the single-frame baseline must be chosen from the
   16 sampled frames only.
4. The displacement for the shuffle baseline must also be defined within that
   same 16-frame clip only.
5. This is important because we do not want to feed the model with frames that
   were outside the original extracted clip.
6. Different backbones must see the same transformed clip for the same video.

## Baseline 1: Single-Frame Repeated

### What to do

1. Take the original 16-frame sampled clip.
2. Pick one frame at random from those 16 frames.
3. Repeat that chosen frame so that the final clip still has length 16.
4. Run evaluation only.
5. Do not train a new probe for this baseline.

### Ground-truth handling

For this baseline, we want to save results under two label scenarios:

1. `all_true`
2. `all_false`

The main intuition is that a clip made by repeating one frame 16 times should
look plausible, so the `all_true` case is the main one we expect to matter.
Still, to be safe, we should save both scenarios.

### Metrics to save

For the single-frame repeated baseline, save evaluation outputs for both
ground-truth settings:

- `all_true`
- `all_false`

For `IntPhys2`, explicitly save:

- accuracy
- ROC
- VOE

For `MVP`, save the corresponding evaluation outputs in the same two scenarios,
with clear naming so that the ground-truth setting is obvious from the result
folder or file name.

## Baseline 2: Displacement

### What to do

1. Take the original 16-frame sampled clip.
2. Apply a displacement within those 16 frame positions.
3. The displacement must be non-zero, otherwise the clip would stay unchanged.
4. Run the evaluation using this displaced clip.

### Consistency rule

The displacement must follow this rule:

- same video -> same displacement across different backbones
- different videos -> different displacement

This means the displacement cannot be generated independently inside each
backbone run. It must be tied to the video identity, so that comparisons across
backbones remain fair.

## Reproducibility

For both baselines, we should save the transformation metadata used for each
video.

For example:

- for `single-frame repeated`: which of the 16 sampled frames was chosen
- for `displacement`: which displacement value was used

This metadata should be stable and reusable across backbones.

## Practical Outcome

At the end, for each dataset and backbone, we should have:

1. a `single-frame repeated` evaluation
2. a `displacement` evaluation

For the `single-frame repeated` baseline, we should also have two explicit
ground-truth evaluation settings:

1. `all_true`
2. `all_false`

## Important Reminder

The key constraint is simple:

- stay inside the original 16-frame extracted clip

That applies both to:

- the random frame used for the repeated-frame baseline
- the displacement used for the shuffle baseline

This keeps the baselines controlled and avoids accidentally testing the models
on frames they never saw in the original extraction setup.
