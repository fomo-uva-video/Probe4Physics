# To do

## V-JEPA 1

1. Run model on all datasets and collect activations:
   - ~~MVP~~
   - ~~Intphys~~
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## V-JEPA 2

1. Run model on all datasets and collect activations:
   - MVP
   - Intphys
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## V-JEPA 2.1

1. Run model on all datasets and collect activations:
   - MVP
   - Intphys
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## Video MAE v1

1. Run model on all datasets and collect activations:
   - MVP
   - Intphys
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## Video MAE v2

1. Run model on all datasets and collect activations:
   - MVP
   - Intphys
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## Diffuser (special procedure)

1. Run model on all datasets and collect activations:
   - MVP
   - Intphys
   - Something something V2
2. Train/eval probes at different encoding levels 25%, 50%, 75%, 100%:
   - Linear
   - MLP
   - Temporal attentive probe over token/frame sequences

## Temporal Control Decisions

These controls are diagnostic baselines for temporal dependence and shortcut use.
They should not be treated as a new plausibility dataset with newly valid labels.

1. Keep the standard clip interface fixed across controls:
   - Use the same canonical 16-frame clip construction as the main experiment.
   - Keep clip length fixed for controls instead of shortening the clip.
   - Apply temporal controls after the standard frame sampling step, not by writing a new video dataset to disk.
2. Separate the static and temporal-order controls:
   - Single-frame baseline: repeat one sampled frame 16 times.
   - Time-shuffled baseline: reorder the sampled frames while keeping the same 16 frames.
3. For the static baseline:
   - Main condition: repeat the middle sampled frame 16 times.
   - Optional sensitivity analysis: compare first / middle / last frame repetition, or averaged random-frame repetition.
4. For the time-shuffled baseline:
   - Use a derangement, not a generic permutation.
   - No sampled frame should remain in its original temporal slot.
   - Shuffle the already sampled 16-frame clip, not the raw decoded video before sampling.
5. Reproducibility and fairness across backbones:
   - Every backbone must see the same sampled frame indices and the same derangement for a given sample.
   - Do not let each backbone create its own shuffle independently.
   - Store a shared per-sample control manifest with sampled frame indices and deranged positions.
   - Derangements should be deterministic from sample_id plus a global seed, or precomputed once and reused everywhere.
6. Interpretation:
   - If performance stays high after temporal derangement, the model is likely using static or endpoint shortcuts rather than genuine temporal physical reasoning.
   - The strongest reading of this control is as a temporal-ablation diagnostic; whether training stays on normal clips or also uses temporally corrupted clips can be decided separately.
