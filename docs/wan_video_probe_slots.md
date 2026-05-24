# Wan Video Probe Slots

This note describes the default Wan slot layout when `backbone.name=wan_video`
and `feature_cache.layer_ids=[]`.

Wan is probed as a diffusion transformer. A probe slot is a flattened
`(noise_level, transformer_depth)` pair. The ordering is noise-major,
depth-minor.

Default config:

- Model: `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- Backbone key: `wan_video`
- Variant: `wan2_1_t2v_14b_diffusers`
- Transformer depth: `40` blocks
- Relative depths: `0.25, 0.5, 0.75, 1.0`
- Block depths: `10, 20, 30, 40`
- Noise levels: `1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1`

| Slot | Noise | Block |
|---:|---:|---:|
| 1 | 1.0 | 10 |
| 2 | 1.0 | 20 |
| 3 | 1.0 | 30 |
| 4 | 1.0 | 40 |
| 5 | 0.9 | 10 |
| 6 | 0.9 | 20 |
| 7 | 0.9 | 30 |
| 8 | 0.9 | 40 |
| 9 | 0.8 | 10 |
| 10 | 0.8 | 20 |
| 11 | 0.8 | 30 |
| 12 | 0.8 | 40 |
| 13 | 0.7 | 10 |
| 14 | 0.7 | 20 |
| 15 | 0.7 | 30 |
| 16 | 0.7 | 40 |
| 17 | 0.6 | 10 |
| 18 | 0.6 | 20 |
| 19 | 0.6 | 30 |
| 20 | 0.6 | 40 |
| 21 | 0.5 | 10 |
| 22 | 0.5 | 20 |
| 23 | 0.5 | 30 |
| 24 | 0.5 | 40 |
| 25 | 0.4 | 10 |
| 26 | 0.4 | 20 |
| 27 | 0.4 | 30 |
| 28 | 0.4 | 40 |
| 29 | 0.3 | 10 |
| 30 | 0.3 | 20 |
| 31 | 0.3 | 30 |
| 32 | 0.3 | 40 |
| 33 | 0.2 | 10 |
| 34 | 0.2 | 20 |
| 35 | 0.2 | 30 |
| 36 | 0.2 | 40 |
| 37 | 0.1 | 10 |
| 38 | 0.1 | 20 |
| 39 | 0.1 | 30 |
| 40 | 0.1 | 40 |

## Wan2.1-T2V-1.3B-Diffusers

The 1.3B variant uses a 30-block transformer (`num_layers=30` in
`transformer/config.json` on Hugging Face). With
`default_relative_depths=[0.25, 0.5, 0.75, 1.0]`, depth anchors resolve to:

- `8, 15, 22, 30`

With the same default 10 noise levels, this still yields 40 probe slots
(10 noises x 4 depths), preserving the shared WAN slot count used by wrappers.

Note on 480P:

- The Hugging Face model card recommends 480P generation for 1.3B, but that is
  generation-time output guidance for `WanPipeline(...)`.
- Probe4Physics extraction remains on benchmark decode/crop defaults
  (`frames_per_clip=16`, `crop_size=224`) unless explicitly overridden.
