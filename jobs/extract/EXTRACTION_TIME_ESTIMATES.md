# Extraction Time Estimates

These estimates use the wall-clock timings from the smoke runs on 2026-04-23.

Formula used:

- `per_sample_seconds = smoke_job_wall_clock_seconds / 2`
- `full_extraction_seconds = per_sample_seconds * dataset_samples`

Important:

- These are wall-clock estimates, not pure model-only forward times.
- They include startup, model load, video decode, and feature write time.
- Because the smoke jobs only processed 2 samples, these full-run estimates are conservative and may overestimate the real full-run time.
- `ltx_video` timings are now from successful smoke runs:
  - `intphys2_ltx_video_extract_22182427.out`
  - `mvp_ltx_video_extract_22181756.out`
- Parameter counts below are from local model artifacts used in this repo:
  - JEPA variants: counted from local checkpoint state dicts (`target_encoder` key)
  - VideoMAE variants: counted from local Hugging Face `model.safetensors`
  - LTX: counted from local Hugging Face `vae/diffusion_pytorch_model.safetensors` (extractor uses VAE only)

## Dataset Sample Counts

| Dataset | Unique full-extraction samples | Sample-runs across 6 listed backbones | Notes |
| --- | ---: | ---: | --- |
| IntPhys2 | 1012 | 6072 | `data/splits/intphys2/manifest.json` -> `stats.n_samples_main` |
| MVP | 9886 | 59316 | `data/splits/mvp/full_60_20_20/manifest.json` -> `stats.n_samples` |
| Overall | 10898 | 65388 | IntPhys2 + MVP |

## IntPhys2

| Backbone | Variant | Params | Per-sample time (s) | Full samples | Estimated full time (s) | Estimated full time |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `jepa_v1` | `vith16_384` | 637,546,240 | 16.0 | 1012 | 16192 | 4h 29m 52s |
| `jepa_v2` | `vitg_384` | 1,012,173,952 | 18.0 | 1012 | 18216 | 5h 03m 36s |
| `jepa_v2_1` | `vitG_384` | 1,845,216,768 | 26.5 | 1012 | 26818 | 7h 26m 58s |
| `videomae` | `vit_huge_16_224` | 632,119,440 | 31.0 | 1012 | 31372 | 8h 42m 52s |
| `videomae_v2` | `vit_giant_16_224` | 1,026,306,560 | 50.5 | 1012 | 51106 | 14h 11m 46s |
| `ltx_video` | `ltxv_13b_0_9_8_distilled` | 1,246,913,778 | 16.5 | 1012 | 16698 | 4h 38m 18s |
| **IntPhys2 total** |  |  |  |  | **160402** | **44h 33m 22s** |

## MVP

| Backbone | Variant | Params | Per-sample time (s) | Full samples | Estimated full time (s) | Estimated full time |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `jepa_v1` | `vith16_384` | 637,546,240 | 93.0 | 9886 | 919398 | 255h 23m 18s |
| `jepa_v2` | `vitg_384` | 1,012,173,952 | 98.5 | 9886 | 973771 | 270h 29m 31s |
| `jepa_v2_1` | `vitG_384` | 1,845,216,768 | 90.0 | 9886 | 889740 | 247h 09m 00s |
| `videomae` | `vit_huge_16_224` | 632,119,440 | 21.5 | 9886 | 212549 | 59h 02m 29s |
| `videomae_v2` | `vit_giant_16_224` | 1,026,306,560 | 36.5 | 9886 | 360839 | 100h 13m 59s |
| `ltx_video` | `ltxv_13b_0_9_8_distilled` | 1,246,913,778 | 26.0 | 9886 | 257036 | 71h 23m 56s |
| **MVP total** |  |  |  |  | **3613333** | **1003h 42m 13s** |

## Totals

| Scope | Unique samples | Sample-runs in listed rows | Estimated full time (s) | Estimated full time |
| --- | ---: | ---: | ---: | --- |
| IntPhys2 total | 1012 | 6072 | 160402 | 44h 33m 22s |
| MVP total | 9886 | 59316 | 3613333 | 1003h 42m 13s |
| **Overall total** | **10898** | **65388** | **3773735** | **1048h 15m 35s** |
