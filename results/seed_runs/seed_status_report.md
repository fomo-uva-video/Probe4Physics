# Seed Run Status Tracker

Generated: 2026-08-11 20:52:29 UTC

This file tracks fixed-config seed reruns and the feature caches required to run them. Regenerate it with:

```bash
source jobs/extract/common.sh
load_probe4physics_env
python scripts/build_seed_status_report.py
```

## Overview

| Run set | Configs | Complete configs | Valid artifact rows | Artifact gaps | Feature cache groups |
| --- | --- | --- | --- | --- | --- |
| Selected-layer main Linear/MLP | 20 | 0/20 | 1/40 | 39 missing, 0 mismatch | 10/10 ready |
| Layerwise main Linear/MLP | 80 | 80/80 | 160/160 | 0 missing, 0 mismatch | 10/10 ready |
| Layerwise main IntPhys2 Attentive | 20 | 20/20 | 40/40 | 0 missing, 0 mismatch | 5/5 ready |
| Layerwise Same-L Linear/MLP | 80 | 0/80 | 160/160 | 0 missing, 0 mismatch | 10/10 ready |
| Layerwise Same-L Attentive | 40 | 40/40 | 80/80 | 0 missing, 0 mismatch | 10/10 ready |
| V-JEPA2 MLP layerwise pilot | 8 | 8/8 | 24/24 | 0 missing, 0 mismatch | 2/2 ready |

## Selected-layer main Linear/MLP

Paper-facing selected layer configs from verified_best_probe_configs.csv.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA | ViT-H/16 | Linear | 24 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | V-JEPA | ViT-H/16 | MLP | 24 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | Linear | 40 | 0/1 | 42:1/1, 101:1/1, 102:0/1 | complete:1, missing_artifact:1 | ready |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | MLP | 40 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | 38 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | 38 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | VideoMAE | ViT-H/16 | Linear | 16 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | VideoMAE | ViT-H/16 | MLP | 24 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-G/16 | Linear | 20 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-G/16 | MLP | 20 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready |
| MVP | V-JEPA | ViT-H/16 | Linear | 32 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | V-JEPA | ViT-H/16 | MLP | 32 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | V-JEPA 2 | ViT-G/16 | Linear | 40 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | V-JEPA 2 | ViT-G/16 | MLP | 40 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | 48 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | 38 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | VideoMAE | ViT-H/16 | Linear | 24 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | VideoMAE | ViT-H/16 | MLP | 24 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-G/16 | Linear | 20 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-G/16 | MLP | 20 | 0/1 | 42:1/1, 101:0/1, 102:0/1 | missing_artifact:2 | ready-compatible |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | 0ffb5c4d39ae2f66 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v1_vith16_384/train-val-test/0ffb5c4d39ae2f66 | valid cache |
| ready | 912a06f251d21d6a | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_1_vitG_384/train-val-test/912a06f251d21d6a | valid cache |
| ready | e8907c44e6a5e86e | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitg_384/train-val-test/e8907c44e6a5e86e | valid cache |
| ready | 049a250ab886a908 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_v2_vit_giant_16_224/train-val-test/049a250ab886a908 | valid cache |
| ready | ea16ae771600cbaf | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_vit_huge_16_224/train-val-test/ea16ae771600cbaf | valid cache |
| ready-compatible | ee823dd0344415ef | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v1_vith16_384/train-val-test/ee823dd0344415ef | compatible cache for expected signature 2645a29c9a1acf8b |
| ready-compatible | 9332321fa633b0d9 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_1_vitG_384/train-val-test/9332321fa633b0d9 | compatible cache for expected signature 06e57ed36c4245fb |
| ready-compatible | c60ddd6219f572f0 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_vitg_384/train-val-test/c60ddd6219f572f0 | compatible cache for expected signature a1c752d24fa0b345 |
| ready-compatible | e14e4af0e1b29964 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_v2_vit_giant_16_224/train-val-test/e14e4af0e1b29964 | compatible cache for expected signature 97517c79ab15e1d6 |
| ready-compatible | 25f6b337642dde89 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_vit_huge_16_224/train-val-test/25f6b337642dde89 | compatible cache for expected signature 3f1b147f99019e88 |

## Layerwise main Linear/MLP

All verified main Linear/MLP layers from verified_layerwise_probe_configs.csv.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA | ViT-H/16 | Linear | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA | ViT-H/16 | MLP | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | Linear | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | 12,24,38,48 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | 12,24,38,48 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-H/16 | Linear | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-H/16 | MLP | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-G/16 | Linear | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| MVP | V-JEPA | ViT-H/16 | Linear | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA | ViT-H/16 | MLP | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2 | ViT-G/16 | Linear | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-Gigantic/16 | Linear | 12,24,38,48 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-Gigantic/16 | MLP | 12,24,38,48 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE | ViT-H/16 | Linear | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE | ViT-H/16 | MLP | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-G/16 | Linear | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | 0ffb5c4d39ae2f66 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v1_vith16_384/train-val-test/0ffb5c4d39ae2f66 | valid cache |
| ready | 912a06f251d21d6a | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_1_vitG_384/train-val-test/912a06f251d21d6a | valid cache |
| ready | e8907c44e6a5e86e | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitg_384/train-val-test/e8907c44e6a5e86e | valid cache |
| ready | 049a250ab886a908 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_v2_vit_giant_16_224/train-val-test/049a250ab886a908 | valid cache |
| ready | ea16ae771600cbaf | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_vit_huge_16_224/train-val-test/ea16ae771600cbaf | valid cache |
| ready-compatible | ee823dd0344415ef | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v1_vith16_384/train-val-test/ee823dd0344415ef | compatible cache for expected signature 2645a29c9a1acf8b |
| ready-compatible | 9332321fa633b0d9 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_1_vitG_384/train-val-test/9332321fa633b0d9 | compatible cache for expected signature 06e57ed36c4245fb |
| ready-compatible | c60ddd6219f572f0 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_vitg_384/train-val-test/c60ddd6219f572f0 | compatible cache for expected signature a1c752d24fa0b345 |
| ready-compatible | e14e4af0e1b29964 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_v2_vit_giant_16_224/train-val-test/e14e4af0e1b29964 | compatible cache for expected signature 97517c79ab15e1d6 |
| ready-compatible | 25f6b337642dde89 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_vit_huge_16_224/train-val-test/25f6b337642dde89 | compatible cache for expected signature 3f1b147f99019e88 |

## Layerwise main IntPhys2 Attentive

All verified main IntPhys2 temporal_attn layers from verified_layerwise_probe_configs.csv.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA | ViT-H/16 | Attentive | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | Attentive | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-Gigantic/16 | Attentive | 12,24,38,48 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-H/16 | Attentive | 8,16,24,32 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-G/16 | Attentive | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | 0ffb5c4d39ae2f66 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v1_vith16_384/train-val-test/0ffb5c4d39ae2f66 | valid cache |
| ready | 912a06f251d21d6a | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_1_vitG_384/train-val-test/912a06f251d21d6a | valid cache |
| ready | e8907c44e6a5e86e | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitg_384/train-val-test/e8907c44e6a5e86e | valid cache |
| ready | 049a250ab886a908 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_v2_vit_giant_16_224/train-val-test/049a250ab886a908 | valid cache |
| ready | ea16ae771600cbaf | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_vit_huge_16_224/train-val-test/ea16ae771600cbaf | valid cache |

## Layerwise Same-L Linear/MLP

All verified same_L ViT-L/16 Linear/MLP layers from verified_layerwise_probe_configs.csv.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready |
| MVP | V-JEPA | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-L/16 | Linear | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-L/16 | MLP | 6,12,18,24 | 0/4 | 42:0/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | 431c15c5e1cd3140 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v1_vitl16_224/train-val-test/431c15c5e1cd3140 | valid cache |
| ready | 2afd7bb2e7011242 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_1_vitl_384/train-val-test/2afd7bb2e7011242 | valid cache |
| ready | 0421db1cbab854c4 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitl_256/train-val-test/0421db1cbab854c4 | valid cache |
| ready | 63eeddee2d3eb278 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_v2_vit_large_16_224/train-val-test/63eeddee2d3eb278 | valid cache |
| ready | cd34e38acee0a26b | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_vit_large_16_224/train-val-test/cd34e38acee0a26b | valid cache |
| ready-compatible | 7afc36c5f7b91d40 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v1_vitl16_224/train-val-test/7afc36c5f7b91d40 | compatible cache for expected signature ff6a4ebdcebbf8e7 |
| ready-compatible | 346b26fd5bbd5549 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_1_vitl_384/train-val-test/346b26fd5bbd5549 | compatible cache for expected signature fab64204850fa44d |
| ready-compatible | 590e60af03af4e96 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_vitl_256/train-val-test/590e60af03af4e96 | compatible cache for expected signature 37efa802f1454036 |
| ready-compatible | 519f1d95f7ff64b2 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_v2_vit_large_16_224/train-val-test/519f1d95f7ff64b2 | compatible cache for expected signature 01ab8c562bd901ac |
| ready-compatible | ceb43d8a3fe860ce | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_vit_large_16_224/train-val-test/ceb43d8a3fe860ce | compatible cache for expected signature a0b31f759852a41f |

## Layerwise Same-L Attentive

All verified same_L ViT-L/16 temporal_attn layers from verified_layerwise_probe_configs.csv.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | V-JEPA 2.1 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| IntPhys2 | VideoMAE-v2 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready |
| MVP | V-JEPA | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | V-JEPA 2.1 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |
| MVP | VideoMAE-v2 | ViT-L/16 | Attentive | 6,12,18,24 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:8 | ready-compatible |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | 431c15c5e1cd3140 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v1_vitl16_224/train-val-test/431c15c5e1cd3140 | valid cache |
| ready | 2afd7bb2e7011242 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_1_vitl_384/train-val-test/2afd7bb2e7011242 | valid cache |
| ready | 0421db1cbab854c4 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitl_256/train-val-test/0421db1cbab854c4 | valid cache |
| ready | 63eeddee2d3eb278 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_v2_vit_large_16_224/train-val-test/63eeddee2d3eb278 | valid cache |
| ready | cd34e38acee0a26b | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/videomae_vit_large_16_224/train-val-test/cd34e38acee0a26b | valid cache |
| ready-compatible | 7afc36c5f7b91d40 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v1_vitl16_224/train-val-test/7afc36c5f7b91d40 | compatible cache for expected signature ff6a4ebdcebbf8e7 |
| ready-compatible | 346b26fd5bbd5549 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_1_vitl_384/train-val-test/346b26fd5bbd5549 | compatible cache for expected signature fab64204850fa44d |
| ready-compatible | 590e60af03af4e96 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_vitl_256/train-val-test/590e60af03af4e96 | compatible cache for expected signature 37efa802f1454036 |
| ready-compatible | 519f1d95f7ff64b2 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_v2_vit_large_16_224/train-val-test/519f1d95f7ff64b2 | compatible cache for expected signature 01ab8c562bd901ac |
| ready-compatible | ceb43d8a3fe860ce | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/videomae_vit_large_16_224/train-val-test/ceb43d8a3fe860ce | compatible cache for expected signature a0b31f759852a41f |

## V-JEPA2 MLP layerwise pilot

Earlier diagnostic pilot; this manifest treats seed 42 as an artifact row, not a source-spreadsheet row.

| Dataset | Model | Backbone | Probe | Layers | Complete configs | Seeds | Artifact status | Feature cache |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntPhys2 | V-JEPA 2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:12 | ready |
| MVP | V-JEPA 2 | ViT-G/16 | MLP | 10,20,30,40 | 4/4 | 42:4/4, 101:4/4, 102:4/4 | complete:12 | ready-compatible |

Feature cache details:

| Status | Signature | Cache dir | Note |
| --- | --- | --- | --- |
| ready | e8907c44e6a5e86e | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/intphys2/jepa_v2_vitg_384/train-val-test/e8907c44e6a5e86e | valid cache |
| ready-compatible | c60ddd6219f572f0 | /gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/features/mvp/jepa_v2_vitg_384/train-val-test/c60ddd6219f572f0 | compatible cache for expected signature a1c752d24fa0b345 |

## Current Next Steps

1. Keep Same-L seed training blocked until the 10 Same-L ViT-L train-val-test feature caches are valid.
2. Submit Same-L Linear/MLP and Attentive wrappers after cache validation.
3. Regenerate result CSV/XLSX exports and this status report after Same-L seed jobs finish.
