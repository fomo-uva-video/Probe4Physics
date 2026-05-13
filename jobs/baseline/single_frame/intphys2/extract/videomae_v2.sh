#!/bin/bash
# Extract the IntPhys2 single-frame baseline with VideoMAE v2 features.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=int_sf_extract_videomae_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="videomae_v2"
BACKBONE_VARIANT="vit_giant_16_224"
PROBE_LAYERS="10,20,30,40"
BASELINE_STAGE="extract"
# Extraction leaves feature_cache.layer_ids empty so the adapter extracts its canonical layers.
export BACKBONE_NAME BACKBONE_VARIANT PROBE_LAYERS BASELINE_STAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME="$(basename "$(dirname "${SCRIPT_DIR}")")"
BASELINE_LABEL="$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")"
BASELINE_NAME="single_frame"
export DATASET_NAME BASELINE_LABEL BASELINE_NAME

exec "${SCRIPT_DIR}/../../../_shared/run_baseline.sh" "$@"
