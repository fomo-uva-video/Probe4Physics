#!/bin/bash
# Train IntPhys2 temporal_attn probes with videomae features over a fixed
# (layer, learning-rate) matrix.
#
# Usage:
#   sbatch jobs/train/intphys2/temporal_attn/videomae_lr_matrix.sh
#
# Default matrix:
#   layers = 8, 16, 24, 32
#   lrs    = 5e-4, 1e-4, 5e-5, 1e-5
#
# Layout:
#   artifacts/probes/intphys2/${GROUP_SUBDIR}/layer_<layer>/lr_<tag>/
#
# The shared runner also incrementally rebuilds:
#   artifacts/probes/intphys2/${GROUP_SUBDIR}/train_eval_summary.json
#   artifacts/probes/intphys2/${GROUP_SUBDIR}/train_eval_summary.csv
# keeping, for each layer, the best completed LR run selected on val VOE.
#
# Quick check after completion:
#   column -s, -t < artifacts/probes/intphys2/intphys2_probe_temporal_attn_videomae_vit_huge_16_224_lr_matrix/train_eval_summary.csv

#SBATCH --partition=gpu_a100
#SBATCH --job-name=intphys2_videomae_attn_lr_matrix
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=output/training/intphys2/attention/intphys2_videomae_attn_lr_matrix_%A_%a.out
#SBATCH --error=output/training/intphys2/attention/intphys2_videomae_attn_lr_matrix_%A_%a.err

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-intphys2}"
PROBE_NAME="${PROBE_NAME:-temporal_attn}"
BACKBONE_NAME="${BACKBONE_NAME:-videomae}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-vit_huge_16_224}"
MATRIX_LAYERS="${MATRIX_LAYERS:-8,16,24,32}"
MATRIX_LRS="${MATRIX_LRS:-5e-4,1e-4,5e-5,1e-5}"
MATRIX_LR_TAGS="${MATRIX_LR_TAGS:-5e-4,1e-4,5e-5,1e-5}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-1}"
PROBE_EVAL_BATCH_SIZE="${PROBE_EVAL_BATCH_SIZE:-1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.01}"
TEMPORAL_NUM_HEADS="${TEMPORAL_NUM_HEADS:-16}"
TEMPORAL_NUM_SELF_ATTN_BLOCKS="${TEMPORAL_NUM_SELF_ATTN_BLOCKS:-1}"
TEMPORAL_MLP_RATIO="${TEMPORAL_MLP_RATIO:-2.0}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.2}"
GROUP_SUBDIR="${GROUP_SUBDIR:-intphys2_probe_temporal_attn_videomae_vit_huge_16_224_lr_matrix}"
WANDB_GROUP="${WANDB_GROUP:-intphys2_temporal_attn_videomae_lr_matrix}"
export DATASET_NAME PROBE_NAME BACKBONE_NAME BACKBONE_VARIANT MATRIX_LAYERS MATRIX_LRS MATRIX_LR_TAGS PROBE_EPOCHS PROBE_BATCH_SIZE PROBE_EVAL_BATCH_SIZE PROBE_WEIGHT_DECAY TEMPORAL_NUM_HEADS TEMPORAL_NUM_SELF_ATTN_BLOCKS TEMPORAL_MLP_RATIO TEMPORAL_DROPOUT GROUP_SUBDIR WANDB_GROUP

SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
exec "${SCRIPT_DIR}/run_lr_matrix.sh"
