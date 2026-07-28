#!/bin/bash
# Train the MVP temporal_attn V-JEPA-v1 random-label control over the same
# fixed (layer, learning-rate) matrix used by the MVP attentive LR matrix.
#
# Usage:
#   sbatch jobs/train/mvp/temporal_attn/jepa_v1_structured_random_lr_matrix.sh
#
# Matrix:
#   layers = 8, 16, 24, 32
#   lrs    = 5e-4, 1e-4, 5e-5, 1e-5
#
# Training:
#   epochs = 30
#   early-stopping patience = 5
#   label control = structured_random, seed 42

#SBATCH --partition=gpu_a100
#SBATCH --job-name=mvp_jepa_v1_attn_random_lr_matrix
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=output/training/mvp/attention/mvp_jepa_v1_attn_random_lr_matrix_%A_%a.out
#SBATCH --error=output/training/mvp/attention/mvp_jepa_v1_attn_random_lr_matrix_%A_%a.err

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-mvp}"
PROBE_NAME="${PROBE_NAME:-temporal_attn}"
BACKBONE_NAME="${BACKBONE_NAME:-jepa_v1}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-vith16_384}"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-tokens}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-2}"
PROBE_EVAL_BATCH_SIZE="${PROBE_EVAL_BATCH_SIZE:-2}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.01}"
PROBE_EARLY_STOPPING_PATIENCE="${PROBE_EARLY_STOPPING_PATIENCE:-5}"
PROBE_LABEL_CONTROL_MODE="${PROBE_LABEL_CONTROL_MODE:-structured_random}"
PROBE_LABEL_CONTROL_SEED="${PROBE_LABEL_CONTROL_SEED:-42}"
TEMPORAL_NUM_HEADS="${TEMPORAL_NUM_HEADS:-16}"
TEMPORAL_NUM_SELF_ATTN_BLOCKS="${TEMPORAL_NUM_SELF_ATTN_BLOCKS:-1}"
TEMPORAL_MLP_RATIO="${TEMPORAL_MLP_RATIO:-2.0}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.2}"
WANDB_MODE="${WANDB_MODE:-offline}"
GROUP_SUBDIR="${GROUP_SUBDIR:-mvp_probe_temporal_attn_jepa_v1_vith16_384_structured_random_lr_matrix_ep30_pat5}"
WANDB_GROUP="${WANDB_GROUP:-mvp_temporal_attn_jepa_v1_structured_random_lr_matrix_ep30_pat5}"
export DATASET_NAME PROBE_NAME BACKBONE_NAME BACKBONE_VARIANT PROBE_FEATURE_VIEW PROBE_DEVICE PROBE_EPOCHS PROBE_BATCH_SIZE PROBE_EVAL_BATCH_SIZE PROBE_WEIGHT_DECAY PROBE_EARLY_STOPPING_PATIENCE PROBE_LABEL_CONTROL_MODE PROBE_LABEL_CONTROL_SEED TEMPORAL_NUM_HEADS TEMPORAL_NUM_SELF_ATTN_BLOCKS TEMPORAL_MLP_RATIO TEMPORAL_DROPOUT WANDB_MODE GROUP_SUBDIR WANDB_GROUP

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
LR_MATRIX=""
for candidate in \
  "${SCRIPT_DIR}/jepa_v1_lr_matrix.sh" \
  "${SLURM_SUBMIT_DIR:-}/jepa_v1_lr_matrix.sh" \
  "${SLURM_SUBMIT_DIR:-}/jobs/train/mvp/temporal_attn/jepa_v1_lr_matrix.sh"
do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    LR_MATRIX="${candidate}"
    break
  fi
done
if [[ -z "${LR_MATRIX}" ]]; then
  echo "ERROR: could not locate jobs/train/mvp/temporal_attn/jepa_v1_lr_matrix.sh" >&2
  exit 2
fi
exec "${LR_MATRIX}"
