#!/bin/bash
# Recover the original seed-42 MVP LTX-13B temporal_attn LR matrix on the
# top four MLP-selected slots.
#
# Layer selection criterion:
#   top 4 MVP LTX-13B MLP slots by test pair-consistency, tie-broken by
#   validation pair-consistency.
#
# Submit with:
#   sbatch jobs/train/mvp/recovery/ltx13b_attentive_seed42_lr_matrix.sh

#SBATCH --partition=gpu_a100
#SBATCH --job-name=recover_mvp_ltx13b_attn
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=output/training/mvp/recovery_ltx13b_attentive/recover_mvp_ltx13b_attn_%A_%a.out
#SBATCH --error=output/training/mvp/recovery_ltx13b_attentive/recover_mvp_ltx13b_attn_%A_%a.err

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_DETAILS="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true)"
  for JOB_FIELD in ${JOB_DETAILS}; do
    case "${JOB_FIELD}" in
      Command=*)
        SCRIPT_PATH="${JOB_FIELD#Command=}"
        break
        ;;
    esac
  done
fi
if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR:-$(pwd)}/${SCRIPT_PATH}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

export DATASET_NAME="mvp"
export PROBE_NAME="temporal_attn"
export BACKBONE_NAME="ltx_video"
export BACKBONE_VARIANT="ltxv_13b_0_9_8_distilled"
export PROBE_FEATURE_VIEW="tokens"
export PROBE_DEVICE="cuda"
export PROBE_EPOCHS="30"
export PROBE_BATCH_SIZE="1"
export PROBE_EVAL_BATCH_SIZE="1"
export PROBE_WEIGHT_DECAY="0.01"
export PROBE_EARLY_STOPPING_PATIENCE="5"
export RUN_SEED="42"
export SPLIT_SEED="42"
export PROBE_LABEL_CONTROL_MODE="original"
export PROBE_LABEL_CONTROL_SEED="42"
export TEMPORAL_NUM_HEADS="16"
export TEMPORAL_NUM_SELF_ATTN_BLOCKS="1"
export TEMPORAL_MLP_RATIO="2.0"
export TEMPORAL_DROPOUT="0.2"
export MATRIX_LAYERS="18,22,26,10"
export MATRIX_LRS="5e-4,1e-4,5e-5,1e-5"
export MATRIX_LR_TAGS="5e-4,1e-4,5e-5,1e-5"
export FEATURE_LAYER_IDS="${MATRIX_LAYERS}"
export PROBE_OUTPUT_DIR="/gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/probes/mvp"
export PROBE_EVAL_OUTPUT_DIR="/gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/results/mvp"
export GROUP_SUBDIR="recovery_seed42_mvp_ltx13b_attentive_lr_matrix"
export WANDB_GROUP="recovery_seed42_mvp_ltx13b_attentive_lr_matrix"
export ENABLE_WANDB="${ENABLE_WANDB:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec "${SCRIPT_DIR}/../temporal_attn/run_lr_matrix.sh" "$@"
