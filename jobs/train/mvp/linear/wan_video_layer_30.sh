#!/bin/bash
# Train MVP linear probes with wan_video block-30 features across all default noise levels.
#
# Usage:
#   sbatch wan_video_layer_30.sh
#   sbatch wan_video_layer_30.sh probe.device=cuda

#SBATCH --partition=gpu_a100
#SBATCH --job-name=mvp_wan_video_linear_l30
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=output/training/mvp/linear/mvp_wan_linear_layer_30_%j.out
#SBATCH --error=output/training/mvp/linear/mvp_wan_linear_layer_30_%j.err

set -euo pipefail

DATASET_NAME="mvp"
PROBE_NAME="linear"
BACKBONE_NAME="wan_video"
BACKBONE_VARIANT=""  # empty uses configs/backbones.yaml default
# Wan default slots are 10 noise levels x 4 depths. These are the block-30 slots.
PROBE_LAYER="3"
PROBE_LAYERS="3, 7, 11, 15, 19, 23, 27, 31, 35, 39"
PROBE_FEATURE_VIEW="pooled"
PROBE_DEVICE="cuda"
ENABLE_WANDB="false"
WANDB_PROJECT="probe4physics"
WANDB_MODE="online"
ENABLE_OPTUNA="true"
OPTUNA_N_TRIALS="20"
OPTUNA_N_JOBS="1"
OPTUNA_TIMEOUT_SECONDS="0"
ENABLE_OPTUNA_PRUNER="true"
OPTUNA_PRUNER_STARTUP_TRIALS="3"
OPTUNA_PRUNER_WARMUP_STEPS="100"
OPTUNA_PRUNER_INTERVAL_STEPS="1"
OPTUNA_SEARCH_OVERRIDES="probe.optuna.search_space.epochs.enabled=true probe.optuna.search_space.epochs.choices=[20,50,100,500,1000,2000]"
export DATASET_NAME PROBE_NAME BACKBONE_NAME BACKBONE_VARIANT PROBE_EPOCHS PROBE_DEVICE PROBE_LAYER PROBE_LAYERS PROBE_FEATURE_VIEW ENABLE_WANDB WANDB_PROJECT WANDB_MODE ENABLE_OPTUNA OPTUNA_N_TRIALS OPTUNA_N_JOBS OPTUNA_TIMEOUT_SECONDS ENABLE_OPTUNA_PRUNER OPTUNA_PRUNER_STARTUP_TRIALS OPTUNA_PRUNER_WARMUP_STEPS OPTUNA_PRUNER_INTERVAL_STEPS OPTUNA_SEARCH_OVERRIDES

JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${JOB_COMMAND}")" && pwd)"
else
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
exec "${SCRIPT_DIR}/run_train.sh" "$@"
