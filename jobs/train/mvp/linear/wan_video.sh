#!/bin/bash
# Train a single MVP linear probe with wan_video features.
#
# Usage:
#   sbatch wan_video.sh
#   sbatch wan_video.sh probe.layers=[1] probe.optuna.enabled=false probe.epochs=5

#SBATCH --partition=gpu_a100
#SBATCH --job-name=mvp_wan_video_linear
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/training/mvp/linear/mvp_wan_linear_smoke_%j.out
#SBATCH --error=output/training/mvp/linear/mvp_wan_linear_smoke_%j.err

set -euo pipefail

DATASET_NAME="mvp"
PROBE_NAME="linear"
BACKBONE_NAME="wan_video"
BACKBONE_VARIANT=""
# Full 40-slot MVP WAN sweeps are split by depth in wan_video_layer_10/20/30/40.sh.
PROBE_LAYER="last"
PROBE_LAYERS="${PROBE_LAYER}"
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
