#!/bin/bash
# Train a MVP mlp probe with ltx_video features.
#
# Usage:
#   sbatch ltx_video.sh
#   sbatch ltx_video.sh probe.device=cpu

#SBATCH --partition=rome
#SBATCH --job-name=mvp_ltx_video_mlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=output/training/mvp/mlp/mvp_ltx_mlp_layers_%j.out
#SBATCH --error=output/training/mvp/mlp/mvp_ltx_mlp_layers_%j.err

set -euo pipefail

DATASET_NAME="mvp"
PROBE_NAME="mlp"
BACKBONE_NAME="ltx_video"
BACKBONE_VARIANT=""  # empty uses configs/backbones.yaml default (currently distilled)
PROBE_EPOCHS="100"
PROBE_LAYER="1"  # sweep values: 1..40 for the default LTX slot grid
PROBE_LAYERS="1,2,4,5"
PROBE_FEATURE_VIEW="pooled"
PROBE_DEVICE="cpu"
ENABLE_WANDB="true"
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
