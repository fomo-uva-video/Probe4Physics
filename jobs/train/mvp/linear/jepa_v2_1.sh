#!/bin/bash
# Train a MVP linear probe with jepa_v2_1 features.
#
# Usage:
#   sbatch jepa_v2_1.sh

#SBATCH --partition=rome
#SBATCH --job-name=mvp_jepa_v2_1_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

DATASET_NAME="mvp"
PROBE_NAME="linear"
BACKBONE_NAME="jepa_v2_1"
BACKBONE_VARIANT="vitG_384"
PROBE_EPOCHS="100"
PROBE_DEVICE="cpu"
PROBE_LAYER="last"  # possible values: last | 12 | 24 | 38 | 48
PROBE_LAYERS="${PROBE_LAYER}"
PROBE_FEATURE_VIEW="pooled"
ENABLE_WANDB="true"
WANDB_PROJECT="probe4physics"
WANDB_MODE="online"
ENABLE_OPTUNA="true"
OPTUNA_N_TRIALS="10"
OPTUNA_N_JOBS="1"
OPTUNA_TIMEOUT_SECONDS="0"
ENABLE_OPTUNA_PRUNER="true"
OPTUNA_PRUNER_STARTUP_TRIALS="3"
OPTUNA_PRUNER_WARMUP_STEPS="5"
OPTUNA_PRUNER_INTERVAL_STEPS="1"
OPTUNA_SEARCH_OVERRIDES=""
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
