#!/bin/bash
# Train a SSv2 temporal_attn probe with videomae_v2 features.
#
# Usage:
#   sbatch videomae_v2.sh
#   sbatch videomae_v2.sh probe.device=cpu

#SBATCH --partition=gpu_mig
#SBATCH --job-name=ssv2_videomae_v2_temporal_attn
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --output=output/training/ssv2/attention/ssv2_videomae_v2_attn_layers_%j.out
#SBATCH --error=output/training/ssv2/attention/ssv2_videomae_v2_attn_layers_%j.err

set -euo pipefail

DATASET_NAME="ssv2"
PROBE_NAME="temporal_attn"
BACKBONE_NAME="videomae_v2"
BACKBONE_VARIANT="vit_giant_16_224"
PROBE_EPOCHS="100"
PROBE_LAYER="last"  # possible values: last | 10 | 20 | 30 | 40
PROBE_LAYERS="${PROBE_LAYER}"
PROBE_FEATURE_VIEW="tokens"
PROBE_DEVICE="cuda"
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
OPTUNA_SEARCH_OVERRIDES="probe.eval_batch_size=1 probe.optuna.search_space.batch_size.choices=[1,2,4]"
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
