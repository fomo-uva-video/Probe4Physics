#!/bin/bash
# Recover the original seed-42 MVP LTX-13B Linear Optuna search.
#
# One array task = one LTX noise x block slot. Submit with:
#   sbatch --array=0-39 jobs/train/mvp/recovery/ltx13b_linear_seed42_array.sh

#SBATCH --partition=rome
#SBATCH --job-name=recover_mvp_ltx13b_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=output/training/mvp/recovery_ltx13b_linear/recover_mvp_ltx13b_linear_%A_%a.out
#SBATCH --error=output/training/mvp/recovery_ltx13b_linear/recover_mvp_ltx13b_linear_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID > 39 )); then
  echo "Unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..39" >&2
  exit 2
fi

SLOT="$((TASK_ID + 1))"
DEPTHS=(12 24 36 48)
NOISE_LEVELS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
DEPTH="${DEPTHS[$(((SLOT - 1) % 4))]}"
NOISE="${NOISE_LEVELS[$(((SLOT - 1) / 4))]}"
SLOT_PADDED="$(printf '%02d' "${SLOT}")"
SLOT_LABEL="noise_${NOISE}_block_${DEPTH}"

DATASET_NAME="mvp"
PROBE_NAME="linear"
BACKBONE_NAME="ltx_video"
BACKBONE_VARIANT="ltxv_13b_0_9_8_distilled"
PROBE_LAYER="${SLOT}"
PROBE_LAYERS="${SLOT}"
PROBE_FEATURE_VIEW="pooled"
PROBE_DEVICE="cpu"
ENABLE_WANDB="false"
WANDB_MODE="disabled"
WANDB_PROJECT="probe4physics"
ENABLE_OPTUNA="true"
OPTUNA_N_TRIALS="20"
OPTUNA_N_JOBS="1"
OPTUNA_TIMEOUT_SECONDS="0"
ENABLE_OPTUNA_PRUNER="true"
OPTUNA_PRUNER_STARTUP_TRIALS="3"
OPTUNA_PRUNER_WARMUP_STEPS="100"
OPTUNA_PRUNER_INTERVAL_STEPS="1"
OPTUNA_STUDY_NAME="recovery_seed42_mvp_ltx13b_linear_slot${SLOT_PADDED}"
WANDB_GROUP="${OPTUNA_STUDY_NAME}"

export DATASET_NAME PROBE_NAME BACKBONE_NAME BACKBONE_VARIANT PROBE_LAYER PROBE_LAYERS
export PROBE_FEATURE_VIEW PROBE_DEVICE ENABLE_WANDB WANDB_MODE WANDB_PROJECT
export ENABLE_OPTUNA OPTUNA_N_TRIALS OPTUNA_N_JOBS OPTUNA_TIMEOUT_SECONDS
export ENABLE_OPTUNA_PRUNER OPTUNA_PRUNER_STARTUP_TRIALS OPTUNA_PRUNER_WARMUP_STEPS
export OPTUNA_PRUNER_INTERVAL_STEPS OPTUNA_STUDY_NAME WANDB_GROUP

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUNNER="${REPO_ROOT}/jobs/train/mvp/linear/run_train.sh"
OUTPUT_SUBDIR="recovery_seed42_mvp_ltx13b_linear_slot${SLOT_PADDED}_${SLOT_LABEL}"

COMMON_ARGS=(
  "seed=42"
  "split.seed=42"
  "feature_cache.dir=/scratch-shared/spunzo1/probe4physics/artifacts/features/mvp"
  "probe.output_dir=/gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/probes/mvp"
  "probe.output_subdir=${OUTPUT_SUBDIR}"
  "probe.eval_output_dir=/gpfs/scratch1/shared/spunzo1/probe4physics/artifacts/results/mvp"
  "probe.eval_output_subdir=${OUTPUT_SUBDIR}"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'TASK_ID=%s\nSLOT=%s\nDEPTH=%s\nNOISE=%s\nSLOT_LABEL=%s\nPROBE_LAYERS=%s\n' \
    "${TASK_ID}" "${SLOT}" "${DEPTH}" "${NOISE}" "${SLOT_LABEL}" "${PROBE_LAYERS}"
  printf 'Runner: %s\n' "${RUNNER}"
  printf 'Args:\n'
  printf '  %q\n' "${COMMON_ARGS[@]}"
  exit 0
fi

exec "${RUNNER}" "${COMMON_ARGS[@]}" "$@"
