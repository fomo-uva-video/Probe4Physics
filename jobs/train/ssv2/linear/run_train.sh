#!/bin/bash
# Shared runner for probe training jobs.
#
# Expected launch style:
#   sbatch jepa_v1.sh
#   sbatch jepa_v2.sh probe.epochs=200

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

DATASET_NAME="${DATASET_NAME:?DATASET_NAME must be set by the wrapper script}"
PROBE_NAME="${PROBE_NAME:?PROBE_NAME must be set by the wrapper script}"
BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"
PROBE_LAYER="${PROBE_LAYER:-last}"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-pooled}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"
ENABLE_OPTUNA="${ENABLE_OPTUNA:-true}"
OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-10}"
OPTUNA_N_JOBS="${OPTUNA_N_JOBS:-1}"
OPTUNA_TIMEOUT_SECONDS="${OPTUNA_TIMEOUT_SECONDS:-0}"

BACKBONE_TAG="${BACKBONE_NAME}"
if [[ -n "${BACKBONE_VARIANT}" ]]; then
  BACKBONE_TAG="${BACKBONE_TAG}_${BACKBONE_VARIANT}"
fi
WANDB_GROUP="${WANDB_GROUP:-${DATASET_NAME}_${PROBE_NAME}_${BACKBONE_TAG}}"
OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-${DATASET_NAME}_${PROBE_NAME}_${BACKBONE_TAG}}"

if [[ "${PROBE_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

echo "===== TRAIN PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "PROBE_NAME=${PROBE_NAME}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${BACKBONE_VARIANT:-<config default>}"
echo "PROBE_EPOCHS=${PROBE_EPOCHS}"
echo "PROBE_DEVICE=${PROBE_DEVICE}"
echo "PROBE_LAYER=${PROBE_LAYER}"
echo "PROBE_FEATURE_VIEW=${PROBE_FEATURE_VIEW}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "ENABLE_OPTUNA=${ENABLE_OPTUNA}"
echo "OPTUNA_N_TRIALS=${OPTUNA_N_TRIALS}"
echo "OPTUNA_N_JOBS=${OPTUNA_N_JOBS}"
echo "OPTUNA_TIMEOUT_SECONDS=${OPTUNA_TIMEOUT_SECONDS}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "============================"

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

cmd=(
  python run.py "train.probe.${DATASET_NAME}"
  "backbone.name=${BACKBONE_NAME}"
  "probe.name=${PROBE_NAME}"
  "probe.epochs=${PROBE_EPOCHS}"
  "probe.device=${PROBE_DEVICE}"
  "probe.layer=${PROBE_LAYER}"
  "probe.feature_view=${PROBE_FEATURE_VIEW}"
  "probe.wandb.enabled=${ENABLE_WANDB}"
  "probe.wandb.project=${WANDB_PROJECT}"
  "probe.wandb.mode=${WANDB_MODE}"
  "probe.wandb.group=${WANDB_GROUP}"
  "probe.optuna.enabled=${ENABLE_OPTUNA}"
  "probe.optuna.n_trials=${OPTUNA_N_TRIALS}"
  "probe.optuna.n_jobs=${OPTUNA_N_JOBS}"
  "probe.optuna.timeout_seconds=${OPTUNA_TIMEOUT_SECONDS}"
  "probe.optuna.study_name=${OPTUNA_STUDY_NAME}"
)

if [[ "${PROBE_FEATURE_VIEW}" == "tokens" ]]; then
  cmd+=("feature_cache.include_tokens=true")
fi

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

cmd+=("$@")
"${cmd[@]}"

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"
