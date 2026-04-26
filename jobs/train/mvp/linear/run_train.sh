#!/bin/bash
# Shared runner for MVP linear-probe training jobs.
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

BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-100}"
LINEAR_PROBE_DEVICE="${LINEAR_PROBE_DEVICE:-cpu}"
LINEAR_PROBE_LAYER="${LINEAR_PROBE_LAYER:-last}"
LINEAR_PROBE_FEATURE_VIEW="${LINEAR_PROBE_FEATURE_VIEW:-pooled}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"

echo "===== TRAIN PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${BACKBONE_VARIANT:-<config default>}"
echo "LINEAR_PROBE_EPOCHS=${LINEAR_PROBE_EPOCHS}"
echo "LINEAR_PROBE_DEVICE=${LINEAR_PROBE_DEVICE}"
echo "LINEAR_PROBE_LAYER=${LINEAR_PROBE_LAYER}"
echo "LINEAR_PROBE_FEATURE_VIEW=${LINEAR_PROBE_FEATURE_VIEW}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "============================"

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

cmd=(
  python run.py train.probe.mvp
  "backbone.name=${BACKBONE_NAME}"
  "probe.epochs=${LINEAR_PROBE_EPOCHS}"
  "probe.device=${LINEAR_PROBE_DEVICE}"
  "probe.layer=${LINEAR_PROBE_LAYER}"
  "probe.feature_view=${LINEAR_PROBE_FEATURE_VIEW}"
  "probe.wandb.enabled=${ENABLE_WANDB}"
  "probe.wandb.project=${WANDB_PROJECT}"
  "probe.wandb.mode=${WANDB_MODE}"
)

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
