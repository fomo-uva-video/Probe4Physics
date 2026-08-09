#!/bin/bash
# Shared launcher for fixed-config seed-rerun manifest jobs.

set -euo pipefail

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

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_main_linear_mlp_v1.csv}"
PROBE_FILTER="${PROBE_FILTER:?PROBE_FILTER must be set by the wrapper script}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"

echo "===== SEED RUNNER PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "PROBE_FILTER=${PROBE_FILTER}"
echo "TASK_INDEX=${TASK_INDEX}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "=================================="

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

python scripts/run_seed_manifest_row.py \
  --manifest "${MANIFEST_PATH}" \
  --task-index "${TASK_INDEX}" \
  --probe "${PROBE_FILTER}" \
  "probe.wandb.enabled=${ENABLE_WANDB}" \
  "probe.wandb.project=${WANDB_PROJECT}" \
  "probe.wandb.mode=${WANDB_MODE}" \
  "$@"

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"
