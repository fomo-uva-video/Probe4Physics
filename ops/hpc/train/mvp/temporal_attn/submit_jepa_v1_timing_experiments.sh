#!/bin/bash
# Submit the three fixed-config MVP JEPA-v1 temporal-attention timing jobs.
#
# Usage:
#   ops/hpc/train/mvp/temporal_attn/submit_jepa_v1_timing_experiments.sh
#   TIMING_GROUP=my_group ops/hpc/train/mvp/temporal_attn/submit_jepa_v1_timing_experiments.sh
#   DRY_RUN=1 ops/hpc/train/mvp/temporal_attn/submit_jepa_v1_timing_experiments.sh

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/jepa_v1_timing_worker.sh"

TIMESTAMP_UTC="${TIMESTAMP_UTC:-$(date -u +"%Y%m%dT%H%M%SZ")}"
TIMING_GROUP="${TIMING_GROUP:-mvp_temporal_attn_jepa_v1_timing_${TIMESTAMP_UTC}}"
DRY_RUN="${DRY_RUN:-0}"

declare -a MODES=(
  baseline
  storage_fix
  storage_fix_bs2
)

cd "${REPO_ROOT}"

echo "Submitting MVP JEPA-v1 temporal-attn timing jobs"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TIMING_GROUP=${TIMING_GROUP}"
echo "WORKER_SCRIPT=${WORKER_SCRIPT}"

for mode in "${MODES[@]}"; do
  job_name="mvp_jepa_v1_timing_${mode}"
  cmd=(
    sbatch
    "--job-name=${job_name}"
    "--export=ALL,EXPERIMENT_MODE=${mode},TIMING_GROUP=${TIMING_GROUP}"
    "${WORKER_SCRIPT}"
  )
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
done
