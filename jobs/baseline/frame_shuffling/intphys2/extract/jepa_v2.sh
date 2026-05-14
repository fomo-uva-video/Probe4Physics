#!/bin/bash
# Extract the IntPhys2 frame-shuffling baseline with V-JEPA v2 features.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=int_shuffle_extract_jepa_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2"
BACKBONE_VARIANT="vitg_384"
PROBE_LAYERS="10,20,30,40"
BASELINE_STAGE="extract"
# Extraction leaves feature_cache.layer_ids empty so the adapter extracts its canonical layers.
export BACKBONE_NAME BACKBONE_VARIANT PROBE_LAYERS BASELINE_STAGE

SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr " " "\n" | sed -n "s/^Command=//p" | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR:-$(pwd)}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
DATASET_NAME="$(basename "$(dirname "${SCRIPT_DIR}")")"
BASELINE_LABEL="$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")"
BASELINE_NAME="displacement"
export DATASET_NAME BASELINE_LABEL BASELINE_NAME

exec "${SCRIPT_DIR}/../../../_shared/run_baseline.sh" "$@"
