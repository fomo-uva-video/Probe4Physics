#!/bin/bash
# Extract IntPhys2 features with V-JEPA v2.
#
# Usage:
#   sbatch jepa_v2.sh
#   MODE=smoke sbatch jepa_v2.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=intphys2_jepa_v2_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2"
export BACKBONE_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXTRACT=""
for candidate in \
  "${SCRIPT_DIR}/run_extract.sh" \
  "${SLURM_SUBMIT_DIR:-}/run_extract.sh" \
  "${SLURM_SUBMIT_DIR:-}/jobs/extract/intphys2/run_extract.sh"; do
  if [[ -x "${candidate}" ]]; then
    RUN_EXTRACT="${candidate}"
    break
  fi
done

if [[ -z "${RUN_EXTRACT}" ]]; then
  echo "ERROR: Could not locate jobs/extract/intphys2/run_extract.sh" >&2
  exit 2
fi

exec "${RUN_EXTRACT}"
