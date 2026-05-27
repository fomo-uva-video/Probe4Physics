#!/bin/bash
# Extract MVP features with V-JEPA v2.
#
# Usage:
#   sbatch jepa_v2.sh
#   MODE=smoke sbatch jepa_v2.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=mvp_jepa_v2_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2"
export BACKBONE_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
exec "${JOB_DIR}/run_extract.sh"
