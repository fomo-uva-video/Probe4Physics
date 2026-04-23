#!/bin/bash
# Extract IntPhys2 features with VideoMAE.
#
# Usage:
#   sbatch videomae.sh
#   MODE=smoke sbatch videomae.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=intphys2_videomae_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="videomae"
export BACKBONE_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
exec "${JOB_DIR}/run_extract.sh"
