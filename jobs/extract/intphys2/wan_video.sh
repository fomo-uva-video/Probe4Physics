#!/bin/bash
# Extract IntPhys2 features with Wan diffusion transformer features.
#
# By default this wrapper follows `wan_video.default_variant` from
# `configs/backbones.yaml`. Override with `BACKBONE_VARIANT=<variant>` at submit
# time when you want a non-default checkpoint.
#
# Usage:
#   sbatch wan_video.sh
#   MODE=smoke sbatch wan_video.sh

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=intphys2_wan_video_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=wan_extract_%x_%j.out
#SBATCH --error=wan_extract_%x_%j.err

set -euo pipefail

BACKBONE_NAME="wan_video"
export BACKBONE_NAME

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -x "${SUBMIT_DIR}/run_extract.sh" ]]; then
  exec "${SUBMIT_DIR}/run_extract.sh"
fi
if [[ -x "${SUBMIT_DIR}/jobs/extract/intphys2/run_extract.sh" ]]; then
  exec "${SUBMIT_DIR}/jobs/extract/intphys2/run_extract.sh"
fi
echo "ERROR: could not locate jobs/extract/intphys2/run_extract.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
exit 2
