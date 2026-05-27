#!/bin/bash
# Extract MVP features with LTX-Video diffusion transformer features.
#
# By default this wrapper follows `ltx_video.default_variant` from
# `configs/backbones.yaml`. Override with `BACKBONE_VARIANT=<variant>` at submit
# time when you want a non-default checkpoint.
#
# Usage:
#   sbatch ltx_video.sh
#   MODE=smoke sbatch ltx_video.sh

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=mvp_ltx_video_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%out/ltx_extract_%x_%j.out
#SBATCH --error=%out/ltx_extract_%x_%j.err

set -euo pipefail

BACKBONE_NAME="ltx_video"
export BACKBONE_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
exec "${JOB_DIR}/run_extract.sh"
