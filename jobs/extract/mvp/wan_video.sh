#!/bin/bash
# Extract MVP features with Wan diffusion transformer features.
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
#SBATCH --job-name=mvp_wan_video_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=%out/wan_extract_%x_%j.out
#SBATCH --error=%out/wan_extract_%x_%j.err

set -euo pipefail

BACKBONE_NAME="wan_video"
# MVP Wan token extraction can hit OOM with large chunk materialization.
# Keep a conservative default and allow submit-time override.
FEATURE_CHUNK_SIZE="${FEATURE_CHUNK_SIZE:-16}"
export BACKBONE_NAME FEATURE_CHUNK_SIZE

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -x "${SUBMIT_DIR}/run_extract.sh" ]]; then
  exec "${SUBMIT_DIR}/run_extract.sh"
fi
if [[ -x "${SUBMIT_DIR}/jobs/extract/mvp/run_extract.sh" ]]; then
  exec "${SUBMIT_DIR}/jobs/extract/mvp/run_extract.sh"
fi
echo "ERROR: could not locate jobs/extract/mvp/run_extract.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
exit 2
