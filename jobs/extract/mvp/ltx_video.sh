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
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="ltx_video"
export BACKBONE_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXTRACT=""
for candidate in \
  "${SCRIPT_DIR}/run_extract.sh" \
  "${SLURM_SUBMIT_DIR:-}/run_extract.sh" \
  "${SLURM_SUBMIT_DIR:-}/jobs/extract/mvp/run_extract.sh"; do
  if [[ -x "${candidate}" ]]; then
    RUN_EXTRACT="${candidate}"
    break
  fi
done

if [[ -z "${RUN_EXTRACT}" ]]; then
  echo "ERROR: Could not locate jobs/extract/mvp/run_extract.sh" >&2
  exit 2
fi

exec "${RUN_EXTRACT}"
