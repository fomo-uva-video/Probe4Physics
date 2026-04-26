#!/bin/bash
# Train an MVP probe with LTX-Video features.
#
# Usage:
#   sbatch ltx_video.sh

#SBATCH --partition=rome
#SBATCH --job-name=mvp_ltx_video_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="ltx_video"
BACKBONE_VARIANT="ltxv_13b_0_9_8_distilled"
LINEAR_PROBE_EPOCHS="100"
LINEAR_PROBE_DEVICE="cpu"
LINEAR_PROBE_LAYER="last"  # possible values: last | 1 | 2 | 4 | 5
LINEAR_PROBE_FEATURE_VIEW="pooled"
ENABLE_WANDB="true"
WANDB_PROJECT="probe4physics"
export BACKBONE_NAME BACKBONE_VARIANT LINEAR_PROBE_EPOCHS LINEAR_PROBE_DEVICE LINEAR_PROBE_LAYER LINEAR_PROBE_FEATURE_VIEW ENABLE_WANDB WANDB_PROJECT

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec "${SCRIPT_DIR}/run_train.sh" "$@"
