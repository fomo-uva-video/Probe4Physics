#!/bin/bash
# Train an IntPhys2 probe with VideoMAE v2 features.
#
# Usage:
#   sbatch videomae_v2.sh
#   sbatch videomae_v2.sh probe.device=cpu

#SBATCH --partition=rome
#SBATCH --job-name=intphys2_videomae_v2_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="videomae_v2"
BACKBONE_VARIANT="vit_giant_16_224"
LINEAR_PROBE_EPOCHS="100"
LINEAR_PROBE_LAYER="last"  # possible values: last | 10 | 20 | 30 | 40
LINEAR_PROBE_FEATURE_VIEW="pooled"
LINEAR_PROBE_DEVICE="cpu"
ENABLE_WANDB="true"
WANDB_PROJECT="probe4physics"
export BACKBONE_NAME BACKBONE_VARIANT LINEAR_PROBE_EPOCHS LINEAR_PROBE_LAYER LINEAR_PROBE_FEATURE_VIEW LINEAR_PROBE_DEVICE ENABLE_WANDB WANDB_PROJECT

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec "${SCRIPT_DIR}/run_train.sh" "$@"
