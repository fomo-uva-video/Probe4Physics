#!/bin/bash
# Train an IntPhys2 linear probe with V-JEPA v2 features.
#
# Usage:
#   sbatch jepa_v2.sh
#   sbatch jepa_v2.sh linear_probe.device=cuda

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=intphys2_jepa_v2_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2"
LINEAR_PROBE_LAYER="last"  # possible values: last | 10 | 20 | 30 | 40
LINEAR_PROBE_FEATURE_VIEW="pooled"
ENABLE_WANDB="true"
WANDB_PROJECT="probe4physics"
export BACKBONE_NAME LINEAR_PROBE_LAYER LINEAR_PROBE_FEATURE_VIEW ENABLE_WANDB WANDB_PROJECT

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec "${SCRIPT_DIR}/run_train.sh" "$@"
