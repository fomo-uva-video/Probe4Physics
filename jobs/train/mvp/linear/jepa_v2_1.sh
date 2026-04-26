#!/bin/bash
# Train an MVP probe with V-JEPA v2.1 features.
#
# Usage:
#   sbatch jepa_v2_1.sh

#SBATCH --partition=rome
#SBATCH --job-name=mvp_jepa_v2_1_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2_1"
BACKBONE_VARIANT="vitG_384"
LINEAR_PROBE_EPOCHS="100"
LINEAR_PROBE_DEVICE="cpu"
LINEAR_PROBE_LAYER="last"  # possible values: last | 12 | 24 | 38 | 48
LINEAR_PROBE_FEATURE_VIEW="pooled"
ENABLE_WANDB="true"
WANDB_PROJECT="probe4physics"
export BACKBONE_NAME BACKBONE_VARIANT LINEAR_PROBE_EPOCHS LINEAR_PROBE_DEVICE LINEAR_PROBE_LAYER LINEAR_PROBE_FEATURE_VIEW ENABLE_WANDB WANDB_PROJECT

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec "${SCRIPT_DIR}/run_train.sh" "$@"
