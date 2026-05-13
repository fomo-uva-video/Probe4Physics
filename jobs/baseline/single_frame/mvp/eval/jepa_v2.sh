#!/bin/bash
# Evaluate the MVP single-frame baseline with V-JEPA v2 features.

#SBATCH --partition=rome
#SBATCH --job-name=mvp_sf_eval_jepa_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BACKBONE_NAME="jepa_v2"
BACKBONE_VARIANT="vitg_384"
PROBE_LAYERS="10,20,30,40"
BASELINE_STAGE="eval"
# Eval loops over PROBE_LAYERS and resolves one normal probe checkpoint per layer.
export BACKBONE_NAME BACKBONE_VARIANT PROBE_LAYERS BASELINE_STAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME="$(basename "$(dirname "${SCRIPT_DIR}")")"
BASELINE_LABEL="$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")"
BASELINE_NAME="single_frame"
export DATASET_NAME BASELINE_LABEL BASELINE_NAME

exec "${SCRIPT_DIR}/../../../_shared/run_baseline.sh" "$@"
