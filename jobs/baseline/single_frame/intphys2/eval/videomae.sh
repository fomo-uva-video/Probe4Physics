#!/bin/bash
# Evaluate the IntPhys2 single-frame baseline with VideoMAE features.
#
# Run this baseline pair from the repository root:
#   mkdir -p output/baseline/single_frame/intphys2/extract output/baseline/single_frame/intphys2/eval
#   extract_jid=$(sbatch --parsable jobs/baseline/single_frame/intphys2/extract/videomae.sh)
#   sbatch --dependency=afterok:${extract_jid} --export=ALL,PROBE_OUTPUT_DIR=/scratch-shared/${USER}/probe4physics/artifacts/probes/intphys2 jobs/baseline/single_frame/intphys2/eval/videomae.sh
# The mkdir is required because Slurm opens stdout/stderr before the script runs.
#
#SBATCH --partition=rome
#SBATCH --job-name=int_sf_eval_videomae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=output/baseline/single_frame/intphys2/eval/%x_%j.out
#SBATCH --error=output/baseline/single_frame/intphys2/eval/%x_%j.err

set -euo pipefail

BACKBONE_NAME="videomae"
BACKBONE_VARIANT="vit_huge_16_224"
PROBE_LAYERS="8,16,24,32"
BASELINE_STAGE="eval"
# Eval loops over PROBE_LAYERS and resolves one normal probe checkpoint per layer.
export BACKBONE_NAME BACKBONE_VARIANT PROBE_LAYERS BASELINE_STAGE

SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr " " "\n" | sed -n "s/^Command=//p" | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR:-$(pwd)}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
DATASET_NAME="$(basename "$(dirname "${SCRIPT_DIR}")")"
BASELINE_LABEL="$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")"
BASELINE_NAME="single_frame"
export DATASET_NAME BASELINE_LABEL BASELINE_NAME

exec "${SCRIPT_DIR}/../../../_shared/run_baseline.sh" "$@"
