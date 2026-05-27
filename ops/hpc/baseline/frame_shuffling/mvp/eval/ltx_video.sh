#!/bin/bash
# Evaluate the MVP frame-shuffling baseline with LTX-Video features.
#
# Run this baseline pair from the repository root:
#   mkdir -p output/baseline/frame_shuffling/mvp/extract output/baseline/frame_shuffling/mvp/eval
#   extract_jid=$(sbatch --parsable ops/hpc/baseline/frame_shuffling/mvp/extract/ltx_video.sh)
#   sbatch --dependency=afterok:${extract_jid} --export=ALL,PROBE_OUTPUT_DIR=${PWD}/artifacts/derived_probe_roots/mvp_probe_linear_ltx_video_complete_40_20260516 ops/hpc/baseline/frame_shuffling/mvp/eval/ltx_video.sh
#   sbatch --dependency=afterok:${extract_jid} --export=ALL,PROBE_NAME=mlp,PROBE_OUTPUT_DIR=/scratch-shared/${USER}/probe4physics/artifacts/probes/mvp/mvp_probe_mlp_ltx_video_complete_40_20260514 ops/hpc/baseline/frame_shuffling/mvp/eval/ltx_video.sh
# The mkdir is required because Slurm opens stdout/stderr before the script runs.
#
#SBATCH --partition=rome
#SBATCH --job-name=mvp_shuffle_eval_ltx_video
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=output/baseline/frame_shuffling/mvp/eval/%x_%j.out
#SBATCH --error=output/baseline/frame_shuffling/mvp/eval/%x_%j.err

set -euo pipefail

BACKBONE_NAME="ltx_video"
BACKBONE_VARIANT="ltxv_13b_0_9_8_distilled"
PROBE_LAYERS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
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
BASELINE_NAME="displacement"
export DATASET_NAME BASELINE_LABEL BASELINE_NAME

exec "${SCRIPT_DIR}/../../../_shared/run_baseline.sh" "$@"
