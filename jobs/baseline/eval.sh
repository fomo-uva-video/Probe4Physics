#!/bin/bash
# Evaluate a normal probe checkpoint on one temporal-baseline test cache.
#
# Expected launch style:
#   DATASET_NAME=mvp BASELINE_NAME=single_frame BACKBONE_NAME=jepa_v1 BACKBONE_VARIANT=vith16_384 sbatch eval.sh
#   PROBE_CHECKPOINT_PATH=/path/to/probe_best.pt DATASET_NAME=intphys2 BASELINE_NAME=displacement BACKBONE_NAME=videomae sbatch eval.sh

#SBATCH --partition=rome
#SBATCH --job-name=baseline_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

DATASET_NAME="${DATASET_NAME:?DATASET_NAME must be mvp or intphys2}"
BASELINE_NAME="${BASELINE_NAME:?BASELINE_NAME must be single_frame or displacement}"
BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
PROBE_NAME="${PROBE_NAME:-linear}"
PROBE_LAYER="${PROBE_LAYER:-last}"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-pooled}"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"
PROBE_CHECKPOINT_PATH="${PROBE_CHECKPOINT_PATH:-}"
FEATURE_DIR="${FEATURE_DIR:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-}"

if [[ "${DATASET_NAME}" != "mvp" && "${DATASET_NAME}" != "intphys2" ]]; then
  echo "ERROR: unsupported DATASET_NAME='${DATASET_NAME}'. Use mvp or intphys2." >&2
  exit 2
fi

if [[ "${BASELINE_NAME}" != "single_frame" && "${BASELINE_NAME}" != "displacement" ]]; then
  echo "ERROR: unsupported BASELINE_NAME='${BASELINE_NAME}'. Use single_frame or displacement." >&2
  exit 2
fi

EFFECTIVE_BACKBONE_VARIANT="$(resolve_backbone_variant "${REPO_ROOT}" "${BACKBONE_NAME}" "${BACKBONE_VARIANT}")"
BACKBONE_TAG="${BACKBONE_NAME}_${EFFECTIVE_BACKBONE_VARIANT}"

if [[ "${DATASET_NAME}" == "mvp" ]]; then
  TRAIN_PREFIX="mvp_probe"
  PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${REPO_ROOT}/artifacts/probes}"
  ANNOTATION_FILE="${ANNOTATION_FILE:-${REPO_ROOT}/data/annotations/mvp_full.jsonl}"
  OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-/scratch-shared/${USER}/probe4physics/data/videos/mvp}"
  CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/cache/videos}"
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/mvp/full_60_20_20}"
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/mvp_${BASELINE_NAME}}"

  [[ -f "${ANNOTATION_FILE}" ]] || { echo "ERROR: MVP annotation file not found: ${ANNOTATION_FILE}" >&2; exit 2; }
  [[ -f "${OFFICIAL_REPO_ROOT}/tasks/mvp/utils.py" ]] || { echo "ERROR: MVP official repo not found: ${OFFICIAL_REPO_ROOT}" >&2; exit 2; }
  [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: MVP videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
  [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_pairs.parquet" ]] || { echo "ERROR: MVP split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }
else
  TRAIN_PREFIX="intphys2_probe"
  PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${REPO_ROOT}/artifacts/probes/intphys2}"
  METADATA_FILE="${METADATA_FILE:-${REPO_ROOT}/data/annotations/intphys2_metadata.csv}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO_ROOT}/data/videos/intphys2}"
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/intphys2}"
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/intphys2_${BASELINE_NAME}}"

  [[ -f "${METADATA_FILE}" ]] || { echo "ERROR: IntPhys2 metadata file not found: ${METADATA_FILE}" >&2; exit 2; }
  [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: IntPhys2 videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
  [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_scenes.parquet" ]] || { echo "ERROR: IntPhys2 split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }
fi

resolve_checkpoint() {
  if [[ -n "${PROBE_CHECKPOINT_PATH}" ]]; then
    printf '%s\n' "${PROBE_CHECKPOINT_PATH}"
    return 0
  fi

  local layer_path=""
  if [[ "${PROBE_LAYER}" != "last" ]]; then
    layer_path="$(find "${PROBE_OUTPUT_DIR}" -path "*/${TRAIN_PREFIX}_${PROBE_NAME}_${BACKBONE_TAG}_*/layer_${PROBE_LAYER}/train/probe_best.pt" -type f 2>/dev/null | sort | tail -n 1)"
    if [[ -n "${layer_path}" ]]; then
      printf '%s\n' "${layer_path}"
      return 0
    fi
  fi

  find "${PROBE_OUTPUT_DIR}" -path "*/${TRAIN_PREFIX}_${PROBE_NAME}_${BACKBONE_TAG}_*/probe_best.pt" -type f 2>/dev/null | sort | tail -n 1
}

RESOLVED_CHECKPOINT="$(resolve_checkpoint)"
if [[ -z "${RESOLVED_CHECKPOINT}" || ! -f "${RESOLVED_CHECKPOINT}" ]]; then
  echo "ERROR: Could not resolve a probe checkpoint." >&2
  echo "Set PROBE_CHECKPOINT_PATH explicitly, or train the normal ${DATASET_NAME} probe first." >&2
  echo "Searched root: ${PROBE_OUTPUT_DIR}" >&2
  echo "Expected backbone tag: ${BACKBONE_TAG}" >&2
  exit 2
fi

EVAL_SUBDIR="${EVAL_SUBDIR:-${DATASET_NAME}_${BASELINE_NAME}_${BACKBONE_TAG}_${PROBE_NAME}_layer_${PROBE_LAYER}_${SLURM_JOB_ID:-manual}}"

cmd=(python run.py "eval.probe.${DATASET_NAME}.${BASELINE_NAME}")

if [[ "${DATASET_NAME}" == "mvp" ]]; then
  cmd+=(
    "annotation_file=${ANNOTATION_FILE}"
    "official_repo_root=${OFFICIAL_REPO_ROOT}"
    "videos_root=${VIDEOS_ROOT}"
    "cache_dir=${CACHE_DIR}"
    "split.dir=${SPLIT_DIR}"
    "annotations.auto_download=false"
  )
else
  cmd+=(
    "metadata_file=${METADATA_FILE}"
    "videos_root=${VIDEOS_ROOT}"
    "split.dir=${SPLIT_DIR}"
  )
fi

cmd+=(
  "backbone.name=${BACKBONE_NAME}"
  "probe.name=${PROBE_NAME}"
  "probe.layer=${PROBE_LAYER}"
  "probe.feature_view=${PROBE_FEATURE_VIEW}"
  "probe.device=${PROBE_DEVICE}"
  "probe.checkpoint_path=${RESOLVED_CHECKPOINT}"
  "probe.eval_output_dir=${EVAL_OUTPUT_DIR}"
  "probe.eval_output_subdir=${EVAL_SUBDIR}"
)

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

if [[ -n "${FEATURE_DIR}" ]]; then
  cmd+=("feature_cache.dir=${FEATURE_DIR}")
fi

mkdir -p output/baseline/eval "${EVAL_OUTPUT_DIR}"

echo "===== BASELINE EVAL PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "BASELINE_NAME=${BASELINE_NAME}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
echo "PROBE_NAME=${PROBE_NAME}"
echo "PROBE_LAYER=${PROBE_LAYER}"
echo "PROBE_FEATURE_VIEW=${PROBE_FEATURE_VIEW}"
echo "PROBE_DEVICE=${PROBE_DEVICE}"
echo "PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR}"
echo "PROBE_CHECKPOINT_PATH=${RESOLVED_CHECKPOINT}"
echo "FEATURE_DIR=${FEATURE_DIR:-<config default>}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "EVAL_SUBDIR=${EVAL_SUBDIR}"
echo "HF_HOME=${HF_HOME}"
echo "Command: ${cmd[*]}"
echo "===================================="

JOB_START_EPOCH="$(date +%s)"
"${cmd[@]}"
JOB_END_EPOCH="$(date +%s)"
echo "JOB_ELAPSED_SECONDS=$((JOB_END_EPOCH - JOB_START_EPOCH))"
date -u
