#!/bin/bash
# Extract temporal-baseline features for one dataset/backbone on Snellius.
#
# Expected launch style:
#   DATASET_NAME=mvp BASELINE_NAME=single_frame BACKBONE_NAME=jepa_v1 BACKBONE_VARIANT=vith16_384 sbatch extract.sh
#   MODE=smoke DATASET_NAME=intphys2 BASELINE_NAME=displacement BACKBONE_NAME=videomae BACKBONE_VARIANT=vit_huge_16_224 sbatch extract.sh

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=baseline_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
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
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
MODE="${MODE:-full}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
FORCE_REEXTRACT="${FORCE_REEXTRACT:-}"
INCLUDE_POOLED="${INCLUDE_POOLED:-}"
INCLUDE_TOKENS="${INCLUDE_TOKENS:-}"
FEATURE_DIR="${FEATURE_DIR:-}"

if [[ "${DATASET_NAME}" != "mvp" && "${DATASET_NAME}" != "intphys2" ]]; then
  echo "ERROR: unsupported DATASET_NAME='${DATASET_NAME}'. Use mvp or intphys2." >&2
  exit 2
fi

if [[ "${BASELINE_NAME}" != "single_frame" && "${BASELINE_NAME}" != "displacement" ]]; then
  echo "ERROR: unsupported BASELINE_NAME='${BASELINE_NAME}'. Use single_frame or displacement." >&2
  exit 2
fi

EFFECTIVE_BACKBONE_VARIANT="$(resolve_backbone_variant "${REPO_ROOT}" "${BACKBONE_NAME}" "${BACKBONE_VARIANT}")"

if [[ "${BACKBONE_NAME}" == "ltx_video" ]]; then
  preflight_ltx_runtime
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_OVERRIDE="feature_cache.max_samples=${MAX_SAMPLES}"
elif [[ "${MODE}" == "full" ]]; then
  MAX_SAMPLES_OVERRIDE=""
elif [[ "${MODE}" == "smoke" ]]; then
  MAX_SAMPLES_OVERRIDE="feature_cache.max_samples=2"
else
  echo "ERROR: unsupported MODE='${MODE}'. Use MODE=full or MODE=smoke." >&2
  exit 2
fi

cmd=(python run.py "extract.${DATASET_NAME}.${BASELINE_NAME}")

if [[ "${DATASET_NAME}" == "mvp" ]]; then
  ANNOTATION_FILE="${ANNOTATION_FILE:-${REPO_ROOT}/data/annotations/mvp_full.jsonl}"
  OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-/scratch-shared/${USER}/probe4physics/data/videos/mvp}"
  CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/cache/videos}"
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/mvp/full_60_20_20}"

  [[ -f "${ANNOTATION_FILE}" ]] || { echo "ERROR: MVP annotation file not found: ${ANNOTATION_FILE}" >&2; exit 2; }
  [[ -f "${OFFICIAL_REPO_ROOT}/tasks/mvp/utils.py" ]] || { echo "ERROR: MVP official repo not found: ${OFFICIAL_REPO_ROOT}" >&2; exit 2; }
  [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: MVP videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
  [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_pairs.parquet" ]] || { echo "ERROR: MVP split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }

  cmd+=(
    "annotation_file=${ANNOTATION_FILE}"
    "official_repo_root=${OFFICIAL_REPO_ROOT}"
    "videos_root=${VIDEOS_ROOT}"
    "cache_dir=${CACHE_DIR}"
    "split.dir=${SPLIT_DIR}"
    "annotations.auto_download=false"
  )
else
  METADATA_FILE="${METADATA_FILE:-${REPO_ROOT}/data/annotations/intphys2_metadata.csv}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO_ROOT}/data/videos/intphys2}"
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/intphys2}"

  [[ -f "${METADATA_FILE}" ]] || { echo "ERROR: IntPhys2 metadata file not found: ${METADATA_FILE}" >&2; exit 2; }
  [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: IntPhys2 videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
  [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_scenes.parquet" ]] || { echo "ERROR: IntPhys2 split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }

  cmd+=(
    "metadata_file=${METADATA_FILE}"
    "videos_root=${VIDEOS_ROOT}"
    "split.dir=${SPLIT_DIR}"
  )
fi

cmd+=(
  "backbone.name=${BACKBONE_NAME}"
  "backbone.kwargs.device=${BACKBONE_DEVICE}"
)

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

if [[ -n "${FEATURE_DIR}" ]]; then
  mkdir -p "${FEATURE_DIR}"
  cmd+=("feature_cache.dir=${FEATURE_DIR}")
fi

if [[ -n "${INCLUDE_POOLED}" ]]; then
  cmd+=("feature_cache.include_pooled=${INCLUDE_POOLED}")
fi

if [[ -n "${INCLUDE_TOKENS}" ]]; then
  cmd+=("feature_cache.include_tokens=${INCLUDE_TOKENS}")
fi

if [[ -n "${FORCE_REEXTRACT}" ]]; then
  cmd+=("feature_cache.force_reextract=${FORCE_REEXTRACT}")
fi

if [[ -n "${MAX_SAMPLES_OVERRIDE}" ]]; then
  cmd+=("${MAX_SAMPLES_OVERRIDE}")
fi

mkdir -p output/baseline/extract

echo "===== BASELINE EXTRACTION PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "BASELINE_NAME=${BASELINE_NAME}"
echo "MODE=${MODE}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<unset>}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
echo "BACKBONE_DEVICE=${BACKBONE_DEVICE}"
echo "FEATURE_DIR=${FEATURE_DIR:-<config default>}"
echo "HF_HOME=${HF_HOME}"
echo "Command: ${cmd[*]}"
echo "=========================================="

JOB_START_EPOCH="$(date +%s)"
"${cmd[@]}"
JOB_END_EPOCH="$(date +%s)"
echo "JOB_ELAPSED_SECONDS=$((JOB_END_EPOCH - JOB_START_EPOCH))"
date -u
