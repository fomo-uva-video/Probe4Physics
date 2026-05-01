#!/bin/bash
# Run IntPhys2 feature extraction for one backbone.
#
# Modes:
#   MODE=full  -> extract with the default config
#   MODE=smoke -> extract only the first 2 ordered samples
#
# Optional:
#   MAX_SAMPLES=<N> overrides MODE-based sample selection and extracts the
#   first N ordered samples. Use this for timing runs such as N=200.
#
# Expected launch style:
#   sbatch jepa_v1.sh
#   MODE=smoke sbatch jepa_v1.sh

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

MODE="${MODE:-full}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
FORCE_REEXTRACT="${FORCE_REEXTRACT:-false}"
INCLUDE_POOLED="${INCLUDE_POOLED:-true}"
INCLUDE_TOKENS="${INCLUDE_TOKENS:-true}"
EFFECTIVE_BACKBONE_VARIANT="$(resolve_backbone_variant "${REPO_ROOT}" "${BACKBONE_NAME}" "${BACKBONE_VARIANT}")"

if [[ "${BACKBONE_NAME}" == "ltx_video" ]]; then
  preflight_ltx_runtime
fi

METADATA_FILE="${METADATA_FILE:-${REPO_ROOT}/data/annotations/intphys2_metadata.csv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO_ROOT}/data/videos/intphys2}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/intphys2}"
FEATURE_DIR="${FEATURE_DIR:-${REPO_ROOT}/artifacts/features/intphys2}"

if [[ ! -f "${METADATA_FILE}" ]]; then
  echo "ERROR: IntPhys2 metadata file not found: ${METADATA_FILE}" >&2
  exit 2
fi

if [[ ! -d "${VIDEOS_ROOT}" ]]; then
  echo "ERROR: IntPhys2 videos root not found: ${VIDEOS_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${SPLIT_DIR}/manifest.json" || ! -f "${SPLIT_DIR}/split_scenes.parquet" ]]; then
  echo "ERROR: IntPhys2 split artifacts not found in ${SPLIT_DIR}" >&2
  echo "Run 'python run.py init.intphys2' first." >&2
  exit 2
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

mkdir -p "${FEATURE_DIR}"

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "MODE=${MODE}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<unset>}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
echo "BACKBONE_DEVICE=${BACKBONE_DEVICE}"
echo "METADATA_FILE=${METADATA_FILE}"
echo "VIDEOS_ROOT=${VIDEOS_ROOT}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "FEATURE_DIR=${FEATURE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "======================"

cmd=(
  python run.py extract.intphys2
  "metadata_file=${METADATA_FILE}"
  "videos_root=${VIDEOS_ROOT}"
  "split.dir=${SPLIT_DIR}"
  "feature_cache.dir=${FEATURE_DIR}"
  "feature_cache.include_pooled=${INCLUDE_POOLED}"
  "feature_cache.include_tokens=${INCLUDE_TOKENS}"
  "feature_cache.force_reextract=${FORCE_REEXTRACT}"
  "backbone.name=${BACKBONE_NAME}"
  "backbone.kwargs.device=${BACKBONE_DEVICE}"
)

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

if [[ -n "${MAX_SAMPLES_OVERRIDE}" ]]; then
  cmd+=("${MAX_SAMPLES_OVERRIDE}")
fi

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

"${cmd[@]}"

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"

echo "Completed at:"
date -u
