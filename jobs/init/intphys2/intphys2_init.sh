#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=IntPhys2InitExtract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=jobs/out/intphys2_init_%j.out
#SBATCH --error=jobs/out/intphys2_init_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
# Some conda activate hooks read unset vars; avoid failing under set -u.
set +u
conda activate probe4physics-gpu
set -u

REPO_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${REPO_ROOT}"

USERNAME="${USERNAME:-$USER}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/${USERNAME}/probe4physics}"

METADATA_FILE="${METADATA_FILE:-${SCRATCH_ROOT}/data/annotations/intphys2_metadata.csv}"
VIDEOS_ROOT="${VIDEOS_ROOT:-${SCRATCH_ROOT}/data/videos/intphys2}"
SPLIT_DIR="${SPLIT_DIR:-${SCRATCH_ROOT}/data/splits/intphys2}"
FEATURE_DIR="${FEATURE_DIR:-${SCRATCH_ROOT}/artifacts/features/intphys2}"

BACKBONE_NAME="${BACKBONE_NAME:-jepa_v1}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
BACKBONE_HF_CACHE_DIR="${BACKBONE_HF_CACHE_DIR:-}"
BACKBONE_EXTRA_OVERRIDES="${BACKBONE_EXTRA_OVERRIDES:-}"
CKPT="${CKPT:-${REPO_ROOT}/data/checkpoints/jepa_v1/vitl16.pth.tar}"

if [[ -z "${BACKBONE_VARIANT}" ]]; then
  if [[ "${BACKBONE_NAME}" == "jepa_v1" ]]; then
    BACKBONE_VARIANT="vitl16_224"
  elif [[ "${BACKBONE_NAME}" == "ltx_video" ]]; then
    BACKBONE_VARIANT="ltx_2b_0_9_8_distilled"
  fi
fi

# Hydra list syntax, for example: [main] or [debug,main]
EXTRACT_SPLITS="${EXTRACT_SPLITS:-[main]}"
INCLUDE_POOLED="${INCLUDE_POOLED:-true}"
INCLUDE_TOKENS="${INCLUDE_TOKENS:-true}"
FORCE_REEXTRACT="${FORCE_REEXTRACT:-false}"

# If metadata is missing, optionally auto-download IntPhys2 first.
AUTO_DOWNLOAD_IF_MISSING="${AUTO_DOWNLOAD_IF_MISSING:-True}"

if [[ ! -f "${REPO_ROOT}/run.py" ]]; then
  echo "ERROR: run.py not found in REPO_ROOT='${REPO_ROOT}'" >&2
  exit 2
fi

if [[ ! -f "${METADATA_FILE}" ]]; then
  if [[ "${AUTO_DOWNLOAD_IF_MISSING}" == "true" ]]; then
    echo "Metadata not found, downloading IntPhys2 first..."
    python run.py download.intphys2 \
      metadata_file="${METADATA_FILE}" \
      videos_root="${VIDEOS_ROOT}" \
      download.local_dir="${VIDEOS_ROOT}"
  else
    echo "ERROR: metadata file not found at METADATA_FILE='${METADATA_FILE}'" >&2
    echo "Run download.intphys2 first or set AUTO_DOWNLOAD_IF_MISSING=true" >&2
    exit 2
  fi
fi

if [[ ! -d "${VIDEOS_ROOT}" ]]; then
  echo "ERROR: videos root not found at VIDEOS_ROOT='${VIDEOS_ROOT}'" >&2
  exit 2
fi

if [[ "${BACKBONE_NAME}" == jepa_* ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found at CKPT='${CKPT}'" >&2
    exit 2
  fi
fi

mkdir -p "${SPLIT_DIR}" "${FEATURE_DIR}"

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "METADATA_FILE=${METADATA_FILE}"
echo "VIDEOS_ROOT=${VIDEOS_ROOT}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "FEATURE_DIR=${FEATURE_DIR}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${BACKBONE_VARIANT}"
if [[ "${BACKBONE_NAME}" == jepa_* ]]; then
  echo "CKPT=${CKPT}"
fi
echo "======================"

echo "Step: init.intphys2"
python run.py init.intphys2 \
  metadata_file="${METADATA_FILE}" \
  videos_root="${VIDEOS_ROOT}" \
  split.dir="${SPLIT_DIR}"