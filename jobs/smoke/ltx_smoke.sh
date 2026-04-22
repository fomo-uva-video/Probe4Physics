#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=LTXSmoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=jobs/out/ltx_smoke_%j.out
#SBATCH --error=jobs/out/ltx_smoke_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate probe4physics-gpu
set -u

REPO_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/run.py" ]]; then
  echo "ERROR: run.py not found in REPO_ROOT='${REPO_ROOT}'. Set PROJECT_ROOT or submit from repo root." >&2
  exit 2
fi

mkdir -p jobs/setup/out

BACKBONE_VARIANT="${BACKBONE_VARIANT:-ltx_2b_0_9_8_distilled}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
BACKBONE_HF_CACHE_DIR="${BACKBONE_HF_CACHE_DIR:-}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-true}"
RUN_EXTRACT_SMOKE="${RUN_EXTRACT_SMOKE:-true}"
EXTRACT_BENCHMARK="${EXTRACT_BENCHMARK:-intphys2}"  # intphys2 | mvp

USERNAME="${USERNAME:-$USER}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/${USERNAME}/probe4physics}"

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "BACKBONE_VARIANT=${BACKBONE_VARIANT}"
echo "BACKBONE_DEVICE=${BACKBONE_DEVICE}"
echo "EXTRACT_BENCHMARK=${EXTRACT_BENCHMARK}"
echo "======================"

echo "Step 1/3: LTX adapter smoke"
smoke_cmd=(
  python experiments/smoke_ltx_video.py
  --variant "${BACKBONE_VARIANT}"
  --device "${BACKBONE_DEVICE}"
  --batch-size 1
)
if [[ -n "${BACKBONE_HF_CACHE_DIR}" ]]; then
  smoke_cmd+=(--hf-cache-dir "${BACKBONE_HF_CACHE_DIR}")
fi
"${smoke_cmd[@]}"

if [[ "${RUN_UNIT_TESTS}" == "true" ]]; then
  echo "Step 2/3: targeted unit tests"
  python -m unittest tests.test_ltx_video_adapter tests.test_run_commands
else
  echo "Step 2/3: skipped unit tests (RUN_UNIT_TESTS=${RUN_UNIT_TESTS})"
fi

if [[ "${RUN_EXTRACT_SMOKE}" != "true" ]]; then
  echo "Step 3/3: skipped extract smoke (RUN_EXTRACT_SMOKE=${RUN_EXTRACT_SMOKE})"
  echo "Completed at:"
  date -u
  exit 0
fi

echo "Step 3/3: extract smoke (${EXTRACT_BENCHMARK})"
if [[ "${EXTRACT_BENCHMARK}" == "intphys2" ]]; then
  METADATA_FILE="${METADATA_FILE:-${SCRATCH_ROOT}/data/annotations/intphys2_metadata.csv}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-${SCRATCH_ROOT}/data/videos/intphys2}"
  SPLIT_DIR="${SPLIT_DIR:-${SCRATCH_ROOT}/data/splits/intphys2}"
  FEATURE_DIR="${FEATURE_DIR:-${SCRATCH_ROOT}/artifacts/features/intphys2_ltx_smoke}"

  if [[ ! -f "${METADATA_FILE}" ]]; then
    echo "ERROR: METADATA_FILE not found: ${METADATA_FILE}" >&2
    exit 2
  fi
  if [[ ! -d "${VIDEOS_ROOT}" ]]; then
    echo "ERROR: VIDEOS_ROOT not found: ${VIDEOS_ROOT}" >&2
    exit 2
  fi

  cmd=(
    python run.py extract.intphys2
    metadata_file="${METADATA_FILE}"
    videos_root="${VIDEOS_ROOT}"
    split.dir="${SPLIT_DIR}"
    feature_cache.dir="${FEATURE_DIR}"
    feature_cache.split_names=[main]
    feature_cache.include_pooled=true
    feature_cache.include_tokens=false
    feature_cache.force_reextract=false
    decode.num_frames=4
    decode.sampling=uniform
    backbone.name=ltx_video
    "backbone.kwargs.device=${BACKBONE_DEVICE}"
    "+backbone.kwargs.variant=${BACKBONE_VARIANT}"
  )

  if [[ -n "${BACKBONE_HF_CACHE_DIR}" ]]; then
    cmd+=("+backbone.kwargs.hf_cache_dir=${BACKBONE_HF_CACHE_DIR}")
  fi

  "${cmd[@]}"
elif [[ "${EXTRACT_BENCHMARK}" == "mvp" ]]; then
  ANN_FILE="${ANN_FILE:-${SCRATCH_ROOT}/data/annotations/mvp_full.jsonl}"
  MVP_OFFICIAL_ROOT="${MVP_OFFICIAL_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs}"
  VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs/videos}"
  SPLIT_DIR="${SPLIT_DIR:-${SCRATCH_ROOT}/data/splits/mvp/full_60_20_20}"
  FEATURE_DIR="${FEATURE_DIR:-${SCRATCH_ROOT}/artifacts/features/mvp_ltx_smoke}"

  if [[ ! -f "${ANN_FILE}" ]]; then
    echo "ERROR: ANN_FILE not found: ${ANN_FILE}" >&2
    exit 2
  fi
  if [[ ! -d "${MVP_OFFICIAL_ROOT}" ]]; then
    echo "ERROR: MVP_OFFICIAL_ROOT not found: ${MVP_OFFICIAL_ROOT}" >&2
    exit 2
  fi
  if [[ ! -d "${VIDEOS_ROOT}" ]]; then
    echo "ERROR: VIDEOS_ROOT not found: ${VIDEOS_ROOT}" >&2
    exit 2
  fi

  cmd=(
    python run.py extract.mvp
    annotation_file="${ANN_FILE}"
    official_repo_root="${MVP_OFFICIAL_ROOT}"
    videos_root="${VIDEOS_ROOT}"
    split.dir="${SPLIT_DIR}"
    feature_cache.dir="${FEATURE_DIR}"
    feature_cache.split_names=[val]
    feature_cache.include_pooled=true
    feature_cache.include_tokens=false
    feature_cache.force_reextract=false
    decode.num_frames=4
    decode.sampling=uniform
    backbone.name=ltx_video
    "backbone.kwargs.device=${BACKBONE_DEVICE}"
    "+backbone.kwargs.variant=${BACKBONE_VARIANT}"
  )

  if [[ -n "${BACKBONE_HF_CACHE_DIR}" ]]; then
    cmd+=("+backbone.kwargs.hf_cache_dir=${BACKBONE_HF_CACHE_DIR}")
  fi

  "${cmd[@]}"
else
  echo "ERROR: Unsupported EXTRACT_BENCHMARK='${EXTRACT_BENCHMARK}'. Use intphys2 or mvp." >&2
  exit 2
fi

echo "Completed at:"
date -u
