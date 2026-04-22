#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=MVP_init
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=jobs/out/mvp_linear.out
#SBATCH --error=jobs/out/mvp_linear.err

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

USERNAME="${USERNAME:-$USER}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/${USERNAME}/probe4physics}"

ANN_FILE="${ANN_FILE:-${SCRATCH_ROOT}/data/annotations/mvp_full.jsonl}"
MVP_OFFICIAL_ROOT="${MVP_OFFICIAL_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs}"
VIDEOS_ROOT="${VIDEOS_ROOT:-/scratch-shared/${USERNAME}/mvp/videos}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-${SCRATCH_ROOT}/data/cache/videos}"

SPLIT_DIR="${SPLIT_DIR:-${SCRATCH_ROOT}/data/splits/mvp/full_60_20_20}"
FEATURE_DIR="${FEATURE_DIR:-${SCRATCH_ROOT}/artifacts/features/mvp}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-${SCRATCH_ROOT}/artifacts/probes}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${SCRATCH_ROOT}/artifacts/results}"

CKPT="/scratch-shared/scur0511/checkpoints/vitl16.pth.tar"
DEVICE="cuda"

TAG="snellius_mvp_jepa_v1_vitl16"

if [[ ! -f "${MVP_OFFICIAL_ROOT}/tasks/mvp/utils.py" ]]; then
	echo "ERROR: Official MVP repo not found at MVP_OFFICIAL_ROOT='${MVP_OFFICIAL_ROOT}'" >&2
	echo "Expected file: ${MVP_OFFICIAL_ROOT}/tasks/mvp/utils.py" >&2
	echo "Tip: set MVP_OFFICIAL_ROOT='${REPO_ROOT}/third_party/minimal_video_pairs'" >&2
	exit 2
fi

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "FEATURE_DIR=${FEATURE_DIR}"
echo "CKPT=${CKPT}"
echo "======================"

python run.py init.mvp \
annotation_file="${ANN_FILE}" \
official_repo_root="${MVP_OFFICIAL_ROOT}" \
videos_root="${VIDEOS_ROOT}" \
cache_dir="${VIDEO_CACHE_DIR}" \
split.dir="${SPLIT_DIR}" \
output_subdir="${TAG}"