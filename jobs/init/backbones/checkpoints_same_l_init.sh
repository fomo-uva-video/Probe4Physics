#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=InitSameLCheckpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=./checkpoints_same_l_init_%j.out
#SBATCH --error=./checkpoints_same_l_init_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local seed rel candidate
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    if candidate="$(cd "${PROJECT_ROOT}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/configs/backbones.yaml" && -f "${candidate}/run.py" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  fi
  for seed in "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${SCRIPT_DIR:-}"; do
    [[ -z "${seed}" ]] && continue
    for rel in . .. ../.. ../../.. ../../../..; do
      if candidate="$(cd "${seed}/${rel}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${candidate}/configs/backbones.yaml" && -f "${candidate}/run.py" ]]; then
          echo "${candidate}"
          return 0
        fi
      fi
    done
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate the Probe4Physics repository root." >&2
  exit 2
fi

cd "${REPO_ROOT}"

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/data/checkpoints}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
mkdir -p "${CHECKPOINTS_ROOT}"

download_one() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -s "${out}" && "${FORCE_DOWNLOAD}" != "true" ]]; then
    echo "SKIP: ${out} already exists"
    return 0
  fi
  local tmp="${out}.tmp"
  rm -f "${tmp}"
  echo "Downloading: ${url}"
  echo " -> ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 5 -o "${tmp}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${url}"
  else
    echo "ERROR: neither curl nor wget is available" >&2
    exit 2
  fi
  if [[ ! -s "${tmp}" ]]; then
    echo "ERROR: download produced empty file: ${tmp}" >&2
    exit 2
  fi
  mv "${tmp}" "${out}"
}

# Official URLs from third_party/jepa_v1/README.md and third_party/vjepa2/README.md.
download_one "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar"   "${CHECKPOINTS_ROOT}/jepa_v1/vitl16.pth.tar"

download_one "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"   "${CHECKPOINTS_ROOT}/jepa_v2/vitl.pth"

download_one "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"   "${CHECKPOINTS_ROOT}/jepa_v2_1/vjepa2_1_vitl_dist_vitG_384.pt"

echo "===== DOWNLOAD SUMMARY ====="
date -u
hostname
echo "REPO_ROOT=${REPO_ROOT}"
echo "CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT}"
echo "FORCE_DOWNLOAD=${FORCE_DOWNLOAD}"
ls -lh   "${CHECKPOINTS_ROOT}/jepa_v1/vitl16.pth.tar"   "${CHECKPOINTS_ROOT}/jepa_v2/vitl.pth"   "${CHECKPOINTS_ROOT}/jepa_v2_1/vjepa2_1_vitl_dist_vitG_384.pt"
