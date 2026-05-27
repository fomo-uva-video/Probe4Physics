#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=InitNonHFCheckpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=./checkpoints_init%j.out
#SBATCH --error=./checkpoints_init%j.err

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
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', SCRIPT_DIR='${SCRIPT_DIR}'" >&2
  exit 2
fi

cd "${REPO_ROOT}"

# Default location matches configs/backbones.yaml paths.
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/data/checkpoints}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"

if [[ ! -f "${REPO_ROOT}/configs/backbones.yaml" ]]; then
  echo "ERROR: configs/backbones.yaml not found in REPO_ROOT='${REPO_ROOT}'" >&2
  exit 2
fi

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

# Official checkpoint URLs were checked against the vendored upstream READMEs:
#   - third_party/jepa_v1/README.md
#   - third_party/vjepa2/README.md
# Only Hydra default variants from configs/backbones.yaml are downloaded here.

# jepa_v1.default_variant = vith16_384
download_one "https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar" \
  "${CHECKPOINTS_ROOT}/jepa_v1/vith16-384.pth.tar"

# jepa_v2.default_variant = vitg_384
download_one "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt" \
  "${CHECKPOINTS_ROOT}/jepa_v2/vitg-384.pth"

# jepa_v2_1.default_variant = vitG_384
download_one "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt" \
  "${CHECKPOINTS_ROOT}/jepa_v2_1/vjepa2_1_vitG_384.pt"

echo "===== DOWNLOAD SUMMARY ====="
date -u
hostname
echo "REPO_ROOT=${REPO_ROOT}"
echo "CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT}"
echo "FORCE_DOWNLOAD=${FORCE_DOWNLOAD}"
echo "Hydra-default non-HF JEPA checkpoints are present."
