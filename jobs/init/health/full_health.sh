#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=FullHealth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=./full_health_%j.out
#SBATCH --error=./full_health_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local seed rel candidate

  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    if candidate="$(cd "${PROJECT_ROOT}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  fi

  for seed in "${SLURM_SUBMIT_DIR:-}" "${PWD:-}"; do
    [[ -z "${seed}" ]] && continue
    for rel in . .. ../.. ../../.. ../../../.. ../../../../..; do
      if candidate="$(cd "${seed}/${rel}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
          echo "${candidate}"
          return 0
        fi
      fi
    done
  done

  for rel in . .. ../.. ../../.. ../../../..; do
    if candidate="$(cd "${SCRIPT_DIR}/${rel}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  done

  return 1
}

CONDA_BIN="$(command -v conda)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "ERROR: 'conda' not found after loading Anaconda3 module." >&2
  exit 2
fi

CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." >/dev/null 2>&1 && pwd)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found under CONDA_BASE='${CONDA_BASE}'." >&2
  exit 2
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate probe4physics-gpu
set -u

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate the Probe4Physics repository root." >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', SCRIPT_DIR='${SCRIPT_DIR}'" >&2
  exit 2
fi

cd "${REPO_ROOT}"

mkdir -p jobs/out

USERNAME="${USERNAME:-$USER}"
HF_HOME="${HF_HOME:-/scratch-shared/${USERNAME}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
HEALTH_DEVICE="${HEALTH_DEVICE:-cuda}"
HEALTH_EXTRA_OVERRIDES="${HEALTH_EXTRA_OVERRIDES:-}"

export HF_HOME
export TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "HEALTH_DEVICE=${HEALTH_DEVICE}"
echo "======================"

echo "===== TORCH RUNTIME ====="
python - <<'PY'
import torch
print("torch_version=", torch.__version__, sep="")
print("torch_cuda_version=", torch.version.cuda, sep="")
print("cuda_available=", torch.cuda.is_available(), sep="")
print("cuda_device_count=", torch.cuda.device_count(), sep="")
PY
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
echo "========================="

if [[ "${HEALTH_DEVICE}" == "cuda" ]]; then
  python - <<'PY'
import sys
import torch

if torch.version.cuda is None:
    print("ERROR: Installed torch is CPU-only (torch.version.cuda is None).", file=sys.stderr)
    raise SystemExit(2)

if not torch.cuda.is_available():
    print("ERROR: torch has CUDA support but torch.cuda.is_available() is False.", file=sys.stderr)
    raise SystemExit(2)
PY
fi

cmd=(
  python run.py health
  synthetic_forward=true
  "device=${HEALTH_DEVICE}"
)

if [[ -n "${HEALTH_EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides=( ${HEALTH_EXTRA_OVERRIDES} )
  cmd+=("${extra_overrides[@]}")
fi

"${cmd[@]}"

echo "Completed at:"
date -u
