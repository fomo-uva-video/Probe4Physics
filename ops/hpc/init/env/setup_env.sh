#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=./setup_env%j.out
#SBATCH --error=./setup_env%j.err

set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local seed rel candidate

  # 1) Explicit override wins if provided.
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    if candidate="$(cd "${PROJECT_ROOT}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  fi

  # 2) On Slurm, submit dir and PWD are reliable; walk up a few levels.
  for seed in "${SLURM_SUBMIT_DIR:-}" "${PWD:-}"; do
    [[ -z "${seed}" ]] && continue
    for rel in . .. ../.. ../../.. ../../../..; do
      if candidate="$(cd "${seed}/${rel}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
          echo "${candidate}"
          return 0
        fi
      fi
    done
  done

  # 3) Fallback to script path walk-up for direct/manual execution.
  for rel in . .. ../.. ../../..; do
    if candidate="$(cd "${SCRIPT_DIR}/${rel}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/environment-gpu.yml" && -f "${candidate}/run.py" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  done

  return 1
}

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate the Probe4Physics repository root with environment-gpu.yml" >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', SCRIPT_DIR='${SCRIPT_DIR}'" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/ops/hpc/setup/out"
mkdir -p "${LOG_DIR}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%s)}"
exec > >(tee -a "${LOG_DIR}/install_env_${JOB_TAG}.out") \
     2> >(tee -a "${LOG_DIR}/install_env_${JOB_TAG}.err" >&2)

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd "${REPO_ROOT}"

if conda env list | awk '$1 == "probe4physics-gpu" {found=1} END {exit !found}'; then
  conda env update -n probe4physics-gpu -f environment-gpu.yml --prune
else
  conda env create -f environment-gpu.yml
fi
