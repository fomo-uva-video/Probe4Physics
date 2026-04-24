#!/bin/bash
# CPU-only health layers job.
#
# Runs:
#   python run.py health.layers strict_exit=true
#
# Logs are written to:
#   logs/init/health/layers_health_cpu_<job_id>.log

#SBATCH --partition=rome
#SBATCH --job-name=HealthLayersCPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=./layers_health_cpu_%j.out
#SBATCH --error=./layers_health_cpu_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  local seed rel candidate

  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    if candidate="$(cd "${PROJECT_ROOT}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${candidate}/run.py" && -d "${candidate}/configs" ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  fi

  for seed in "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${SCRIPT_DIR:-}"; do
    [[ -z "${seed}" ]] && continue
    for rel in . .. ../.. ../../.. ../../../..; do
      if candidate="$(cd "${seed}/${rel}" >/dev/null 2>&1 && pwd)"; then
        if [[ -f "${candidate}/run.py" && -d "${candidate}/configs" ]]; then
          echo "${candidate}"
          return 0
        fi
      fi
    done
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
conda activate "${CONDA_ENV_NAME:-probe4physics-gpu}"
set -u

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: Could not locate the Probe4Physics repository root." >&2
  echo "Checked: PROJECT_ROOT='${PROJECT_ROOT:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', PWD='${PWD:-}', SCRIPT_DIR='${SCRIPT_DIR}'" >&2
  exit 2
fi

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs/init/health"
mkdir -p "${LOG_DIR}"

JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_DIR}/layers_health_cpu_${JOB_ID}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===== PROVENANCE ====="
date -u
hostname
git rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME:-probe4physics-gpu}"
echo "======================"

python run.py health.layers strict_exit=true ${HEALTH_LAYERS_EXTRA_OVERRIDES:-}

echo "Completed at:"
date -u
