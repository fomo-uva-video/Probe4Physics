#!/bin/bash
# Validate exact Same-L / ViT-L feature caches before releasing seed training arrays.

#SBATCH --partition=rome
#SBATCH --job-name=validate_same_l_caches
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=output/training/seed_runs/layerwise_same_l_validate/validate_same_l_caches_%j.out
#SBATCH --error=output/training/seed_runs/layerwise_same_l_validate/validate_same_l_caches_%j.err

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_DETAILS="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true)"
  for JOB_FIELD in ${JOB_DETAILS}; do
    case "${JOB_FIELD}" in
      Command=*)
        SCRIPT_PATH="${JOB_FIELD#Command=}"
        break
        ;;
    esac
  done
fi
if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR:-$(pwd)}/${SCRIPT_PATH}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

mkdir -p output/training/seed_runs/layerwise_same_l_validate

python scripts/validate_seed_manifest_feature_caches.py   --manifest results/seed_runs/seed_manifest_layerwise_same_l_linear_mlp_v1.csv   --manifest results/seed_runs/seed_manifest_layerwise_same_l_attentive_v1.csv
