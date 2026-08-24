#!/bin/bash
# Fixed-config seed reruns for backbone-sweep Linear probes.
# Manifest: results/seed_runs/seed_manifest_layerwise_backbone_sweep_linear_mlp_v1.csv

#SBATCH --partition=rome
#SBATCH --job-name=seed_bs_linear
#SBATCH --array=0-47%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/training/seed_runs/layerwise_backbone_sweep_linear/seed_bs_linear_%A_%a.out
#SBATCH --error=output/training/seed_runs/layerwise_backbone_sweep_linear/seed_bs_linear_%A_%a.err

set -euo pipefail

PROBE_FILTER="linear"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_layerwise_backbone_sweep_linear_mlp_v1.csv}"
export PROBE_FILTER SEED_MANIFEST_PATH

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

exec "${SCRIPT_DIR}/run_manifest_task.sh" "$@"
