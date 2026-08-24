#!/bin/bash
# Fixed-config seed reruns for MVP V-JEPA 2 / ViT-G/16 temporal_attn.
#
# Manifest rows:
#   results/seed_runs/seed_manifest_mvp_jepa_v2_attentive_layerwise_v1.csv
#
# Scope:
#   layers 10, 20, 30, 40 x seeds 101, 102 = 8 tasks.
#
# These rows use the recovered per-layer LR winners, train for the full
# recovered 30-epoch budget, and disable early stopping.
#
# Usage:
#   sbatch jobs/train/seed_runs/mvp_jepa_v2_attentive_layerwise_gpu_array.sh
#   sbatch jobs/train/seed_runs/mvp_jepa_v2_attentive_layerwise_gpu_array.sh --dry-run

#SBATCH --partition=gpu_a100
#SBATCH --job-name=seed_mvp_j2_attn
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=output/training/seed_runs/mvp_jepa_v2_attentive_layerwise/seed_mvp_j2_attn_%A_%a.out
#SBATCH --error=output/training/seed_runs/mvp_jepa_v2_attentive_layerwise/seed_mvp_j2_attn_%A_%a.err

set -euo pipefail

PROBE_FILTER="temporal_attn"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_mvp_jepa_v2_attentive_layerwise_v1.csv}"
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
