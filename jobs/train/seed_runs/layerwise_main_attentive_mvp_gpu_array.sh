#!/bin/bash
# Fixed-config seed reruns for verified main MVP temporal_attn probes.
#
# Manifest rows selected by probe_hydra=temporal_attn:
#   results/seed_runs/seed_manifest_layerwise_main_attentive_mvp_v1.csv
#
# These rows match the recovered MVP main attentive config epoch budget, but
# disable early stopping so seeds 101/102 run the full 30 epochs.
#
# Usage:
#   sbatch jobs/train/seed_runs/layerwise_main_attentive_mvp_gpu_array.sh
#   sbatch jobs/train/seed_runs/layerwise_main_attentive_mvp_gpu_array.sh --dry-run

#SBATCH --partition=gpu_a100
#SBATCH --job-name=seed_main_attn_mvp
#SBATCH --array=0-23%4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=output/training/seed_runs/layerwise_main_attentive_mvp/seed_main_attn_mvp_%A_%a.out
#SBATCH --error=output/training/seed_runs/layerwise_main_attentive_mvp/seed_main_attn_mvp_%A_%a.err

set -euo pipefail

PROBE_FILTER="temporal_attn"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_layerwise_main_attentive_mvp_v1.csv}"
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
