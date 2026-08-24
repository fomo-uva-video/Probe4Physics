#!/bin/bash
# Fixed-config seed reruns for all verified main IntPhys2 temporal_attn probes.
#
# Manifest rows selected by probe_hydra=temporal_attn:
#   results/seed_runs/seed_manifest_layerwise_main_attentive_intphys2_v1.csv
#
# Time limit basis from prior comparable IntPhys2 temporal_attn 90ep/pat20 logs:
#   output/training/intphys2/attention/*_attn_*_lr_matrix_*.out
#   n=96 completed tasks, max observed elapsed = 10224s (2.84h), slowest on jepa_v2_1.
# This wrapper uses 06:00:00, which is >2x the observed max while avoiding the old
# 24h blanket used by broad LR-matrix jobs.
#
# Usage:
#   sbatch jobs/train/seed_runs/layerwise_main_attentive_intphys2_gpu_array.sh
#   sbatch jobs/train/seed_runs/layerwise_main_attentive_intphys2_gpu_array.sh --dry-run

#SBATCH --partition=gpu_a100
#SBATCH --job-name=seed_lw_attn_intphys2
#SBATCH --array=0-39%4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=output/training/seed_runs/layerwise_attentive_intphys2/seed_lw_attn_intphys2_%A_%a.out
#SBATCH --error=output/training/seed_runs/layerwise_attentive_intphys2/seed_lw_attn_intphys2_%A_%a.err

set -euo pipefail

PROBE_FILTER="temporal_attn"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_layerwise_main_attentive_intphys2_v1.csv}"
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
