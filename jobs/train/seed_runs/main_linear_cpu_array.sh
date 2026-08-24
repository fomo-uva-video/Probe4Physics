#!/bin/bash
# Fixed-config seed reruns for the main Linear probes.
#
# Manifest rows selected by probe_hydra=linear:
#   results/seed_runs/seed_manifest_main_linear_mlp_v1.csv
#
# Time limit basis from real completed main Linear jobs:
#   max observed elapsed = 00:42:21, old job limit = 02:00:00.
# This wrapper keeps 02:00:00 as a conservative fixed-config limit.
#
# Usage:
#   sbatch jobs/train/seed_runs/main_linear_cpu_array.sh
#   sbatch jobs/train/seed_runs/main_linear_cpu_array.sh --dry-run

#SBATCH --partition=rome
#SBATCH --job-name=seed_main_linear
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/training/seed_runs/linear/seed_main_linear_%A_%a.out
#SBATCH --error=output/training/seed_runs/linear/seed_main_linear_%A_%a.err

set -euo pipefail

PROBE_FILTER="linear"
export PROBE_FILTER

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
