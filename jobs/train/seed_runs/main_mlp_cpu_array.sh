#!/bin/bash
# Fixed-config seed reruns for the main MLP probes.
#
# Manifest rows selected by probe_hydra=mlp:
#   results/seed_runs/seed_manifest_main_linear_mlp_v1.csv
#
# Time limit basis from real completed main MLP jobs:
#   max observed elapsed = 03:35:58, old job limits were 02:00:00-12:00:00.
# This wrapper uses 06:00:00 as a conservative fixed-config limit.
#
# Usage:
#   sbatch jobs/train/seed_runs/main_mlp_cpu_array.sh
#   sbatch jobs/train/seed_runs/main_mlp_cpu_array.sh --dry-run

#SBATCH --partition=rome
#SBATCH --job-name=seed_main_mlp
#SBATCH --array=0-19%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=output/training/seed_runs/mlp/seed_main_mlp_%A_%a.out
#SBATCH --error=output/training/seed_runs/mlp/seed_main_mlp_%A_%a.err

set -euo pipefail

PROBE_FILTER="mlp"
export PROBE_FILTER

SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

exec "${SCRIPT_DIR}/run_manifest_task.sh" "$@"
