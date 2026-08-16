#!/bin/bash
# Fixed-config seed reruns for backbone-sweep temporal_attn probes.
# Manifest: results/seed_runs/seed_manifest_layerwise_backbone_sweep_attentive_v1.csv

#SBATCH --partition=gpu_a100
#SBATCH --job-name=seed_bs_attn
#SBATCH --array=0-47%4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=output/training/seed_runs/layerwise_backbone_sweep_attentive/seed_bs_attn_%A_%a.out
#SBATCH --error=output/training/seed_runs/layerwise_backbone_sweep_attentive/seed_bs_attn_%A_%a.err

set -euo pipefail

PROBE_FILTER="temporal_attn"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-results/seed_runs/seed_manifest_layerwise_backbone_sweep_attentive_v1.csv}"
export PROBE_FILTER SEED_MANIFEST_PATH

SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr ' ' '
' | sed -n 's/^Command=//p' | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

exec "${SCRIPT_DIR}/run_manifest_task.sh" "$@"
