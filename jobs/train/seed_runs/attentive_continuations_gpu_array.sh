#!/bin/bash
# Continue attentive seed runs from official probe_last.pt checkpoints to
# their manifest-defined target epoch, with early stopping disabled.
#
# Build and validate first:
#   python scripts/build_attentive_continuation_manifest.py
#   python scripts/validate_attentive_continuation_manifest.py
#
# Submit with an explicit array range from the manifest row count, for example:
#   N=$(python - <<'PY'
# import csv
# with open("results/seed_runs/attentive_continuation_manifest_v1.csv", newline="") as f:
#     print(sum(1 for _ in csv.DictReader(f)))
# PY
# )
#   sbatch --array=0-$((N - 1))%4 jobs/train/seed_runs/attentive_continuations_gpu_array.sh
#
# The 16h limit is based on the slow MVP same-L V-JEPA 2.1 continuation
# diagnostics: ~3.5h for ~24 epochs, so the worst early-stopped continuations
# can plausibly require around 12h. This keeps margin without using a 24h
# blanket job.

#SBATCH --partition=gpu_a100
#SBATCH --job-name=attn_continue
#SBATCH --array=0-0%4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=output/training/seed_runs/attentive_continuations/attn_continue_%A_%a.out
#SBATCH --error=output/training/seed_runs/attentive_continuations/attn_continue_%A_%a.err

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

MANIFEST_PATH="${ATTENTIVE_CONTINUATION_MANIFEST_PATH:-results/seed_runs/attentive_continuation_manifest_v1.csv}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p output/training/seed_runs/attentive_continuations

echo "===== ATTENTIVE CONTINUATION PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "TASK_INDEX=${TASK_INDEX}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "============================================="

python scripts/run_attentive_continuation_row.py \
  --manifest "${MANIFEST_PATH}" \
  --task-index "${TASK_INDEX}" \
  "probe.wandb.enabled=${ENABLE_WANDB}" \
  "probe.wandb.project=${WANDB_PROJECT}" \
  "probe.wandb.mode=${WANDB_MODE}" \
  "$@"
