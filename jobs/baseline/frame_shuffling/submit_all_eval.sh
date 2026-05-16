#!/bin/bash
# Submit frame-shuffling baseline eval jobs for MLP probes.
#
# Run after the matching extraction jobs have completed:
#   bash jobs/baseline/frame_shuffling/submit_all_eval.sh
#
# This mirrors the eval-side sbatch command documented at the top of each
# eval wrapper: it submits only eval jobs and exports PROBE_NAME plus the
# normal-probe checkpoint root via PROBE_OUTPUT_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

SLURM_DEVICE="${SLURM_DEVICE:-gpu_h100}"
SLURM_GPUS="${SLURM_GPUS:-1}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"

DATASETS=(
#  mvp
  intphys2
)

BACKBONES=(
  jepa_v1
  jepa_v2
  jepa_v2_1
  videomae
  videomae_v2
#  ltx_video
)

probe_output_dir() {
  local dataset="${1}"
  local backbone="${2}"

  if [[ "${dataset}" == "mvp" && "${backbone}" == "ltx_video" ]]; then
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/mvp/mvp_probe_mlp_ltx_video_complete_40_20260514\n' "${USER}"
  elif [[ "${dataset}" == "intphys2" && "${backbone}" == "ltx_video" ]]; then
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/intphys2/intphys2_probe_mlp_ltx_video_ltxv_13b_0_9_8_distilled_20260509T102816Z\n' "${USER}"
  else
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/%s\n' "${USER}" "${dataset}"
  fi
}

for dataset in "${DATASETS[@]}"; do
  mkdir -p "output/baseline/frame_shuffling/${dataset}/eval"

  for backbone in "${BACKBONES[@]}"; do
    eval_job="jobs/baseline/frame_shuffling/${dataset}/eval/${backbone}.sh"
    output_dir="$(probe_output_dir "${dataset}" "${backbone}")"

    if [[ ! -f "${eval_job}" ]]; then
      echo "ERROR: eval job not found: ${eval_job}" >&2
      exit 2
    fi

    echo "Submitting ${eval_job} with PROBE_NAME=temporal_attn PROBE_OUTPUT_DIR=${output_dir} SLURM_DEVICE=${SLURM_DEVICE} PROBE_DEVICE=${PROBE_DEVICE}"
    sbatch --partition="${SLURM_DEVICE}" --gpus="${SLURM_GPUS}" --export=ALL,PROBE_NAME=temporal_attn,PROBE_OUTPUT_DIR="${output_dir}",PROBE_DEVICE="${PROBE_DEVICE}" "${eval_job}"
  done
done
