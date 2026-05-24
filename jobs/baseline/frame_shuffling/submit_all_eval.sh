#!/bin/bash
# Submit frame-shuffling baseline eval jobs.
#
# Run after the matching extraction jobs have completed:
#   bash jobs/baseline/frame_shuffling/submit_all_eval.sh
#   PROBE_NAME=mlp bash jobs/baseline/frame_shuffling/submit_all_eval.sh
#   PROBE_NAME=linear BACKBONES_CSV=ltx_video bash jobs/baseline/frame_shuffling/submit_all_eval.sh
#   MISSING_ONLY=0 bash jobs/baseline/frame_shuffling/submit_all_eval.sh
#
# This mirrors the eval-side sbatch command documented at the top of each
# eval wrapper: it submits only eval jobs and exports PROBE_NAME plus the
# normal-probe checkpoint root via PROBE_OUTPUT_DIR.
#
# By default this launcher targets the currently missing IntPhys2
# frame-shuffling evals and skips backbone/probe combinations whose expected
# layer summaries already exist. Set MISSING_ONLY=0 to force resubmission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

SLURM_DEVICE="${SLURM_DEVICE:-gpu_h100}"
SLURM_GPUS="${SLURM_GPUS:-1}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_NAME="${PROBE_NAME:-temporal_attn}"
MISSING_ONLY="${MISSING_ONLY:-1}"
if [[ -z "${PROBE_FEATURE_VIEW:-}" ]]; then
  if [[ "${PROBE_NAME}" == "temporal_attn" ]]; then
    PROBE_FEATURE_VIEW="tokens"
  else
    PROBE_FEATURE_VIEW="pooled"
  fi
fi

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV:-intphys2}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV:-jepa_v1,jepa_v2,jepa_v2_1,videomae,videomae_v2,ltx_video,wan_video}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

expected_eval_count() {
  local backbone="${1}"

  if [[ "${backbone}" == "ltx_video" || "${backbone}" == "wan_video" ]]; then
    printf '40\n'
  else
    printf '4\n'
  fi
}

matches_backbone_result() {
  local dataset="${1}"
  local backbone="${2}"
  local probe="${3}"
  local result_name="${4}"

  case "${backbone}" in
    jepa_v1)
      [[ "${result_name}" == "${dataset}_frame_shuffling_jepa_v1_"*"_${probe}_"* ]]
      ;;
    jepa_v2)
      [[ "${result_name}" == "${dataset}_frame_shuffling_jepa_v2_"*"_${probe}_"* \
        && "${result_name}" != "${dataset}_frame_shuffling_jepa_v2_1_"* ]]
      ;;
    jepa_v2_1)
      [[ "${result_name}" == "${dataset}_frame_shuffling_jepa_v2_1_"*"_${probe}_"* ]]
      ;;
    videomae)
      [[ "${result_name}" == "${dataset}_frame_shuffling_videomae_vit_"*"_${probe}_"* ]]
      ;;
    videomae_v2)
      [[ "${result_name}" == "${dataset}_frame_shuffling_videomae_v2_"*"_${probe}_"* ]]
      ;;
    ltx_video)
      [[ "${result_name}" == "${dataset}_frame_shuffling_ltx_video_"*"_${probe}_"* ]]
      ;;
    wan_video)
      [[ "${result_name}" == "${dataset}_frame_shuffling_wan_video_"*"_${probe}_"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

completed_eval_count() {
  local dataset="${1}"
  local backbone="${2}"
  local probe="${3}"
  local result_root="artifacts/results/${dataset}_frame_shuffling"
  local result_dir result_name summary layer_label

  declare -A seen_layers=()

  [[ -d "${result_root}" ]] || {
    printf '0\n'
    return 0
  }

  shopt -s nullglob
  for result_dir in "${result_root}"/*; do
    [[ -d "${result_dir}" ]] || continue
    result_name="$(basename "${result_dir}")"
    matches_backbone_result "${dataset}" "${backbone}" "${probe}" "${result_name}" || continue

    for summary in "${result_dir}"/layer_*/probe_eval_summary.json; do
      layer_label="$(basename "$(dirname "${summary}")")"
      seen_layers["${layer_label}"]=1
    done
  done
  shopt -u nullglob

  printf '%s\n' "${#seen_layers[@]}"
}

probe_output_dir() {
  local dataset="${1}"
  local backbone="${2}"
  local probe="${3}"

  if [[ "${backbone}" == "ltx_video" && "${probe}" == "linear" && "${dataset}" == "mvp" ]]; then
    printf '%s/artifacts/derived_probe_roots/mvp_probe_linear_ltx_video_complete_40_20260516\n' "${REPO_ROOT}"
  elif [[ "${backbone}" == "ltx_video" && "${probe}" == "linear" && "${dataset}" == "intphys2" ]]; then
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/intphys2/ltx_linear_all_40_optuna\n' "${USER}"
  elif [[ "${backbone}" == "ltx_video" && "${probe}" == "mlp" && "${dataset}" == "mvp" ]]; then
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/mvp/mvp_probe_mlp_ltx_video_complete_40_20260514\n' "${USER}"
  elif [[ "${backbone}" == "ltx_video" && "${probe}" == "mlp" && "${dataset}" == "intphys2" ]]; then
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/intphys2/intphys2_probe_mlp_ltx_video_ltxv_13b_0_9_8_distilled_20260509T102816Z\n' "${USER}"
  else
    printf '/scratch-shared/%s/probe4physics/artifacts/probes/%s\n' "${USER}" "${dataset}"
  fi
}

for dataset in "${DATASETS[@]}"; do
  mkdir -p "output/baseline/frame_shuffling/${dataset}/eval"

  for backbone in "${BACKBONES[@]}"; do
    eval_job="jobs/baseline/frame_shuffling/${dataset}/eval/${backbone}.sh"
    output_dir="$(probe_output_dir "${dataset}" "${backbone}" "${PROBE_NAME}")"
    expected_count="$(expected_eval_count "${backbone}")"
    completed_count="$(completed_eval_count "${dataset}" "${backbone}" "${PROBE_NAME}")"

    if [[ ! -f "${eval_job}" ]]; then
      echo "ERROR: eval job not found: ${eval_job}" >&2
      exit 2
    fi

    if truthy "${MISSING_ONLY}" && (( completed_count >= expected_count )); then
      echo "Skipping ${eval_job}: ${dataset}/${backbone}/${PROBE_NAME} already has ${completed_count}/${expected_count} eval summaries."
      continue
    fi

    echo "Submitting ${eval_job} with PROBE_NAME=${PROBE_NAME} PROBE_FEATURE_VIEW=${PROBE_FEATURE_VIEW} PROBE_OUTPUT_DIR=${output_dir} SLURM_DEVICE=${SLURM_DEVICE} PROBE_DEVICE=${PROBE_DEVICE}"
    sbatch --partition="${SLURM_DEVICE}" --gpus="${SLURM_GPUS}" --export=ALL,PROBE_NAME="${PROBE_NAME}",PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW}",PROBE_OUTPUT_DIR="${output_dir}",PROBE_DEVICE="${PROBE_DEVICE}" "${eval_job}"
  done
done
