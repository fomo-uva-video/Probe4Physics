#!/bin/bash
# Submit baseline extraction jobs for MVP and IntPhys2 across all configured backbones.
#
# Optional filters:
#   ONLY_DATASET=mvp|intphys2
#   ONLY_BASELINE=single_frame|displacement
#   ONLY_BACKBONE=jepa_v1|jepa_v2|jepa_v2_1|videomae|videomae_v2|ltx_video
#   MODE=smoke|full
#   DRY_RUN=true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/extract.sh"

DATASETS=(mvp intphys2)
BASELINES=(single_frame displacement)
BACKBONES=(jepa_v1 jepa_v2 jepa_v2_1 videomae videomae_v2 ltx_video)

variant_for() {
  case "$1" in
    jepa_v1) echo "vith16_384" ;;
    jepa_v2) echo "vitg_384" ;;
    jepa_v2_1) echo "vitG_384" ;;
    videomae) echo "vit_huge_16_224" ;;
    videomae_v2) echo "vit_giant_16_224" ;;
    ltx_video) echo "ltxv_13b_0_9_8_dev" ;;
    *) echo "" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  [[ -z "${ONLY_DATASET:-}" || "${dataset}" == "${ONLY_DATASET}" ]] || continue
  for baseline in "${BASELINES[@]}"; do
    [[ -z "${ONLY_BASELINE:-}" || "${baseline}" == "${ONLY_BASELINE}" ]] || continue
    for backbone in "${BACKBONES[@]}"; do
      [[ -z "${ONLY_BACKBONE:-}" || "${backbone}" == "${ONLY_BACKBONE}" ]] || continue
      variant="$(variant_for "${backbone}")"
      job_name="${dataset}_${baseline}_${backbone}_baseline_extract"
      cmd=(
        sbatch
        "--job-name=${job_name}"
        "--export=ALL,DATASET_NAME=${dataset},BASELINE_NAME=${baseline},BACKBONE_NAME=${backbone},BACKBONE_VARIANT=${variant}"
        "${JOB_SCRIPT}"
      )
      echo "${cmd[*]}"
      if [[ "${DRY_RUN:-false}" != "true" ]]; then
        "${cmd[@]}"
      fi
    done
  done
done
