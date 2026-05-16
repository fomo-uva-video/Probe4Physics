#!/bin/bash
# Shared runner for temporal-baseline extraction and eval jobs.
#
# Expected launch style from the repository root:
#   mkdir -p output/baseline/single_frame/mvp/extract output/baseline/single_frame/mvp/eval
#   extract_jid=$(sbatch --parsable jobs/baseline/single_frame/mvp/extract/jepa_v1.sh)
#   sbatch --dependency=afterok:${extract_jid} --export=ALL,PROBE_OUTPUT_DIR=/scratch-shared/${USER}/probe4physics/artifacts/probes/mvp jobs/baseline/single_frame/mvp/eval/jepa_v1.sh
#
# The mkdir is required because Slurm opens stdout/stderr before this script runs.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

DATASET_NAME="${DATASET_NAME:?DATASET_NAME must be set by the wrapper script}"
BASELINE_NAME="${BASELINE_NAME:?BASELINE_NAME must be set by the wrapper script}"
BASELINE_LABEL="${BASELINE_LABEL:-${BASELINE_NAME}}"
BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
BASELINE_STAGE="${BASELINE_STAGE:-extract_eval}"
MODE="${MODE:-full}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
FORCE_REEXTRACT="${FORCE_REEXTRACT:-}"
INCLUDE_POOLED="${INCLUDE_POOLED:-}"
INCLUDE_TOKENS="${INCLUDE_TOKENS:-}"
FEATURE_DIR="${FEATURE_DIR:-}"
PROBE_NAME="${PROBE_NAME:-linear}"
PROBE_LAYER="${PROBE_LAYER:-last}"
PROBE_LAYERS="${PROBE_LAYERS:-${PROBE_LAYER}}"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-pooled}"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"
PROBE_CHECKPOINT_PATH="${PROBE_CHECKPOINT_PATH:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_SUBDIR="${EVAL_SUBDIR:-}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-}"
EXTRA_ARGS=("$@")

if [[ "${DATASET_NAME}" != "mvp" && "${DATASET_NAME}" != "intphys2" ]]; then
  echo "ERROR: unsupported DATASET_NAME='${DATASET_NAME}'. Use mvp or intphys2." >&2
  exit 2
fi

if [[ "${BASELINE_NAME}" != "single_frame" && "${BASELINE_NAME}" != "displacement" ]]; then
  echo "ERROR: unsupported BASELINE_NAME='${BASELINE_NAME}'. Use single_frame or displacement." >&2
  exit 2
fi

if [[ "${BASELINE_STAGE}" != "extract" && "${BASELINE_STAGE}" != "eval" && "${BASELINE_STAGE}" != "extract_eval" ]]; then
  echo "ERROR: unsupported BASELINE_STAGE='${BASELINE_STAGE}'. Use extract, eval, or extract_eval." >&2
  exit 2
fi

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "ERROR: unsupported MODE='${MODE}'. Use MODE=full or MODE=smoke." >&2
  exit 2
fi

EFFECTIVE_BACKBONE_VARIANT="$(resolve_backbone_variant "${REPO_ROOT}" "${BACKBONE_NAME}" "${BACKBONE_VARIANT}")"
BACKBONE_TAG="${BACKBONE_NAME}_${EFFECTIVE_BACKBONE_VARIANT}"

if [[ "${BACKBONE_NAME}" == "ltx_video" && "${BASELINE_STAGE}" != "eval" ]]; then
  preflight_ltx_runtime
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_OVERRIDE="feature_cache.max_samples=${MAX_SAMPLES}"
elif [[ "${MODE}" == "smoke" ]]; then
  MAX_SAMPLES_OVERRIDE="feature_cache.max_samples=2"
else
  MAX_SAMPLES_OVERRIDE=""
fi

parse_probe_layers() {
  local raw="${PROBE_LAYERS// /}"
  if [[ -z "${raw}" ]]; then
    raw="${PROBE_LAYER}"
  fi
  IFS=',' read -r -a PARSED_PROBE_LAYERS <<< "${raw}"
  if [[ "${#PARSED_PROBE_LAYERS[@]}" -eq 0 ]]; then
    echo "ERROR: PROBE_LAYERS resolved to an empty list." >&2
    exit 2
  fi
}

DATASET_ARGS=()

load_dataset_args() {
  DATASET_ARGS=()

  if [[ "${DATASET_NAME}" == "mvp" ]]; then
    ANNOTATION_FILE="${ANNOTATION_FILE:-${REPO_ROOT}/data/annotations/mvp_full.jsonl}"
    OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-${REPO_ROOT}/third_party/minimal_video_pairs}"
    VIDEOS_ROOT="${VIDEOS_ROOT:-/scratch-shared/${USER}/probe4physics/data/videos/mvp}"
    CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/cache/videos}"
    SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/mvp/full_60_20_20}"

    [[ -f "${ANNOTATION_FILE}" ]] || { echo "ERROR: MVP annotation file not found: ${ANNOTATION_FILE}" >&2; exit 2; }
    [[ -f "${OFFICIAL_REPO_ROOT}/tasks/mvp/utils.py" ]] || { echo "ERROR: MVP official repo not found: ${OFFICIAL_REPO_ROOT}" >&2; exit 2; }
    [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: MVP videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
    [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_pairs.parquet" ]] || { echo "ERROR: MVP split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }

    DATASET_ARGS+=(
      "annotation_file=${ANNOTATION_FILE}"
      "official_repo_root=${OFFICIAL_REPO_ROOT}"
      "videos_root=${VIDEOS_ROOT}"
      "cache_dir=${CACHE_DIR}"
      "split.dir=${SPLIT_DIR}"
      "annotations.auto_download=false"
    )
  else
    METADATA_FILE="${METADATA_FILE:-${REPO_ROOT}/data/annotations/intphys2_metadata.csv}"
    VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO_ROOT}/data/videos/intphys2}"
    SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits/intphys2}"

    [[ -f "${METADATA_FILE}" ]] || { echo "ERROR: IntPhys2 metadata file not found: ${METADATA_FILE}" >&2; exit 2; }
    [[ -d "${VIDEOS_ROOT}" ]] || { echo "ERROR: IntPhys2 videos root not found: ${VIDEOS_ROOT}" >&2; exit 2; }
    [[ -f "${SPLIT_DIR}/manifest.json" && -f "${SPLIT_DIR}/split_scenes.parquet" ]] || { echo "ERROR: IntPhys2 split artifacts not found in ${SPLIT_DIR}" >&2; exit 2; }

    DATASET_ARGS+=(
      "metadata_file=${METADATA_FILE}"
      "videos_root=${VIDEOS_ROOT}"
      "split.dir=${SPLIT_DIR}"
    )
  fi
}

default_probe_output_dir() {
  if [[ "${DATASET_NAME}" == "mvp" ]]; then
    printf '%s\n' "${REPO_ROOT}/artifacts/probes/mvp"
  else
    printf '%s\n' "${REPO_ROOT}/artifacts/probes/intphys2"
  fi
}

normalize_probe_output_dir() {
  local root="${PROBE_OUTPUT_DIR:-$(default_probe_output_dir)}"
  if [[ -d "${root}" ]]; then
    (cd "${root}" && pwd -P)
  else
    printf '%s\n' "${root}"
  fi
}

probe_layer_label() {
  local layer="${1}"
  python -c '
import sys
from models.cache_metadata import resolve_backbone_layer_label

name, variant, layer = sys.argv[1:4]
try:
    parsed_layer = int(layer)
except ValueError:
    parsed_layer = layer
kwargs = {"variant": variant} if variant else {}
print(resolve_backbone_layer_label(name, kwargs, parsed_layer))
' "${BACKBONE_NAME}" "${EFFECTIVE_BACKBONE_VARIANT}" "${layer}"
}

run_extract() {
  local cmd=(python run.py "extract.${DATASET_NAME}.${BASELINE_NAME}")
  load_dataset_args
  cmd+=("${DATASET_ARGS[@]}")

  cmd+=(
    "backbone.name=${BACKBONE_NAME}"
    "backbone.kwargs.device=${BACKBONE_DEVICE}"
  )

  if [[ -n "${BACKBONE_VARIANT}" ]]; then
    cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
  fi

  if [[ -n "${FEATURE_DIR}" ]]; then
    mkdir -p "${FEATURE_DIR}"
    cmd+=("feature_cache.dir=${FEATURE_DIR}")
  fi

  if [[ -n "${INCLUDE_POOLED}" ]]; then
    cmd+=("feature_cache.include_pooled=${INCLUDE_POOLED}")
  fi

  if [[ -n "${INCLUDE_TOKENS}" ]]; then
    cmd+=("feature_cache.include_tokens=${INCLUDE_TOKENS}")
  fi

  if [[ -n "${FORCE_REEXTRACT}" ]]; then
    cmd+=("feature_cache.force_reextract=${FORCE_REEXTRACT}")
  fi

  if [[ -n "${MAX_SAMPLES_OVERRIDE}" ]]; then
    cmd+=("${MAX_SAMPLES_OVERRIDE}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  echo "===== BASELINE EXTRACTION PROVENANCE ====="
  date -u
  hostname
  git -C "${REPO_ROOT}" rev-parse HEAD
  python --version
  which python
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "DATASET_NAME=${DATASET_NAME}"
  echo "BASELINE_LABEL=${BASELINE_LABEL}"
  echo "BASELINE_NAME=${BASELINE_NAME}"
  echo "BASELINE_STAGE=${BASELINE_STAGE}"
  echo "MODE=${MODE}"
  echo "MAX_SAMPLES=${MAX_SAMPLES:-<unset>}"
  echo "BACKBONE_NAME=${BACKBONE_NAME}"
  echo "BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
  echo "CANONICAL_PROBE_LAYERS=${PROBE_LAYERS}"
  echo "BACKBONE_DEVICE=${BACKBONE_DEVICE}"
  echo "FEATURE_DIR=${FEATURE_DIR:-<config default>}"
  echo "HF_HOME=${HF_HOME}"
  echo "Command: ${cmd[*]}"
  echo "=========================================="

  "${cmd[@]}"
}

resolve_checkpoint() {
  local layer="${1}"

  if [[ -n "${PROBE_CHECKPOINT_PATH}" ]]; then
    printf '%s\n' "${PROBE_CHECKPOINT_PATH}"
    return 0
  fi

  local train_prefix=""
  if [[ "${DATASET_NAME}" == "mvp" ]]; then
    train_prefix="mvp_probe"
  else
    train_prefix="intphys2_probe"
  fi

  local layer_path=""
  if [[ "${layer}" != "last" ]]; then
    local layer_label=""
    layer_label="$(probe_layer_label "${layer}")"
    local optuna_summary=""
    optuna_summary="$(find -L "${PROBE_OUTPUT_DIR}" -path "*/${train_prefix}_${PROBE_NAME}_${BACKBONE_TAG}_*/layer_${layer_label}/train/optuna_summary.json" -type f 2>/dev/null | sort | tail -n 1)"
    if [[ -z "${optuna_summary}" ]]; then
      optuna_summary="$(find -L "${PROBE_OUTPUT_DIR}" -path "${PROBE_OUTPUT_DIR}/layer_${layer_label}/train/optuna_summary.json" -type f 2>/dev/null | sort | tail -n 1)"
    fi
    if [[ -n "${optuna_summary}" ]]; then
      local best_trial=""
      best_trial="$(python -c 'import json, sys; data=json.load(open(sys.argv[1])); value=data.get("best_trial_number"); print("" if value is None else int(value))' "${optuna_summary}" 2>/dev/null || true)"
      if [[ -n "${best_trial}" ]]; then
        local best_trial_dir=""
        printf -v best_trial_dir 'trial_%04d' "${best_trial}"
        layer_path="$(dirname "${optuna_summary}")/${best_trial_dir}/probe_best.pt"
        if [[ -f "${layer_path}" ]]; then
          printf '%s\n' "${layer_path}"
          return 0
        fi
        echo "ERROR: Optuna best checkpoint not found: ${layer_path}" >&2
        return 0
      fi
    fi

    layer_path="$(find -L "${PROBE_OUTPUT_DIR}" \( -path "*/${train_prefix}_${PROBE_NAME}_${BACKBONE_TAG}_*/layer_${layer_label}/train/probe_best.pt" -o -path "*/${train_prefix}_${PROBE_NAME}_${BACKBONE_TAG}_*/layer_${layer_label}/train/*/probe_best.pt" \) -type f 2>/dev/null | sort | tail -n 1)"
    if [[ -z "${layer_path}" ]]; then
      layer_path="$(find -L "${PROBE_OUTPUT_DIR}" \( -path "${PROBE_OUTPUT_DIR}/layer_${layer_label}/train/probe_best.pt" -o -path "${PROBE_OUTPUT_DIR}/layer_${layer_label}/train/*/probe_best.pt" \) -type f 2>/dev/null | sort | tail -n 1)"
    fi
    if [[ -n "${layer_path}" ]]; then
      printf '%s\n' "${layer_path}"
      return 0
    fi
    return 0
  fi

  find -L "${PROBE_OUTPUT_DIR}" -path "*/${train_prefix}_${PROBE_NAME}_${BACKBONE_TAG}_*/probe_best.pt" -type f 2>/dev/null | sort | tail -n 1
}

run_eval_for_layer() {
  local layer="${1}"
  local layer_label=""
  layer_label="$(probe_layer_label "${layer}")"
  local resolved_checkpoint=""
  resolved_checkpoint="$(resolve_checkpoint "${layer}")"

  if [[ -z "${resolved_checkpoint}" || ! -f "${resolved_checkpoint}" ]]; then
    echo "ERROR: Could not resolve a normal probe checkpoint." >&2
    echo "Set PROBE_CHECKPOINT_PATH explicitly, or train the normal ${DATASET_NAME} probe first." >&2
    echo "Searched root: ${PROBE_OUTPUT_DIR}" >&2
    echo "Expected backbone tag: ${BACKBONE_TAG}" >&2
    echo "Expected probe layer: ${layer}" >&2
    echo "Expected probe layer label: ${layer_label}" >&2
    exit 2
  fi

  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/${DATASET_NAME}_${BASELINE_LABEL}}"
  local eval_subdir=""
  if [[ -n "${EVAL_SUBDIR}" && "${#PARSED_PROBE_LAYERS[@]}" -eq 1 ]]; then
    eval_subdir="${EVAL_SUBDIR}"
  elif [[ -n "${EVAL_SUBDIR}" ]]; then
    eval_subdir="${EVAL_SUBDIR}/layer_${layer_label}"
  else
    eval_subdir="${DATASET_NAME}_${BASELINE_LABEL}_${BACKBONE_TAG}_${PROBE_NAME}_${SLURM_JOB_ID:-manual}/layer_${layer_label}"
  fi

  local cmd=(python run.py "eval.probe.${DATASET_NAME}.${BASELINE_NAME}")
  load_dataset_args
  cmd+=("${DATASET_ARGS[@]}")

  cmd+=(
    "backbone.name=${BACKBONE_NAME}"
    "probe.name=${PROBE_NAME}"
    "probe.layer=${layer}"
    "probe.feature_view=${PROBE_FEATURE_VIEW}"
    "probe.device=${PROBE_DEVICE}"
    "probe.checkpoint_path=${resolved_checkpoint}"
    "probe.eval_output_dir=${EVAL_OUTPUT_DIR}"
    "probe.eval_output_subdir=${eval_subdir}"
  )

  if [[ -n "${BACKBONE_VARIANT}" ]]; then
    cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
  fi

  if [[ -n "${FEATURE_DIR}" ]]; then
    cmd+=("feature_cache.dir=${FEATURE_DIR}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  mkdir -p "${EVAL_OUTPUT_DIR}"

  echo "===== BASELINE EVAL PROVENANCE ====="
  date -u
  hostname
  git -C "${REPO_ROOT}" rev-parse HEAD
  python --version
  which python
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "DATASET_NAME=${DATASET_NAME}"
  echo "BASELINE_LABEL=${BASELINE_LABEL}"
  echo "BASELINE_NAME=${BASELINE_NAME}"
  echo "BASELINE_STAGE=${BASELINE_STAGE}"
  echo "BACKBONE_NAME=${BACKBONE_NAME}"
  echo "BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
  echo "PROBE_NAME=${PROBE_NAME}"
  echo "PROBE_LAYER=${layer}"
  echo "PROBE_LAYER_LABEL=${layer_label}"
  echo "PROBE_LAYERS=${PROBE_LAYERS}"
  echo "PROBE_FEATURE_VIEW=${PROBE_FEATURE_VIEW}"
  echo "PROBE_DEVICE=${PROBE_DEVICE}"
  echo "PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR}"
  echo "PROBE_CHECKPOINT_PATH=${resolved_checkpoint}"
  echo "FEATURE_DIR=${FEATURE_DIR:-<config default>}"
  echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
  echo "EVAL_SUBDIR=${eval_subdir}"
  echo "HF_HOME=${HF_HOME}"
  echo "Command: ${cmd[*]}"
  echo "===================================="

  "${cmd[@]}"
}

run_eval() {
  parse_probe_layers
  PROBE_OUTPUT_DIR="$(normalize_probe_output_dir)"

  if [[ -n "${PROBE_CHECKPOINT_PATH}" && "${#PARSED_PROBE_LAYERS[@]}" -ne 1 ]]; then
    echo "ERROR: PROBE_CHECKPOINT_PATH can only be used with one PROBE_LAYER/PROBE_LAYERS value." >&2
    echo "For multi-layer baseline eval, leave PROBE_CHECKPOINT_PATH empty and let the job resolve each layer checkpoint." >&2
    exit 2
  fi

  for layer in "${PARSED_PROBE_LAYERS[@]}"; do
    run_eval_for_layer "${layer}"
  done
}

mkdir -p "output/baseline/${BASELINE_LABEL}/${DATASET_NAME}"

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "JOB_START_UTC=${JOB_START_UTC}"

if [[ "${BASELINE_STAGE}" == "extract" || "${BASELINE_STAGE}" == "extract_eval" ]]; then
  run_extract
fi

if [[ "${BASELINE_STAGE}" == "eval" || "${BASELINE_STAGE}" == "extract_eval" ]]; then
  run_eval
fi

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"
