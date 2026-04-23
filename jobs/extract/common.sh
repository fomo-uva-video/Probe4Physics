#!/bin/bash
# Shared helpers for SLURM extraction jobs.
#
# This file is sourced by benchmark-specific job runners. It keeps the job
# wrappers short and makes the runtime assumptions explicit in one place.

set -euo pipefail


resolve_repo_root() {
  local source_path="${1}"
  local script_dir
  script_dir="$(cd "$(dirname "${source_path}")" && pwd)"
  local candidate="${script_dir}"

  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/run.py" && -d "${candidate}/configs" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done

  echo "ERROR: Could not locate the Probe4Physics repository root from ${source_path}" >&2
  return 1
}


load_probe4physics_env() {
  module purge
  module load 2025
  module load Anaconda3/2025.06-1

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "${CONDA_ENV_NAME:-probe4physics-gpu}"
  set -u
}


configure_hf_cache() {
  local username="${USERNAME:-$USER}"
  local default_hf_home="/scratch-shared/${username}/.cache/huggingface"

  export HF_HOME="${HF_HOME:-${default_hf_home}}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

  mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"
}


resolve_backbone_variant() {
  local repo_root="${1}"
  local backbone_name="${2}"
  local requested_variant="${3:-}"

  if [[ -n "${requested_variant}" ]]; then
    printf '%s\n' "${requested_variant}"
    return 0
  fi

  python - <<'PY' "${repo_root}" "${backbone_name}"
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1])
backbone_name = sys.argv[2]
cfg_path = repo_root / "configs" / "backbones.yaml"

if not cfg_path.exists():
    print("<unknown>")
    raise SystemExit(0)

with cfg_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}

section = payload.get(backbone_name, {})
if isinstance(section, dict):
    print(str(section.get("default_variant", "<unknown>")))
else:
    print("<unknown>")
PY
}
