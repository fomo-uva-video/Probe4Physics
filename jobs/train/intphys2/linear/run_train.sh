#!/bin/bash
# Shared runner for IntPhys2 linear-probe training jobs.
#
# Expected launch style:
#   sbatch jepa_v1.sh
#   sbatch jepa_v2.sh linear_probe.device=cuda

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"

python run.py train.linear.intphys2 "backbone.name=${BACKBONE_NAME}" "$@"
