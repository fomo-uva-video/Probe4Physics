#!/bin/bash
# Validate and submit the attentive continuation manifest with the exact Slurm
# array range implied by the current CSV. This script does not build the
# manifest; regenerate it explicitly before submitting when inputs change.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST_PATH="${ATTENTIVE_CONTINUATION_MANIFEST_PATH:-results/seed_runs/attentive_continuation_manifest_v1.csv}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
DRY_RUN="false"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN="true"
      ;;
    *)
      echo "Unsupported argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

python scripts/validate_attentive_continuation_manifest.py --manifest "${MANIFEST_PATH}"

ROW_COUNT="$(python - "${MANIFEST_PATH}" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"

if [[ "${ROW_COUNT}" -le 0 ]]; then
  echo "No pending attentive continuation rows in ${MANIFEST_PATH}" >&2
  exit 1
fi

ARRAY_SPEC="0-$((ROW_COUNT - 1))%${MAX_CONCURRENT}"
SBATCH_CMD=(
  sbatch
  --array="${ARRAY_SPEC}"
  --export="ALL,ATTENTIVE_CONTINUATION_MANIFEST_PATH=${MANIFEST_PATH}"
  jobs/train/seed_runs/attentive_continuations_gpu_array.sh
)

echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "ROW_COUNT=${ROW_COUNT}"
echo "ARRAY_SPEC=${ARRAY_SPEC}"
printf 'COMMAND='
printf '%q ' "${SBATCH_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
