#!/bin/bash
# Train MVP temporal_attn probes with jepa_v1 features over a fixed
# (layer, learning-rate) matrix.
#
# Usage:
#   

# Default matrix:
#   layers = 8, 16, 24, 32
#   lrs    = 5e-4, 1e-4, 5e-5, 1e-5
#
# Layout:
#   artifacts/probes/mvp/${GROUP_SUBDIR}/layer_<layer>/lr_<tag>/
#
# The script also incrementally rebuilds:
#   artifacts/probes/mvp/${GROUP_SUBDIR}/train_eval_summary.json
#   artifacts/probes/mvp/${GROUP_SUBDIR}/train_eval_summary.csv
# keeping, for each layer, the best completed LR run selected on validation.
#
# Quick check after completion:
#   column -s, -t < artifacts/probes/mvp/mvp_probe_temporal_attn_jepa_v1_vith16_384_lr_matrix/train_eval_summary.csv

#SBATCH --partition=gpu_a100
#SBATCH --job-name=mvp_jepa_v1_attn_lr_matrix
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=output/training/mvp/attention/mvp_jepa_v1_attn_lr_matrix_%A_%a.out
#SBATCH --error=output/training/mvp/attention/mvp_jepa_v1_attn_lr_matrix_%A_%a.err

set -euo pipefail

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

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../extract/common.sh"

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

load_probe4physics_env
configure_hf_cache

DATASET_NAME="${DATASET_NAME:-mvp}"
PROBE_NAME="${PROBE_NAME:-temporal_attn}"
BACKBONE_NAME="${BACKBONE_NAME:-jepa_v1}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-vith16_384}"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-tokens}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-2}"
PROBE_EVAL_BATCH_SIZE="${PROBE_EVAL_BATCH_SIZE:-2}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.01}"
PROBE_EARLY_STOPPING_PATIENCE="${PROBE_EARLY_STOPPING_PATIENCE:-5}"
PROBE_LABEL_CONTROL_MODE="${PROBE_LABEL_CONTROL_MODE:-original}"
PROBE_LABEL_CONTROL_SEED="${PROBE_LABEL_CONTROL_SEED:-42}"
TEMPORAL_NUM_HEADS="${TEMPORAL_NUM_HEADS:-16}"
TEMPORAL_NUM_SELF_ATTN_BLOCKS="${TEMPORAL_NUM_SELF_ATTN_BLOCKS:-1}"
TEMPORAL_MLP_RATIO="${TEMPORAL_MLP_RATIO:-2.0}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.2}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-artifacts/probes/mvp}"
PROBE_EVAL_OUTPUT_DIR="${PROBE_EVAL_OUTPUT_DIR:-artifacts/results}"
GROUP_SUBDIR="${GROUP_SUBDIR:-mvp_probe_temporal_attn_jepa_v1_vith16_384_lr_matrix}"

LAYER_VALUES=("8" "16" "24" "32")
LR_VALUES=("5e-4" "1e-4" "5e-5" "1e-5")
LR_TAGS=("5e-4" "1e-4" "5e-5" "1e-5")
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_LAYERS="${#LAYER_VALUES[@]}"
NUM_LRS="${#LR_VALUES[@]}"
TOTAL_TASKS="$((NUM_LAYERS * NUM_LRS))"

if [[ "${TASK_INDEX}" -lt 0 || "${TASK_INDEX}" -ge "${TOTAL_TASKS}" ]]; then
  echo "Invalid matrix index '${TASK_INDEX}'. Expected 0..$((TOTAL_TASKS - 1))." >&2
  exit 2
fi

LAYER_INDEX="$((TASK_INDEX / NUM_LRS))"
LR_INDEX="$((TASK_INDEX % NUM_LRS))"

PROBE_LAYER="${LAYER_VALUES[${LAYER_INDEX}]}"
LR_VALUE="${LR_VALUES[${LR_INDEX}]}"
LR_TAG="${LR_TAGS[${LR_INDEX}]}"

RUN_SUBDIR="${GROUP_SUBDIR}/layer_${PROBE_LAYER}/lr_${LR_TAG}"
TRAIN_SUBDIR="${RUN_SUBDIR}/train"
EVAL_SUBDIR="${RUN_SUBDIR}/eval"
RUN_ROOT="${REPO_ROOT}/${PROBE_OUTPUT_DIR}/${RUN_SUBDIR}"
GROUP_ROOT="${REPO_ROOT}/${PROBE_OUTPUT_DIR}/${GROUP_SUBDIR}"
TRAIN_ROOT="${REPO_ROOT}/${PROBE_OUTPUT_DIR}/${TRAIN_SUBDIR}"
EVAL_ROOT="${REPO_ROOT}/${PROBE_EVAL_OUTPUT_DIR}/${EVAL_SUBDIR}"
CHECKPOINT_PATH="${TRAIN_ROOT}/probe_best.pt"

WANDB_GROUP="${WANDB_GROUP:-mvp_temporal_attn_jepa_v1_lr_matrix}"
WANDB_NAME="${WANDB_NAME:-mvp_jepa_v1_layer_${PROBE_LAYER}_lr_${LR_TAG}}"

echo "===== TRAIN PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "GROUP_SUBDIR=${GROUP_SUBDIR}"
echo "RUN_SUBDIR=${RUN_SUBDIR}"
echo "PROBE_LAYER=${PROBE_LAYER}"
echo "LR_VALUE=${LR_VALUE}"
echo "PROBE_EPOCHS=${PROBE_EPOCHS}"
echo "PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE}"
echo "PROBE_EVAL_BATCH_SIZE=${PROBE_EVAL_BATCH_SIZE}"
echo "PROBE_WEIGHT_DECAY=${PROBE_WEIGHT_DECAY}"
echo "PROBE_EARLY_STOPPING_PATIENCE=${PROBE_EARLY_STOPPING_PATIENCE}"
echo "PROBE_LABEL_CONTROL_MODE=${PROBE_LABEL_CONTROL_MODE}"
echo "PROBE_LABEL_CONTROL_SEED=${PROBE_LABEL_CONTROL_SEED}"
echo "TEMPORAL_NUM_HEADS=${TEMPORAL_NUM_HEADS}"
echo "TEMPORAL_NUM_SELF_ATTN_BLOCKS=${TEMPORAL_NUM_SELF_ATTN_BLOCKS}"
echo "TEMPORAL_MLP_RATIO=${TEMPORAL_MLP_RATIO}"
echo "TEMPORAL_DROPOUT=${TEMPORAL_DROPOUT}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "============================"

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

train_cmd=(
  python run.py "train.probe.${DATASET_NAME}"
  "backbone.name=${BACKBONE_NAME}"
  "+backbone.kwargs.variant=${BACKBONE_VARIANT}"
  "probe.name=${PROBE_NAME}"
  "probe.device=${PROBE_DEVICE}"
  "probe.layer=${PROBE_LAYER}"
  "probe.feature_view=${PROBE_FEATURE_VIEW}"
  "probe.epochs=${PROBE_EPOCHS}"
  "probe.lr=${LR_VALUE}"
  "probe.weight_decay=${PROBE_WEIGHT_DECAY}"
  "probe.batch_size=${PROBE_BATCH_SIZE}"
  "probe.eval_batch_size=${PROBE_EVAL_BATCH_SIZE}"
  "probe.early_stopping.enabled=true"
  "probe.early_stopping.patience=${PROBE_EARLY_STOPPING_PATIENCE}"
  "probe.label_control.mode=${PROBE_LABEL_CONTROL_MODE}"
  "probe.label_control.seed=${PROBE_LABEL_CONTROL_SEED}"
  "probe.temporal_attn.num_heads=${TEMPORAL_NUM_HEADS}"
  "probe.temporal_attn.num_self_attn_blocks=${TEMPORAL_NUM_SELF_ATTN_BLOCKS}"
  "probe.temporal_attn.mlp_ratio=${TEMPORAL_MLP_RATIO}"
  "probe.temporal_attn.dropout=${TEMPORAL_DROPOUT}"
  "probe.output_dir=${PROBE_OUTPUT_DIR}"
  "probe.output_subdir=${TRAIN_SUBDIR}"
  "probe.eval_output_dir=${PROBE_EVAL_OUTPUT_DIR}"
  "probe.wandb.enabled=${ENABLE_WANDB}"
  "probe.wandb.project=${WANDB_PROJECT}"
  "probe.wandb.mode=${WANDB_MODE}"
  "probe.wandb.group=${WANDB_GROUP}"
  "probe.wandb.name=${WANDB_NAME}"
  "probe.wandb.tags=[mvp,jepa_v1,temporal_attn,lr_matrix,label_${PROBE_LABEL_CONTROL_MODE},layer_${PROBE_LAYER},lr_${LR_TAG}]"
  "probe.optuna.enabled=false"
  "feature_cache.include_tokens=true"
)

echo "==> Launching training command:"
printf '  %q' "${train_cmd[@]}"
printf '\n'
"${train_cmd[@]}"

eval_cmd=(
  python run.py "eval.probe.${DATASET_NAME}"
  "backbone.name=${BACKBONE_NAME}"
  "+backbone.kwargs.variant=${BACKBONE_VARIANT}"
  "probe.name=${PROBE_NAME}"
  "probe.device=${PROBE_DEVICE}"
  "probe.layer=${PROBE_LAYER}"
  "probe.feature_view=${PROBE_FEATURE_VIEW}"
  "probe.checkpoint_path=${CHECKPOINT_PATH}"
  "probe.eval_output_dir=${PROBE_EVAL_OUTPUT_DIR}"
  "probe.eval_output_subdir=${EVAL_SUBDIR}"
  "probe.eval_batch_size=${PROBE_EVAL_BATCH_SIZE}"
  "probe.label_control.mode=${PROBE_LABEL_CONTROL_MODE}"
  "probe.label_control.seed=${PROBE_LABEL_CONTROL_SEED}"
  "feature_cache.include_tokens=true"
)

echo "==> Launching evaluation command:"
printf '  %q' "${eval_cmd[@]}"
printf '\n'
"${eval_cmd[@]}"

export GROUP_ROOT RUN_ROOT EVAL_ROOT PROBE_LAYER LR_VALUE LR_TAG PROBE_LABEL_CONTROL_MODE PROBE_LABEL_CONTROL_SEED
mkdir -p "${GROUP_ROOT}"
exec 9>"${GROUP_ROOT}/.summary.lock"
flock 9
python - <<'PY'
import json
import os
from pathlib import Path

from training.run_probe import _write_train_eval_summary_csv


def parse_int(value: object) -> object:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metric_value(item: dict) -> float:
    return float(item.get("objective_metric", 0.0))


group_root = Path(os.environ["GROUP_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
probe_layer = str(os.environ["PROBE_LAYER"])
lr_value = str(os.environ["LR_VALUE"])
lr_tag = str(os.environ["LR_TAG"])
label_control = {
    "mode": os.environ.get("PROBE_LABEL_CONTROL_MODE", "original"),
    "seed": parse_int(os.environ.get("PROBE_LABEL_CONTROL_SEED", "42")),
}

train_summary = read_json(run_root / "train" / "train_summary.json")
eval_summary = read_json(eval_root / "probe_eval_summary.json")

single_layer_summary = {
    "dataset": "mvp",
    "probe_name": "temporal_attn",
    "feature_view": "tokens",
    "train_split": "train",
    "objective_metric_name": "pair_consistency",
    "model_selection_split": "val",
    "split_name": eval_summary.get("split_name", "test"),
    "reported_splits": list(eval_summary.get("reported_splits", [])),
    "label_control": label_control,
    "sweep_dir": str(run_root),
    "requested_layers": [probe_layer],
    "layers": [
        {
            "layer": probe_layer,
            "layer_label": probe_layer,
            "learning_rate": lr_value,
            "learning_rate_tag": lr_tag,
            "checkpoint": str(train_summary["checkpoint"]),
            "objective_metric": float(eval_summary["objective_metric"]),
            "train": train_summary,
            "eval": eval_summary,
        }
    ],
    "best_layer": probe_layer,
    "best_layer_label": probe_layer,
    "best_objective_metric": float(eval_summary["objective_metric"]),
}

single_summary_path = run_root / "train_eval_summary.json"
write_json(single_summary_path, single_layer_summary)
_write_train_eval_summary_csv(run_root / "train_eval_summary.csv", single_layer_summary)

best_by_layer: dict[str, dict] = {}
for candidate_path in sorted(group_root.glob("layer_*/lr_*/train_eval_summary.json")):
    payload = read_json(candidate_path)
    layers = payload.get("layers", [])
    if not isinstance(layers, list) or not layers:
        continue
    layer_entry = layers[0]
    if not isinstance(layer_entry, dict):
        continue
    layer_label = str(layer_entry.get("layer_label", "")).strip()
    if not layer_label:
        continue
    previous = best_by_layer.get(layer_label)
    if previous is None or metric_value(layer_entry) > metric_value(previous):
        best_by_layer[layer_label] = layer_entry


def sort_key(label: str) -> tuple[int, str]:
    return (0, f"{int(label):08d}") if label.isdigit() else (1, label)


group_layers = [best_by_layer[label] for label in sorted(best_by_layer, key=sort_key)]
group_summary = {
    "dataset": "mvp",
    "probe_name": "temporal_attn",
    "feature_view": "tokens",
    "train_split": "train",
    "objective_metric_name": "pair_consistency",
    "model_selection_split": "val",
    "split_name": "test",
    "reported_splits": ["train", "val", "test"],
    "label_control": label_control,
    "sweep_dir": str(group_root),
    "requested_layers": [layer.get("layer", "") for layer in group_layers],
    "layers": group_layers,
    "best_layer": None,
    "best_layer_label": None,
    "best_objective_metric": None,
}
if group_layers:
    best_overall = max(group_layers, key=metric_value)
    group_summary["best_layer"] = best_overall.get("layer")
    group_summary["best_layer_label"] = best_overall.get("layer_label")
    group_summary["best_objective_metric"] = metric_value(best_overall)

group_summary_path = group_root / "train_eval_summary.json"
write_json(group_summary_path, group_summary)
_write_train_eval_summary_csv(group_root / "train_eval_summary.csv", group_summary)
PY
flock -u 9

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"
