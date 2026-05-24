#!/bin/bash
# Shared runner for IntPhys2 temporal_attn fixed-LR matrix jobs.

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

DATASET_NAME="${DATASET_NAME:-intphys2}"
PROBE_NAME="${PROBE_NAME:-temporal_attn}"
BACKBONE_NAME="${BACKBONE_NAME:?BACKBONE_NAME must be set by the wrapper script}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-}"
EFFECTIVE_BACKBONE_VARIANT="$(resolve_backbone_variant "${REPO_ROOT}" "${BACKBONE_NAME}" "${BACKBONE_VARIANT}")"
PROBE_FEATURE_VIEW="${PROBE_FEATURE_VIEW:-tokens}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-1}"
PROBE_EVAL_BATCH_SIZE="${PROBE_EVAL_BATCH_SIZE:-${PROBE_BATCH_SIZE}}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.01}"
TEMPORAL_NUM_HEADS="${TEMPORAL_NUM_HEADS:-16}"
TEMPORAL_NUM_SELF_ATTN_BLOCKS="${TEMPORAL_NUM_SELF_ATTN_BLOCKS:-1}"
TEMPORAL_MLP_RATIO="${TEMPORAL_MLP_RATIO:-2.0}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.2}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-artifacts/probes/intphys2}"
PROBE_EVAL_OUTPUT_DIR="${PROBE_EVAL_OUTPUT_DIR:-artifacts/results/intphys2}"
MATRIX_LAYERS="${MATRIX_LAYERS:?MATRIX_LAYERS must be set by the wrapper script}"
MATRIX_LRS="${MATRIX_LRS:-5e-4,1e-4,5e-5,1e-5}"
MATRIX_LR_TAGS="${MATRIX_LR_TAGS:-${MATRIX_LRS}}"

if [[ "${DATASET_NAME}" != "intphys2" ]]; then
  echo "run_lr_matrix.sh is only intended for DATASET_NAME=intphys2." >&2
  exit 2
fi
if [[ "${PROBE_NAME}" != "temporal_attn" ]]; then
  echo "run_lr_matrix.sh is only intended for PROBE_NAME=temporal_attn." >&2
  exit 2
fi

BACKBONE_TAG="${BACKBONE_NAME}"
if [[ -n "${EFFECTIVE_BACKBONE_VARIANT}" && "${EFFECTIVE_BACKBONE_VARIANT}" != "<unknown>" ]]; then
  BACKBONE_TAG="${BACKBONE_TAG}_${EFFECTIVE_BACKBONE_VARIANT}"
fi
GROUP_SUBDIR="${GROUP_SUBDIR:-intphys2_probe_temporal_attn_${BACKBONE_TAG}_lr_matrix}"

MATRIX_LAYERS_COMPACT="${MATRIX_LAYERS// /}"
MATRIX_LRS_COMPACT="${MATRIX_LRS// /}"
MATRIX_LR_TAGS_COMPACT="${MATRIX_LR_TAGS// /}"
IFS=',' read -r -a LAYER_VALUES <<< "${MATRIX_LAYERS_COMPACT}"
IFS=',' read -r -a LR_VALUES <<< "${MATRIX_LRS_COMPACT}"
IFS=',' read -r -a LR_TAGS <<< "${MATRIX_LR_TAGS_COMPACT}"

NUM_LAYERS="${#LAYER_VALUES[@]}"
NUM_LRS="${#LR_VALUES[@]}"
if [[ "${NUM_LAYERS}" -eq 0 || "${NUM_LRS}" -eq 0 ]]; then
  echo "MATRIX_LAYERS and MATRIX_LRS must not be empty." >&2
  exit 2
fi
if [[ "${#LR_TAGS[@]}" -ne "${NUM_LRS}" ]]; then
  echo "MATRIX_LR_TAGS must contain the same number of entries as MATRIX_LRS." >&2
  exit 2
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
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

WANDB_GROUP="${WANDB_GROUP:-intphys2_temporal_attn_${BACKBONE_NAME}_lr_matrix}"
WANDB_NAME="${WANDB_NAME:-intphys2_${BACKBONE_NAME}_layer_${PROBE_LAYER}_lr_${LR_TAG}}"

echo "===== TRAIN PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "GROUP_SUBDIR=${GROUP_SUBDIR}"
echo "RUN_SUBDIR=${RUN_SUBDIR}"
echo "BACKBONE_NAME=${BACKBONE_NAME}"
echo "BACKBONE_VARIANT=${BACKBONE_VARIANT:-<config default>}"
echo "EFFECTIVE_BACKBONE_VARIANT=${EFFECTIVE_BACKBONE_VARIANT}"
echo "PROBE_LAYER=${PROBE_LAYER}"
echo "LR_VALUE=${LR_VALUE}"
echo "PROBE_EPOCHS=${PROBE_EPOCHS}"
echo "PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE}"
echo "PROBE_EVAL_BATCH_SIZE=${PROBE_EVAL_BATCH_SIZE}"
echo "PROBE_WEIGHT_DECAY=${PROBE_WEIGHT_DECAY}"
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
  "probe.early_stopping.patience=30"
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
  "probe.wandb.tags=[intphys2,${BACKBONE_NAME},temporal_attn,lr_matrix,layer_${PROBE_LAYER},lr_${LR_TAG}]"
  "probe.optuna.enabled=false"
  "feature_cache.include_tokens=true"
)

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  train_cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

echo "==> Launching training command:"
printf '  %q' "${train_cmd[@]}"
printf '\n'
"${train_cmd[@]}"

eval_cmd=(
  python run.py "eval.probe.${DATASET_NAME}"
  "backbone.name=${BACKBONE_NAME}"
  "probe.name=${PROBE_NAME}"
  "probe.device=${PROBE_DEVICE}"
  "probe.layer=${PROBE_LAYER}"
  "probe.feature_view=${PROBE_FEATURE_VIEW}"
  "probe.checkpoint_path=${CHECKPOINT_PATH}"
  "probe.eval_output_dir=${PROBE_EVAL_OUTPUT_DIR}"
  "probe.eval_output_subdir=${EVAL_SUBDIR}"
  "probe.eval_batch_size=${PROBE_EVAL_BATCH_SIZE}"
  "feature_cache.include_tokens=true"
)

if [[ -n "${BACKBONE_VARIANT}" ]]; then
  eval_cmd+=("+backbone.kwargs.variant=${BACKBONE_VARIANT}")
fi

echo "==> Launching evaluation command:"
printf '  %q' "${eval_cmd[@]}"
printf '\n'
"${eval_cmd[@]}"

export GROUP_ROOT RUN_ROOT EVAL_ROOT PROBE_LAYER LR_VALUE LR_TAG
mkdir -p "${GROUP_ROOT}"
exec 9>"${GROUP_ROOT}/.summary.lock"
flock 9
python - <<'PY'
import csv
import json
import os
from pathlib import Path
from typing import Any


DATASET = "intphys2"
PROBE_NAME = "temporal_attn"
FEATURE_VIEW = "tokens"
OBJECTIVE_METRIC_NAME = "voe_accuracy"
MODEL_SELECTION_SPLIT = "val"
PRIMARY_SPLIT = "test"
REPORTED_SPLITS = ["train", "val", "test"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scalar_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): value
        for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def selection_metric(eval_summary: dict[str, Any]) -> float:
    metrics_by_split = eval_summary.get("metrics_by_split", {})
    if isinstance(metrics_by_split, dict):
        val_metrics = metrics_by_split.get(MODEL_SELECTION_SPLIT, {})
        if isinstance(val_metrics, dict) and OBJECTIVE_METRIC_NAME in val_metrics:
            return float(val_metrics[OBJECTIVE_METRIC_NAME])
    return float(eval_summary.get("objective_metric", 0.0))


def metric_value(item: dict[str, Any]) -> float:
    return float(item.get("selection_metric", item.get("objective_metric", 0.0)))


def write_matrix_csv(path: Path, summary: dict[str, Any]) -> None:
    report_splits = [str(item) for item in summary.get("reported_splits", [])]
    layer_summaries = summary.get("layers", [])
    metric_keys_by_split: dict[str, set[str]] = {split: set() for split in report_splits}
    extra_splits: set[str] = set()
    for layer_summary in layer_summaries:
        if not isinstance(layer_summary, dict):
            continue
        eval_summary = layer_summary.get("eval", {})
        metrics_by_split = (
            eval_summary.get("metrics_by_split", {})
            if isinstance(eval_summary, dict)
            else {}
        )
        if not isinstance(metrics_by_split, dict):
            continue
        for split_name, metrics in metrics_by_split.items():
            split_label = str(split_name)
            target = metric_keys_by_split.get(split_label)
            if target is None:
                extra_splits.add(split_label)
                target = metric_keys_by_split.setdefault(split_label, set())
            target.update(scalar_metrics(metrics).keys())

    ordered_splits = report_splits + sorted(extra_splits - set(report_splits))
    metric_columns = [
        f"{split}_{metric}"
        for split in ordered_splits
        for metric in sorted(metric_keys_by_split.get(split, set()))
    ]
    fieldnames = [
        "layer",
        "layer_label",
        "selected_lr",
        "selected_lr_tag",
        "selection_metric_name",
        "selection_metric",
        "objective_metric_name",
        "objective_metric",
        "checkpoint",
    ] + metric_columns

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_summary in layer_summaries:
            if not isinstance(layer_summary, dict):
                continue
            eval_summary = layer_summary.get("eval", {})
            metrics_by_split = (
                eval_summary.get("metrics_by_split", {})
                if isinstance(eval_summary, dict)
                else {}
            )
            row = {
                "layer": layer_summary.get("layer", ""),
                "layer_label": layer_summary.get("layer_label", ""),
                "selected_lr": layer_summary.get("learning_rate", ""),
                "selected_lr_tag": layer_summary.get("learning_rate_tag", ""),
                "selection_metric_name": layer_summary.get(
                    "selection_metric_name", OBJECTIVE_METRIC_NAME
                ),
                "selection_metric": layer_summary.get("selection_metric", ""),
                "objective_metric_name": summary.get(
                    "objective_metric_name", OBJECTIVE_METRIC_NAME
                ),
                "objective_metric": layer_summary.get("objective_metric", ""),
                "checkpoint": layer_summary.get("checkpoint", ""),
            }
            if isinstance(metrics_by_split, dict):
                for split_name, metrics in metrics_by_split.items():
                    for metric_name, metric in scalar_metrics(metrics).items():
                        row[f"{split_name}_{metric_name}"] = metric
            writer.writerow(row)


group_root = Path(os.environ["GROUP_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
probe_layer = str(os.environ["PROBE_LAYER"])
lr_value = str(os.environ["LR_VALUE"])
lr_tag = str(os.environ["LR_TAG"])

train_summary = read_json(run_root / "train" / "train_summary.json")
eval_summary = read_json(eval_root / "probe_eval_summary.json")
selection_value = selection_metric(eval_summary)

single_layer = {
    "layer": probe_layer,
    "layer_label": probe_layer,
    "learning_rate": lr_value,
    "learning_rate_tag": lr_tag,
    "checkpoint": str(train_summary["checkpoint"]),
    "selection_metric_name": OBJECTIVE_METRIC_NAME,
    "selection_metric": selection_value,
    "objective_metric": float(eval_summary["objective_metric"]),
    "train": train_summary,
    "eval": eval_summary,
}

single_layer_summary = {
    "dataset": DATASET,
    "probe_name": PROBE_NAME,
    "feature_view": FEATURE_VIEW,
    "train_split": "train",
    "objective_metric_name": OBJECTIVE_METRIC_NAME,
    "model_selection_split": MODEL_SELECTION_SPLIT,
    "split_name": eval_summary.get("split_name", PRIMARY_SPLIT),
    "reported_splits": list(eval_summary.get("reported_splits", REPORTED_SPLITS)),
    "sweep_dir": str(run_root),
    "requested_layers": [probe_layer],
    "layers": [single_layer],
    "best_layer": probe_layer,
    "best_layer_label": probe_layer,
    "best_selection_metric": selection_value,
    "best_objective_metric": float(eval_summary["objective_metric"]),
}

write_json(run_root / "train_eval_summary.json", single_layer_summary)
write_matrix_csv(run_root / "train_eval_summary.csv", single_layer_summary)

best_by_layer: dict[str, dict[str, Any]] = {}
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
    "dataset": DATASET,
    "probe_name": PROBE_NAME,
    "feature_view": FEATURE_VIEW,
    "train_split": "train",
    "objective_metric_name": OBJECTIVE_METRIC_NAME,
    "model_selection_split": MODEL_SELECTION_SPLIT,
    "split_name": PRIMARY_SPLIT,
    "reported_splits": REPORTED_SPLITS,
    "sweep_dir": str(group_root),
    "requested_layers": [layer.get("layer", "") for layer in group_layers],
    "layers": group_layers,
    "best_layer": None,
    "best_layer_label": None,
    "best_selection_metric": None,
    "best_objective_metric": None,
}
if group_layers:
    best_overall = max(group_layers, key=metric_value)
    group_summary["best_layer"] = best_overall.get("layer")
    group_summary["best_layer_label"] = best_overall.get("layer_label")
    group_summary["best_selection_metric"] = metric_value(best_overall)
    group_summary["best_objective_metric"] = float(best_overall.get("objective_metric", 0.0))

write_json(group_root / "train_eval_summary.json", group_summary)
write_matrix_csv(group_root / "train_eval_summary.csv", group_summary)
PY
flock -u 9

JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
JOB_ELAPSED_SECONDS="$((JOB_END_EPOCH - JOB_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=${JOB_ELAPSED_SECONDS}"
