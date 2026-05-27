#!/bin/bash
# Run fixed-config MVP JEPA-v1 temporal-attention timing experiments on Snellius.
#
# Modes:
#   baseline: shared cache, batch_size=1, eval_batch_size=1
#   storage_fix: layer-32 token-only cache on scratch-shared, copied into TMPDIR
#   storage_fix_bs2: storage_fix plus batch_size=2 and eval_batch_size=2
#   storage_fix_bs4: storage_fix plus batch_size=4 and eval_batch_size=4
#   storage_fix_bs8: storage_fix plus batch_size=8 and eval_batch_size=8
#
# Usage:
#   sbatch ops/hpc/train/mvp/temporal_attn/jepa_v1_timing_worker.sh
#   sbatch --export=ALL,EXPERIMENT_MODE=storage_fix ops/hpc/train/mvp/temporal_attn/jepa_v1_timing_worker.sh
#   sbatch --export=ALL,EXPERIMENT_MODE=storage_fix_bs4 ops/hpc/train/mvp/temporal_attn/jepa_v1_timing_worker.sh
#   sbatch --export=ALL,EXPERIMENT_MODE=storage_fix_bs8 ops/hpc/train/mvp/temporal_attn/jepa_v1_timing_worker.sh

#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node
#SBATCH --job-name=mvp_jepa_v1_timing
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=output/training/mvp/attention/%x_%j.out
#SBATCH --error=output/training/mvp/attention/%x_%j.err

set -euo pipefail

JOB_COMMAND=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -n 1)"
fi
if [[ -n "${JOB_COMMAND}" && "${JOB_COMMAND}" != /* ]]; then
  JOB_COMMAND="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
if [[ -n "${JOB_COMMAND}" ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../extract/common.sh"

require_scratch_node_tmpdir() {
  if [[ -z "${TMPDIR:-}" ]]; then
    echo "ERROR: TMPDIR is not set." >&2
    echo "Expected a scratch-node allocation where TMPDIR points under /scratch-node." >&2
    exit 2
  fi
  if [[ "${TMPDIR}" != /scratch-node/* ]]; then
    echo "ERROR: TMPDIR is '${TMPDIR}', expected /scratch-node/... for node-local staging." >&2
    echo "Make sure this job is submitted with '#SBATCH --constraint=scratch-node'." >&2
    exit 2
  fi
}

REPO_ROOT="${PROJECT_ROOT:-$(resolve_repo_root "${SCRIPT_PATH}")}"
cd "${REPO_ROOT}"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-baseline}"
case "${EXPERIMENT_MODE}" in
  baseline|storage_fix|storage_fix_bs2|storage_fix_bs4|storage_fix_bs8) ;;
  *)
    echo "ERROR: Unsupported EXPERIMENT_MODE='${EXPERIMENT_MODE}'." >&2
    echo "Supported values: baseline | storage_fix | storage_fix_bs2 | storage_fix_bs4 | storage_fix_bs8" >&2
    exit 2
    ;;
esac

require_scratch_node_tmpdir
load_probe4physics_env
configure_hf_cache

FIXED_LAYER="${FIXED_LAYER:-32}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-probe4physics}"
WANDB_MODE="${WANDB_MODE:-online}"
BACKBONE_VARIANT="${BACKBONE_VARIANT:-vith16_384}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-cuda}"
SHARED_FEATURE_ROOT="${SHARED_FEATURE_ROOT:-/scratch-shared/${USER}/probe4physics/artifacts/features/mvp}"
SHARED_PROBE_ROOT="${SHARED_PROBE_ROOT:-/scratch-shared/${USER}/probe4physics/artifacts/probes/mvp_timing}"
SHARED_RESULT_ROOT="${SHARED_RESULT_ROOT:-/scratch-shared/${USER}/probe4physics/artifacts/results/mvp_timing}"
LOCAL_FEATURE_ROOT="${LOCAL_FEATURE_ROOT:-${TMPDIR}/${USER}_probe4physics_features_mvp_${SLURM_JOB_ID:-manual}}"
TIMESTAMP_UTC="${TIMESTAMP_UTC:-$(date -u +"%Y%m%dT%H%M%SZ")}"
TIMING_GROUP="${TIMING_GROUP:-mvp_temporal_attn_jepa_v1_timing_${TIMESTAMP_UTC}}"
RUN_STEM="${TIMING_GROUP}_${EXPERIMENT_MODE}_${SLURM_JOB_ID:-manual}"

mkdir -p "${SHARED_PROBE_ROOT}" "${SHARED_RESULT_ROOT}"

TRAIN_BATCH_SIZE="1"
EVAL_BATCH_SIZE="1"
STORAGE_STRATEGY="shared_cache_default_signature"
USE_LOCAL_LAYER32_CACHE="false"
if [[ "${EXPERIMENT_MODE}" == "storage_fix" ]]; then
  STORAGE_STRATEGY="scratch_node_local_layer32_token_cache"
  USE_LOCAL_LAYER32_CACHE="true"
elif [[ "${EXPERIMENT_MODE}" == "storage_fix_bs2" ]]; then
  TRAIN_BATCH_SIZE="2"
  EVAL_BATCH_SIZE="2"
  STORAGE_STRATEGY="scratch_node_local_layer32_token_cache"
  USE_LOCAL_LAYER32_CACHE="true"
elif [[ "${EXPERIMENT_MODE}" == "storage_fix_bs4" ]]; then
  TRAIN_BATCH_SIZE="4"
  EVAL_BATCH_SIZE="4"
  STORAGE_STRATEGY="scratch_node_local_layer32_token_cache"
  USE_LOCAL_LAYER32_CACHE="true"
elif [[ "${EXPERIMENT_MODE}" == "storage_fix_bs8" ]]; then
  TRAIN_BATCH_SIZE="8"
  EVAL_BATCH_SIZE="8"
  STORAGE_STRATEGY="scratch_node_local_layer32_token_cache"
  USE_LOCAL_LAYER32_CACHE="true"
fi

SOURCE_CACHE_DIR=""
SOURCE_SIGNATURE=""
TARGET_CACHE_DIR=""
TARGET_SIGNATURE=""
TARGET_CACHE_STATUS=""
LOCAL_CACHE_DIR=""

resolve_cache_layout() {
  local mode="${1}"
  local line key value
  while IFS='=' read -r key value; do
    case "${key}" in
      SOURCE_CACHE_DIR) SOURCE_CACHE_DIR="${value}" ;;
      SOURCE_SIGNATURE) SOURCE_SIGNATURE="${value}" ;;
      TARGET_CACHE_DIR) TARGET_CACHE_DIR="${value}" ;;
      TARGET_SIGNATURE) TARGET_SIGNATURE="${value}" ;;
      TARGET_CACHE_STATUS) TARGET_CACHE_STATUS="${value}" ;;
    esac
  done < <(
    MODE="${mode}" \
    REPO_ROOT="${REPO_ROOT}" \
    SHARED_FEATURE_ROOT="${SHARED_FEATURE_ROOT}" \
    FIXED_LAYER="${FIXED_LAYER}" \
    BACKBONE_VARIANT="${BACKBONE_VARIANT}" \
    BACKBONE_DEVICE="${BACKBONE_DEVICE}" \
    python - <<'PY'
import copy
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fcntl
import torch
import yaml

from benchmarks.mvp.features import (
    _feature_cfg,
    _find_compatible_feature_cache,
    has_valid_feature_cache,
    resolve_expected_feature_cache_paths,
)


def build_cfg(*, repo_root: Path, feature_root: Path, layer_ids: list[int], include_pooled: bool, include_tokens: bool) -> dict:
    cfg = yaml.safe_load((repo_root / "configs" / "mvp.yaml").read_text(encoding="utf-8"))
    cfg["backbone"] = {
        "name": "jepa_v1",
        "kwargs": {
            "device": os.environ["BACKBONE_DEVICE"],
            "variant": os.environ["BACKBONE_VARIANT"],
        },
    }
    feature_cfg = cfg.setdefault("feature_cache", {})
    feature_cfg["dir"] = str(feature_root)
    feature_cfg["split_names"] = ["train", "val", "test"]
    feature_cfg["layer_ids"] = list(layer_ids)
    feature_cfg["include_pooled"] = bool(include_pooled)
    feature_cfg["include_tokens"] = bool(include_tokens)
    return cfg


def ensure_source_cache(repo_root: Path, feature_root: Path) -> tuple[Path, str]:
    source_cfg = build_cfg(
        repo_root=repo_root,
        feature_root=feature_root,
        layer_ids=[],
        include_pooled=True,
        include_tokens=True,
    )
    source_paths = resolve_expected_feature_cache_paths(source_cfg)
    if not has_valid_feature_cache(source_cfg):
        compatible = _find_compatible_feature_cache(
            source_cfg,
            exact_paths=source_paths,
            feature_cfg=_feature_cfg(source_cfg),
        )
        if compatible is None:
            raise SystemExit(
                "Missing source MVP token cache for JEPA-v1. "
                "Expected an exact or compatible cache under "
                f"{feature_root}."
            )
        source_paths = compatible
    return source_paths.cache_dir, source_paths.signature


def ensure_target_cache(repo_root: Path, feature_root: Path, layer: int) -> tuple[Path, str, str]:
    source_cache_dir, source_signature = ensure_source_cache(repo_root, feature_root)
    source_manifest_path = source_cache_dir / "manifest.json"
    source_state_path = source_cache_dir / ".resume" / "state.json"

    target_cfg = build_cfg(
        repo_root=repo_root,
        feature_root=feature_root,
        layer_ids=[layer],
        include_pooled=False,
        include_tokens=True,
    )
    target_paths = resolve_expected_feature_cache_paths(target_cfg)
    status = "reused"

    if not has_valid_feature_cache(target_cfg):
        target_paths.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target_paths.cache_dir.parent / f".{target_paths.signature}.build.lock"
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if not has_valid_feature_cache(target_cfg):
                status = "built"
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
                source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
                completed_chunks = sorted(
                    list(source_state.get("completed_chunks", [])),
                    key=lambda chunk: int(chunk.get("start_offset", 0)),
                )
                if not completed_chunks:
                    raise SystemExit("Source chunked token cache has no completed chunks.")

                tmp_dir = target_paths.cache_dir.parent / f".{target_paths.signature}.tmp-{os.getpid()}"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                resume_dir = tmp_dir / ".resume"
                chunks_root = resume_dir / "chunks"
                chunks_root.mkdir(parents=True, exist_ok=True)

                shutil.copy2(source_cache_dir / "index.parquet", tmp_dir / "index.parquet")

                new_completed_chunks = []
                for chunk in completed_chunks:
                    source_tokens_ref = str(chunk.get("tokens_path", "")).strip()
                    if not source_tokens_ref:
                        raise SystemExit("Source chunked token cache is missing tokens_path entries.")
                    source_tokens_path = Path(source_tokens_ref)
                    if not source_tokens_path.is_absolute():
                        source_tokens_path = (source_cache_dir / source_tokens_path).resolve()
                    if not source_tokens_path.exists():
                        raise SystemExit(f"Missing source chunk payload: {source_tokens_path}")

                    payload = torch.load(str(source_tokens_path), map_location="cpu", weights_only=False)
                    by_layer = payload.get("by_layer", {})
                    if layer in by_layer:
                        layer_tensor = by_layer[layer]
                    elif str(layer) in by_layer:
                        layer_tensor = by_layer[str(layer)]
                    else:
                        raise SystemExit(
                            f"Source chunk {source_tokens_path} does not contain layer {layer}."
                        )

                    chunk_dir_name = source_tokens_path.parent.name
                    chunk_rel_dir = Path(".resume") / "chunks" / chunk_dir_name
                    chunk_dir = tmp_dir / chunk_rel_dir
                    chunk_dir.mkdir(parents=True, exist_ok=True)

                    source_index_ref = str(chunk.get("index_path", "")).strip()
                    if source_index_ref:
                        source_index_path = Path(source_index_ref)
                        if not source_index_path.is_absolute():
                            source_index_path = (source_cache_dir / source_index_path).resolve()
                        if source_index_path.exists():
                            shutil.copy2(source_index_path, chunk_dir / "index.parquet")

                    target_tokens_rel = chunk_rel_dir / "features_tokens.pt"
                    torch.save(
                        {"selected_layers": [layer], "by_layer": {layer: layer_tensor}},
                        tmp_dir / target_tokens_rel,
                    )

                    new_chunk = dict(chunk)
                    new_chunk["index_path"] = str(chunk_rel_dir / "index.parquet")
                    new_chunk["pooled_path"] = ""
                    new_chunk["tokens_path"] = str(target_tokens_rel)
                    new_completed_chunks.append(new_chunk)

                new_state = copy.deepcopy(source_state)
                new_state["selected_layers"] = [layer]
                new_state["completed_chunks"] = new_completed_chunks
                new_state["status"] = "complete"
                (resume_dir / "state.json").write_text(
                    json.dumps(new_state, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                (resume_dir / "events.jsonl").touch()

                target_manifest = copy.deepcopy(source_manifest)
                target_manifest["created_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
                target_manifest["signature"] = target_paths.signature
                target_manifest.setdefault("backbone", {}).setdefault("kwargs", {})
                target_manifest["backbone"]["kwargs"]["device"] = os.environ["BACKBONE_DEVICE"]
                target_manifest["backbone"]["kwargs"]["variant"] = os.environ["BACKBONE_VARIANT"]
                target_manifest.setdefault("features", {})
                target_manifest["features"]["include_pooled"] = False
                target_manifest["features"]["include_tokens"] = True
                target_manifest["features"]["selected_layers"] = [layer]
                target_manifest.setdefault("files", {})
                target_manifest["files"]["index"] = "index.parquet"
                target_manifest["files"]["pooled"] = ""
                target_manifest["files"]["tokens"] = ""
                target_manifest["storage"] = {
                    "tokens": {
                        "format": "chunked_resume",
                        "state": ".resume/state.json",
                        "chunks_dir": ".resume/chunks",
                    }
                }
                target_manifest["resume"] = {
                    "enabled": True,
                    "resumed": True,
                    "reused_samples": int(target_manifest.get("features", {}).get("n_samples", 0)),
                    "new_samples": 0,
                    "resume_start_offset": int(target_manifest.get("features", {}).get("n_samples", 0)),
                    "source_state_path": ".resume/state.json",
                    "events_path": ".resume/events.jsonl",
                }
                (tmp_dir / "manifest.json").write_text(
                    json.dumps(target_manifest, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                if target_paths.cache_dir.exists():
                    shutil.rmtree(target_paths.cache_dir)
                tmp_dir.rename(target_paths.cache_dir)

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    if not has_valid_feature_cache(target_cfg):
        raise SystemExit(f"Repacked target cache is still invalid: {target_paths.cache_dir}")

    print(f"SOURCE_CACHE_DIR={source_cache_dir}")
    print(f"SOURCE_SIGNATURE={source_signature}")
    print(f"TARGET_CACHE_DIR={target_paths.cache_dir}")
    print(f"TARGET_SIGNATURE={target_paths.signature}")
    print(f"TARGET_CACHE_STATUS={status}")


repo_root = Path(os.environ["REPO_ROOT"]).resolve()
feature_root = Path(os.environ["SHARED_FEATURE_ROOT"]).expanduser().resolve()
mode = str(os.environ["MODE"]).strip()
layer = int(os.environ["FIXED_LAYER"])

if mode == "baseline":
    source_cache_dir, source_signature = ensure_source_cache(repo_root, feature_root)
    print(f"SOURCE_CACHE_DIR={source_cache_dir}")
    print(f"SOURCE_SIGNATURE={source_signature}")
    print("TARGET_CACHE_DIR=")
    print("TARGET_SIGNATURE=")
    print("TARGET_CACHE_STATUS=")
else:
    ensure_target_cache(repo_root, feature_root, layer)
PY
  )
}

stage_local_layer32_cache() {
  local shared_feature_root_resolved
  local target_cache_dir_resolved
  local relative_cache_path
  if [[ -z "${TARGET_CACHE_DIR}" ]]; then
    echo "ERROR: TARGET_CACHE_DIR is empty; cannot stage local cache." >&2
    exit 2
  fi
  shared_feature_root_resolved="$(readlink -f "${SHARED_FEATURE_ROOT}")"
  target_cache_dir_resolved="$(readlink -f "${TARGET_CACHE_DIR}")"
  relative_cache_path="${target_cache_dir_resolved#${shared_feature_root_resolved}/}"
  if [[ "${relative_cache_path}" == "${target_cache_dir_resolved}" ]]; then
    echo "ERROR: Could not derive cache path relative to SHARED_FEATURE_ROOT." >&2
    echo "SHARED_FEATURE_ROOT=${SHARED_FEATURE_ROOT}" >&2
    echo "SHARED_FEATURE_ROOT_RESOLVED=${shared_feature_root_resolved}" >&2
    echo "TARGET_CACHE_DIR=${TARGET_CACHE_DIR}" >&2
    echo "TARGET_CACHE_DIR_RESOLVED=${target_cache_dir_resolved}" >&2
    exit 2
  fi
  LOCAL_CACHE_DIR="${LOCAL_FEATURE_ROOT}/${relative_cache_path}"
  mkdir -p "$(dirname "${LOCAL_CACHE_DIR}")"
  rsync -ah --delete --info=progress2 "${target_cache_dir_resolved}/" "${LOCAL_CACHE_DIR}/"
}

echo "===== TIMING PROVENANCE ====="
date -u
hostname
git -C "${REPO_ROOT}" rev-parse HEAD
python --version
which python
echo "REPO_ROOT=${REPO_ROOT}"
echo "EXPERIMENT_MODE=${EXPERIMENT_MODE}"
echo "TIMING_GROUP=${TIMING_GROUP}"
echo "RUN_STEM=${RUN_STEM}"
echo "FIXED_LAYER=${FIXED_LAYER}"
echo "PROBE_EPOCHS=${PROBE_EPOCHS}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "STORAGE_STRATEGY=${STORAGE_STRATEGY}"
echo "SHARED_FEATURE_ROOT=${SHARED_FEATURE_ROOT}"
echo "SHARED_PROBE_ROOT=${SHARED_PROBE_ROOT}"
echo "SHARED_RESULT_ROOT=${SHARED_RESULT_ROOT}"
echo "TMPDIR=${TMPDIR}"
echo "LOCAL_FEATURE_ROOT=${LOCAL_FEATURE_ROOT}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "============================="

JOB_START_EPOCH="$(date +%s)"
JOB_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "JOB_START_UTC=${JOB_START_UTC}"

CACHE_PREP_START_EPOCH="$(date +%s)"
resolve_cache_layout "${EXPERIMENT_MODE}"
echo "SOURCE_CACHE_DIR=${SOURCE_CACHE_DIR}"
echo "SOURCE_SIGNATURE=${SOURCE_SIGNATURE}"
if [[ "${USE_LOCAL_LAYER32_CACHE}" == "true" ]]; then
  echo "TARGET_CACHE_DIR=${TARGET_CACHE_DIR}"
  echo "TARGET_SIGNATURE=${TARGET_SIGNATURE}"
  echo "TARGET_CACHE_STATUS=${TARGET_CACHE_STATUS}"
  LOCAL_STAGE_START_EPOCH="$(date +%s)"
  stage_local_layer32_cache
  LOCAL_STAGE_END_EPOCH="$(date +%s)"
  echo "LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR}"
  echo "LOCAL_STAGE_ELAPSED_SECONDS=$((LOCAL_STAGE_END_EPOCH - LOCAL_STAGE_START_EPOCH))"
fi
CACHE_PREP_END_EPOCH="$(date +%s)"
echo "CACHE_PREP_ELAPSED_SECONDS=$((CACHE_PREP_END_EPOCH - CACHE_PREP_START_EPOCH))"

FEATURE_CACHE_DIR_OVERRIDE="${SHARED_FEATURE_ROOT}"
FEATURE_CACHE_LAYER_IDS_OVERRIDE="feature_cache.layer_ids=[]"
FEATURE_CACHE_INCLUDE_POOLED_OVERRIDE="feature_cache.include_pooled=true"
if [[ "${USE_LOCAL_LAYER32_CACHE}" == "true" ]]; then
  FEATURE_CACHE_DIR_OVERRIDE="${LOCAL_FEATURE_ROOT}"
  FEATURE_CACHE_LAYER_IDS_OVERRIDE="feature_cache.layer_ids=[${FIXED_LAYER}]"
  FEATURE_CACHE_INCLUDE_POOLED_OVERRIDE="feature_cache.include_pooled=false"
fi

cmd=(
  python run.py "train_eval.probe.mvp"
  "backbone.name=jepa_v1"
  "+backbone.kwargs.variant=${BACKBONE_VARIANT}"
  "backbone.kwargs.device=${BACKBONE_DEVICE}"
  "probe.name=temporal_attn"
  "probe.device=cuda"
  "probe.layer=${FIXED_LAYER}"
  "probe.feature_view=tokens"
  "probe.epochs=${PROBE_EPOCHS}"
  "probe.lr=5.9750279999602906e-05"
  "probe.weight_decay=9.44351568796269e-05"
  "probe.batch_size=${TRAIN_BATCH_SIZE}"
  "probe.eval_batch_size=${EVAL_BATCH_SIZE}"
  "probe.temporal_attn.num_heads=16"
  "probe.temporal_attn.num_self_attn_blocks=2"
  "probe.temporal_attn.mlp_ratio=2.0"
  "probe.temporal_attn.dropout=0.17936999364332554"
  "probe.output_dir=${SHARED_PROBE_ROOT}"
  "probe.output_subdir=${RUN_STEM}"
  "probe.eval_output_dir=${SHARED_RESULT_ROOT}"
  "probe.eval_output_subdir=${RUN_STEM}"
  "probe.wandb.enabled=true"
  "probe.wandb.project=${WANDB_PROJECT}"
  "probe.wandb.mode=${WANDB_MODE}"
  "probe.wandb.group=${TIMING_GROUP}"
  "probe.wandb.name=${RUN_STEM}"
  "probe.wandb.tags=[timing,snellius,scratch-node,mvp,jepa_v1,temporal_attn,${EXPERIMENT_MODE}]"
  "probe.optuna.enabled=false"
  "feature_cache.dir=${FEATURE_CACHE_DIR_OVERRIDE}"
  "${FEATURE_CACHE_LAYER_IDS_OVERRIDE}"
  "${FEATURE_CACHE_INCLUDE_POOLED_OVERRIDE}"
  "feature_cache.include_tokens=true"
)

echo "==> Launching training command:"
printf '  %q' "${cmd[@]}"
echo

TRAIN_START_EPOCH="$(date +%s)"
"${cmd[@]}"
TRAIN_END_EPOCH="$(date +%s)"
JOB_END_EPOCH="$(date +%s)"
JOB_END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "TRAIN_ELAPSED_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))"
echo "JOB_END_UTC=${JOB_END_UTC}"
echo "JOB_ELAPSED_SECONDS=$((JOB_END_EPOCH - JOB_START_EPOCH))"
