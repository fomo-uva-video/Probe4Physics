from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.mvp.eval import run_mvp_eval
from benchmarks.mvp.features import load_feature_cache_for_config
from models.cache_metadata import resolve_backbone_cache_metadata
from probes import LinearProbe, create_probe
from training.wandb_utils import init_wandb_train_logger


class LinearProbeConfigError(ValueError):
    pass


REQUIRED_MVP_SEMANTIC_COLUMNS = (
    "plausibility_label",
    "yes_choice_idx",
    "no_choice_idx",
)


def run_mvp_train_linear(config: dict[str, Any]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)
    _require_semantic_feature_index(index)

    probe_cfg = _linear_cfg(config)
    features = _select_feature_tensor(bundle, probe_cfg)
    labels = torch.tensor(index["plausibility_label"].tolist(), dtype=torch.long)

    train_mask = index["split"].astype(str) == "train"
    val_mask = index["split"].astype(str) == "val"

    if not train_mask.any():
        raise LinearProbeConfigError("No train samples in feature cache index.")

    x_train = features[torch.tensor(train_mask.to_numpy())]
    y_train = labels[torch.tensor(train_mask.to_numpy())]

    x_val = None
    y_val = None
    if val_mask.any():
        x_val = features[torch.tensor(val_mask.to_numpy())]
        y_val = labels[torch.tensor(val_mask.to_numpy())]

    input_dim = int(x_train.shape[1])
    num_classes = int(labels.max().item()) + 1

    probe = create_probe(
        "linear",
        input_dim=input_dim,
        num_classes=max(num_classes, 2),
        device=probe_cfg["device"],
    )

    output_dir = _resolve_train_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = init_wandb_train_logger(
        config,
        benchmark="mvp",
        output_dir=output_dir,
        metadata={
            "feature_signature": str(manifest.get("signature", "")),
            "feature_view": probe_cfg["feature_view"],
            "layer": probe_cfg["layer"],
            "target_type": "semantic_plausibility",
        },
    )

    train_summary: dict[str, Any] | None = None
    try:
        fit = probe.fit(
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=probe_cfg["epochs"],
            lr=probe_cfg["lr"],
            batch_size=probe_cfg["batch_size"],
            weight_decay=probe_cfg["weight_decay"],
            seed=int(config.get("seed", 42)),
            epoch_logger=None if logger is None else logger.log_epoch,
        )
        checkpoint_last_path = output_dir / "linear_probe_last.pt"
        checkpoint_best_path = output_dir / "linear_probe_best.pt"
        metadata = {
            "feature_signature": str(manifest.get("signature", "")),
            "feature_cache_dir": str(bundle["paths"].cache_dir),
            "feature_view": probe_cfg["feature_view"],
            "layer": probe_cfg["layer"],
            "target_type": "semantic_plausibility",
            "positive_label": 1,
            "negative_label": 0,
            "seed": int(config.get("seed", 42)),
            "checkpoint_kind": "last",
        }
        probe.save(checkpoint_last_path, metadata=metadata)

        best_metadata = dict(metadata)
        best_metadata["checkpoint_kind"] = "best"
        if fit.best_epoch is not None:
            best_metadata["best_epoch"] = int(fit.best_epoch)
        if fit.best_val_accuracy is not None:
            best_metadata["best_val_accuracy"] = float(fit.best_val_accuracy)
        if fit.best_val_loss is not None:
            best_metadata["best_val_loss"] = float(fit.best_val_loss)

        best_state_dict = getattr(probe, "best_fit_state_dict", None)
        best_state = best_state_dict() if callable(best_state_dict) else None
        if best_state is not None and hasattr(probe, "model"):
            last_state = copy.deepcopy(probe.model.state_dict())
            probe.model.load_state_dict(best_state)
            probe.save(checkpoint_best_path, metadata=best_metadata)
            probe.model.load_state_dict(last_state)
        else:
            probe.save(checkpoint_best_path, metadata=best_metadata)

        train_summary = {
            "checkpoint": str(checkpoint_best_path),
            "checkpoint_last": str(checkpoint_last_path),
            "checkpoint_best": str(checkpoint_best_path),
            "output_dir": str(output_dir),
            "feature_signature": str(manifest.get("signature", "")),
            "fit": {
                "train_loss": fit.train_loss,
                "train_accuracy": fit.train_accuracy,
                "val_loss": fit.val_loss,
                "val_accuracy": fit.val_accuracy,
                "n_epochs": fit.n_epochs,
                "best_epoch": fit.best_epoch,
                "best_val_loss": fit.best_val_loss,
                "best_val_accuracy": fit.best_val_accuracy,
                "history": fit.history,
            },
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "input_dim": input_dim,
            "elapsed_seconds": max(0.0, time.perf_counter() - wall_start),
        }
        train_summary["seconds_per_train_sample"] = (
            train_summary["elapsed_seconds"] / float(train_summary["n_train"])
            if train_summary["n_train"] > 0
            else 0.0
        )
        total_seen = int(train_summary["n_train"]) + int(train_summary["n_val"])
        train_summary["seconds_per_labeled_sample"] = (
            train_summary["elapsed_seconds"] / float(total_seen)
            if total_seen > 0
            else 0.0
        )

        (output_dir / "train_summary.json").write_text(
            json.dumps(train_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return train_summary
    finally:
        if logger is not None:
            try:
                if train_summary is not None:
                    logger.log_summary(train_summary)
            finally:
                logger.finish()


def run_mvp_eval_linear(config: dict[str, Any]) -> dict[str, Any]:
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)
    _require_semantic_feature_index(index)

    probe_cfg = _linear_cfg(config)
    split_name = str(config.get("split_name", "test"))

    checkpoint_path = _resolve_checkpoint_path(config)
    probe = LinearProbe.load(checkpoint_path, device=probe_cfg["device"])

    ckpt_payload = torch.load(str(checkpoint_path), map_location="cpu")
    ckpt_meta = ckpt_payload.get("metadata", {}) if isinstance(ckpt_payload, dict) else {}
    checkpoint_signature = str(ckpt_meta.get("feature_signature", ""))
    current_signature = str(manifest.get("signature", ""))
    if checkpoint_signature and checkpoint_signature != current_signature:
        raise LinearProbeConfigError(
            "Linear checkpoint feature signature mismatch. "
            f"checkpoint={checkpoint_signature}, current={current_signature}."
        )
    _require_semantic_checkpoint(ckpt_meta)

    features = _select_feature_tensor(bundle, probe_cfg)
    split_mask = index["split"].astype(str) == split_name
    if not split_mask.any():
        raise LinearProbeConfigError(f"No samples found for split_name='{split_name}' in cache index.")

    split_idx = torch.tensor(split_mask.to_numpy())
    x_eval = features[split_idx]
    split_frame = index.loc[split_mask].copy()

    semantic_pred = probe.predict(x_eval, batch_size=probe_cfg["eval_batch_size"]).tolist()
    pred_idx = _semantic_predictions_to_choice_indices(split_frame, semantic_pred)
    sample_ids = split_frame["sample_id"].tolist()
    if len(sample_ids) != len(pred_idx):
        raise LinearProbeConfigError(
            "MVP linear eval produced a prediction count mismatch. "
            f"n_samples={len(sample_ids)}, n_predictions={len(pred_idx)}."
        )
    pred_by_sample = {
        str(sample_id): int(pred)
        for sample_id, pred in zip(sample_ids, pred_idx)
    }

    output_dir = _resolve_eval_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = output_dir / "linear_predictions.json"
    pred_file.write_text(json.dumps(pred_by_sample, indent=2, sort_keys=True), encoding="utf-8")

    eval_cfg = dict(config)
    eval_cfg["predictor"] = {
        "mode": "from_file",
        "prediction_file": str(pred_file),
    }
    eval_cfg["output_subdir"] = output_dir.name
    eval_cfg["output_dir"] = str(output_dir.parent)

    result = run_mvp_eval(eval_cfg)
    summary = {
        "linear_eval_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "prediction_file": str(pred_file),
        "feature_signature": current_signature,
        "base_eval": result,
    }
    (output_dir / "linear_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _select_feature_tensor(bundle: dict[str, Any], probe_cfg: dict[str, Any]) -> torch.Tensor:
    feature_view = probe_cfg["feature_view"]
    layer = probe_cfg["layer"]

    pooled = bundle.get("pooled")
    tokens = bundle.get("tokens")

    if feature_view == "pooled":
        if pooled is None:
            raise LinearProbeConfigError("Pooled features are not available in cache.")
        selected = _resolve_layer(layer, pooled)
        tensor = pooled["by_layer"][selected]
        if tensor.ndim != 2:
            raise LinearProbeConfigError(f"Expected pooled tensor [N, D], got {tuple(tensor.shape)}")
        return tensor.to(dtype=torch.float32)

    if feature_view == "tokens_mean":
        if tokens is None:
            raise LinearProbeConfigError("Token features are not available in cache.")
        selected = _resolve_layer(layer, tokens)
        tensor = tokens["by_layer"][selected]
        if tensor.ndim != 3:
            raise LinearProbeConfigError(f"Expected token tensor [N, T, D], got {tuple(tensor.shape)}")
        return tensor.mean(dim=1).to(dtype=torch.float32)

    raise LinearProbeConfigError(
        f"Unsupported linear_probe.feature_view='{feature_view}'. Use pooled|tokens_mean."
    )


def _resolve_layer(layer: int | str, payload: dict[str, Any]) -> int:
    selected_layers = [int(item) for item in payload.get("selected_layers", [])]
    if not selected_layers:
        selected_layers = sorted(int(k) for k in payload.get("by_layer", {}).keys())
    if not selected_layers:
        raise LinearProbeConfigError("No layers found in feature payload.")

    if isinstance(layer, str) and layer == "last":
        return int(selected_layers[-1])

    resolved = int(layer)
    if resolved not in selected_layers:
        raise LinearProbeConfigError(f"Requested layer {resolved} not in cache layers {selected_layers}")
    return resolved


def _linear_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("linear_probe", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LinearProbeConfigError("linear_probe must be a dictionary")

    layer_raw = raw.get("layer", "last")
    layer: int | str
    if isinstance(layer_raw, str):
        layer_norm = layer_raw.strip().lower()
        if layer_norm in {"", "auto", "last"}:
            layer = "last"
        else:
            layer = int(layer_raw)
    else:
        layer = int(layer_raw)

    return {
        "feature_view": str(raw.get("feature_view", "pooled")).strip().lower(),
        "layer": layer,
        "epochs": int(raw.get("epochs", 30)),
        "lr": float(raw.get("lr", 1e-3)),
        "batch_size": int(raw.get("batch_size", 128)),
        "eval_batch_size": int(raw.get("eval_batch_size", 1024)),
        "weight_decay": float(raw.get("weight_decay", 0.0)),
        "device": str(raw.get("device", "cpu")),
    }


def _resolve_train_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("output_dir", "artifacts/probes")))
    subdir = str(raw.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"mvp_linear_{_backbone_run_label(config)}_{timestamp}"


def _resolve_eval_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("eval_output_dir", "artifacts/results")))
    subdir = str(raw.get("eval_output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"linear_eval_{timestamp}"


def _resolve_checkpoint_path(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    explicit = str(raw.get("checkpoint_path", "")).strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"linear_probe.checkpoint_path not found: {path}")
        return path

    train_root = Path(str(raw.get("output_dir", "artifacts/probes")))
    patterns = (
        "*/linear_probe_best.pt",
        "*/linear_probe_last.pt",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates = sorted(train_root.glob(pattern))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            "No linear checkpoint found automatically. "
            "Set linear_probe.checkpoint_path explicitly or run train.linear.mvp first."
        )
    return candidates[-1]


def _backbone_run_label(config: dict[str, Any]) -> str:
    raw = config.get("backbone", {})
    if not isinstance(raw, dict):
        return "backbone"
    name = str(raw.get("name", "backbone")).strip() or "backbone"
    kwargs = raw.get("kwargs", {})
    if not isinstance(kwargs, dict):
        kwargs = {}
    metadata = resolve_backbone_cache_metadata(name, kwargs)
    variant = str(metadata.get("variant") or kwargs.get("variant", "")).strip()
    return f"{name}_{variant}" if variant else name


def _require_semantic_feature_index(index) -> None:
    """Fail fast when a feature cache predates semantic MVP label support.

    Mixed choice ordering makes positional `answer_idx` unusable as a stable
    binary target. Training or evaluating the linear probe without these
    semantic columns would silently regress to the original bug, so this guard
    points the user back to `extract.mvp` immediately.
    """

    missing = [column for column in REQUIRED_MVP_SEMANTIC_COLUMNS if column not in index.columns]
    if missing:
        raise LinearProbeConfigError(
            "MVP feature cache index is missing semantic label columns "
            f"{missing}. Re-run `python run.py extract.mvp` to regenerate the cache."
        )


def _require_semantic_checkpoint(metadata: dict[str, Any]) -> None:
    """Reject checkpoints trained against the old positional target semantics."""

    target_type = str(metadata.get("target_type", "")).strip()
    if target_type == "semantic_plausibility":
        return
    raise LinearProbeConfigError(
        "Linear checkpoint target_type is incompatible with MVP semantic plausibility eval. "
        "Re-run `python run.py train.linear.mvp` after regenerating features with "
        "`python run.py extract.mvp`."
    )


def _semantic_predictions_to_choice_indices(split_frame, semantic_pred: list[int]) -> list[int]:
    """Translate semantic plausibility predictions back to sample-local `A/B`.

    The probe always predicts the stable semantic convention `1=yes/plausible`
    and `0=no/implausible`. Official MVP scoring, however, still expects a
    positional prediction index per sample. This helper performs the final,
    deterministic conversion with the per-row `yes_choice_idx`/`no_choice_idx`
    metadata stored in the feature cache.
    """

    if len(split_frame) != len(semantic_pred):
        raise LinearProbeConfigError(
            "MVP semantic prediction count mismatch during A/B translation. "
            f"n_rows={len(split_frame)}, n_predictions={len(semantic_pred)}."
        )

    pred_idx: list[int] = []
    for (_, row), pred in zip(split_frame.iterrows(), semantic_pred):
        semantic_label = int(pred)
        if semantic_label == 1:
            pred_idx.append(int(row["yes_choice_idx"]))
            continue
        if semantic_label == 0:
            pred_idx.append(int(row["no_choice_idx"]))
            continue
        raise LinearProbeConfigError(
            "MVP semantic plausibility predictions must be binary 0/1. "
            f"Got {semantic_label} for sample_id={row['sample_id']}."
        )
    return pred_idx
