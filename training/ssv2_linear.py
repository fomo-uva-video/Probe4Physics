from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.ssv2.data import SSV2_NUM_CLASSES
from benchmarks.ssv2.eval import run_ssv2_eval
from benchmarks.ssv2.features import load_feature_cache_for_config
from models.cache_metadata import resolve_backbone_cache_metadata
from probes import LinearProbe, create_probe
from training.wandb_utils import init_wandb_train_logger


class LinearProbeConfigError(ValueError):
    pass


def run_ssv2_train_linear(config: dict[str, Any]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)

    probe_cfg = _linear_cfg(config)
    features = _select_feature_tensor(bundle, probe_cfg)
    labels = torch.tensor(index["label_idx"].tolist(), dtype=torch.long)

    train_split = str(probe_cfg.get("train_split", "train"))
    train_mask = index["split"].astype(str) == train_split
    val_mask = index["split"].astype(str) == "val"

    # If no explicit train split found, fall back to all available samples.
    if not train_mask.any():
        available = sorted(index["split"].astype(str).unique().tolist())
        raise LinearProbeConfigError(
            f"No training samples found for split='{train_split}' in feature cache index. "
            f"Available splits: {available}. "
            "Ensure feature_cache.split_names includes a 'train' split."
        )

    x_train = features[torch.tensor(train_mask.to_numpy())]
    y_train = labels[torch.tensor(train_mask.to_numpy())]

    x_val = None
    y_val = None
    if val_mask.any():
        x_val = features[torch.tensor(val_mask.to_numpy())]
        y_val = labels[torch.tensor(val_mask.to_numpy())]

    input_dim = int(x_train.shape[1])
    n_classes = int(manifest.get("stats", {}).get("n_classes", SSV2_NUM_CLASSES))

    probe = create_probe(
        "linear",
        input_dim=input_dim,
        num_classes=n_classes,
        device=probe_cfg["device"],
    )

    output_dir = _resolve_train_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = init_wandb_train_logger(
        config,
        benchmark="ssv2",
        output_dir=output_dir,
        metadata={
            "feature_signature": str(manifest.get("signature", "")),
            "feature_view": probe_cfg["feature_view"],
            "layer": probe_cfg["layer"],
            "n_classes": n_classes,
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
            "seed": int(config.get("seed", 42)),
            "benchmark": "ssv2",
            "n_classes": n_classes,
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
            "n_classes": n_classes,
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


def run_ssv2_eval_linear(config: dict[str, Any]) -> dict[str, Any]:
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)

    probe_cfg = _linear_cfg(config)
    split_name = str(config.get("split_name", "val"))

    checkpoint_path = _resolve_checkpoint_path(config)
    probe = LinearProbe.load(checkpoint_path, device=probe_cfg["device"])

    ckpt_payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ckpt_meta = ckpt_payload.get("metadata", {}) if isinstance(ckpt_payload, dict) else {}
    checkpoint_signature = str(ckpt_meta.get("feature_signature", ""))
    current_signature = str(manifest.get("signature", ""))
    if checkpoint_signature and checkpoint_signature != current_signature:
        raise LinearProbeConfigError(
            "Linear checkpoint feature signature mismatch. "
            f"checkpoint={checkpoint_signature}, current={current_signature}."
        )

    features = _select_feature_tensor(bundle, probe_cfg)
    split_mask = index["split"].astype(str) == split_name
    if not split_mask.any():
        raise LinearProbeConfigError(
            f"No samples found for split_name='{split_name}' in cache index."
        )

    split_idx = torch.tensor(split_mask.to_numpy())
    x_eval = features[split_idx]
    split_frame = index.loc[split_mask].copy()

    # Get logits → softmax scores for all 174 classes (enables Top-5 evaluation).
    logits = probe.predict_logits(x_eval, batch_size=probe_cfg["eval_batch_size"])
    probs = torch.softmax(logits, dim=1)
    pred_idx_list = logits.argmax(dim=1).tolist()
    scores_list = probs.tolist()  # list of per-class probability vectors

    pred_by_sample = {
        str(sample_id): {
            "pred_idx": int(pred),
            "scores": scores,
        }
        for sample_id, pred, scores in zip(
            split_frame["sample_id"].tolist(), pred_idx_list, scores_list, strict=True
        )
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

    result = run_ssv2_eval(eval_cfg)
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
            raise LinearProbeConfigError(
                f"Expected pooled tensor [N, D], got {tuple(tensor.shape)}"
            )
        return tensor.to(dtype=torch.float32)

    if feature_view == "tokens_mean":
        if tokens is None:
            raise LinearProbeConfigError("Token features are not available in cache.")
        selected = _resolve_layer(layer, tokens)
        tensor = tokens["by_layer"][selected]
        if tensor.ndim != 3:
            raise LinearProbeConfigError(
                f"Expected token tensor [N, T, D], got {tuple(tensor.shape)}"
            )
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
        raise LinearProbeConfigError(
            f"Requested layer {resolved} not in cache layers {selected_layers}"
        )
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
        "train_split": str(raw.get("train_split", "train")),
        "epochs": int(raw.get("epochs", 30)),
        "lr": float(raw.get("lr", 1e-3)),
        "batch_size": int(raw.get("batch_size", 128)),
        "eval_batch_size": int(raw.get("eval_batch_size", 1024)),
        "weight_decay": float(raw.get("weight_decay", 0.0)),
        "device": str(raw.get("device", "cpu")),
    }


def _resolve_train_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("output_dir", "artifacts/probes/ssv2")))
    subdir = str(raw.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"ssv2_linear_{_backbone_run_label(config)}_{timestamp}"


def _resolve_eval_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("eval_output_dir", "artifacts/results/ssv2")))
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

    train_root = Path(str(raw.get("output_dir", "artifacts/probes/ssv2")))
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
            "Set linear_probe.checkpoint_path explicitly or run train.linear.ssv2 first."
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
