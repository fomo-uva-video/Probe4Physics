from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.intphys2.eval import run_intphys2_eval
from benchmarks.intphys2.features import load_feature_cache_for_config
from probes import LinearProbe, create_probe


class LinearProbeConfigError(ValueError):
    pass


def run_intphys2_train_linear(config: dict[str, Any]) -> dict[str, Any]:
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)

    probe_cfg = _linear_cfg(config)
    features = _select_feature_tensor(bundle, probe_cfg)
    labels = torch.tensor(index["plausibility"].tolist(), dtype=torch.long)

    train_mask = index["split"].astype(str) == "train"
    val_mask = index["split"].astype(str) == "val"

    # IntPhys2 main split may not have an explicit train/val partition;
    # fall back to using the entire available split for training.
    if not train_mask.any():
        split_name = str(probe_cfg.get("train_split", "main"))
        train_mask = index["split"].astype(str) == split_name
        if not train_mask.any():
            raise LinearProbeConfigError(
                "No training samples found in feature cache index. "
                "Ensure feature_cache.split_names includes a train or main split."
            )

    x_train = features[torch.tensor(train_mask.to_numpy())]
    y_train = labels[torch.tensor(train_mask.to_numpy())]

    x_val = None
    y_val = None
    if val_mask.any():
        x_val = features[torch.tensor(val_mask.to_numpy())]
        y_val = labels[torch.tensor(val_mask.to_numpy())]

    input_dim = int(x_train.shape[1])

    probe = create_probe(
        "linear",
        input_dim=input_dim,
        num_classes=2,  # binary: impossible (0) vs possible (1)
        device=probe_cfg["device"],
    )

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
    )

    output_dir = _resolve_train_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "linear_probe.pt"
    metadata = {
        "feature_signature": str(manifest.get("signature", "")),
        "feature_cache_dir": str(bundle["paths"].cache_dir),
        "feature_view": probe_cfg["feature_view"],
        "layer": probe_cfg["layer"],
        "seed": int(config.get("seed", 42)),
        "benchmark": "intphys2",
    }
    probe.save(checkpoint_path, metadata=metadata)

    train_summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "feature_signature": str(manifest.get("signature", "")),
        "fit": {
            "train_loss": fit.train_loss,
            "val_loss": fit.val_loss,
            "n_epochs": fit.n_epochs,
            "history": fit.history,
        },
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "input_dim": input_dim,
    }

    (output_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return train_summary


def run_intphys2_eval_linear(config: dict[str, Any]) -> dict[str, Any]:
    bundle = load_feature_cache_for_config(config)
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)

    probe_cfg = _linear_cfg(config)
    split_name = str(config.get("split_name", "main"))

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

    # Obtain both hard predictions and soft scores (P(possible)) for VOE
    logits = probe.predict_logits(x_eval, batch_size=probe_cfg["eval_batch_size"])
    probs = torch.softmax(logits, dim=1)
    pred_idx_list = logits.argmax(dim=1).tolist()
    score_list = probs[:, 1].tolist()  # P(plausible=1)

    pred_by_sample = {
        str(sample_id): {"pred_idx": int(pred), "score": float(score)}
        for sample_id, pred, score in zip(
            split_frame["sample_id"].tolist(), pred_idx_list, score_list, strict=True
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

    result = run_intphys2_eval(eval_cfg)
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
        "train_split": str(raw.get("train_split", "main")),
        "epochs": int(raw.get("epochs", 30)),
        "lr": float(raw.get("lr", 1e-3)),
        "batch_size": int(raw.get("batch_size", 128)),
        "eval_batch_size": int(raw.get("eval_batch_size", 1024)),
        "weight_decay": float(raw.get("weight_decay", 0.0)),
        "device": str(raw.get("device", "cpu")),
    }


def _resolve_train_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("output_dir", "artifacts/probes/intphys2")))
    subdir = str(raw.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"linear_train_{timestamp}"


def _resolve_eval_output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("linear_probe", {})
    root = Path(str(raw.get("eval_output_dir", "artifacts/results/intphys2")))
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

    train_root = Path(str(raw.get("output_dir", "artifacts/probes/intphys2")))
    candidates = sorted(train_root.glob("linear_train_*/linear_probe.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No linear checkpoint found automatically. "
            "Set linear_probe.checkpoint_path explicitly or run train.linear.intphys2 first."
        )
    return candidates[-1]
