from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from benchmarks.intphys2.eval import run_intphys2_eval
from benchmarks.intphys2.features import (
    load_feature_cache_for_config as load_intphys2_feature_cache,
)
from benchmarks.mvp.eval import run_mvp_eval
from benchmarks.mvp.features import load_feature_cache_for_config as load_mvp_feature_cache
from benchmarks.ssv2.data import SSV2_NUM_CLASSES
from benchmarks.ssv2.eval import run_ssv2_eval
from benchmarks.ssv2.features import load_feature_cache_for_config as load_ssv2_feature_cache
from models.cache_metadata import resolve_backbone_cache_metadata
from probes import (
    LinearProbe,
    MLPProbe,
    TemporalAttentiveProbe,
    create_probe,
    get_registered_probes,
)
from training.wandb_utils import WandbTrainLogger, init_wandb_train_logger

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_SUPPORTED_PROBE_VIEWS = {
    "linear": {"pooled", "tokens_mean"},
    "mlp": {"pooled", "tokens_mean"},
    "temporal_attn": {"tokens"},
}
_PROBE_CLASSES = {
    "linear": LinearProbe,
    "mlp": MLPProbe,
    "temporal_attn": TemporalAttentiveProbe,
}
_INTPHYS2_EXPECTED_SPLITS = {"train", "val", "test"}
_REQUIRED_MVP_SEMANTIC_COLUMNS = (
    "plausibility_label",
    "yes_choice_idx",
    "no_choice_idx",
)


class ProbeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    load_bundle: Callable[[dict[str, Any], str], dict[str, Any]]
    eval_runner: Callable[[dict[str, Any]], dict[str, Any]]
    default_train_output_dir: str
    default_eval_output_dir: str
    default_eval_split: str
    train_prefix: str
    eval_prefix: str
    objective_metric: str


DATASET_SPECS = {
    "mvp": DatasetSpec(
        name="mvp",
        load_bundle=lambda config, _feature_view: load_mvp_feature_cache(config),
        eval_runner=lambda config: run_mvp_eval(config),
        default_train_output_dir="artifacts/probes",
        default_eval_output_dir="artifacts/results",
        default_eval_split="test",
        train_prefix="mvp_probe",
        eval_prefix="mvp_probe_eval",
        objective_metric="accuracy",
    ),
    "intphys2": DatasetSpec(
        name="intphys2",
        load_bundle=lambda config, feature_view: load_intphys2_feature_cache(
            config,
            feature_view=feature_view,
        ),
        eval_runner=lambda config: run_intphys2_eval(config),
        default_train_output_dir="artifacts/probes/intphys2",
        default_eval_output_dir="artifacts/results/intphys2",
        default_eval_split="val",
        train_prefix="intphys2_probe",
        eval_prefix="intphys2_probe_eval",
        objective_metric="voe_accuracy",
    ),
    "ssv2": DatasetSpec(
        name="ssv2",
        load_bundle=lambda config, _feature_view: load_ssv2_feature_cache(config),
        eval_runner=lambda config: run_ssv2_eval(config),
        default_train_output_dir="artifacts/probes/ssv2",
        default_eval_output_dir="artifacts/results/ssv2",
        default_eval_split="val",
        train_prefix="ssv2_probe",
        eval_prefix="ssv2_probe_eval",
        objective_metric="top1_accuracy",
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe4Physics probe runner")
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--dataset", required=True, choices=tuple(sorted(DATASET_SPECS)))
    parser.add_argument("--probe", required=True, choices=tuple(sorted(_SUPPORTED_PROBE_VIEWS)))
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(argv)

    overrides = list(args.overrides)
    overrides.append(f"probe.name={args.probe}")
    cfg = _compose_config(args.dataset, overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ProbeConfigError(f"Config must resolve to a dict, got {type(config)!r}")

    if args.mode == "train":
        result = run_probe_train(args.dataset, config)
    else:
        result = run_probe_eval(args.dataset, config)

    print(json.dumps(result, indent=2, sort_keys=True))
    if isinstance(result, dict):
        exit_code = int(result.get("exit_code", 0))
        if exit_code != 0:
            raise SystemExit(exit_code)


def run_mvp_train_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_train("mvp", config)


def run_mvp_eval_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_eval("mvp", config)


def run_intphys2_train_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_train("intphys2", config)


def run_intphys2_eval_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_eval("intphys2", config)


def run_ssv2_train_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_train("ssv2", config)


def run_ssv2_eval_probe(config: dict[str, Any]) -> dict[str, Any]:
    return run_probe_eval("ssv2", config)


def run_probe_train(dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    _require_dataset(dataset)
    probe_cfg = _probe_cfg(config)
    if probe_cfg["optuna"]["enabled"]:
        return _run_optuna_study(dataset, config, probe_cfg)

    summary, _logger = _run_single_train(dataset, config, probe_cfg, finish_logger=True)
    return summary


def run_probe_eval(dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    _require_dataset(dataset)
    probe_cfg = _probe_cfg(config)
    return _run_single_eval(dataset, config, probe_cfg)


def _run_single_train(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
    *,
    output_dir: Path | None = None,
    finish_logger: bool = True,
) -> tuple[dict[str, Any], WandbTrainLogger | None]:
    wall_start = time.perf_counter()
    spec = DATASET_SPECS[dataset]
    bundle = spec.load_bundle(config, probe_cfg["feature_view"])
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)
    _validate_dataset_index(dataset, index)

    features = _select_feature_tensor(bundle, probe_cfg)
    labels = _resolve_labels(dataset, index)
    train_mask, val_mask = _resolve_split_masks(dataset, index, probe_cfg)

    if not train_mask.any():
        raise ProbeConfigError(_missing_train_split_message(dataset, probe_cfg["train_split"], index))

    train_idx = torch.tensor(train_mask.to_numpy())
    val_idx = torch.tensor(val_mask.to_numpy())

    x_train = features[train_idx]
    y_train = labels[train_idx]

    x_val = None
    y_val = None
    if val_mask.any():
        x_val = features[val_idx]
        y_val = labels[val_idx]

    input_dim = int(x_train.shape[-1])
    num_classes = _resolve_num_classes(dataset, labels, manifest)
    probe = _create_probe_instance(probe_cfg, input_dim=input_dim, num_classes=num_classes)

    train_output_dir = output_dir or _resolve_train_output_dir(dataset, config, probe_cfg)
    train_output_dir.mkdir(parents=True, exist_ok=True)

    logger = init_wandb_train_logger(
        config,
        benchmark=dataset,
        output_dir=train_output_dir,
        metadata=_training_metadata(dataset, config, probe_cfg, manifest, bundle, num_classes),
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

        checkpoint_last_path = train_output_dir / "probe_last.pt"
        checkpoint_best_path = train_output_dir / "probe_best.pt"
        metadata = _training_metadata(dataset, config, probe_cfg, manifest, bundle, num_classes)
        metadata["checkpoint_kind"] = "last"
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
            "output_dir": str(train_output_dir),
            "feature_signature": str(manifest.get("signature", "")),
            "dataset": dataset,
            "probe_name": probe_cfg["name"],
            "feature_view": probe_cfg["feature_view"],
            "layer": probe_cfg["layer"],
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
            "num_classes": num_classes,
            "elapsed_seconds": max(0.0, time.perf_counter() - wall_start),
            "probe_hparams": _probe_hparam_summary(probe_cfg, num_classes),
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

        (train_output_dir / "train_summary.json").write_text(
            json.dumps(train_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return train_summary, None if finish_logger else logger
    finally:
        if logger is not None:
            try:
                if train_summary is not None:
                    logger.log_summary(train_summary)
            finally:
                if finish_logger or train_summary is None:
                    logger.finish()


def _run_single_eval(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
    *,
    output_dir: Path | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    bundle = spec.load_bundle(config, probe_cfg["feature_view"])
    manifest = bundle["manifest"]
    index = bundle["index"].sort_values("feature_index").reset_index(drop=True)
    _validate_dataset_index(dataset, index)

    eval_split = str(split_name or config.get("split_name", spec.default_eval_split))
    checkpoint_path = _resolve_checkpoint_path(dataset, config, probe_cfg)
    probe, ckpt_payload = load_probe_from_checkpoint(
        checkpoint_path,
        device=probe_cfg["device"],
        expected_probe_name=probe_cfg["name"],
    )

    ckpt_meta = ckpt_payload.get("metadata", {}) if isinstance(ckpt_payload, dict) else {}
    checkpoint_signature = str(ckpt_meta.get("feature_signature", ""))
    current_signature = str(manifest.get("signature", ""))
    if checkpoint_signature and checkpoint_signature != current_signature:
        raise ProbeConfigError(
            "Probe checkpoint feature signature mismatch. "
            f"checkpoint={checkpoint_signature}, current={current_signature}."
        )
    if dataset == "mvp":
        _require_semantic_checkpoint(ckpt_meta)

    features = _select_feature_tensor(bundle, probe_cfg)
    split_mask = index["split"].astype(str) == eval_split
    if not split_mask.any():
        raise ProbeConfigError(
            f"No samples found for split_name='{eval_split}' in cache index."
        )

    split_idx = torch.tensor(split_mask.to_numpy())
    x_eval = features[split_idx]
    split_frame = index.loc[split_mask].copy()

    eval_output_dir = output_dir or _resolve_eval_output_dir(dataset, config, probe_cfg)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = eval_output_dir / "probe_predictions.json"

    if dataset == "mvp":
        semantic_pred = probe.predict(x_eval, batch_size=probe_cfg["eval_batch_size"]).tolist()
        pred_idx = _semantic_predictions_to_choice_indices(split_frame, semantic_pred)
        sample_ids = split_frame["sample_id"].tolist()
        if len(sample_ids) != len(pred_idx):
            raise ProbeConfigError(
                "MVP probe eval produced a prediction count mismatch. "
                f"n_samples={len(sample_ids)}, n_predictions={len(pred_idx)}."
            )
        pred_payload = {
            str(sample_id): int(pred)
            for sample_id, pred in zip(sample_ids, pred_idx)
        }
    elif dataset == "intphys2":
        logits = probe.predict_logits(x_eval, batch_size=probe_cfg["eval_batch_size"])
        probs = torch.softmax(logits, dim=1)
        pred_payload = {
            str(sample_id): {
                "pred_idx": int(pred),
                "score": float(score),
            }
            for sample_id, pred, score in zip(
                split_frame["sample_id"].tolist(),
                logits.argmax(dim=1).tolist(),
                probs[:, 1].tolist(),
            )
        }
    elif dataset == "ssv2":
        logits = probe.predict_logits(x_eval, batch_size=probe_cfg["eval_batch_size"])
        probs = torch.softmax(logits, dim=1)
        pred_payload = {
            str(sample_id): {
                "pred_idx": int(pred),
                "scores": scores,
            }
            for sample_id, pred, scores in zip(
                split_frame["sample_id"].tolist(),
                logits.argmax(dim=1).tolist(),
                probs.tolist(),
            )
        }
    else:
        raise ProbeConfigError(f"Unsupported dataset for eval: {dataset}")

    pred_file.write_text(
        json.dumps(pred_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    eval_cfg = copy.deepcopy(config)
    eval_cfg["split_name"] = eval_split
    eval_cfg["predictor"] = {
        "mode": "from_file",
        "prediction_file": str(pred_file),
    }
    eval_cfg["output_subdir"] = eval_output_dir.name
    eval_cfg["output_dir"] = str(eval_output_dir.parent)

    result = spec.eval_runner(eval_cfg)
    metric_value = _extract_objective_metric(dataset, result.get("metrics", {}), probe_cfg)
    summary = {
        "probe_eval_dir": str(eval_output_dir),
        "checkpoint": str(checkpoint_path),
        "prediction_file": str(pred_file),
        "feature_signature": current_signature,
        "dataset": dataset,
        "probe_name": probe_cfg["name"],
        "split_name": eval_split,
        "objective_metric": metric_value,
        "base_eval": result,
    }
    (eval_output_dir / "probe_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _run_optuna_study(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> dict[str, Any]:
    optuna = _import_optuna()
    optuna_cfg = probe_cfg["optuna"]
    study_root = _resolve_train_output_dir(dataset, config, probe_cfg)
    study_root.mkdir(parents=True, exist_ok=True)

    study_name = str(optuna_cfg.get("study_name", "")).strip() or study_root.name
    storage = str(optuna_cfg.get("storage", "")).strip()
    if not storage:
        storage = f"sqlite:///{(study_root / 'optuna_study.sqlite3').resolve()}"

    sampler = optuna.samplers.TPESampler(seed=int(optuna_cfg["sampler_seed"]))
    study = optuna.create_study(
        study_name=study_name,
        direction=str(optuna_cfg["direction"]),
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    trial_summaries: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        trial_cfg = copy.deepcopy(config)
        _apply_trial_parameters(trial_cfg, dataset, probe_cfg, trial)
        resolved_trial_probe_cfg = _probe_cfg(trial_cfg)
        trial_dir = study_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        wandb_cfg = trial_cfg.get("probe", {}).get("wandb", {})
        if isinstance(wandb_cfg, dict) and bool(wandb_cfg.get("enabled", False)):
            if not str(wandb_cfg.get("group", "")).strip():
                wandb_cfg["group"] = study_name
            if not str(wandb_cfg.get("name", "")).strip():
                wandb_cfg["name"] = f"{study_name}/trial_{trial.number:04d}"

        logger: WandbTrainLogger | None = None
        try:
            train_summary, logger = _run_single_train(
                dataset,
                trial_cfg,
                resolved_trial_probe_cfg,
                output_dir=trial_dir,
                finish_logger=False,
            )

            eval_cfg = copy.deepcopy(trial_cfg)
            eval_cfg["split_name"] = DATASET_SPECS[dataset].default_eval_split
            eval_probe_cfg = _probe_cfg(eval_cfg)
            eval_probe_cfg["checkpoint_path"] = str(train_summary["checkpoint"])
            eval_summary = _run_single_eval(
                dataset,
                _with_checkpoint_path(eval_cfg, train_summary["checkpoint"]),
                eval_probe_cfg,
                output_dir=trial_dir / "val_eval",
                split_name=DATASET_SPECS[dataset].default_eval_split,
            )

            metric_value = float(eval_summary["objective_metric"])
            trial_summary = {
                "trial_number": int(trial.number),
                "value": metric_value,
                "params": dict(trial.params),
                "output_dir": str(trial_dir),
                "checkpoint": train_summary["checkpoint"],
                "eval_output_dir": eval_summary["probe_eval_dir"],
            }
            trial_summaries.append(trial_summary)
            trial.set_user_attr("output_dir", str(trial_dir))
            trial.set_user_attr("checkpoint", str(train_summary["checkpoint"]))
            trial.set_user_attr("eval_output_dir", str(eval_summary["probe_eval_dir"]))

            if logger is not None:
                logger.run.summary["objective_metric"] = metric_value
                logger.run.summary["optuna_study_name"] = study_name
                logger.run.summary["optuna_trial_number"] = int(trial.number)
            return metric_value
        finally:
            if logger is not None:
                logger.finish()

    timeout_seconds = int(optuna_cfg["timeout_seconds"])
    study.optimize(
        objective,
        n_trials=int(optuna_cfg["n_trials"]),
        n_jobs=int(optuna_cfg["n_jobs"]),
        timeout=None if timeout_seconds <= 0 else timeout_seconds,
    )

    best_trial = study.best_trial
    best_summary = {
        "dataset": dataset,
        "probe_name": probe_cfg["name"],
        "study_name": study.study_name,
        "storage": storage,
        "direction": str(optuna_cfg["direction"]),
        "objective_metric": _resolve_objective_metric_name(dataset, probe_cfg),
        "n_trials": len(study.trials),
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "best_output_dir": str(best_trial.user_attrs.get("output_dir", "")),
        "best_checkpoint": str(best_trial.user_attrs.get("checkpoint", "")),
        "trials": trial_summaries,
    }
    (study_root / "optuna_summary.json").write_text(
        json.dumps(best_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (study_root / "best_params.json").write_text(
        json.dumps(dict(best_trial.params), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return best_summary


def load_probe_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
    expected_probe_name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ProbeConfigError(f"Invalid checkpoint payload at {path}")

    probe_name = str(
        payload.get("type")
        or payload.get("probe_name")
        or payload.get("metadata", {}).get("probe_name", "")
    ).strip()
    if not probe_name:
        raise ProbeConfigError(
            f"Checkpoint does not declare a probe type: {path}"
        )
    if expected_probe_name and expected_probe_name != probe_name:
        raise ProbeConfigError(
            "Configured probe name does not match checkpoint type. "
            f"config={expected_probe_name}, checkpoint={probe_name}."
        )

    probe_class = _PROBE_CLASSES.get(probe_name)
    if probe_class is None:
        known = ", ".join(sorted(_PROBE_CLASSES))
        raise ProbeConfigError(
            f"Unsupported checkpoint probe type '{probe_name}'. Known: {known}"
        )
    return probe_class.load(path, device=device), payload


def _compose_config(config_name: str, overrides: list[str]) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ProbeConfigError(f"Unknown dataset '{dataset}'. Known: {known}")


def _validate_dataset_index(dataset: str, index: Any) -> None:
    if dataset == "mvp":
        _require_semantic_feature_index(index)
        return
    if dataset == "intphys2":
        _require_fresh_intphys2_split_index(index)
        return
    if dataset == "ssv2":
        return
    raise ProbeConfigError(f"Unsupported dataset '{dataset}'.")


def _resolve_labels(dataset: str, index: Any) -> torch.Tensor:
    if dataset == "mvp":
        return torch.tensor(index["plausibility_label"].tolist(), dtype=torch.long)
    if dataset == "intphys2":
        return torch.tensor(index["plausibility"].tolist(), dtype=torch.long)
    if dataset == "ssv2":
        return torch.tensor(index["label_idx"].tolist(), dtype=torch.long)
    raise ProbeConfigError(f"Unsupported dataset '{dataset}'.")


def _resolve_num_classes(
    dataset: str,
    labels: torch.Tensor,
    manifest: dict[str, Any],
) -> int:
    if dataset == "ssv2":
        return int(manifest.get("stats", {}).get("n_classes", SSV2_NUM_CLASSES))
    if dataset == "mvp":
        return max(int(labels.max().item()) + 1, 2)
    if dataset == "intphys2":
        return 2
    raise ProbeConfigError(f"Unsupported dataset '{dataset}'.")


def _resolve_split_masks(
    dataset: str,
    index: Any,
    probe_cfg: dict[str, Any],
) -> tuple[Any, Any]:
    train_split = probe_cfg["train_split"]
    train_mask = index["split"].astype(str) == train_split
    val_mask = index["split"].astype(str) == "val"

    if dataset == "ssv2" and not train_mask.any():
        available = sorted(index["split"].astype(str).unique().tolist())
        raise ProbeConfigError(
            f"No training samples found for split='{train_split}' in feature cache index. "
            f"Available splits: {available}. "
            "Ensure feature_cache.split_names includes a 'train' split."
        )
    return train_mask, val_mask


def _missing_train_split_message(dataset: str, train_split: str, index: Any) -> str:
    if dataset == "intphys2":
        return (
            "No training samples found in feature cache index. "
            "Ensure feature_cache.split_names includes 'train' and re-run extract.intphys2."
        )
    available = sorted(index["split"].astype(str).unique().tolist())
    return (
        f"No training samples found for split='{train_split}' in feature cache index. "
        f"Available splits: {available}."
    )


def _create_probe_instance(
    probe_cfg: dict[str, Any],
    *,
    input_dim: int,
    num_classes: int,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "device": probe_cfg["device"],
    }
    if probe_cfg["name"] == "mlp":
        kwargs.update(probe_cfg["mlp"])
    elif probe_cfg["name"] == "temporal_attn":
        kwargs.update(probe_cfg["temporal_attn"])
    return create_probe(probe_cfg["name"], **kwargs)


def _select_feature_tensor(bundle: dict[str, Any], probe_cfg: dict[str, Any]) -> torch.Tensor:
    probe_name = probe_cfg["name"]
    feature_view = probe_cfg["feature_view"]
    allowed_views = _SUPPORTED_PROBE_VIEWS.get(probe_name)
    if allowed_views is None:
        known = ", ".join(sorted(_SUPPORTED_PROBE_VIEWS))
        raise ProbeConfigError(
            f"Unsupported probe '{probe_name}'. Supported probes: {known}"
        )
    if feature_view not in allowed_views:
        allowed = ", ".join(sorted(allowed_views))
        raise ProbeConfigError(
            f"Unsupported probe.feature_view='{feature_view}' for probe '{probe_name}'. "
            f"Allowed values: {allowed}."
        )

    layer = probe_cfg["layer"]
    pooled = bundle.get("pooled")
    tokens = bundle.get("tokens")

    if feature_view == "pooled":
        if pooled is None:
            raise ProbeConfigError("Pooled features are not available in cache.")
        selected = _resolve_layer(layer, pooled)
        tensor = pooled["by_layer"][selected]
        if tensor.ndim != 2:
            raise ProbeConfigError(f"Expected pooled tensor [N, D], got {tuple(tensor.shape)}")
        return tensor.to(dtype=torch.float32)

    if feature_view == "tokens_mean":
        if tokens is None:
            raise ProbeConfigError("Token features are not available in cache.")
        selected = _resolve_layer(layer, tokens)
        tensor = tokens["by_layer"][selected]
        if tensor.ndim != 3:
            raise ProbeConfigError(f"Expected token tensor [N, T, D], got {tuple(tensor.shape)}")
        return tensor.mean(dim=1).to(dtype=torch.float32)

    if feature_view == "tokens":
        if tokens is None:
            raise ProbeConfigError("Token features are not available in cache.")
        selected = _resolve_layer(layer, tokens)
        tensor = tokens["by_layer"][selected]
        if tensor.ndim != 3:
            raise ProbeConfigError(f"Expected token tensor [N, T, D], got {tuple(tensor.shape)}")
        return tensor.to(dtype=torch.float32)

    raise ProbeConfigError(
        f"Unsupported probe.feature_view='{feature_view}'. Use pooled|tokens_mean|tokens."
    )


def _resolve_layer(layer: int | str, payload: dict[str, Any]) -> int:
    selected_layers = [int(item) for item in payload.get("selected_layers", [])]
    if not selected_layers:
        selected_layers = sorted(int(k) for k in payload.get("by_layer", {}).keys())
    if not selected_layers:
        raise ProbeConfigError("No layers found in feature payload.")

    if isinstance(layer, str) and layer == "last":
        return int(selected_layers[-1])

    resolved = int(layer)
    if resolved not in selected_layers:
        raise ProbeConfigError(f"Requested layer {resolved} not in cache layers {selected_layers}")
    return resolved


def _probe_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("probe", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProbeConfigError("probe must be a dictionary")

    probe_name = str(raw.get("name", "linear")).strip().lower()
    known_probes = set(get_registered_probes())
    if probe_name not in known_probes:
        known = ", ".join(sorted(known_probes))
        raise ProbeConfigError(f"Unknown probe '{probe_name}'. Registered probes: {known}")

    layer_raw = raw.get("layer", "last")
    if isinstance(layer_raw, str):
        layer_norm = layer_raw.strip().lower()
        layer: int | str = "last" if layer_norm in {"", "auto", "last"} else int(layer_raw)
    else:
        layer = int(layer_raw)

    feature_view_raw = str(raw.get("feature_view", "auto")).strip().lower() or "auto"
    if feature_view_raw == "auto":
        feature_view = "tokens" if probe_name == "temporal_attn" else "pooled"
    else:
        feature_view = feature_view_raw

    mlp_raw = raw.get("mlp", {})
    if mlp_raw is None:
        mlp_raw = {}
    if not isinstance(mlp_raw, dict):
        raise ProbeConfigError("probe.mlp must be a dictionary")

    temporal_raw = raw.get("temporal_attn", {})
    if temporal_raw is None:
        temporal_raw = {}
    if not isinstance(temporal_raw, dict):
        raise ProbeConfigError("probe.temporal_attn must be a dictionary")

    return {
        "name": probe_name,
        "feature_view": feature_view,
        "layer": layer,
        "train_split": str(raw.get("train_split", "train")).strip() or "train",
        "epochs": int(raw.get("epochs", 30)),
        "lr": float(raw.get("lr", 1e-3)),
        "batch_size": int(raw.get("batch_size", 128)),
        "eval_batch_size": int(raw.get("eval_batch_size", 1024)),
        "weight_decay": float(raw.get("weight_decay", 0.0)),
        "device": str(raw.get("device", "cpu")),
        "output_dir": str(raw.get("output_dir", "")),
        "output_subdir": str(raw.get("output_subdir", "")),
        "checkpoint_path": str(raw.get("checkpoint_path", "")),
        "eval_output_dir": str(raw.get("eval_output_dir", "")),
        "eval_output_subdir": str(raw.get("eval_output_subdir", "")),
        "mlp": {
            "hidden_dims": [int(item) for item in mlp_raw.get("hidden_dims", [512])],
            "dropout": float(mlp_raw.get("dropout", 0.0)),
        },
        "temporal_attn": {
            "embed_dim": _optional_int(temporal_raw.get("embed_dim")),
            "num_heads": int(temporal_raw.get("num_heads", 16)),
            "num_self_attn_blocks": int(temporal_raw.get("num_self_attn_blocks", 3)),
            "mlp_ratio": float(temporal_raw.get("mlp_ratio", 4.0)),
            "dropout": float(temporal_raw.get("dropout", 0.0)),
        },
        "optuna": _optuna_cfg(raw.get("optuna")),
    }


def _optuna_cfg(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProbeConfigError("probe.optuna must be a dictionary")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "n_trials": int(raw.get("n_trials", 20)),
        "n_jobs": int(raw.get("n_jobs", 1)),
        "timeout_seconds": int(raw.get("timeout_seconds", 0)),
        "storage": str(raw.get("storage", "")),
        "study_name": str(raw.get("study_name", "")),
        "direction": str(raw.get("direction", "maximize")),
        "metric": str(raw.get("metric", "")).strip(),
        "sampler_seed": int(raw.get("sampler_seed", 42)),
    }


def _resolve_train_output_dir(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> Path:
    spec = DATASET_SPECS[dataset]
    root_value = probe_cfg["output_dir"] or spec.default_train_output_dir
    root = Path(str(root_value))
    subdir = str(probe_cfg["output_subdir"]).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{spec.train_prefix}_{probe_cfg['name']}_{_backbone_run_label(config)}_{timestamp}"


def _resolve_eval_output_dir(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> Path:
    spec = DATASET_SPECS[dataset]
    root_value = probe_cfg["eval_output_dir"] or spec.default_eval_output_dir
    root = Path(str(root_value))
    subdir = str(probe_cfg["eval_output_subdir"]).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{spec.eval_prefix}_{probe_cfg['name']}_{timestamp}"


def _resolve_checkpoint_path(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> Path:
    explicit = str(probe_cfg["checkpoint_path"]).strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"probe.checkpoint_path not found: {path}")
        return path

    train_root_value = probe_cfg["output_dir"] or DATASET_SPECS[dataset].default_train_output_dir
    train_root = Path(str(train_root_value))
    patterns = (
        "*/probe_best.pt",
        "*/probe_last.pt",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates = sorted(train_root.glob(pattern))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            "No probe checkpoint found automatically. "
            f"Set probe.checkpoint_path explicitly or run train.probe.{dataset} first."
        )
    return candidates[-1]


def _training_metadata(
    dataset: str,
    config: dict[str, Any],
    probe_cfg: dict[str, Any],
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    num_classes: int,
) -> dict[str, Any]:
    metadata = {
        "dataset": dataset,
        "probe_name": probe_cfg["name"],
        "feature_signature": str(manifest.get("signature", "")),
        "feature_cache_dir": str(bundle["paths"].cache_dir),
        "feature_view": probe_cfg["feature_view"],
        "layer": probe_cfg["layer"],
        "seed": int(config.get("seed", 42)),
        "num_classes": int(num_classes),
        "probe_hparams": _probe_hparam_summary(probe_cfg, num_classes),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    if dataset == "mvp":
        metadata["target_type"] = "semantic_plausibility"
        metadata["positive_label"] = 1
        metadata["negative_label"] = 0
    return metadata


def _probe_hparam_summary(probe_cfg: dict[str, Any], num_classes: int) -> dict[str, Any]:
    summary = {
        "name": probe_cfg["name"],
        "feature_view": probe_cfg["feature_view"],
        "layer": probe_cfg["layer"],
        "epochs": probe_cfg["epochs"],
        "lr": probe_cfg["lr"],
        "batch_size": probe_cfg["batch_size"],
        "eval_batch_size": probe_cfg["eval_batch_size"],
        "weight_decay": probe_cfg["weight_decay"],
        "device": probe_cfg["device"],
        "num_classes": int(num_classes),
    }
    if probe_cfg["name"] == "mlp":
        summary.update(probe_cfg["mlp"])
    elif probe_cfg["name"] == "temporal_attn":
        summary.update(probe_cfg["temporal_attn"])
    return summary


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


def _resolve_objective_metric_name(dataset: str, probe_cfg: dict[str, Any]) -> str:
    override = str(probe_cfg["optuna"]["metric"]).strip()
    if override:
        return override
    return DATASET_SPECS[dataset].objective_metric


def _extract_objective_metric(
    dataset: str,
    metrics: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> float:
    metric_name = _resolve_objective_metric_name(dataset, probe_cfg)
    if metric_name not in metrics:
        known = ", ".join(sorted(metrics))
        raise ProbeConfigError(
            f"Objective metric '{metric_name}' not found in eval metrics. Available: {known}"
        )
    return float(metrics[metric_name])


def _apply_trial_parameters(
    trial_cfg: dict[str, Any],
    dataset: str,
    probe_cfg: dict[str, Any],
    trial: Any,
) -> None:
    probe_section = trial_cfg.setdefault("probe", {})
    if not isinstance(probe_section, dict):
        raise ProbeConfigError("probe must be a dictionary")

    probe_section["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    probe_section["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    probe_section["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    probe_section["epochs"] = trial.suggest_categorical("epochs", [20, 50, 100])

    if probe_cfg["name"] == "mlp":
        mlp_section = probe_section.setdefault("mlp", {})
        if not isinstance(mlp_section, dict):
            raise ProbeConfigError("probe.mlp must be a dictionary")
        mlp_section["hidden_dims"] = trial.suggest_categorical(
            "hidden_dims",
            [[256], [512], [1024], [512, 256]],
        )
        mlp_section["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    elif probe_cfg["name"] == "temporal_attn":
        temporal_section = probe_section.setdefault("temporal_attn", {})
        if not isinstance(temporal_section, dict):
            raise ProbeConfigError("probe.temporal_attn must be a dictionary")
        temporal_section["num_heads"] = trial.suggest_categorical("num_heads", [4, 8, 16])
        temporal_section["num_self_attn_blocks"] = trial.suggest_categorical(
            "num_self_attn_blocks",
            [1, 2, 3],
        )
        temporal_section["mlp_ratio"] = trial.suggest_categorical("mlp_ratio", [2.0, 4.0])
        temporal_section["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    probe_section.setdefault("optuna", {})
    if dataset == "mvp" and str(probe_section.get("feature_view", "")).strip().lower() == "auto":
        probe_section["feature_view"] = "tokens" if probe_cfg["name"] == "temporal_attn" else "pooled"


def _with_checkpoint_path(config: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    probe_section = updated.setdefault("probe", {})
    if not isinstance(probe_section, dict):
        raise ProbeConfigError("probe must be a dictionary")
    probe_section["checkpoint_path"] = str(checkpoint_path)
    return updated


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ProbeConfigError(
            "probe.optuna.enabled=true but the 'optuna' package is not installed. "
            "Install dependencies from environment.yml or run `pip install optuna`."
        ) from exc
    return optuna


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _require_fresh_intphys2_split_index(index: Any) -> None:
    existing = set(index["split"].astype(str).unique().tolist())
    if not existing.intersection(_INTPHYS2_EXPECTED_SPLITS):
        raise ProbeConfigError(
            f"IntPhys2 feature cache has stale split labels: {sorted(existing)}. "
            "This cache was built before the train/val/test split was introduced. "
            "Delete the old cache and re-run: python run.py extract.intphys2"
        )


def _require_semantic_feature_index(index: Any) -> None:
    missing = [column for column in _REQUIRED_MVP_SEMANTIC_COLUMNS if column not in index.columns]
    if missing:
        raise ProbeConfigError(
            "MVP feature cache index is missing semantic label columns "
            f"{missing}. Re-run `python run.py extract.mvp` to regenerate the cache."
        )


def _require_semantic_checkpoint(metadata: dict[str, Any]) -> None:
    target_type = str(metadata.get("target_type", "")).strip()
    if target_type == "semantic_plausibility":
        return
    raise ProbeConfigError(
        "Probe checkpoint target_type is incompatible with MVP semantic plausibility eval. "
        "Re-run `python run.py train.probe.mvp` after regenerating features with "
        "`python run.py extract.mvp`."
    )


def _semantic_predictions_to_choice_indices(split_frame: Any, semantic_pred: list[int]) -> list[int]:
    if len(split_frame) != len(semantic_pred):
        raise ProbeConfigError(
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
        raise ProbeConfigError(
            "MVP semantic plausibility predictions must be binary 0/1. "
            f"Got {semantic_label} for sample_id={row['sample_id']}."
        )
    return pred_idx


if __name__ == "__main__":
    main()
