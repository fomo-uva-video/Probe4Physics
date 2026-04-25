from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WandbConfigError(RuntimeError):
    pass


def wandb_enabled(config: dict[str, Any]) -> bool:
    raw = _wandb_cfg(config)
    return bool(raw.get("enabled", False))


@dataclass
class WandbTrainLogger:
    run: Any

    def log_epoch(self, metrics: dict[str, float]) -> None:
        payload = dict(metrics)
        step = payload.get("epoch")
        if isinstance(step, float) and step.is_integer():
            step = int(step)
        if isinstance(step, int):
            self.run.log(payload, step=step)
            return
        self.run.log(payload)

    def log_summary(self, summary: dict[str, Any]) -> None:
        fit = summary.get("fit", {})
        if isinstance(fit, dict):
            payload = {
                "final_train_loss": fit.get("train_loss"),
                "final_train_accuracy": fit.get("train_accuracy"),
                "final_val_loss": fit.get("val_loss"),
                "final_val_accuracy": fit.get("val_accuracy"),
                "n_epochs": fit.get("n_epochs"),
            }
            self.run.summary.update({k: v for k, v in payload.items() if v is not None})

        self.run.summary.update(
            {
                "checkpoint_path": summary.get("checkpoint"),
                "output_dir": summary.get("output_dir"),
                "feature_signature": summary.get("feature_signature"),
                "n_train": summary.get("n_train"),
                "n_val": summary.get("n_val"),
                "input_dim": summary.get("input_dim"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "seconds_per_train_sample": summary.get("seconds_per_train_sample"),
                "seconds_per_labeled_sample": summary.get("seconds_per_labeled_sample"),
            }
        )
        if summary.get("n_classes") is not None:
            self.run.summary["n_classes"] = summary["n_classes"]

    def finish(self) -> None:
        self.run.finish()


def init_wandb_train_logger(
    config: dict[str, Any],
    *,
    benchmark: str,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> WandbTrainLogger | None:
    raw = _wandb_cfg(config)
    if not bool(raw.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise WandbConfigError(
            "linear_probe.wandb.enabled=true but the 'wandb' package is not installed. "
            "Install dependencies from environment.yml or run `pip install wandb`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = {
        "project": str(raw.get("project", "probe4physics")),
        "entity": _optional_str(raw.get("entity", "")),
        "name": _default_run_name(config, benchmark, output_dir, raw),
        "group": _optional_str(raw.get("group", "")),
        "job_type": str(raw.get("job_type", "probe_train")),
        "tags": _string_list(raw.get("tags", [])),
        "notes": _optional_str(raw.get("notes", "")),
        "mode": str(raw.get("mode", "online")),
        "dir": str(Path(str(raw.get("dir", output_dir))).resolve()),
        "config": _build_run_config(config, benchmark, output_dir, metadata),
    }

    if raw.get("resume") is not None:
        init_kwargs["resume"] = raw.get("resume")
    if _optional_str(raw.get("id", "")):
        init_kwargs["id"] = str(raw.get("id")).strip()

    cleaned_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, [], "")}
    run = wandb.init(**cleaned_kwargs)
    return WandbTrainLogger(run=run)


def _build_run_config(
    config: dict[str, Any],
    benchmark: str,
    output_dir: Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "seed": config.get("seed", 42),
        "output_dir": str(output_dir),
        "linear_probe": copy.deepcopy(config.get("linear_probe", {})),
    }
    if "backbone" in config:
        payload["backbone"] = copy.deepcopy(config.get("backbone", {}))
    if "feature_cache" in config:
        payload["feature_cache"] = copy.deepcopy(config.get("feature_cache", {}))
    if metadata:
        payload["probe_metadata"] = copy.deepcopy(metadata)

    linear_probe = payload.get("linear_probe")
    if isinstance(linear_probe, dict):
        linear_probe.pop("checkpoint_path", None)
        wandb_cfg = linear_probe.get("wandb")
        if isinstance(wandb_cfg, dict):
            wandb_cfg.pop("id", None)
    return payload


def _default_run_name(
    config: dict[str, Any],
    benchmark: str,
    output_dir: Path,
    raw: dict[str, Any],
) -> str:
    explicit = _optional_str(raw.get("name", ""))
    if explicit:
        return explicit

    experiment = config.get("experiment", {})
    experiment_name = ""
    if isinstance(experiment, dict):
        experiment_name = _optional_str(experiment.get("name", "")) or ""
    feature_view = ""
    linear_probe = config.get("linear_probe", {})
    if isinstance(linear_probe, dict):
        feature_view = _optional_str(linear_probe.get("feature_view", "")) or ""
    pieces = [piece for piece in (experiment_name, benchmark, feature_view, output_dir.name) if piece]
    return "/".join(pieces)


def _wandb_cfg(config: dict[str, Any]) -> dict[str, Any]:
    linear_probe = config.get("linear_probe", {})
    if linear_probe is None:
        return {}
    if not isinstance(linear_probe, dict):
        raise WandbConfigError("linear_probe must be a dictionary")
    raw = linear_probe.get("wandb", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise WandbConfigError("linear_probe.wandb must be a dictionary")
    return raw


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []
