from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from benchmarks.mvp.eval import run_mvp_eval
from benchmarks.mvp.features import has_valid_feature_cache
from benchmarks.mvp.init import run_mvp_init
from experiments.registry import get_experiment, list_experiments
from training.mvp_extract import run_mvp_extract
from training.mvp_linear import run_mvp_eval_linear, run_mvp_train_linear

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_COMMAND = "eval.mvp"


@dataclass(frozen=True)
class CommandSpec:
    config_name: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    description: str


COMMANDS: dict[str, CommandSpec] = {}

ALIASES = {
    "init": "init.mvp",
    "mvp": "eval.mvp",
    "eval": "eval.mvp",
}


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])

    if args and args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command_name, overrides = _parse_command_and_overrides(args)
    command_name = ALIASES.get(command_name, command_name)

    if command_name not in COMMANDS:
        known = ", ".join(sorted(list(COMMANDS.keys()) + list(ALIASES.keys())))
        raise SystemExit(f"Unknown command '{command_name}'. Known commands: {known}")

    command = COMMANDS[command_name]
    cfg = _compose_config(command.config_name, overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Config must resolve to a dict, got: {type(config)!r}")

    result = command.handler(config)
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_exp_list(config: dict[str, Any]) -> dict[str, Any]:
    _ = config
    specs = list_experiments()
    return {
        "experiments": [
            {
                "name": spec.name,
                "description": spec.description,
                "pipeline": list(spec.pipeline),
                "config_overrides": spec.config_overrides,
            }
            for spec in specs
        ],
    }


def _run_exp(config: dict[str, Any]) -> dict[str, Any]:
    name = _resolve_experiment_name(config)
    spec = get_experiment(name)

    merged_config = _deep_merge_dict(copy.deepcopy(config), spec.config_overrides)
    feature_cache_cfg = merged_config.get("feature_cache", {})
    force_reextract = bool(
        feature_cache_cfg.get("force_reextract", False)
        if isinstance(feature_cache_cfg, dict)
        else False
    )

    step_results: list[dict[str, Any]] = []
    for step in spec.pipeline:
        if step not in COMMANDS:
            raise KeyError(f"Experiment '{name}' references unknown command '{step}'")
        if step.startswith("exp."):
            raise ValueError("Nested experiment commands are not allowed.")

        if step == "extract.mvp" and has_valid_feature_cache(merged_config) and not force_reextract:
            step_results.append(
                {
                    "step": step,
                    "skipped": True,
                    "reason": "valid_feature_cache_exists",
                }
            )
            continue

        handler = COMMANDS[step].handler
        result = handler(merged_config)
        step_results.append({"step": step, "skipped": False, "result": result})

    return {
        "experiment": spec.name,
        "description": spec.description,
        "pipeline": step_results,
    }


def _resolve_experiment_name(config: dict[str, Any]) -> str:
    top_level_name = str(config.get("name", "")).strip()
    if top_level_name:
        return top_level_name

    exp_cfg = config.get("experiment", {})
    if isinstance(exp_cfg, dict):
        nested_name = str(exp_cfg.get("name", "")).strip()
        if nested_name:
            return nested_name

    return "mvp.jepa_v1.linear"


def _deep_merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge_dict(dict(base[key]), value)
        else:
            base[key] = value
    return base


COMMANDS = {
    "init.mvp": CommandSpec(
        config_name="mvp",
        handler=run_mvp_init,
        description="Initialize MVP full annotations + selection + split artifacts.",
    ),
    "eval.mvp": CommandSpec(
        config_name="mvp",
        handler=run_mvp_eval,
        description="Run MVP evaluation pipeline.",
    ),
    "extract.mvp": CommandSpec(
        config_name="mvp",
        handler=run_mvp_extract,
        description="Extract and cache frozen backbone features for MVP.",
    ),
    "train.linear.mvp": CommandSpec(
        config_name="mvp",
        handler=run_mvp_train_linear,
        description="Train a linear probe from cached MVP features.",
    ),
    "eval.linear.mvp": CommandSpec(
        config_name="mvp",
        handler=run_mvp_eval_linear,
        description="Evaluate linear probe predictions with official MVP scoring.",
    ),
    "exp.list": CommandSpec(
        config_name="mvp",
        handler=_run_exp_list,
        description="List available experiment recipes.",
    ),
    "exp.run": CommandSpec(
        config_name="mvp",
        handler=_run_exp,
        description="Run an experiment recipe (extract -> train -> eval).",
    ),
}


def _parse_command_and_overrides(args: list[str]) -> tuple[str, list[str]]:
    if not args:
        return DEFAULT_COMMAND, []

    first = args[0]
    # If first arg looks like a hydra override, keep default command.
    if first.startswith("-") or "=" in first:
        return DEFAULT_COMMAND, args

    return first, args[1:]


def _compose_config(config_name: str, overrides: list[str]) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _print_help() -> None:
    lines = [
        "Probe4Physics command runner",
        "",
        "Usage:",
        "  python run.py                          # default: eval.mvp",
        "  python run.py <command> [hydra_overrides]",
        "",
        "Common commands:",
        "  python run.py init.mvp",
        "  python run.py extract.mvp",
        "  python run.py train.linear.mvp",
        "  python run.py eval.linear.mvp",
        "  python run.py exp.list",
        "  python run.py exp.run name=mvp.jepa_v1.linear",
        "",
        "Commands:",
    ]
    for name, spec in sorted(COMMANDS.items()):
        lines.append(f"  {name:<18} {spec.description}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
