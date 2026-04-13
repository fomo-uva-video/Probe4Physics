from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from evaluation.mvp.eval import run_mvp_eval
from evaluation.mvp.init import run_mvp_init

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_COMMAND = "eval.mvp"


@dataclass(frozen=True)
class CommandSpec:
    config_name: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    description: str


COMMANDS: dict[str, CommandSpec] = {
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
}

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
        "  python run.py init.mvp [hydra_overrides]",
        "  python run.py eval.mvp [hydra_overrides]",
        "  python run.py mvp [hydra_overrides]     # alias",
        "",
        "Examples:",
        "  python run.py init.mvp",
        "  python run.py eval.mvp",
        "  python run.py eval.mvp split_name=val predictor.mode=random",
        "",
        "Commands:",
    ]
    for name, spec in sorted(COMMANDS.items()):
        lines.append(f"  {name:<10} {spec.description}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
