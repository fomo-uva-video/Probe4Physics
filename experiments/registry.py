from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    pipeline: tuple[str, ...]
    config_overrides: dict[str, Any]


_EXPERIMENTS: dict[str, ExperimentSpec] = {
    "mvp.jepa_v1.linear": ExperimentSpec(
        name="mvp.jepa_v1.linear",
        description="MVP frozen-feature linear probe baseline with JEPA v1.",
        pipeline=("extract.mvp", "train.linear.mvp", "eval.linear.mvp"),
        config_overrides={},
    ),
}


def list_experiments() -> tuple[ExperimentSpec, ...]:
    return tuple(_EXPERIMENTS[name] for name in sorted(_EXPERIMENTS))


def get_experiment(name: str) -> ExperimentSpec:
    key = str(name).strip()
    if key not in _EXPERIMENTS:
        known = ", ".join(sorted(_EXPERIMENTS)) or "<none>"
        raise KeyError(f"Unknown experiment '{key}'. Known experiments: {known}")
    return _EXPERIMENTS[key]
