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
    "mvp.ltx_video.linear": ExperimentSpec(
        name="mvp.ltx_video.linear",
        description="MVP frozen-feature linear probe baseline with LTX-Video VAE features.",
        pipeline=("extract.mvp", "train.linear.mvp", "eval.linear.mvp"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "intphys2.jepa_v1.linear": ExperimentSpec(
        name="intphys2.jepa_v1.linear",
        description="IntPhys2 frozen-feature linear probe baseline with JEPA v1.",
        pipeline=("extract.intphys2", "train.linear.intphys2", "eval.linear.intphys2"),
        config_overrides={},
    ),
    "intphys2.ltx_video.linear": ExperimentSpec(
        name="intphys2.ltx_video.linear",
        description="IntPhys2 frozen-feature linear probe baseline with LTX-Video VAE features.",
        pipeline=("extract.intphys2", "train.linear.intphys2", "eval.linear.intphys2"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "ssv2.jepa_v1.linear": ExperimentSpec(
        name="ssv2.jepa_v1.linear",
        description="SSv2 frozen-feature 174-class linear probe baseline with JEPA v1 (control task).",
        pipeline=("extract.ssv2", "train.linear.ssv2", "eval.linear.ssv2"),
        config_overrides={},
    ),
    "ssv2.ltx_video.linear": ExperimentSpec(
        name="ssv2.ltx_video.linear",
        description="SSv2 frozen-feature 174-class linear probe baseline with LTX-Video VAE features.",
        pipeline=("extract.ssv2", "train.linear.ssv2", "eval.linear.ssv2"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
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
