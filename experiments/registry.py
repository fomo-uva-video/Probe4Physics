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
    "mvp.jepa_v1.probe": ExperimentSpec(
        name="mvp.jepa_v1.probe",
        description="MVP frozen-feature probe recipe with JEPA v1.",
        pipeline=("extract.mvp", "train.probe.mvp", "eval.probe.mvp"),
        config_overrides={},
    ),
    "mvp.ltx_video.probe": ExperimentSpec(
        name="mvp.ltx_video.probe",
        description="MVP frozen-feature probe recipe with LTX-Video diffusion transformer features.",
        pipeline=("extract.mvp", "train.probe.mvp", "eval.probe.mvp"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "mvp.wan_video.probe": ExperimentSpec(
        name="mvp.wan_video.probe",
        description="MVP frozen-feature probe recipe with Wan diffusion transformer features.",
        pipeline=("extract.mvp", "train.probe.mvp", "eval.probe.mvp"),
        config_overrides={
            "backbone": {
                "name": "wan_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "intphys2.jepa_v1.probe": ExperimentSpec(
        name="intphys2.jepa_v1.probe",
        description="IntPhys2 frozen-feature probe recipe with JEPA v1.",
        pipeline=("extract.intphys2", "train.probe.intphys2", "eval.probe.intphys2"),
        config_overrides={},
    ),
    "intphys2.ltx_video.probe": ExperimentSpec(
        name="intphys2.ltx_video.probe",
        description="IntPhys2 frozen-feature probe recipe with LTX-Video diffusion transformer features.",
        pipeline=("extract.intphys2", "train.probe.intphys2", "eval.probe.intphys2"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "intphys2.wan_video.probe": ExperimentSpec(
        name="intphys2.wan_video.probe",
        description="IntPhys2 frozen-feature probe recipe with Wan diffusion transformer features.",
        pipeline=("extract.intphys2", "train.probe.intphys2", "eval.probe.intphys2"),
        config_overrides={
            "backbone": {
                "name": "wan_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "intphys2.jepa_v1.displacement": ExperimentSpec(
        name="intphys2.jepa_v1.displacement",
        description="IntPhys2 test-only displacement baseline eval recipe with JEPA v1.",
        pipeline=(
            "extract.intphys2.displacement",
            "eval.probe.intphys2.displacement",
        ),
        config_overrides={},
    ),
    "intphys2.ltx_video.displacement": ExperimentSpec(
        name="intphys2.ltx_video.displacement",
        description="IntPhys2 test-only displacement baseline eval recipe with LTX-Video diffusion transformer features.",
        pipeline=(
            "extract.intphys2.displacement",
            "eval.probe.intphys2.displacement",
        ),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "intphys2.jepa_v1.single_frame": ExperimentSpec(
        name="intphys2.jepa_v1.single_frame",
        description="IntPhys2 single-frame repeated baseline with JEPA v1.",
        pipeline=(
            "extract.intphys2.single_frame",
            "eval.probe.intphys2.single_frame",
        ),
        config_overrides={},
    ),
    "intphys2.ltx_video.single_frame": ExperimentSpec(
        name="intphys2.ltx_video.single_frame",
        description="IntPhys2 single-frame repeated baseline with LTX-Video diffusion transformer features.",
        pipeline=(
            "extract.intphys2.single_frame",
            "eval.probe.intphys2.single_frame",
        ),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "ssv2.jepa_v1.probe": ExperimentSpec(
        name="ssv2.jepa_v1.probe",
        description="SSv2 frozen-feature probe recipe with JEPA v1 (control task).",
        pipeline=("extract.ssv2", "train.probe.ssv2", "eval.probe.ssv2"),
        config_overrides={},
    ),
    "ssv2.ltx_video.probe": ExperimentSpec(
        name="ssv2.ltx_video.probe",
        description="SSv2 frozen-feature probe recipe with LTX-Video diffusion transformer features.",
        pipeline=("extract.ssv2", "train.probe.ssv2", "eval.probe.ssv2"),
        config_overrides={
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "device": "cuda",
                },
            }
        },
    ),
    "ssv2.wan_video.probe": ExperimentSpec(
        name="ssv2.wan_video.probe",
        description="SSv2 frozen-feature probe recipe with Wan diffusion transformer features.",
        pipeline=("extract.ssv2", "train.probe.ssv2", "eval.probe.ssv2"),
        config_overrides={
            "backbone": {
                "name": "wan_video",
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
