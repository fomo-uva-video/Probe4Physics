from __future__ import annotations

from benchmarks.intphys2.features_single_frame import run_intphys2_single_frame_extraction


def run_intphys2_single_frame_extract(config: dict[str, Any]) -> dict[str, Any]:
    return run_intphys2_single_frame_extraction(config)
