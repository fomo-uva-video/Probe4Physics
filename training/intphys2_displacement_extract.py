from __future__ import annotations

from typing import Any

from benchmarks.intphys2.features_displacement import run_intphys2_displacement_extraction


def run_intphys2_displacement_extract(config: dict[str, Any]) -> dict[str, Any]:
    return run_intphys2_displacement_extraction(config)
