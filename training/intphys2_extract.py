from __future__ import annotations

from typing import Any

from benchmarks.intphys2.features import run_intphys2_feature_extraction


def run_intphys2_extract(config: dict[str, Any]) -> dict[str, Any]:
    return run_intphys2_feature_extraction(config)
