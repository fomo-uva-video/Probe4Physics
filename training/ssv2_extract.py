from __future__ import annotations

from typing import Any

from benchmarks.ssv2.features import run_ssv2_feature_extraction


def run_ssv2_extract(config: dict[str, Any]) -> dict[str, Any]:
    return run_ssv2_feature_extraction(config)
