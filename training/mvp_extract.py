from __future__ import annotations

from typing import Any

from benchmarks.mvp.features import run_mvp_feature_extraction


def run_mvp_extract(config: dict[str, Any]) -> dict[str, Any]:
    return run_mvp_feature_extraction(config)
