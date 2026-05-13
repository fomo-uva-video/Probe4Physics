from __future__ import annotations

from benchmarks.mvp.features_displacement import run_mvp_displacement_extraction


def run_mvp_displacement_extract(config: dict) -> dict:
    return run_mvp_displacement_extraction(config)
