from __future__ import annotations

import copy
from typing import Any

MVP_BASELINE_SPLIT_NAMES = ("test",)


def with_mvp_baseline_test_config(
    config: dict[str, Any],
    baseline_tag: str,
) -> dict[str, Any]:
    """Return a baseline config pinned to the MVP test split only."""

    next_config = copy.deepcopy(config)
    feature_cache = next_config.get("feature_cache", {})
    if not isinstance(feature_cache, dict):
        feature_cache = {}
    else:
        feature_cache = dict(feature_cache)

    feature_cache["split_names"] = list(MVP_BASELINE_SPLIT_NAMES)
    next_config["feature_cache"] = feature_cache
    next_config["baseline_tag"] = str(baseline_tag)
    next_config["split_name"] = MVP_BASELINE_SPLIT_NAMES[0]
    return next_config
