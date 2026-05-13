from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from uuid import uuid4

import torch

from benchmarks.mvp.baseline_config import with_mvp_baseline_test_config
from benchmarks.mvp.features import (
    _decode_video_clip,
    has_valid_feature_cache,
    resolve_expected_feature_cache_paths,
    run_mvp_feature_extraction,
)

_BASELINE_TAG = "displacement"


def _displacement_for_sample(sample_id: str, num_frames: int) -> int:
    """Deterministic non-zero displacement in [1, num_frames-1] tied to sample_id."""
    if num_frames <= 1:
        raise ValueError(f"num_frames must be > 1 for displacement, got {num_frames}")
    digest = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16)
    return digest % (num_frames - 1) + 1


def _make_displacement_clip_fn():
    def clip_fn(record: dict[str, Any], num_frames: int, crop_size: int) -> torch.Tensor:
        clip = _decode_video_clip(record["video_path"], num_frames, crop_size)
        d = _displacement_for_sample(str(record["sample_id"]), num_frames)
        return torch.roll(clip, shifts=d, dims=2)

    return clip_fn


def run_mvp_displacement_extraction(config: dict[str, Any]) -> dict[str, Any]:
    config = with_mvp_baseline_test_config(config, _BASELINE_TAG)
    result = run_mvp_feature_extraction(config, clip_fn=_make_displacement_clip_fn())

    paths = resolve_expected_feature_cache_paths(config)
    if paths.index_path.exists():
        import pandas as pd

        index = pd.read_parquet(paths.index_path)
        num_frames = int(config.get("decode", {}).get("num_frames", 16))
        displacements = {
            str(sid): _displacement_for_sample(str(sid), num_frames)
            for sid in index["sample_id"].tolist()
        }
        metadata: dict[str, Any] = {
            "baseline": _BASELINE_TAG,
            "description": (
                "displacement = SHA256(sample_id) % (num_frames - 1) + 1; "
                "non-zero circular shift applied along the time axis"
            ),
            "num_frames": num_frames,
            "displacements": displacements,
        }
        metadata_path = paths.cache_dir / "baseline_metadata.json"
        tmp = metadata_path.parent / f".{metadata_path.name}.tmp.{os.getpid()}.{uuid4().hex}"
        tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(tmp, metadata_path)

    return result


def has_valid_displacement_cache(config: dict[str, Any]) -> bool:
    return has_valid_feature_cache(with_mvp_baseline_test_config(config, _BASELINE_TAG))
