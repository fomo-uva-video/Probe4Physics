from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SSV2_NUM_CLASSES = 174
SSV2_SPLITS = ["train", "val"]


def load_ssv2_labels(labels_file: str | Path) -> dict[str, int]:
    """Load the label-name → class-index mapping from the official labels.json."""
    path = Path(labels_file)
    if not path.exists():
        raise FileNotFoundError(f"SSv2 labels file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")

    return {str(k): int(v) for k, v in payload.items()}


def load_ssv2_annotations(annotation_file: str | Path) -> list[dict[str, Any]]:
    """Load an official SSv2 annotation JSON (train, val, or test).

    Each row contains at minimum ``id`` and ``label`` (test split omits label).
    """
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"SSv2 annotation file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}")

    return payload


def derive_sample_id(row: dict[str, Any]) -> str:
    """Return the stable sample_id for an SSv2 row (the official video id field)."""
    return str(row["id"])


def resolve_video_path(video_id: str, videos_root: str | Path) -> str:
    """Resolve an SSv2 video id to its expected local .webm path."""
    return str(Path(videos_root) / f"{video_id}.webm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
