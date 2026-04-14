from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

INTPHYS2_SPLITS = ["debug", "main", "held_out"]
INTPHYS2_CONDITIONS = ["permanence", "immutability", "continuity", "solidity"]

# Column name aliases for flexible CSV/JSON parsing
_VIDEO_PATH_KEYS = ("video_path", "filename", "file", "video_file", "video_ref")
_SCENE_ID_KEYS = ("scene_id", "scene", "quadruplet_id", "group_id")
_CONDITION_KEYS = ("condition", "principle", "physics_condition", "category")
_PLAUSIBILITY_KEYS = ("plausibility", "label", "is_plausible", "possible")
_SPLIT_KEYS = ("split", "subset", "partition")
_DIFFICULTY_KEYS = ("difficulty", "hard", "level")
_CAMERA_KEYS = ("camera", "cam", "view", "viewpoint")


def load_intphys2_rows(metadata_file: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_file)
    suffix = path.suffix.lower()

    if suffix not in (".csv", ".jsonl", ".json", ".parquet"):
        raise ValueError(
            f"Unsupported metadata format: {suffix}. Use .csv, .jsonl, .json, or .parquet"
        )

    if not path.exists():
        raise FileNotFoundError(f"IntPhys2 metadata file not found: {path}")

    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    return _load_parquet(path)


def normalize_intphys2_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the row with canonical field names populated."""
    result = dict(row)

    if "video_path" not in result:
        for key in _VIDEO_PATH_KEYS:
            if key in row and str(row[key]).strip():
                result["video_path"] = str(row[key]).strip()
                break
        else:
            result["video_path"] = ""

    if "scene_id" not in result:
        for key in _SCENE_ID_KEYS:
            if key in row and str(row[key]).strip():
                result["scene_id"] = str(row[key]).strip()
                break
        else:
            result["scene_id"] = ""

    if "condition" not in result:
        for key in _CONDITION_KEYS:
            if key in row and str(row[key]).strip():
                result["condition"] = str(row[key]).strip()
                break
        else:
            result["condition"] = ""

    if "plausibility" not in result:
        for key in _PLAUSIBILITY_KEYS:
            if key in row:
                result["plausibility"] = int(row[key])
                break
        else:
            result["plausibility"] = -1
    else:
        result["plausibility"] = int(result["plausibility"])

    if "split" not in result:
        for key in _SPLIT_KEYS:
            if key in row and str(row[key]).strip():
                result["split"] = str(row[key]).strip()
                break
        else:
            result["split"] = ""

    if "difficulty" not in result:
        for key in _DIFFICULTY_KEYS:
            if key in row and str(row[key]).strip():
                result["difficulty"] = str(row[key]).strip()
                break
        else:
            result["difficulty"] = ""

    if "camera" not in result:
        for key in _CAMERA_KEYS:
            if key in row and str(row[key]).strip():
                result["camera"] = str(row[key]).strip()
                break
        else:
            result["camera"] = ""

    return result


def derive_sample_id(row: dict[str, Any]) -> str:
    """Derive a stable sample_id from an IntPhys2 row."""
    if "sample_id" in row and str(row["sample_id"]).strip():
        return str(row["sample_id"]).strip()

    video_path = str(row.get("video_path", "")).strip()
    if video_path:
        return hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]

    payload = json.dumps(row, sort_keys=True, default=str)
    return "s_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


def resolve_video_path(
    video_ref: str,
    videos_root: str | Path,
    cache_dir: str | Path,
    materialize_missing: bool,
    timeout_seconds: int,
) -> str:
    """Resolve a video reference to a local file path."""
    if not video_ref:
        return ""

    videos_root_path = Path(videos_root)
    normalized_ref = str(video_ref).lstrip("/\\")

    # Try rooted-relative path under videos_root first
    rooted_relative = videos_root_path / normalized_ref
    if rooted_relative.exists():
        return str(rooted_relative)

    direct = Path(video_ref)
    if direct.is_absolute() and direct.exists():
        return str(direct)

    local = videos_root_path / normalized_ref
    if local.exists():
        return str(local)

    parsed = urllib.parse.urlparse(video_ref)
    is_url = parsed.scheme in {"http", "https"}
    if not is_url:
        return str(local)

    if not materialize_missing:
        return video_ref

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ext = Path(parsed.path).suffix or ".mp4"
    digest = hashlib.sha1(video_ref.encode("utf-8")).hexdigest()
    target = cache / f"{digest}{ext}"
    if target.exists():
        return str(target)

    with urllib.request.urlopen(video_ref, timeout=timeout_seconds) as response:
        with target.open("wb") as output:
            shutil.copyfileobj(response, output)
    return str(target)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "items", "rows", "samples"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError("JSON metadata file must contain a list or a dict with list field.")


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading .parquet metadata requires pandas. Install it with: pip install pandas"
        ) from exc

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")
