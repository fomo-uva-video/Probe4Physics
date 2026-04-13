from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MVP_SUBSETS = [
    "human_object_interactions",
    "intuitive_physics",
    "robot_object_interactions",
    "temporal_reasoning",
]


def ensure_mvp_annotation_file(
    annotation_file: str | Path,
    split_hint: str,
    auto_download: bool,
    dataset_id: str,
    subsets: list[str] | tuple[str, ...],
    hf_cache_dir: str | Path,
    target: str,
) -> str:
    path = Path(annotation_file)
    if path.exists():
        return str(path)

    if not auto_download:
        raise FileNotFoundError(
            "Annotation file not found and auto-download is disabled: "
            f"{path}. Enable annotations.auto_download or create the file manually."
        )

    resolved_target = _resolve_target(target, path, split_hint)
    download_mvp_annotations(
        output_file=path,
        dataset_id=dataset_id,
        target=resolved_target,
        subsets=list(subsets),
        hf_cache_dir=hf_cache_dir,
    )

    if not path.exists():
        raise RuntimeError(f"Auto-download completed but file was not created: {path}")
    return str(path)


def download_mvp_annotations(
    output_file: str | Path,
    dataset_id: str,
    target: str,
    subsets: list[str],
    hf_cache_dir: str | Path,
) -> None:
    if target not in {"mvp", "mvp_mini"}:
        raise ValueError(f"target must be 'mvp' or 'mvp_mini', got {target!r}")

    split = "full" if target == "mvp" else "mini"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = Path(hf_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install it inside the active env with: "
            "python -m pip install datasets"
        ) from exc

    rows: list[dict[str, Any]] = []
    for subset in subsets:
        ds = load_dataset(
            dataset_id,
            subset,
            split=split,
            cache_dir=str(cache_path),
        )

        for row in ds:
            item = dict(row)
            item["subset"] = str(item.get("subset", subset))
            item["mvp_split"] = split
            rows.append(item)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def load_mvp_rows(annotation_file: str | Path) -> list[dict[str, Any]]:
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".parquet":
        return _load_parquet(path)

    raise ValueError(
        f"Unsupported annotation format: {suffix}. Use .jsonl, .json, .csv, or .parquet"
    )


def resolve_video_path(
    video_ref: str,
    videos_root: str | Path,
    cache_dir: str | Path,
    materialize_missing: bool,
    timeout_seconds: int,
) -> str:
    if not video_ref:
        return ""

    videos_root_path = Path(videos_root)
    direct = Path(video_ref)
    if direct.is_absolute() and direct.exists():
        return str(direct)

    local = videos_root_path / video_ref
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


def _resolve_target(target: str, path: Path, split_hint: str) -> str:
    value = str(target or "auto").strip().lower()
    if value in {"mvp", "mvp_mini"}:
        return value
    if value != "auto":
        raise ValueError(f"annotations.target must be auto|mvp|mvp_mini, got {target!r}")

    hint = str(split_hint).strip().lower()
    stem = path.stem.lower()
    if "mini" in hint or "mini" in stem:
        return "mvp_mini"
    return "mvp"


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

    raise ValueError("JSON annotation file must contain a list or a dict with list field.")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading .parquet annotations requires pandas. Install it with: pip install pandas"
        ) from exc

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")
