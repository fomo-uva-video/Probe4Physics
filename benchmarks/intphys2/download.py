"""
HuggingFace downloader for the facebook/IntPhys2 dataset.

Downloads video files and metadata via huggingface_hub.snapshot_download,
normalises the HuggingFace column schema to the canonical intphys2 format,
and writes a metadata CSV that the rest of the pipeline (init → extract →
eval) can consume directly.

Usage (via run.py):
    python run.py download.intphys2
    python run.py download.intphys2 download.splits=[Debug]
    python run.py download.intphys2 download.local_dir=/path/to/my/storage
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

# HuggingFace split names → our canonical split names
_HF_SPLIT_MAP: dict[str, str] = {
    "debug": "debug",
    "Debug": "debug",
    "main": "main",
    "Main": "main",
    "heldout": "held_out",
    "HeldOut": "held_out",
    "held_out": "held_out",
}

# Expected columns from the HuggingFace facebook/IntPhys2 parquet files
_HF_SCENE_KEYS = ("SceneIndex", "scene_index", "sceneindex")
_HF_FILE_KEYS = ("file_name", "filename", "file")
_HF_CONDITION_KEYS = ("condition", "Condition")
_HF_TYPE_KEYS = ("type", "Type", "plausibility")
_HF_DIFFICULTY_KEYS = ("Difficulty", "difficulty")
_HF_CAMERA_KEYS = ("Camera", "camera")


class DownloadError(RuntimeError):
    pass


class DownloadConfigError(ValueError):
    pass


def run_intphys2_download(config: dict[str, Any]) -> dict[str, Any]:
    """Download IntPhys2 from HuggingFace and write a normalised metadata CSV.

    After this command succeeds, run ``python run.py init.intphys2`` to build
    the split artifacts.
    """
    _validate_config(config)
    hf_cfg = _hf_cfg(config)

    repo_id: str = hf_cfg["repo_id"]
    hf_splits: list[str] = hf_cfg["splits"]
    local_dir: Path = _resolve_local_dir(config, hf_cfg)
    local_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(config["metadata_file"])

    _log("Starting download", repo_id=repo_id, splits=hf_splits, local_dir=str(local_dir))

    # Step 1 — download snapshot (videos + metadata files for requested splits)
    snapshot_path = _snapshot_download(repo_id, local_dir, hf_splits)

    # Step 2 — load and normalise metadata from the snapshot
    all_rows: list[dict[str, Any]] = []
    for hf_split in hf_splits:
        rows = _load_split_metadata(Path(snapshot_path), hf_split)
        all_rows.extend(rows)
        _log(f"Loaded {hf_split}", n_rows=len(rows))

    if not all_rows:
        raise DownloadError(
            f"No metadata rows found in snapshot at {snapshot_path}. "
            "Check that the split names are correct and the download completed."
        )

    # Step 3 — write normalised metadata CSV
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    _write_metadata_csv(metadata_file, all_rows)

    n_per_split: dict[str, int] = {}
    for row in all_rows:
        s = str(row.get("split", "unknown"))
        n_per_split[s] = n_per_split.get(s, 0) + 1

    _log(
        "Download complete — run `python run.py init.intphys2` next",
        metadata_file=str(metadata_file),
        videos_root=str(local_dir),
        n_total=len(all_rows),
    )

    return {
        "metadata_file": str(metadata_file),
        "videos_root": str(local_dir),
        "snapshot_dir": str(snapshot_path),
        "n_rows": len(all_rows),
        "n_rows_per_split": n_per_split,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_config(config: dict[str, Any]) -> None:
    for key in ("metadata_file", "videos_root"):
        if key not in config:
            raise DownloadConfigError(f"Config must contain '{key}'")


def _hf_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("download", {})
    if not isinstance(raw, dict):
        raise DownloadConfigError("'download' config must be a dictionary")
    return {
        "repo_id": str(raw.get("repo_id", "facebook/IntPhys2")),
        "splits": list(raw.get("splits", ["Debug", "Main"])),
        "local_dir": str(raw.get("local_dir", "")).strip(),
    }


def _resolve_local_dir(config: dict[str, Any], hf_cfg: dict[str, Any]) -> Path:
    """Return the directory where the HF snapshot will be stored.

    Defaults to ``videos_root`` so that no separate copy step is needed.
    """
    explicit = hf_cfg.get("local_dir", "").strip()
    if explicit:
        return Path(explicit)
    return Path(str(config.get("videos_root", "data/videos/intphys2")))


def _snapshot_download(repo_id: str, local_dir: Path, hf_splits: list[str]) -> str:
    """Download the requested splits from HuggingFace and return the local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download IntPhys2. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    # Build allow_patterns to fetch only what we need:
    #   - HF-generated parquet metadata (data/*.parquet)
    #   - Video files for each requested split ({Split}/Videos/*.mp4)
    allow_patterns: list[str] = ["README.md", "data/*.parquet"]
    for s in hf_splits:
        allow_patterns.extend([
            f"{s}/Videos/*.mp4",
            f"{s}/Videos/*.webm",
            f"{s}/*.csv",
            f"{s}/*.json",
            f"{s}/*.parquet",
        ])

    result = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    return result


def _load_split_metadata(snapshot_dir: Path, hf_split: str) -> list[dict[str, Any]]:
    """Find and parse metadata for one HF split from the downloaded snapshot."""
    our_split = _HF_SPLIT_MAP.get(hf_split, hf_split.lower())

    # Priority 1 — HF auto-generated parquet in data/ directory
    # Naming: data/{Split}-00000-of-00001.parquet (case-insensitive match)
    parquet_candidates = _find_split_parquet(snapshot_dir, hf_split)
    if parquet_candidates:
        rows = _rows_from_parquet(parquet_candidates)
        return [_normalize_hf_row(r, hf_split, our_split) for r in rows]

    # Priority 2 — CSV file(s) inside the split subdirectory
    split_dir = snapshot_dir / hf_split
    if split_dir.exists():
        csv_files = sorted(split_dir.glob("*.csv"))
        if csv_files:
            rows = _rows_from_csv(csv_files)
            return [_normalize_hf_row(r, hf_split, our_split) for r in rows]

    # Priority 3 — JSON file(s) inside the split subdirectory
    if split_dir.exists():
        json_files = sorted(split_dir.glob("*.json"))
        if json_files:
            rows = _rows_from_json(json_files)
            return [_normalize_hf_row(r, hf_split, our_split) for r in rows]

    raise DownloadError(
        f"Could not find metadata for split '{hf_split}' in {snapshot_dir}.\n"
        f"Looked in: {snapshot_dir / 'data'}/*.parquet, "
        f"{snapshot_dir / hf_split}/*.csv, "
        f"{snapshot_dir / hf_split}/*.json\n"
        "The HuggingFace repo structure may have changed."
    )


def _find_split_parquet(snapshot_dir: Path, hf_split: str) -> list[Path]:
    data_dir = snapshot_dir / "data"
    if not data_dir.exists():
        return []
    # Try exact name prefix first, then case-insensitive scan
    candidates = sorted(data_dir.glob(f"{hf_split}-*.parquet"))
    if not candidates:
        candidates = sorted(
            p for p in data_dir.glob("*.parquet")
            if hf_split.lower() in p.stem.lower()
        )
    return candidates


def _rows_from_parquet(files: list[Path]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read parquet metadata files. "
            "Install it with: pip install pandas pyarrow"
        ) from exc

    frames = [pd.read_parquet(f) for f in files]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return df.to_dict(orient="records")


def _rows_from_csv(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def _rows_from_json(files: list[Path]) -> list[dict[str, Any]]:
    import json
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            for key in ("data", "items", "rows", "samples"):
                if key in payload and isinstance(payload[key], list):
                    rows.extend(payload[key])
                    break
    return rows


def _normalize_hf_row(row: dict[str, Any], hf_split: str, our_split: str) -> dict[str, Any]:
    """Map HuggingFace facebook/IntPhys2 columns to our canonical schema."""
    # --- scene_id: use split-prefixed SceneIndex for global uniqueness ---
    scene_index = _pick(row, _HF_SCENE_KEYS, default="")
    scene_id = f"{hf_split}_{scene_index}" if scene_index else ""

    # --- video_path relative to videos_root (= snapshot local_dir) ---
    file_name = _pick(row, _HF_FILE_KEYS, default="")
    video_path = _resolve_video_path(file_name, hf_split)

    # --- condition ---
    condition = _pick(row, _HF_CONDITION_KEYS, default="").lower().strip()

    # --- plausibility from type column ---
    type_val = _pick(row, _HF_TYPE_KEYS, default=None)
    plausibility = _parse_plausibility(type_val) if type_val is not None else -1

    # --- optional metadata ---
    difficulty = _pick(row, _HF_DIFFICULTY_KEYS, default="")
    camera = _pick(row, _HF_CAMERA_KEYS, default="")

    return {
        "video_path": video_path,
        "scene_id": scene_id,
        "condition": condition,
        "plausibility": plausibility,
        "split": our_split,
        "difficulty": difficulty,
        "camera": camera,
    }


def _pick(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """Return the first non-empty value from a set of candidate column names."""
    for key in keys:
        if key in row:
            val = row[key]
            if val is not None and str(val).strip():
                return val
    return default


def _resolve_video_path(file_name: str, hf_split: str) -> str:
    """Build a video_path relative to the snapshot root (= videos_root).

    HF file_name is typically 'Videos/{64-char-hash}.mp4'.
    The full relative path within the snapshot is '{hf_split}/{file_name}'.
    """
    if not file_name:
        return ""
    fn = Path(str(file_name))
    # Normalise: ensure path starts with Videos/
    parts = fn.parts
    if parts and parts[0].lower() == "videos":
        return f"{hf_split}/{fn.as_posix()}"
    # file_name might just be the hash filename
    return f"{hf_split}/Videos/{fn.name}"


def _parse_plausibility(value: Any) -> int:
    """Convert HF 'type' column value to 0 (impossible) or 1 (possible).

    HF values observed: "Possible", "Impossible", "Possible_1", "Impossible_2"
    Also handles integer/float 0/1 for robustness.
    """
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    # Check "impossible" before "possible" — the latter is a substring of the former.
    if "impossible" in s:
        return 0
    if "possible" in s or "plausible" in s or s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return -1


def _write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["video_path", "scene_id", "condition", "plausibility", "split", "difficulty", "camera"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log(message: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    if payload:
        print(f"[download.intphys2] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[download.intphys2] {message}", file=sys.stderr)
