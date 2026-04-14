from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.intphys2.data import (
    derive_sample_id,
    load_intphys2_rows,
    normalize_intphys2_row,
)


class InitConfigError(ValueError):
    pass


def run_intphys2_init(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    metadata_file = Path(config["metadata_file"])
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"IntPhys2 metadata file not found: {metadata_file}. "
            "Download the IntPhys2 dataset and point metadata_file to the CSV/JSONL path."
        )

    rows = load_intphys2_rows(metadata_file)
    norm_rows = [normalize_intphys2_row(row) for row in rows]

    _validate_rows(norm_rows)

    split_dir = _resolve_split_dir(config)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_scene_rows = _build_split_scene_rows(norm_rows)
    if not split_scene_rows:
        raise InitConfigError(
            "No valid scenes found in metadata file. "
            "Ensure rows have scene_id, split, and plausibility (0/1) fields."
        )

    split_scenes_path = split_dir / "split_scenes.parquet"
    manifest_path = split_dir / "manifest.json"

    _write_split_scenes_parquet(split_scenes_path, split_scene_rows)

    manifest = _build_manifest(metadata_file, norm_rows, split_scene_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    scenes_by_split = dict(sorted(Counter(row["split"] for row in split_scene_rows).items()))

    _log_init(
        "IntPhys2 split artifacts written",
        split_dir=str(split_dir),
        n_scenes=len(split_scene_rows),
        n_samples=len(norm_rows),
        splits=list(scenes_by_split.keys()),
    )

    return {
        "split_dir": str(split_dir),
        "metadata_file": str(metadata_file),
        "manifest": str(manifest_path),
        "n_scenes": len(split_scene_rows),
        "n_samples": len(norm_rows),
        "scenes_by_split": scenes_by_split,
    }


def _log_init(message: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        print(f"[init.intphys2] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[init.intphys2] {message}", file=sys.stderr)


def _validate_config(config: dict[str, Any]) -> None:
    required = ["metadata_file", "split"]
    missing = [key for key in required if key not in config]
    if missing:
        raise InitConfigError(f"Missing config keys: {missing}")

    split = config.get("split", {})
    if not isinstance(split, dict):
        raise InitConfigError("split config must be a dictionary")


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise InitConfigError("Metadata file is empty.")

    invalid_plausibility = [
        row for row in rows
        if int(row.get("plausibility", -1)) not in (0, 1)
    ]
    if invalid_plausibility:
        first = invalid_plausibility[0]
        raise InitConfigError(
            f"Invalid plausibility value: {first.get('plausibility')!r}. "
            "Expected 0 (impossible) or 1 (possible)."
        )

    missing_scene_id = [row for row in rows if not str(row.get("scene_id", "")).strip()]
    if missing_scene_id:
        raise InitConfigError(
            f"{len(missing_scene_id)} rows are missing a scene_id. "
            "Each row must have a scene_id grouping it into a quadruplet."
        )

    missing_split = [row for row in rows if not str(row.get("split", "")).strip()]
    if missing_split:
        raise InitConfigError(
            f"{len(missing_split)} rows are missing a split label. "
            "Expected values like: debug, main, held_out."
        )


def _build_split_scene_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_id = str(row.get("scene_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if scene_id and split:
            scenes[(scene_id, split)].append(row)

    result: list[dict[str, Any]] = []
    for (scene_id, split), scene_rows in sorted(scenes.items()):
        sample_ids = [derive_sample_id(row) for row in scene_rows]
        n_possible = sum(1 for row in scene_rows if int(row.get("plausibility", -1)) == 1)
        n_impossible = sum(1 for row in scene_rows if int(row.get("plausibility", -1)) == 0)
        conditions = sorted(
            {str(row.get("condition", "")).strip() for row in scene_rows}
            - {""}
        )
        result.append(
            {
                "scene_id": scene_id,
                "split": split,
                "n_samples": len(scene_rows),
                "n_possible": n_possible,
                "n_impossible": n_impossible,
                "condition": conditions[0] if len(conditions) == 1 else "|".join(conditions),
                "sample_ids_json": json.dumps(sample_ids, ensure_ascii=True),
            }
        )

    return result


def _resolve_split_dir(config: dict[str, Any]) -> Path:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise InitConfigError("split config must be a dictionary")
    return Path(str(raw.get("dir", "data/splits/intphys2")))


def _write_split_scenes_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing split_scenes.parquet requires pandas. "
            "Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _build_manifest(
    metadata_file: Path,
    rows: list[dict[str, Any]],
    split_scene_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scenes_by_split = dict(sorted(Counter(row["split"] for row in split_scene_rows).items()))
    samples_by_split = dict(
        sorted(Counter(str(row.get("split", "")) for row in rows).items())
    )
    conditions_counter = dict(
        sorted(Counter(str(row.get("condition", "")) for row in rows).items())
    )

    return {
        "version": 1,
        "kind": "intphys2_split",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "metadata_file": str(metadata_file),
        "metadata_sha256": _sha256_file(metadata_file),
        "stats": {
            "n_samples": len(rows),
            "n_scenes": len(split_scene_rows),
            "scenes_by_split": scenes_by_split,
            "samples_by_split": samples_by_split,
            "samples_by_condition": conditions_counter,
        },
        "packages": _package_versions(["pandas", "pyarrow"]),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as metadata
    except Exception:
        return versions

    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
