from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.intphys2.data import (
    derive_sample_id,
    load_intphys2_rows,
    normalize_intphys2_row,
)
from benchmarks.splitting import (
    build_strata,
    build_units,
    split_units,
)


class InitConfigError(ValueError):
    pass


def run_intphys2_init(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    metadata_file = Path(config["metadata_file"])
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"IntPhys2 metadata file not found: {metadata_file}. "
            "Download the IntPhys2 dataset first: python run.py download.intphys2"
        )

    rows = load_intphys2_rows(metadata_file)
    norm_rows = [normalize_intphys2_row(row) for row in rows]
    _validate_rows(norm_rows)

    # Keep only "main" — drop debug (calibration noise only) and held_out (no labels).
    main_rows = [r for r in norm_rows if str(r.get("split", "")).strip().lower() == "main"]
    if not main_rows:
        raise InitConfigError(
            "No rows with split='main' found in metadata. "
            "Ensure the metadata file contains Main-split data from facebook/IntPhys2."
        )

    split_cfg = _split_cfg(config)

    # Group by scene_id so that all 4 clips of a quadruplet always land in the same split.
    units = build_units(main_rows, group_key="scene_id")

    # Stratify by physics condition (permanence / immutability / continuity / solidity)
    # so each condition is represented proportionally in train/val/test.
    unit_to_stratum = build_strata(
        units,
        keys=split_cfg["stratify_keys"],
        rare_strata_min_units=split_cfg["rare_strata_min_units"],
        other_label=split_cfg["rare_bucket_label"],
    )
    for unit in units:
        unit["stratum"] = unit_to_stratum[unit["unit_id"]]

    unit_split_map = split_units(
        units,
        ratios=split_cfg["ratios"],
        seed=split_cfg["seed"],
        stratified=True,
    )

    split_scene_rows = _build_split_scene_rows(units, unit_split_map)
    if not split_scene_rows:
        raise InitConfigError("No valid scenes produced after splitting. Check metadata.")

    # Optional video presence check.
    videos_root = Path(str(config.get("videos_root", ""))).expanduser()
    video_report = _check_video_presence(main_rows, videos_root)

    split_dir = _resolve_split_dir(split_cfg)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_scenes_path = split_dir / "split_scenes.parquet"
    manifest_path = split_dir / "manifest.json"

    _write_split_scenes_parquet(split_scenes_path, split_scene_rows)

    manifest = _build_manifest(
        metadata_file=metadata_file,
        all_rows=norm_rows,
        main_rows=main_rows,
        split_scene_rows=split_scene_rows,
        split_cfg=split_cfg,
        video_report=video_report,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    scenes_by_split = dict(sorted(Counter(row["split"] for row in split_scene_rows).items()))

    _log_init(
        "IntPhys2 split artifacts written",
        split_dir=str(split_dir),
        n_scenes=len(split_scene_rows),
        n_samples=len(main_rows),
        splits=list(scenes_by_split.keys()),
    )

    if video_report["n_missing"] > 0:
        _log_init(
            "WARNING: missing video files",
            n_missing=video_report["n_missing"],
            n_present=video_report["n_present"],
            n_total=video_report["n_total"],
        )

    return {
        "split_dir": str(split_dir),
        "metadata_file": str(metadata_file),
        "manifest": str(manifest_path),
        "n_scenes": len(split_scene_rows),
        "n_samples": len(main_rows),
        "scenes_by_split": scenes_by_split,
        "video_check": video_report,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_init(message: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
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


def _split_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise InitConfigError("split config must be a dictionary")

    ratios = raw.get("ratios", {})
    if not isinstance(ratios, dict):
        raise InitConfigError("split.ratios must be a dictionary")

    return {
        "dir": str(raw.get("dir", "data/splits/intphys2")),
        "stratify_keys": list(raw.get("stratify_keys", ["condition"])),
        "ratios": {
            "train": float(ratios.get("train", 0.6)),
            "val": float(ratios.get("val", 0.2)),
            "test": float(ratios.get("test", 0.2)),
        },
        "seed": int(raw.get("seed", 42)),
        "rare_strata_min_units": int(raw.get("rare_strata_min_units", 3)),
        "rare_bucket_label": str(raw.get("rare_bucket_label", "OTHER")),
    }


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise InitConfigError("Metadata file is empty.")

    invalid_plausibility = [
        row for row in rows if int(row.get("plausibility", -1)) not in (0, 1)
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
            "Expected 'main' (or 'debug') from the HuggingFace download."
        )


def _build_split_scene_rows(
    units: list[dict[str, Any]],
    unit_split_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Build scene-level rows (one per scene quadruplet) with train/val/test labels.

    Mirrors MVP's _build_split_pair_rows: the unit (scene) is the atom of
    splitting so all 4 clips of a quadruplet always land in the same split.
    """
    rows: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda u: str(u["unit_id"])):
        scene_id = str(unit["unit_id"])
        split_name = unit_split_map[scene_id]
        scene_rows = unit["rows"]

        sample_ids = [derive_sample_id(row) for row in scene_rows]
        conditions = sorted(
            {str(r.get("condition", "")).strip() for r in scene_rows} - {""}
        )
        n_possible = sum(1 for r in scene_rows if int(r.get("plausibility", -1)) == 1)
        n_impossible = sum(1 for r in scene_rows if int(r.get("plausibility", -1)) == 0)

        rows.append(
            {
                "scene_id": scene_id,
                "split": split_name,
                "stratum": str(unit.get("stratum", "unknown")),
                "condition": conditions[0] if len(conditions) == 1 else "|".join(conditions),
                "n_samples": len(scene_rows),
                "n_possible": n_possible,
                "n_impossible": n_impossible,
                "sample_ids_json": json.dumps(sample_ids, ensure_ascii=True),
            }
        )
    return rows


def _check_video_presence(
    rows: list[dict[str, Any]],
    videos_root: Path,
) -> dict[str, Any]:
    """Count how many video files in the metadata are actually present on disk."""
    if not videos_root or not videos_root.exists():
        return {
            "n_total": len(rows),
            "n_present": 0,
            "n_missing": len(rows),
            "n_skipped": 0,
            "videos_root_exists": False,
            "first_missing": [],
        }

    present = 0
    missing: list[str] = []
    skipped = 0
    for row in rows:
        vp = str(row.get("video_path", "")).strip()
        if not vp:
            skipped += 1
            continue
        if (videos_root / vp).exists():
            present += 1
        else:
            missing.append(vp)

    return {
        "n_total": len(rows),
        "n_present": present,
        "n_missing": len(missing),
        "n_skipped": skipped,
        "videos_root_exists": True,
        "first_missing": missing[:5],
    }


def _resolve_split_dir(split_cfg: dict[str, Any]) -> Path:
    return Path(split_cfg["dir"])


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
    all_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    split_scene_rows: list[dict[str, Any]],
    split_cfg: dict[str, Any],
    video_report: dict[str, Any],
) -> dict[str, Any]:
    scenes_by_split = dict(sorted(Counter(row["split"] for row in split_scene_rows).items()))
    conditions_counter = dict(
        sorted(Counter(str(row.get("condition", "")) for row in main_rows).items())
    )
    conditions_by_split: dict[str, dict[str, int]] = {}
    for row in split_scene_rows:
        split = str(row["split"])
        cond = str(row.get("condition", ""))
        conditions_by_split.setdefault(split, {})
        conditions_by_split[split][cond] = conditions_by_split[split].get(cond, 0) + 1

    return {
        "version": 2,
        "kind": "intphys2_split",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "metadata_file": str(metadata_file),
        "metadata_sha256": _sha256_file(metadata_file),
        "split_config": {
            "source_filter": "main",
            "ratios": split_cfg["ratios"],
            "seed": split_cfg["seed"],
            "stratify_keys": split_cfg["stratify_keys"],
        },
        "stats": {
            "n_samples_all": len(all_rows),
            "n_samples_main": len(main_rows),
            "n_scenes": len(split_scene_rows),
            "scenes_by_split": scenes_by_split,
            "scenes_by_condition_split": {
                split: dict(sorted(conds.items()))
                for split, conds in sorted(conditions_by_split.items())
            },
            "samples_by_condition": conditions_counter,
        },
        "video_check": video_report,
        "packages": _package_versions(["pandas", "pyarrow", "scikit-learn"]),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

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
