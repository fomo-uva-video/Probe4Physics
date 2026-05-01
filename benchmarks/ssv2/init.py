from __future__ import annotations

import json
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.ssv2.data import (
    SSV2_NUM_CLASSES,
    derive_sample_id,
    load_ssv2_annotations,
    load_ssv2_labels,
    resolve_video_path,
    sha256_file,
)
from benchmarks.splitting import (
    build_strata,
    build_units,
    split_units,
)


class InitConfigError(ValueError):
    pass


def run_ssv2_init(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    train_file = Path(config["train_annotation_file"])
    val_file = Path(config["val_annotation_file"])
    labels_file = Path(config["labels_file"])

    for path in (train_file, val_file, labels_file):
        if not path.exists():
            raise FileNotFoundError(
                f"SSv2 file not found: {path}. "
                "Download the SSv2 dataset and set the paths in configs/ssv2.yaml."
            )

    label_map = load_ssv2_labels(labels_file)
    _validate_label_map(label_map)

    train_rows = load_ssv2_annotations(train_file)
    val_rows = load_ssv2_annotations(val_file)

    split_cfg = _split_cfg(config)
    seed = int(config.get("seed", 42))
    max_per_class = int(split_cfg["max_samples_per_class"])

    # Combine official train + val, normalise, validate, then apply our own
    # 60/20/20 stratified split so SSv2 is consistent with IntPhys2 and MVP.
    all_norm = _normalize_rows(train_rows + val_rows, label_map)
    _validate_normalized_rows(all_norm)

    # Subsample first (keeps class balance), then split.
    all_subset = _subset_by_class(all_norm, max_per_class, seed=seed)

    if not all_subset:
        raise InitConfigError("No samples remain after subsetting. Check max_samples_per_class.")

    # Each clip is its own splitting unit; stratify by label class.
    units = build_units(all_subset, group_key="sample_id")
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
        seed=seed,
        stratified=True,
    )

    # Materialise split labels back onto rows.
    all_rows = []
    for unit in units:
        row = dict(unit["rows"][0])
        row["split"] = unit_split_map[unit["unit_id"]]
        all_rows.append(row)

    split_dir = _resolve_split_dir(split_cfg)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_clips_path = split_dir / "split_clips.parquet"
    manifest_path = split_dir / "manifest.json"

    _write_split_clips_parquet(split_clips_path, all_rows)

    manifest = _build_manifest(
        train_file=train_file,
        val_file=val_file,
        labels_file=labels_file,
        label_map=label_map,
        all_rows=all_rows,
        split_cfg=split_cfg,
        seed=seed,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    samples_by_split = dict(sorted(Counter(row["split"] for row in all_rows).items()))

    _log_init(
        "SSv2 split artifacts written",
        split_dir=str(split_dir),
        n_train=samples_by_split.get("train", 0),
        n_val=samples_by_split.get("val", 0),
        n_test=samples_by_split.get("test", 0),
        n_classes=len(label_map),
        max_per_class=max_per_class,
    )

    return {
        "split_dir": str(split_dir),
        "manifest": str(manifest_path),
        "n_samples": len(all_rows),
        "samples_by_split": samples_by_split,
        "n_classes": len(label_map),
    }


def _log_init(message: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        print(f"[init.ssv2] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[init.ssv2] {message}", file=sys.stderr)


def _validate_config(config: dict[str, Any]) -> None:
    required = ["train_annotation_file", "val_annotation_file", "labels_file", "split"]
    missing = [key for key in required if key not in config]
    if missing:
        raise InitConfigError(f"Missing config keys: {missing}")

    split = config.get("split", {})
    if not isinstance(split, dict):
        raise InitConfigError("split config must be a dictionary")


def _validate_label_map(label_map: dict[str, int]) -> None:
    if not label_map:
        raise InitConfigError("Labels file is empty.")

    indices = sorted(label_map.values())
    if indices[0] != 0 or indices[-1] != len(label_map) - 1:
        raise InitConfigError(
            f"Label indices are not contiguous 0-based integers. "
            f"Found range [{indices[0]}, {indices[-1]}] for {len(label_map)} classes."
        )


def _validate_normalized_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise InitConfigError("No rows found after normalization.")

    invalid = [row for row in rows if row.get("label_idx", -1) < 0]
    if invalid:
        raise InitConfigError(
            f"{len(invalid)} rows have an unknown label. "
            "Ensure label names in annotation JSON match labels.json exactly."
        )


def _normalize_rows(
    rows: list[dict[str, Any]],
    label_map: dict[str, int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = derive_sample_id(row)
        # labels.json uses plain "something" but annotation templates use "[something]".
        # Strip brackets so both forms resolve to the same key.
        raw_template = str(row.get("template", row.get("label", "")))
        label_name = re.sub(r"\[([^\]]+)\]", r"\1", raw_template)
        label_idx = label_map.get(label_name, -1)
        result.append(
            {
                "sample_id": sample_id,
                "label_idx": label_idx,
                "label_name": label_name,
                "template": raw_template,
                "video_ref": f"{sample_id}.webm",
            }
        )
    return result


def _subset_by_class(
    rows: list[dict[str, Any]],
    max_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically keep at most ``max_per_class`` samples per label class."""
    if max_per_class <= 0:
        return rows

    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["label_idx"])].append(row)

    result: list[dict[str, Any]] = []
    for label_idx in sorted(by_class):
        class_rows = by_class[label_idx]
        if len(class_rows) > max_per_class:
            class_rows = rng.sample(class_rows, max_per_class)
        result.extend(class_rows)

    return result


def _resolve_split_dir(split_cfg: dict[str, Any]) -> Path:
    return Path(split_cfg["dir"])


def _write_split_clips_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing split_clips.parquet requires pandas. "
            "Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _build_manifest(
    train_file: Path,
    val_file: Path,
    labels_file: Path,
    label_map: dict[str, int],
    all_rows: list[dict[str, Any]],
    split_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    samples_by_split = dict(sorted(Counter(row["split"] for row in all_rows).items()))

    return {
        "version": 2,
        "kind": "ssv2_split",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "annotation_sha256": {
            "train": sha256_file(train_file),
            "val": sha256_file(val_file),
            "labels": sha256_file(labels_file),
        },
        "split_config": {
            "ratios": split_cfg["ratios"],
            "seed": seed,
            "max_samples_per_class": split_cfg["max_samples_per_class"],
            "stratify_keys": split_cfg["stratify_keys"],
        },
        "stats": {
            "n_samples": len(all_rows),
            "n_classes": len(label_map),
            "samples_by_split": samples_by_split,
        },
        "packages": _package_versions(["pandas", "pyarrow", "scikit-learn"]),
    }


def _split_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise InitConfigError("split config must be a dictionary")

    ratios = raw.get("ratios", {})
    if not isinstance(ratios, dict):
        raise InitConfigError("split.ratios must be a dictionary")

    return {
        "dir": str(raw.get("dir", "data/splits/ssv2")),
        "max_samples_per_class": int(raw.get("max_samples_per_class", 200)),
        "stratify_keys": list(raw.get("stratify_keys", ["label_name"])),
        "ratios": {
            "train": float(ratios.get("train", 0.6)),
            "val": float(ratios.get("val", 0.2)),
            "test": float(ratios.get("test", 0.2)),
        },
        "rare_strata_min_units": int(raw.get("rare_strata_min_units", 3)),
        "rare_bucket_label": str(raw.get("rare_bucket_label", "OTHER")),
    }


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
