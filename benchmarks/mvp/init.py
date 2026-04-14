from __future__ import annotations

import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from benchmarks.mvp.data import DEFAULT_MVP_SUBSETS, ensure_mvp_annotation_file, load_mvp_rows
from benchmarks.mvp.selection import (
    apply_mvp_selection,
    derive_pair_id,
    derive_sample_id,
    normalize_question_template,
    write_selection_artifacts,
)
from benchmarks.splitting import (
    build_strata,
    build_units,
    materialize_sample_assignments,
    split_units,
)


class InitConfigError(ValueError):
    pass


def run_mvp_init(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    official_repo_root = Path(config["official_repo_root"])
    official_utils = official_repo_root / "tasks" / "mvp" / "utils.py"
    if not official_utils.exists():
        raise FileNotFoundError(
            "Official MVP submodule is missing. Expected file: "
            f"{official_utils}. Run: git submodule update --init --recursive"
        )

    annotations_cfg = _annotations_cfg(config)
    annotation_file = ensure_mvp_annotation_file(
        annotation_file=config["annotation_file"],
        split_hint="full",
        auto_download=bool(annotations_cfg["auto_download"]),
        dataset_id=str(annotations_cfg["dataset_id"]),
        subsets=list(annotations_cfg["subsets"]),
        hf_cache_dir=str(annotations_cfg["hf_cache_dir"]),
        target="mvp",
    )

    rows = load_mvp_rows(annotation_file)
    kept_rows, dropped_rows, selection_report = apply_mvp_selection(rows, config["selection"])

    if not kept_rows:
        raise InitConfigError("Selection produced 0 rows; cannot build split.")

    enriched_rows = [_enrich_row_for_split(row) for row in kept_rows]
    split_cfg = _split_cfg(config)

    units = build_units(enriched_rows, split_cfg["group_key"])
    unit_to_stratum = build_strata(
        units,
        split_cfg["stratify_keys"],
        int(split_cfg["rare_strata_min_units"]),
        str(split_cfg["rare_bucket_label"]),
    )
    for unit in units:
        unit["stratum"] = unit_to_stratum[unit["unit_id"]]

    unit_split_map = split_units(
        units,
        ratios=split_cfg["ratios"],
        seed=int(split_cfg["seed"]),
        stratified=True,
    )

    assignments = materialize_sample_assignments(
        enriched_rows,
        unit_split_map,
        group_key=split_cfg["group_key"],
    )

    split_dir = _resolve_split_dir(split_cfg)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_pairs_path = split_dir / "split_pairs.parquet"
    manifest_path = split_dir / "manifest.json"

    split_pair_rows = _build_split_pair_rows(units, unit_split_map)
    if bool(split_cfg["template_coverage_check"]):
        _assert_template_coverage(split_pair_rows)
    _write_split_pairs_parquet(split_pairs_path, split_pair_rows)

    manifest = _build_manifest(
        annotation_file=Path(annotation_file),
        kept_rows=kept_rows,
        dropped_rows=dropped_rows,
        selection_report=selection_report,
        split_cfg=split_cfg,
        split_pair_rows=split_pair_rows,
        assignments=assignments,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if bool(config["selection"]["artifacts"]["enabled"]):
        write_selection_artifacts(split_dir, kept_rows, dropped_rows, selection_report)

    return {
        "split_dir": str(split_dir),
        "annotation_file": str(annotation_file),
        "manifest": str(manifest_path),
        "n_pairs": len(split_pair_rows),
        "n_samples": len(assignments),
        "pairs_by_split": dict(Counter(row["split"] for row in split_pair_rows)),
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "annotation_file",
        "official_repo_root",
        "selection",
        "split",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise InitConfigError(f"Missing config keys: {missing}")

    selection = config.get("selection", {})
    if not isinstance(selection, dict):
        raise InitConfigError("selection must be a dictionary")
    if not bool(selection.get("drop_incomplete_pairs", True)):
        raise InitConfigError("selection.drop_incomplete_pairs=false is not allowed")


def _split_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise InitConfigError("split config must be a dictionary")

    ratios = raw.get("ratios", {})
    if not isinstance(ratios, dict):
        raise InitConfigError("split.ratios must be a dictionary")

    return {
        "dir": str(raw.get("dir", "data/splits/mvp/full_60_20_20")),
        "group_key": str(raw.get("group_key", "pair_id")),
        "stratify_keys": list(raw.get("stratify_keys", ["source", "question_template"])),
        "ratios": {
            "train": float(ratios.get("train", 0.6)),
            "val": float(ratios.get("val", 0.2)),
            "test": float(ratios.get("test", 0.2)),
        },
        "seed": int(raw.get("seed", 42)),
        "rare_strata_min_units": int(raw.get("rare_strata_min_units", 3)),
        "rare_bucket_label": str(raw.get("rare_bucket_label", "OTHER")),
        "template_coverage_check": bool(raw.get("template_coverage_check", False)),
    }


def _annotations_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("annotations", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InitConfigError("annotations must be a dictionary")

    subsets = raw.get("subsets", DEFAULT_MVP_SUBSETS)
    if not isinstance(subsets, (list, tuple)) or not subsets:
        raise InitConfigError("annotations.subsets must be a non-empty list")

    subset_values = [str(item) for item in subsets if str(item).strip()]
    if not subset_values:
        raise InitConfigError("annotations.subsets resolved to empty list")

    return {
        "auto_download": bool(raw.get("auto_download", True)),
        "dataset_id": str(raw.get("dataset_id", "facebook/minimal_video_pairs")),
        "hf_cache_dir": str(raw.get("hf_cache_dir", ".cache/hf_datasets")),
        "subsets": subset_values,
    }


def _enrich_row_for_split(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = derive_sample_id(row)
    pair_id = derive_pair_id(row, sample_id)
    question = str(row.get("question", "")).strip()
    question_template = normalize_question_template(question)
    source = str(row.get("source", "")).strip().lower() or "unknown"

    payload = dict(row)
    payload["sample_id"] = sample_id
    payload["pair_id"] = pair_id
    payload["source"] = source
    payload["question_template"] = question_template
    return payload


def _resolve_split_dir(split_cfg: dict[str, Any]) -> Path:
    return Path(split_cfg["dir"])


def _build_split_pair_rows(
    units: list[dict[str, Any]],
    unit_split_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: item["unit_id"]):
        unit_id = str(unit["unit_id"])
        split_name = unit_split_map[unit_id]

        source_values = sorted({str(row.get("source", "unknown")) for row in unit["rows"]})
        template_values = sorted({str(row.get("question_template", "unknown")) for row in unit["rows"]})
        sample_ids = [str(row.get("sample_id", "")) for row in unit["rows"]]

        rows.append(
            {
                "pair_id": unit_id,
                "split": split_name,
                "stratum": str(unit.get("stratum", "unknown")),
                "source": source_values[0] if len(source_values) == 1 else "|".join(source_values),
                "question_template": template_values[0]
                if len(template_values) == 1
                else "|".join(template_values),
                "n_samples": len(sample_ids),
                "sample_ids_json": json.dumps(sample_ids, ensure_ascii=True),
            }
        )

    return rows


def _write_split_pairs_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Writing split parquet requires pandas. Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _build_manifest(
    annotation_file: Path,
    kept_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    selection_report: dict[str, Any],
    split_cfg: dict[str, Any],
    split_pair_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_counter = Counter(row["split"] for row in split_pair_rows)
    sample_counter = Counter(row["split"] for row in assignments)

    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    template_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in split_pair_rows:
        split_name = str(row["split"])
        source_counts[split_name][str(row["source"])] += 1
        template_counts[split_name][str(row["question_template"])] += 1

    return {
        "version": 1,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "annotation_file": str(annotation_file),
        "annotation_sha256": _sha256_file(annotation_file),
        "selection_sha256": _sha256_rows(kept_rows),
        "selection": {
            "kept_rows": len(kept_rows),
            "dropped_rows": len(dropped_rows),
            "report": selection_report,
        },
        "split_config": split_cfg,
        "stats": {
            "n_pairs": len(split_pair_rows),
            "n_samples": len(assignments),
            "pairs_by_split": dict(sorted(pair_counter.items())),
            "samples_by_split": dict(sorted(sample_counter.items())),
            "pairs_by_source_split": {
                split: dict(sorted(counter.items())) for split, counter in sorted(source_counts.items())
            },
            "pairs_by_template_split": {
                split: dict(sorted(counter.items())) for split, counter in sorted(template_counts.items())
            },
        },
        "packages": _package_versions(["datasets", "hydra-core", "omegaconf", "pandas", "pyarrow"]),
    }


def _assert_template_coverage(split_pair_rows: list[dict[str, Any]]) -> None:
    required = {"train", "val", "test"}
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in split_pair_rows:
        coverage[str(row["question_template"])].add(str(row["split"]))

    missing_by_template = {
        template: sorted(required - present)
        for template, present in coverage.items()
        if not required.issubset(present)
    }
    if missing_by_template:
        first_template, missing = next(iter(missing_by_template.items()))
        raise InitConfigError(
            "Template coverage check failed. "
            f"Template '{first_template}' missing splits: {missing}. "
            "Disable split.template_coverage_check or adjust split seed/ratios."
        )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
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
