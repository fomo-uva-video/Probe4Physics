from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

import run
from benchmarks.intphys2.features import (
    FeatureCachePaths as IntPhys2FeatureCachePaths,
    has_valid_feature_cache as has_valid_intphys2_feature_cache,
    resolve_expected_feature_cache_paths as resolve_intphys2_feature_cache_paths,
)
from benchmarks.mvp.features import (
    FeatureCachePaths as MVPFeatureCachePaths,
    _feature_cfg as mvp_feature_cfg,
    _find_compatible_feature_cache as find_compatible_mvp_feature_cache,
    _is_valid_cache as is_valid_mvp_feature_cache,
    resolve_expected_feature_cache_paths as resolve_mvp_feature_cache_paths,
)


DEFAULT_OUTPUT = Path("results/seed_runs/seed_status_report.md")
SELECTED_MANIFEST = Path("results/seed_runs/seed_manifest_main_linear_mlp_v1.csv")
SELECTED_LONG = Path("results/seed_runs/seed_results_long.csv")
SELECTED_SUMMARY = Path("results/seed_runs/seed_summary.csv")
LAYERWISE_MANIFEST = Path("results/seed_runs/seed_manifest_layerwise_main_linear_mlp_v1.csv")
LAYERWISE_LONG = Path("results/seed_runs/layerwise_seed_results_long.csv")
LAYERWISE_SUMMARY = Path("results/seed_runs/layerwise_seed_summary.csv")
ATTENTIVE_INTPHYS2_MANIFEST = Path("results/seed_runs/seed_manifest_layerwise_main_attentive_intphys2_v1.csv")
ATTENTIVE_INTPHYS2_LONG = Path("results/seed_runs/layerwise_attentive_intphys2_seed_results_long.csv")
ATTENTIVE_INTPHYS2_SUMMARY = Path("results/seed_runs/layerwise_attentive_intphys2_seed_summary.csv")
SAME_L_LINEAR_MLP_MANIFEST = Path("results/seed_runs/seed_manifest_layerwise_same_l_linear_mlp_v1.csv")
SAME_L_LINEAR_MLP_LONG = Path("results/seed_runs/layerwise_same_l_seed_results_long.csv")
SAME_L_LINEAR_MLP_SUMMARY = Path("results/seed_runs/layerwise_same_l_seed_summary.csv")
SAME_L_ATTENTIVE_MANIFEST = Path("results/seed_runs/seed_manifest_layerwise_same_l_attentive_v1.csv")
SAME_L_ATTENTIVE_LONG = Path("results/seed_runs/layerwise_same_l_attentive_seed_results_long.csv")
SAME_L_ATTENTIVE_SUMMARY = Path("results/seed_runs/layerwise_same_l_attentive_seed_summary.csv")
PILOT_MANIFEST = Path("results/seed_runs/seed_manifest_jepa_v2_mlp_layerwise_v1.csv")
EXPECTED_FINAL_SEEDS = (42, 101, 102)


@dataclass(frozen=True)
class RunSet:
    name: str
    manifest_path: Path
    long_path: Path | None
    summary_path: Path | None
    expected_seeds: tuple[int, ...]
    notes: str


@dataclass(frozen=True)
class FeatureStatus:
    status: str
    cache_dir: str
    signature: str
    note: str


RUN_SETS = [
    RunSet(
        name="Selected-layer main Linear/MLP",
        manifest_path=SELECTED_MANIFEST,
        long_path=SELECTED_LONG,
        summary_path=SELECTED_SUMMARY,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="Paper-facing selected layer configs from verified_best_probe_configs.csv.",
    ),
    RunSet(
        name="Layerwise main Linear/MLP",
        manifest_path=LAYERWISE_MANIFEST,
        long_path=LAYERWISE_LONG,
        summary_path=LAYERWISE_SUMMARY,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="All verified main Linear/MLP layers from verified_layerwise_probe_configs.csv.",
    ),
    RunSet(
        name="Layerwise main IntPhys2 Attentive",
        manifest_path=ATTENTIVE_INTPHYS2_MANIFEST,
        long_path=ATTENTIVE_INTPHYS2_LONG,
        summary_path=ATTENTIVE_INTPHYS2_SUMMARY,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="All verified main IntPhys2 temporal_attn layers from verified_layerwise_probe_configs.csv.",
    ),
    RunSet(
        name="Layerwise Same-L Linear/MLP",
        manifest_path=SAME_L_LINEAR_MLP_MANIFEST,
        long_path=SAME_L_LINEAR_MLP_LONG,
        summary_path=SAME_L_LINEAR_MLP_SUMMARY,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="All verified same_L ViT-L/16 Linear/MLP layers from verified_layerwise_probe_configs.csv.",
    ),
    RunSet(
        name="Layerwise Same-L Attentive",
        manifest_path=SAME_L_ATTENTIVE_MANIFEST,
        long_path=SAME_L_ATTENTIVE_LONG,
        summary_path=SAME_L_ATTENTIVE_SUMMARY,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="All verified same_L ViT-L/16 temporal_attn layers from verified_layerwise_probe_configs.csv.",
    ),
    RunSet(
        name="V-JEPA2 MLP layerwise pilot",
        manifest_path=PILOT_MANIFEST,
        long_path=None,
        summary_path=None,
        expected_seeds=EXPECTED_FINAL_SEEDS,
        notes="Earlier diagnostic pilot; this manifest treats seed 42 as an artifact row, not a source-spreadsheet row.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown status report for seed reruns and feature caches.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2, sort_keys=True))


def build_report() -> str:
    feature_cache: dict[tuple[str, str, str, str, str], FeatureStatus] = {}
    sections: list[str] = []
    sections.append("# Seed Run Status Tracker")
    sections.append("")
    sections.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    sections.append("")
    sections.append("This file tracks fixed-config seed reruns and the feature caches required to run them. Regenerate it with:")
    sections.append("")
    sections.append("```bash")
    sections.append("source jobs/extract/common.sh")
    sections.append("load_probe4physics_env")
    sections.append("python scripts/build_seed_status_report.py")
    sections.append("```")
    sections.append("")

    overview_rows: list[list[str]] = []
    detailed_sections: list[str] = []
    for run_set in RUN_SETS:
        manifest_rows = _read_csv_if_exists(run_set.manifest_path)
        long_rows = _read_csv_if_exists(run_set.long_path) if run_set.long_path else []
        summary_rows = _read_csv_if_exists(run_set.summary_path) if run_set.summary_path else []
        if not manifest_rows:
            overview_rows.append([run_set.name, "missing manifest", "NA", "NA", "NA", "NA"])
            continue

        long_by_config_seed = _long_rows_by_config_seed(long_rows)
        planned_configs = _ordered_config_ids(manifest_rows)
        planned_new_rows = len(manifest_rows)
        summary_by_config = {_value(row.get("config_id")): row for row in summary_rows if _value(row.get("config_id"))}
        complete_configs = _complete_config_count(
            planned_configs,
            summary_by_config,
            long_by_config_seed,
            run_set.expected_seeds,
            manifest_rows=manifest_rows,
        )
        valid_artifact_rows = _valid_artifact_row_count(manifest_rows, long_by_config_seed)
        missing_artifact_rows = _missing_artifact_row_count(manifest_rows, long_by_config_seed)
        mismatch_rows = _artifact_mismatch_row_count(manifest_rows, long_by_config_seed)
        feature_statuses = _feature_statuses_for_rows(manifest_rows, feature_cache)
        feature_ready = sum(1 for item in feature_statuses.values() if _feature_is_ready(item))

        overview_rows.append(
            [
                run_set.name,
                str(len(planned_configs)),
                f"{complete_configs}/{len(planned_configs)}",
                f"{valid_artifact_rows}/{planned_new_rows}",
                f"{missing_artifact_rows} missing, {mismatch_rows} mismatch",
                f"{feature_ready}/{len(feature_statuses)} ready",
            ]
        )
        detailed_sections.append(_run_set_section(run_set, manifest_rows, long_by_config_seed, summary_by_config, feature_cache))

    sections.append("## Overview")
    sections.append("")
    sections.append(_markdown_table(
        ["Run set", "Configs", "Complete configs", "Valid artifact rows", "Artifact gaps", "Feature cache groups"],
        overview_rows,
    ))
    sections.append("")
    sections.extend(detailed_sections)

    sections.append("## Current Next Steps")
    sections.append("")
    sections.append("1. Keep Same-L seed training blocked until the 10 Same-L ViT-L train-val-test feature caches are valid.")
    sections.append("2. Submit Same-L Linear/MLP and Attentive wrappers after cache validation.")
    sections.append("3. Regenerate result CSV/XLSX exports and this status report after Same-L seed jobs finish.")
    sections.append("")
    return "\n".join(sections)



def _feature_is_ready(status: FeatureStatus) -> bool:
    return status.status in {"ready", "ready-compatible"}

def _run_set_section(
    run_set: RunSet,
    manifest_rows: list[dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
    summary_by_config: dict[str, dict[str, str]],
    feature_cache: dict[tuple[str, str, str, str, str], FeatureStatus],
) -> str:
    lines: list[str] = []
    lines.append(f"## {run_set.name}")
    lines.append("")
    lines.append(run_set.notes)
    lines.append("")

    grouped = _group_manifest_rows(manifest_rows)
    rows: list[list[str]] = []
    for key in sorted(grouped):
        items = grouped[key]
        first = items[0]
        config_ids = _ordered_config_ids(items)
        layers = _compact_layers([row.get("layer", "") for row in items])
        seed_counts = _seed_counts(config_ids, long_by_config_seed, items, run_set.expected_seeds)
        complete_configs = _complete_config_count(
            config_ids,
            summary_by_config,
            long_by_config_seed,
            run_set.expected_seeds,
            manifest_rows=items,
        )
        artifact_status = _artifact_status_summary(items, long_by_config_seed)
        feature_status = _feature_status_for_row(first, feature_cache)
        rows.append(
            [
                _value(first.get("dataset")) or _value(first.get("dataset_hydra")),
                _value(first.get("model")),
                _value(first.get("backbone")),
                _value(first.get("probe")) or _value(first.get("probe_hydra")),
                layers,
                f"{complete_configs}/{len(config_ids)}",
                seed_counts,
                artifact_status,
                feature_status.status,
            ]
        )
    lines.append(_markdown_table(
        ["Dataset", "Model", "Backbone", "Probe", "Layers", "Complete configs", "Seeds", "Artifact status", "Feature cache"],
        rows,
    ))
    lines.append("")

    feature_rows: list[list[str]] = []
    for status in sorted(_feature_statuses_for_rows(manifest_rows, feature_cache).values(), key=lambda item: item.cache_dir):
        feature_rows.append([status.status, status.signature, status.cache_dir, status.note])
    lines.append("Feature cache details:")
    lines.append("")
    lines.append(_markdown_table(["Status", "Signature", "Cache dir", "Note"], feature_rows))
    lines.append("")
    return "\n".join(lines)


def _feature_statuses_for_rows(
    rows: list[dict[str, str]],
    feature_cache: dict[tuple[str, str, str, str, str], FeatureStatus],
) -> dict[tuple[str, str, str, str, str], FeatureStatus]:
    statuses: dict[tuple[str, str, str, str, str], FeatureStatus] = {}
    for row in rows:
        key = _feature_key(row)
        statuses[key] = _feature_status_for_row(row, feature_cache)
    return statuses


def _feature_status_for_row(
    row: dict[str, str],
    feature_cache: dict[tuple[str, str, str, str, str], FeatureStatus],
) -> FeatureStatus:
    key = _feature_key(row)
    if key not in feature_cache:
        feature_cache[key] = _check_feature_cache(row)
    return feature_cache[key]


def _feature_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        _value(row.get("dataset_hydra")),
        _value(row.get("backbone_name")),
        _value(row.get("backbone_variant")),
        _value(row.get("feature_view")),
    )


def _check_feature_cache(row: dict[str, str]) -> FeatureStatus:
    dataset = _value(row.get("dataset_hydra"))
    overrides = [
        "split.seed=42",
        f"backbone.name={_value(row.get('backbone_name'))}",
        f"+backbone.kwargs.variant={_value(row.get('backbone_variant'))}",
        f"probe.name={_value(row.get('probe_hydra'))}",
        f"probe.feature_view={_value(row.get('feature_view'))}",
        f"probe.layer={_value(row.get('layer'))}",
    ]
    manifest_overrides = _parse_json_list(_value(row.get("hydra_overrides_json")))
    overrides.extend(
        item for item in manifest_overrides if item.startswith("feature_cache.layer_ids=")
    )
    try:
        cfg = run._compose_config("mvp" if dataset == "mvp" else "intphys2", overrides)
        config = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config, dict):
            raise ValueError(f"resolved config is not a dict: {type(config)!r}")
        if dataset == "mvp":
            paths = resolve_mvp_feature_cache_paths(config)
            if is_valid_mvp_feature_cache(paths):
                return _feature_status_from_paths(paths, valid=True)
            compatible = find_compatible_mvp_feature_cache(
                config,
                exact_paths=paths,
                feature_cfg=mvp_feature_cfg(config),
            )
            if compatible is not None and is_valid_mvp_feature_cache(compatible):
                return FeatureStatus(
                    "ready-compatible",
                    str(compatible.cache_dir),
                    str(compatible.signature),
                    f"compatible cache for expected signature {paths.signature}",
                )
            return _feature_status_from_paths(paths, valid=False)
        elif dataset == "intphys2":
            paths = resolve_intphys2_feature_cache_paths(config)
            valid = has_valid_intphys2_feature_cache(config)
        else:
            return FeatureStatus("unknown", "", "", f"unsupported dataset_hydra={dataset!r}")
        return _feature_status_from_paths(paths, valid=valid)
    except Exception as exc:
        return FeatureStatus("missing", "", "", str(exc))



def _parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]

def _feature_status_from_paths(paths: MVPFeatureCachePaths | IntPhys2FeatureCachePaths, *, valid: bool) -> FeatureStatus:
    status = "ready" if valid else "missing"
    note = "valid cache" if valid else "cache missing or invalid"
    return FeatureStatus(status, str(paths.cache_dir), str(paths.signature), note)


def _group_manifest_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            _value(row.get("dataset")) or _value(row.get("dataset_hydra")),
            _value(row.get("model")),
            _value(row.get("backbone")),
            _value(row.get("probe")) or _value(row.get("probe_hydra")),
        )
        grouped[key].append(row)
    return grouped


def _seed_counts(
    config_ids: list[str],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
    manifest_rows: list[dict[str, str]],
    expected_seeds: tuple[int, ...],
) -> str:
    parts: list[str] = []
    for seed in expected_seeds:
        count = 0
        for config_id in config_ids:
            row = long_by_config_seed.get((config_id, str(seed)))
            if row and row.get("status") == "complete":
                count += 1
            elif not long_by_config_seed and _manifest_artifact_exists(config_id, str(seed), manifest_rows):
                count += 1
        parts.append(f"{seed}:{count}/{len(config_ids)}")
    return ", ".join(parts)


def _artifact_status_summary(
    manifest_rows: list[dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
) -> str:
    if not long_by_config_seed:
        statuses = Counter("complete" if _row_artifact_exists(row) else "missing_artifact" for row in manifest_rows)
    else:
        statuses = Counter()
        for row in manifest_rows:
            config_id = _value(row.get("config_id"))
            seed = _value(row.get("seed"))
            status = long_by_config_seed.get((config_id, seed), {}).get("status", "not_collected")
            statuses[status] += 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(statuses.items()))


def _valid_artifact_row_count(
    manifest_rows: list[dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
) -> int:
    if not long_by_config_seed:
        return sum(1 for row in manifest_rows if _row_artifact_exists(row))
    return sum(
        1
        for row in manifest_rows
        if long_by_config_seed.get((_value(row.get("config_id")), _value(row.get("seed"))), {}).get("status") == "complete"
    )


def _missing_artifact_row_count(
    manifest_rows: list[dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
) -> int:
    if not long_by_config_seed:
        return sum(1 for row in manifest_rows if not _row_artifact_exists(row))
    return sum(
        1
        for row in manifest_rows
        if long_by_config_seed.get((_value(row.get("config_id")), _value(row.get("seed"))), {}).get("status")
        == "missing_artifact"
    )


def _artifact_mismatch_row_count(
    manifest_rows: list[dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
) -> int:
    if not long_by_config_seed:
        return 0
    return sum(
        1
        for row in manifest_rows
        if long_by_config_seed.get((_value(row.get("config_id")), _value(row.get("seed"))), {}).get("status")
        == "artifact_mismatch"
    )


def _complete_config_count(
    config_ids: list[str],
    summary_by_config: dict[str, dict[str, str]],
    long_by_config_seed: dict[tuple[str, str], dict[str, str]],
    expected_seeds: tuple[int, ...],
    *,
    manifest_rows: list[dict[str, str]],
) -> int:
    if summary_by_config:
        return sum(1 for config_id in config_ids if summary_by_config.get(config_id, {}).get("complete") == "true")
    count = 0
    for config_id in config_ids:
        if long_by_config_seed:
            complete = all(
                long_by_config_seed.get((config_id, str(seed)), {}).get("status") == "complete"
                for seed in expected_seeds
            )
        else:
            complete = all(_manifest_artifact_exists(config_id, str(seed), manifest_rows) for seed in expected_seeds)
        if complete:
            count += 1
    return count


def _manifest_artifact_exists(config_id: str, seed: str, rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if _value(row.get("config_id")) == config_id and _value(row.get("seed")) == seed:
            return _row_artifact_exists(row)
    return False


def _row_artifact_exists(row: dict[str, str]) -> bool:
    output_dir = _value(row.get("probe_output_dir"))
    output_subdir = _value(row.get("probe_output_subdir"))
    return bool(output_dir and output_subdir and (Path(output_dir) / output_subdir / "train_eval_summary.json").exists())


def _long_rows_by_config_seed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config_id = _value(row.get("config_id"))
        seed = _value(row.get("seed"))
        if config_id and seed:
            result[(config_id, seed)] = row
    return result


def _ordered_config_ids(rows: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        config_id = _value(row.get("config_id"))
        if config_id and config_id not in seen:
            ordered.append(config_id)
            seen.add(config_id)
    return ordered


def _compact_layers(raw_layers: list[str]) -> str:
    layers = sorted({int(layer) for layer in raw_layers if _value(layer).lstrip("-").isdigit()})
    if not layers:
        return "NA"
    return ",".join(str(layer) for layer in layers)


def _read_csv_if_exists(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    escaped_headers = [_escape_cell(item) for item in headers]
    lines = ["| " + " | ".join(escaped_headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    if not rows:
        lines.append("| " + " | ".join("NA" for _ in headers) + " |")
        return "\n".join(lines)
    for row in rows:
        padded = [*row, *[""] * (len(headers) - len(row))]
        lines.append("| " + " | ".join(_escape_cell(item) for item in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _escape_cell(raw: Any) -> str:
    text = str(raw).replace("\n", " ").strip()
    return text.replace("|", "\\|") or "NA"


def _value(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text in {"", "NULL", "null", "None", "none", "NA", "N/A"} else text


if __name__ == "__main__":
    main()
