from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from benchmarks.intphys2.eval import run_intphys2_eval
from benchmarks.mvp.eval import run_mvp_eval
from models.cache_metadata import resolve_backbone_layer_label


LABEL_MODES: tuple[tuple[str, int | None], ...] = (
    ("original", None),
    ("all_true", 1),
    ("all_false", 0),
)
DEFAULT_BASELINE_RESULT_ROOTS = (
    "artifacts/results/mvp_single_frame",
    "artifacts/results/mvp_displacement",
    "artifacts/results/intphys2_single_frame",
    "artifacts/results/intphys2_displacement",
)
BASELINE_CSV_NAME = "baseline_label_metrics.csv"
BASELINE_AGGREGATE_CSV_NAME = "baseline_label_metrics_all.csv"
_KNOWN_BACKBONES = (
    "jepa_v2_1",
    "videomae_v2",
    "ltx_video",
    "jepa_v1",
    "jepa_v2",
    "videomae",
)
_CSV_FIELDNAMES = [
    "dataset",
    "baseline",
    "backbone",
    "backbone_variant",
    "probe",
    "run_id",
    "layer",
    "layer_label",
    "original_accuracy",
    "original_pair_consistency",
    "original_voe_accuracy",
    "original_roc_auc",
    "all_true_accuracy",
    "all_true_pair_consistency",
    "all_true_voe_accuracy",
    "all_false_accuracy",
    "all_false_pair_consistency",
    "all_false_voe_accuracy",
    "prediction_file",
    "summary_path",
]


class BaselinePostprocessError(ValueError):
    pass


def run_baseline_label_backfill(config: dict[str, Any]) -> dict[str, Any]:
    """Backfill original/all-true/all-false baseline metrics from saved predictions."""

    cfg = config.get("baseline_backfill", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise BaselinePostprocessError("baseline_backfill must be a dictionary")

    roots = cfg.get("roots", DEFAULT_BASELINE_RESULT_ROOTS)
    if isinstance(roots, (str, Path)):
        root_paths = [Path(str(roots))]
    elif isinstance(roots, (list, tuple)):
        root_paths = [Path(str(item)) for item in roots]
    else:
        raise BaselinePostprocessError("baseline_backfill.roots must be a path or list of paths")

    force = bool(cfg.get("force", False))
    processed_roots: dict[str, Any] = {}
    totals = {
        "roots_seen": 0,
        "roots_existing": 0,
        "summaries_seen": 0,
        "summaries_updated": 0,
        "summaries_skipped": 0,
    }

    for root in root_paths:
        totals["roots_seen"] += 1
        if not root.exists():
            processed_roots[str(root)] = {"exists": False, "reason": "missing"}
            continue
        if not root.is_dir():
            processed_roots[str(root)] = {"exists": False, "reason": "not_a_directory"}
            continue

        totals["roots_existing"] += 1
        root_report = _process_baseline_root(root, force=force)
        processed_roots[str(root)] = root_report
        totals["summaries_seen"] += int(root_report.get("summaries_seen", 0))
        totals["summaries_updated"] += int(root_report.get("summaries_updated", 0))
        totals["summaries_skipped"] += int(root_report.get("summaries_skipped", 0))

    return {
        "baseline_backfill": {
            "force": force,
            "roots": processed_roots,
            "totals": totals,
        }
    }


def write_baseline_metric_csvs_for_run(run_dir: Path) -> dict[str, Any]:
    rows = _rows_for_run(run_dir)
    output_path = run_dir / BASELINE_CSV_NAME
    _write_csv(output_path, rows)
    return {"run_dir": str(run_dir), "csv": str(output_path), "rows": len(rows)}


def write_baseline_metric_csvs_for_root(root: Path) -> dict[str, Any]:
    run_dirs = _baseline_run_dirs(root)
    rows: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_report = write_baseline_metric_csvs_for_run(run_dir)
        run_reports.append(run_report)
        rows.extend(_rows_for_run(run_dir))

    output_path = root / BASELINE_AGGREGATE_CSV_NAME
    _write_csv(output_path, rows)
    return {
        "root": str(root),
        "csv": str(output_path),
        "rows": len(rows),
        "runs": run_reports,
    }


def _process_baseline_root(root: Path, *, force: bool) -> dict[str, Any]:
    summary_paths = _baseline_summary_paths(root)
    updated = 0
    skipped = 0
    summary_reports: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        try:
            result = _process_baseline_summary(summary_path, force=force)
        except Exception as exc:
            skipped += 1
            summary_reports.append(
                {
                    "summary_path": str(summary_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if bool(result.get("updated", False)):
            updated += 1
        else:
            skipped += 1
        summary_reports.append(result)

    csv_report = write_baseline_metric_csvs_for_root(root)
    report = {
        "exists": True,
        "summaries_seen": len(summary_paths),
        "summaries_updated": updated,
        "summaries_skipped": skipped,
        "csv": csv_report,
        "summaries": summary_reports,
    }
    (root / "baseline_label_backfill_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _process_baseline_summary(summary_path: Path, *, force: bool) -> dict[str, Any]:
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise BaselinePostprocessError(f"Summary is not a JSON object: {summary_path}")

    prediction_file = Path(str(summary.get("prediction_file", "")))
    if not prediction_file.exists():
        raise BaselinePostprocessError(f"Prediction file not found: {prediction_file}")

    dataset = _summary_dataset(summary, summary_path)
    baseline_name = _baseline_name(dataset, summary, summary_path)
    alias_key = f"{baseline_name}_evals"
    eval_runner = _eval_runner_for_dataset(dataset)
    summary_dir = summary_path.parent
    config_snapshot = _find_config_snapshot(summary, summary_dir)
    if config_snapshot is None:
        raise BaselinePostprocessError(
            f"No eval config snapshot found for summary: {summary_path}"
        )
    base_config = _load_yaml_config(config_snapshot)

    scenario_summaries = _existing_scenario_summaries(summary, alias_key)
    metrics_by_label_mode = _existing_metrics_by_label_mode(summary)
    changed = False

    for label_mode, label_value in LABEL_MODES:
        scenario_dir = summary_dir / f"{baseline_name}_{label_mode}"
        existing_summary = scenario_summaries.get(label_mode, {})
        existing_metrics = metrics_by_label_mode.get(label_mode, {})
        existing_metrics_path = scenario_dir / "metrics.json"
        if (
            not force
            and isinstance(existing_metrics, dict)
            and existing_metrics
            and existing_metrics_path.exists()
        ):
            metrics = existing_metrics
            result = existing_summary.get("base_eval", {})
        else:
            eval_cfg = _config_for_label_mode(
                base_config,
                prediction_file=prediction_file,
                scenario_dir=scenario_dir,
                label_value=label_value,
            )
            result = eval_runner(eval_cfg)
            metrics = result.get("metrics", {})
            if not isinstance(metrics, dict):
                raise BaselinePostprocessError(
                    f"Eval metrics for label_mode='{label_mode}' must be a dictionary."
                )
            changed = True

        scenario_summaries[label_mode] = {
            "label_mode": label_mode,
            "label_value": None if label_value is None else int(label_value),
            "probe_eval_dir": str(scenario_dir),
            "metrics": metrics,
            "base_eval": result,
        }
        metrics_by_label_mode[label_mode] = metrics

    primary_metrics = metrics_by_label_mode["original"]
    metric_name = str(
        summary.get("objective_metric_name")
        or ("pair_consistency" if dataset.startswith("mvp") else "voe_accuracy")
    )
    summary.update(
        {
            "dataset": dataset,
            "baseline_name": baseline_name,
            "primary_label_mode": "original",
            "label_modes": [label_mode for label_mode, _ in LABEL_MODES],
            "objective_metric_name": metric_name,
            "objective_metric": float(primary_metrics.get(metric_name, 0.0) or 0.0),
            "metrics": primary_metrics,
            "metrics_by_label_mode": metrics_by_label_mode,
            "baseline_evals": scenario_summaries,
            alias_key: scenario_summaries,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "summary_path": str(summary_path),
        "status": "updated" if changed else "already_complete",
        "updated": changed,
        "dataset": dataset,
        "baseline": baseline_name,
        "label_modes": list(metrics_by_label_mode),
    }


def _config_for_label_mode(
    config: dict[str, Any],
    *,
    prediction_file: Path,
    scenario_dir: Path,
    label_value: int | None,
) -> dict[str, Any]:
    eval_cfg = dict(config)
    if label_value is None:
        eval_cfg.pop("label_override", None)
    else:
        eval_cfg["label_override"] = int(label_value)
    eval_cfg["predictor"] = {
        "mode": "from_file",
        "prediction_file": str(prediction_file),
    }
    eval_cfg["output_dir"] = str(scenario_dir.parent)
    eval_cfg["output_subdir"] = scenario_dir.name
    return eval_cfg


def _find_config_snapshot(summary: dict[str, Any], summary_dir: Path) -> Path | None:
    candidate_dirs: list[Path] = []
    for value in _iter_scenario_dicts(summary):
        if not isinstance(value, dict):
            continue
        raw_dir = str(value.get("probe_eval_dir", "")).strip()
        if raw_dir:
            candidate_dirs.append(Path(raw_dir))
        base_eval = value.get("base_eval", {})
        if isinstance(base_eval, dict):
            raw_output_dir = str(base_eval.get("output_dir", "")).strip()
            if raw_output_dir:
                candidate_dirs.append(Path(raw_output_dir))

    base_eval = summary.get("base_eval", {})
    if isinstance(base_eval, dict):
        raw_output_dir = str(base_eval.get("output_dir", "")).strip()
        if raw_output_dir:
            candidate_dirs.append(Path(raw_output_dir))
    candidate_dirs.append(summary_dir)

    for directory in candidate_dirs:
        candidate = directory / "run_config.snapshot.yaml"
        if candidate.exists():
            return candidate
    return None


def _load_yaml_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise BaselinePostprocessError(f"Config snapshot is not a dictionary: {path}")
    return dict(payload)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_scenario_summaries(summary: dict[str, Any], alias_key: str) -> dict[str, dict[str, Any]]:
    candidates = (
        summary.get("baseline_evals"),
        summary.get(alias_key),
        summary.get("single_frame_evals"),
        summary.get("displacement_evals"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {
                str(key): value
                for key, value in candidate.items()
                if isinstance(value, dict)
            }
    return {}


def _existing_metrics_by_label_mode(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = summary.get("metrics_by_label_mode", {})
    if isinstance(raw, dict):
        return {
            str(key): value
            for key, value in raw.items()
            if isinstance(value, dict)
        }
    return {}


def _iter_scenario_dicts(summary: dict[str, Any]):
    for key in ("baseline_evals", "single_frame_evals", "displacement_evals"):
        value = summary.get(key)
        if isinstance(value, dict):
            yield from value.values()


def _eval_runner_for_dataset(dataset: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if dataset.startswith("mvp"):
        return run_mvp_eval
    if dataset.startswith("intphys2"):
        return run_intphys2_eval
    raise BaselinePostprocessError(f"Unsupported baseline dataset: {dataset}")


def _summary_dataset(summary: dict[str, Any], summary_path: Path) -> str:
    raw = str(summary.get("dataset", "")).strip()
    if raw:
        return raw
    root_name = _baseline_root_for_summary(summary_path).name
    if root_name in {
        "mvp_single_frame",
        "mvp_displacement",
        "intphys2_single_frame",
        "intphys2_displacement",
    }:
        return root_name
    raise BaselinePostprocessError(f"Cannot infer baseline dataset for {summary_path}")


def _baseline_name(dataset: str, summary: dict[str, Any], summary_path: Path) -> str:
    raw = str(summary.get("baseline_name", "")).strip()
    if raw:
        return raw
    if dataset.endswith("_single_frame"):
        return "single_frame"
    if dataset.endswith("_displacement"):
        return "displacement"
    root_name = _baseline_root_for_summary(summary_path).name
    if root_name.endswith("_single_frame"):
        return "single_frame"
    if root_name.endswith("_displacement"):
        return "displacement"
    raise BaselinePostprocessError(f"Cannot infer baseline name for {summary_path}")


def _baseline_root_for_summary(summary_path: Path) -> Path:
    path = summary_path.resolve()
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "artifacts" and idx + 2 < len(parts) and parts[idx + 1] == "results":
            return Path(*parts[: idx + 3])
    return summary_path.parents[2] if len(summary_path.parents) >= 3 else summary_path.parent


def _baseline_summary_paths(root: Path) -> list[Path]:
    paths = set(root.glob("*/layer_*/probe_eval_summary.json"))
    paths.update(root.glob("*/probe_eval_summary.json"))
    paths.discard(root / "probe_eval_summary.json")
    return sorted(paths)


def _baseline_run_dirs(root: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for summary_path in _baseline_summary_paths(root):
        parent = summary_path.parent
        if parent.name.startswith("layer_"):
            run_dirs.add(parent.parent)
        else:
            run_dirs.add(parent)
    return sorted(run_dirs)


def _rows_for_run(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    summary_paths = set(run_dir.glob("layer_*/probe_eval_summary.json"))
    summary_paths.update(run_dir.glob("*/probe_eval_summary.json"))
    if (run_dir / "probe_eval_summary.json").exists():
        summary_paths.add(run_dir / "probe_eval_summary.json")

    for summary_path in sorted(summary_paths):
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        rows.append(_csv_row_for_summary(run_dir, summary_path, summary))
    return rows


def _csv_row_for_summary(run_dir: Path, summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    dataset = _summary_dataset(summary, summary_path)
    baseline = _baseline_name(dataset, summary, summary_path)
    probe = str(summary.get("probe_name", "")).strip()
    run_info = _parse_run_dir(run_dir.name, dataset=dataset, probe=probe)
    layer_label = _layer_label_for_summary(summary_path, summary)
    layer = _layer_slot_for_label(run_info["backbone"], run_info["backbone_variant"], layer_label)
    metrics_by_label_mode = _existing_metrics_by_label_mode(summary)
    if not metrics_by_label_mode and isinstance(summary.get("metrics"), dict):
        metrics_by_label_mode = {"original": summary["metrics"]}

    row = {
        "dataset": dataset,
        "baseline": baseline,
        "backbone": run_info["backbone"],
        "backbone_variant": run_info["backbone_variant"],
        "probe": probe,
        "run_id": run_info["run_id"],
        "layer": layer,
        "layer_label": layer_label,
        "prediction_file": str(summary.get("prediction_file", "")),
        "summary_path": str(summary_path),
    }
    for label_mode in ("original", "all_true", "all_false"):
        metrics = metrics_by_label_mode.get(label_mode, {})
        row[f"{label_mode}_accuracy"] = _metric_value(metrics, "accuracy")
        row[f"{label_mode}_pair_consistency"] = _metric_value(metrics, "pair_consistency")
        row[f"{label_mode}_voe_accuracy"] = _metric_value(metrics, "voe_accuracy")
    row["original_roc_auc"] = _metric_value(metrics_by_label_mode.get("original", {}), "roc_auc")
    return row


def _parse_run_dir(run_name: str, *, dataset: str, probe: str) -> dict[str, str]:
    prefix = f"{dataset}_"
    remaining = run_name[len(prefix) :] if run_name.startswith(prefix) else run_name

    run_id = ""
    tail = remaining.rsplit("_", 1)
    if len(tail) == 2 and tail[1].isdigit():
        remaining, run_id = tail

    probe_suffix = f"_{probe}"
    if probe and remaining.endswith(probe_suffix):
        remaining = remaining[: -len(probe_suffix)]

    backbone = remaining
    variant = ""
    for candidate in _KNOWN_BACKBONES:
        if remaining == candidate:
            backbone = candidate
            break
        candidate_prefix = f"{candidate}_"
        if remaining.startswith(candidate_prefix):
            backbone = candidate
            variant = remaining[len(candidate_prefix) :]
            break

    return {"backbone": backbone, "backbone_variant": variant, "run_id": run_id}


def _layer_label_for_summary(summary_path: Path, summary: dict[str, Any]) -> str:
    raw = summary.get("layer_label")
    if raw is not None:
        return str(raw)
    parent_name = summary_path.parent.name
    if parent_name.startswith("layer_"):
        return parent_name[len("layer_") :]
    raw_layer = summary.get("layer")
    if raw_layer is not None:
        return str(raw_layer)
    return ""


def _layer_slot_for_label(backbone: str, variant: str, layer_label: str) -> str:
    if layer_label.isdigit():
        return layer_label
    if backbone != "ltx_video" or not layer_label:
        return ""

    kwargs = {"variant": variant} if variant else {}
    for slot in range(1, 257):
        if resolve_backbone_layer_label(backbone, kwargs, slot) == layer_label:
            return str(slot)
    return ""


def _metric_value(metrics: Any, key: str) -> Any:
    if not isinstance(metrics, dict):
        return ""
    value = metrics.get(key, "")
    return "" if value is None else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
