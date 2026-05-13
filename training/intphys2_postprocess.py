from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.intphys2.core import compute_binary_roc_auc
from training.run_probe import _write_train_eval_summary_csv


def run_intphys2_backfill_roc_auc(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    postprocess_cfg = config.get("postprocess", {})
    if not isinstance(postprocess_cfg, dict):
        postprocess_cfg = {}

    intphys_root = Path(
        postprocess_cfg.get(
            "artifacts_root",
            repo_root / "artifacts" / "probes" / "intphys2",
        )
    )
    best_summary_csvs = _path_list(
        postprocess_cfg.get("best_summary_csvs"),
        default=[intphys_root / "intphys2_optuna_best_summary.csv"],
    )
    copied_summary_csvs = _path_list(
        postprocess_cfg.get("copied_summary_csvs"),
        default=[repo_root / "docs" / "train_eval_summary.csv"],
    )

    eval_prediction_files = sorted(intphys_root.glob("**/predictions.csv"))
    updated_eval_dirs = 0
    skipped_eval_dirs = 0
    for predictions_csv in eval_prediction_files:
        if _backfill_probe_eval_dir(predictions_csv.parent):
            updated_eval_dirs += 1
        else:
            skipped_eval_dirs += 1

    train_eval_summary_jsons = sorted(intphys_root.glob("**/train_eval_summary.json"))
    updated_train_eval_summaries = 0
    source_summary_csvs: list[Path] = []
    for summary_json in train_eval_summary_jsons:
        if _backfill_train_eval_summary(summary_json):
            updated_train_eval_summaries += 1
        source_summary_csvs.append(summary_json.with_suffix(".csv"))

    updated_best_summary_csvs = 0
    for best_summary_csv in best_summary_csvs:
        if best_summary_csv.exists() and _backfill_best_summary_csv(best_summary_csv):
            updated_best_summary_csvs += 1

    updated_copied_summary_csvs = 0
    for copied_summary_csv in copied_summary_csvs:
        if copied_summary_csv.exists() and _backfill_copied_summary_csv(
            copied_summary_csv,
            source_summary_csvs,
        ):
            updated_copied_summary_csvs += 1

    return {
        "artifacts_root": str(intphys_root),
        "updated_eval_dirs": updated_eval_dirs,
        "skipped_eval_dirs": skipped_eval_dirs,
        "updated_train_eval_summaries": updated_train_eval_summaries,
        "updated_best_summary_csvs": updated_best_summary_csvs,
        "updated_copied_summary_csvs": updated_copied_summary_csvs,
        "eval_prediction_files": len(eval_prediction_files),
        "train_eval_summary_jsons": len(train_eval_summary_jsons),
    }


def _path_list(raw: Any, *, default: list[Path]) -> list[Path]:
    if raw is None:
        return list(default)
    if isinstance(raw, (str, Path)):
        return [Path(raw)]
    if isinstance(raw, list):
        return [Path(item) for item in raw]
    return list(default)


def _compute_roc_auc_from_predictions_csv(path: Path) -> float | None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        labels: list[int] = []
        scores: list[float] = []
        for row in reader:
            labels.append(int(row["plausibility"]))
            score_raw = str(row.get("score", "")).strip()
            if score_raw:
                scores.append(float(score_raw))
            else:
                pred_raw = str(row.get("pred_idx", "")).strip()
                if not pred_raw:
                    return None
                scores.append(float(int(pred_raw)))
    return compute_binary_roc_auc(labels, scores)


def _backfill_probe_eval_dir(probe_eval_dir: Path) -> bool:
    predictions_csv = probe_eval_dir / "predictions.csv"
    if not predictions_csv.exists():
        return False

    roc_auc = _compute_roc_auc_from_predictions_csv(predictions_csv)
    if roc_auc is None:
        return False

    changed = False
    metrics_path = probe_eval_dir / "metrics.json"
    if metrics_path.exists():
        metrics_payload = _read_json_dict(metrics_path)
        if _set_roc_auc(metrics_payload, roc_auc):
            metrics_path.write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            changed = True

    summary_path = probe_eval_dir / "probe_eval_summary.json"
    if summary_path.exists():
        summary_payload = _read_json_dict(summary_path)
        summary_changed = False
        summary_changed |= _set_roc_auc(summary_payload.get("metrics"), roc_auc)
        summary_changed |= _set_nested_metric(summary_payload.get("base_eval"), roc_auc)
        if summary_changed:
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            changed = True

    return changed


def _backfill_train_eval_summary(summary_path: Path) -> bool:
    summary = _read_json_dict(summary_path)
    top_level_split = str(summary.get("split_name", "")).strip()
    changed = False

    layers = summary.get("layers", [])
    if not isinstance(layers, list):
        return False

    for layer_summary in layers:
        if not isinstance(layer_summary, dict):
            continue
        eval_summary = layer_summary.get("eval")
        if not isinstance(eval_summary, dict):
            continue
        layer_label = str(layer_summary.get("layer_label", "")).strip()

        split_eval_dirs = eval_summary.get("split_eval_dirs", {})
        if not isinstance(split_eval_dirs, dict):
            split_eval_dirs = {}
        split_evals = eval_summary.get("split_evals", {})
        if not isinstance(split_evals, dict):
            split_evals = {}
        metrics_by_split = eval_summary.get("metrics_by_split", {})
        if not isinstance(metrics_by_split, dict):
            metrics_by_split = {}
            eval_summary["metrics_by_split"] = metrics_by_split

        selected_split = str(eval_summary.get("split_name", "")).strip() or top_level_split

        for split_name in _ordered_split_names(split_eval_dirs, split_evals, metrics_by_split):
            probe_eval_dir = _resolve_split_eval_dir(
                split_name,
                split_eval_dirs,
                split_evals,
                summary_root=summary_path.parent,
                layer_label=layer_label,
            )
            if probe_eval_dir is None:
                continue
            predictions_csv = probe_eval_dir / "predictions.csv"
            if not predictions_csv.exists():
                continue
            roc_auc = _compute_roc_auc_from_predictions_csv(predictions_csv)
            if roc_auc is None:
                continue

            changed |= _backfill_probe_eval_dir(probe_eval_dir)

            split_metrics = metrics_by_split.get(split_name)
            if not isinstance(split_metrics, dict):
                split_metrics = {}
                metrics_by_split[split_name] = split_metrics
            changed |= _set_roc_auc(split_metrics, roc_auc)

            split_eval_summary = split_evals.get(split_name)
            if isinstance(split_eval_summary, dict):
                changed |= _set_roc_auc(split_eval_summary.get("metrics"), roc_auc)
                changed |= _set_nested_metric(split_eval_summary.get("base_eval"), roc_auc)

            if split_name == selected_split:
                changed |= _set_roc_auc(eval_summary.get("metrics"), roc_auc)
                changed |= _set_nested_metric(eval_summary.get("base_eval"), roc_auc)

    if changed:
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    csv_path = summary_path.with_suffix(".csv")
    previous_csv = csv_path.read_text(encoding="utf-8") if csv_path.exists() else None
    _write_train_eval_summary_csv(csv_path, summary)
    current_csv = csv_path.read_text(encoding="utf-8")
    csv_changed = previous_csv != current_csv
    return changed or csv_changed


def _ordered_split_names(
    split_eval_dirs: dict[str, Any],
    split_evals: dict[str, Any],
    metrics_by_split: dict[str, Any],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for source in (split_eval_dirs, split_evals, metrics_by_split):
        for key in source.keys():
            split_name = str(key)
            if split_name in seen:
                continue
            seen.add(split_name)
            ordered.append(split_name)
    return ordered


def _resolve_split_eval_dir(
    split_name: str,
    split_eval_dirs: dict[str, Any],
    split_evals: dict[str, Any],
    *,
    summary_root: Path,
    layer_label: str,
) -> Path | None:
    raw_dir = split_eval_dirs.get(split_name)
    if raw_dir:
        candidate = Path(str(raw_dir))
        if candidate.exists():
            return candidate

    split_eval_summary = split_evals.get(split_name)
    if isinstance(split_eval_summary, dict):
        probe_eval_dir = split_eval_summary.get("probe_eval_dir")
        if probe_eval_dir:
            candidate = Path(str(probe_eval_dir))
            if candidate.exists():
                return candidate

    if layer_label:
        fallback = summary_root / f"layer_{layer_label}" / "eval" / split_name
        if fallback.exists():
            return fallback
    return None


def _backfill_best_summary_csv(path: Path) -> bool:
    rows, fieldnames = _read_csv_rows(path)
    if not rows:
        return False

    new_columns = ["train_roc_auc", "val_roc_auc", "test_roc_auc"]
    output_fieldnames = list(fieldnames)
    for column in new_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    changed = False
    for row in rows:
        train_dir_raw = str(row.get("train_dir", "")).strip()
        if not train_dir_raw:
            continue
        train_dir = Path(train_dir_raw)
        layer_dir = train_dir.parent
        for split_name in ("train", "val", "test"):
            predictions_csv = layer_dir / "eval" / split_name / "predictions.csv"
            if not predictions_csv.exists():
                continue
            roc_auc = _compute_roc_auc_from_predictions_csv(predictions_csv)
            if roc_auc is None:
                continue
            changed |= _set_csv_value(row, f"{split_name}_roc_auc", roc_auc)

    if changed or output_fieldnames != fieldnames:
        _write_csv_rows(path, output_fieldnames, rows)
        return True
    return False


def _backfill_copied_summary_csv(path: Path, source_summary_csvs: list[Path]) -> bool:
    rows, fieldnames = _read_csv_rows(path)
    if not rows:
        return False

    matches: list[tuple[list[str], list[dict[str, Any]]]] = []
    for source_csv in source_summary_csvs:
        if not source_csv.exists():
            continue
        source_rows, source_fieldnames = _read_csv_rows(source_csv)
        if len(source_rows) != len(rows):
            continue
        projected_source_rows = [
            {key: source_row.get(key, "") for key in fieldnames}
            for source_row in source_rows
        ]
        if projected_source_rows == rows:
            matches.append((source_fieldnames, source_rows))

    if len(matches) != 1:
        return False

    source_fieldnames, source_rows = matches[0]
    if source_fieldnames == fieldnames and source_rows == rows:
        return False
    _write_csv_rows(path, source_fieldnames, source_rows)
    return True


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(payload)!r}.")
    return payload


def _set_roc_auc(metrics: Any, roc_auc: float) -> bool:
    if not isinstance(metrics, dict):
        return False
    current = metrics.get("roc_auc")
    if current == roc_auc:
        return False
    metrics["roc_auc"] = roc_auc
    return True


def _set_nested_metric(base_eval: Any, roc_auc: float) -> bool:
    if not isinstance(base_eval, dict):
        return False
    metrics = base_eval.get("metrics")
    return _set_roc_auc(metrics, roc_auc)


def _set_csv_value(row: dict[str, Any], key: str, value: float) -> bool:
    current = row.get(key, "")
    new_value = str(value)
    if current == new_value:
        return False
    row[key] = new_value
    return True


def _read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
