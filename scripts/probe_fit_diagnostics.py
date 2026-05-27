#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("mvp", "intphys2")
PROBES = ("linear", "mlp", "temporal_attn")
OBJECTIVE_METRICS = {
    "mvp": "pair_consistency",
    "intphys2": "voe_accuracy",
}
RESULT_ROOTS = (
    Path("artifacts/results"),
    Path("artifacts/results/intphys2"),
)
DEFAULT_PROBE_ROOTS = (
    Path("artifacts/probes"),
)


@dataclass(frozen=True)
class SummaryRecord:
    path: Path
    dataset: str
    probe: str
    row: dict[str, Any]


EvalIndex = dict[tuple[str, str, str], dict[str, Any]]
BestCheckpointIndex = set[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val fit diagnostics for Probe4Physics probe runs."
    )
    parser.add_argument(
        "--probe-root",
        action="append",
        type=Path,
        default=None,
        help="Probe artifact root to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/probe_fit_diagnostics"),
        help="Directory where the six CSV files will be written.",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=5,
        help="Number of trailing epochs used for simple trend diagnostics.",
    )
    args = parser.parse_args()

    roots = args.probe_root or next_default_roots()
    if not roots:
        raise SystemExit("No probe roots found. Pass --probe-root explicitly.")

    eval_index, best_checkpoint_index = build_indexes(roots)
    records = collect_records(
        roots,
        tail_window=max(2, int(args.tail_window)),
        eval_index=eval_index,
        best_checkpoint_index=best_checkpoint_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for probe in PROBES:
            rows = [r.row for r in records if r.dataset == dataset and r.probe == probe]
            rows.sort(
                key=lambda row: (
                    str(row.get("run_name", "")),
                    _sort_layer(row.get("layer")),
                    str(row.get("lr", "")),
                    str(row.get("summary_path", "")),
                )
            )
            out_path = args.output_dir / f"{dataset}_{probe}_fit_diagnostics.csv"
            write_csv(out_path, rows)
            compact_path = args.output_dir / f"{dataset}_{probe}_fit_diagnostics_compact.csv"
            write_best_config_compact_csv(compact_path, rows)
            print(f"{out_path}: {len(rows)} rows")


def next_default_roots() -> list[Path]:
    for root in DEFAULT_PROBE_ROOTS:
        if root.exists():
            return [root]
    return []


def collect_records(
    roots: list[Path],
    *,
    tail_window: int,
    eval_index: EvalIndex,
    best_checkpoint_index: BestCheckpointIndex,
) -> list[SummaryRecord]:
    seen: set[Path] = set()
    records: list[SummaryRecord] = []
    for root in roots:
        for dataset in DATASETS:
            dataset_root = root / dataset
            if not dataset_root.exists():
                continue
            for path in iter_candidate_train_summaries(dataset_root, dataset):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                record = parse_train_summary(
                    path,
                    tail_window=tail_window,
                    eval_index=eval_index,
                    best_checkpoint_index=best_checkpoint_index,
                )
                if record is not None:
                    records.append(record)
    return records


def iter_candidate_train_summaries(dataset_root: Path, dataset: str) -> list[Path]:
    summaries: list[Path] = []
    for run_dir in iter_candidate_run_dirs(dataset_root, dataset):
        summaries.extend(run_dir.rglob("train_summary.json"))
    return summaries


def iter_candidate_run_dirs(dataset_root: Path, dataset: str) -> list[Path]:
    run_dirs: set[Path] = set()
    patterns: list[str] = []
    for probe in PROBES:
        patterns.extend(
            [
                f"{dataset}_probe_{probe}_*",
                f"*_{probe}_*",
                f"old*/{dataset}_probe_{probe}_*",
                f"old*/*_{probe}_*",
            ]
        )
    for pattern in patterns:
        for path in dataset_root.glob(pattern):
            if path.is_dir():
                run_dirs.add(path)
    return sorted(run_dirs)


def build_indexes(roots: list[Path]) -> tuple[EvalIndex, BestCheckpointIndex]:
    index: EvalIndex = {}
    best_checkpoints: BestCheckpointIndex = set()
    for root in roots:
        for dataset in DATASETS:
            dataset_root = root / dataset
            if not dataset_root.exists():
                continue
            for run_dir in iter_candidate_run_dirs(dataset_root, dataset):
                for summary_path in run_dir.rglob("train_eval_summary.json"):
                    add_train_eval_summary_to_indexes(
                        summary_path,
                        index,
                        best_checkpoints,
                    )
    return index, best_checkpoints


def add_train_eval_summary_to_indexes(
    summary_path: Path,
    index: EvalIndex,
    best_checkpoints: BestCheckpointIndex,
) -> None:
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    layers = summary.get("layers")
    if not isinstance(layers, list):
        return
    for layer_summary in layers:
        if not isinstance(layer_summary, dict):
            continue
        checkpoint = str(layer_summary.get("checkpoint", "")).strip()
        eval_summary = layer_summary.get("eval")
        if not checkpoint or not isinstance(eval_summary, dict):
            continue
        best_checkpoints.update(checkpoint_keys(checkpoint))
        metrics_by_split = eval_summary.get("metrics_by_split")
        if not isinstance(metrics_by_split, dict):
            continue
        for split, metrics in metrics_by_split.items():
            if split in {"train", "val", "test"} and isinstance(metrics, dict):
                for key in checkpoint_keys(checkpoint):
                    index[("__checkpoint__", key, str(split))] = metrics


def parse_train_summary(
    path: Path,
    *,
    tail_window: int,
    eval_index: EvalIndex,
    best_checkpoint_index: BestCheckpointIndex,
) -> SummaryRecord | None:
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    dataset = str(summary.get("dataset", "")).strip()
    probe = str(summary.get("probe_name", "")).strip()
    if dataset not in DATASETS or probe not in PROBES:
        return None

    fit = summary.get("fit")
    if not isinstance(fit, dict):
        return None
    history = fit.get("history")
    if not isinstance(history, list) or not history:
        return None

    best_epoch = _to_int(fit.get("best_epoch"))
    n_epochs = _to_int(fit.get("n_epochs")) or len(history)
    best_row = _history_row_for_epoch(history, best_epoch) or {}
    final_row = history[-1] if isinstance(history[-1], dict) else {}
    tail_start = history[max(0, len(history) - tail_window)]
    if not isinstance(tail_start, dict):
        tail_start = {}

    best_train_acc = _to_float(best_row.get("train_accuracy"))
    best_val_acc = _to_float(fit.get("best_val_accuracy"))
    if best_val_acc is None:
        best_val_acc = _to_float(best_row.get("val_accuracy"))
    final_train_acc = _to_float(final_row.get("train_accuracy"))
    final_val_acc = _to_float(final_row.get("val_accuracy"))

    best_train_loss = _to_float(best_row.get("train_loss"))
    best_val_loss = _to_float(fit.get("best_val_loss"))
    if best_val_loss is None:
        best_val_loss = _to_float(best_row.get("val_loss"))
    final_train_loss = _to_float(final_row.get("train_loss"))
    final_val_loss = _to_float(final_row.get("val_loss"))

    accuracy_train_val_gap_at_best = _subtract(best_train_acc, best_val_acc)
    train_val_gap_final = _subtract(final_train_acc, final_val_acc)
    val_acc_drop_after_best = _subtract(best_val_acc, final_val_acc)
    val_loss_rise_after_best = _subtract(final_val_loss, best_val_loss)
    best_epoch_fraction = (
        float(best_epoch) / float(n_epochs) if best_epoch is not None and n_epochs else None
    )

    train_acc_tail_delta = _subtract(
        _to_float(final_row.get("train_accuracy")),
        _to_float(tail_start.get("train_accuracy")),
    )
    val_acc_tail_delta = _subtract(
        _to_float(final_row.get("val_accuracy")),
        _to_float(tail_start.get("val_accuracy")),
    )
    val_loss_tail_delta = _subtract(
        _to_float(final_row.get("val_loss")),
        _to_float(tail_start.get("val_loss")),
    )

    objective_metric_name = OBJECTIVE_METRICS[dataset]
    checkpoint = str(summary.get("checkpoint", "")).strip()
    is_selected_best_config = bool(
        checkpoint and checkpoint_keys(checkpoint) & best_checkpoint_index
    )
    eval_metrics = load_eval_metrics(
        path,
        dataset=dataset,
        probe=probe,
        eval_index=eval_index,
        checkpoint=checkpoint,
    )
    test_metric = _metric_from_eval(eval_metrics.get("test"), objective_metric_name)
    val_metric = _metric_from_eval(eval_metrics.get("val"), objective_metric_name)
    train_metric = _metric_from_eval(eval_metrics.get("train"), objective_metric_name)
    diagnostic_train_metric = train_metric if train_metric is not None else best_train_acc
    diagnostic_val_metric = val_metric if val_metric is not None else best_val_acc
    diagnostic_test_metric = test_metric
    diagnostic_metric_source = (
        "dataset_eval"
        if val_metric is not None
        else "epoch_accuracy_fallback"
    )
    diagnostic_train_val_gap = _subtract(diagnostic_train_metric, diagnostic_val_metric)
    diagnostic_test_val_gap = _subtract(diagnostic_test_metric, diagnostic_val_metric)

    row = {
        "dataset": dataset,
        "probe": probe,
        "run_name": infer_run_name(path, dataset, probe),
        "summary_path": str(path),
        "checkpoint": str(summary.get("checkpoint", "")),
        "is_selected_best_config": _flag(is_selected_best_config),
        "layer": summary.get("layer", ""),
        "feature_view": summary.get("feature_view", ""),
        "n_train": summary.get("n_train", ""),
        "n_val": summary.get("n_val", ""),
        "lr": _hparam(summary, "lr"),
        "weight_decay": _hparam(summary, "weight_decay"),
        "batch_size": _hparam(summary, "batch_size"),
        "epochs_configured": _hparam(summary, "epochs"),
        "hyperpars": format_hyperpars(summary),
        "n_epochs_ran": n_epochs,
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_epoch_fraction": _round(best_epoch_fraction),
        "early_stopped": fit.get("early_stopped", ""),
        "early_stopping_patience": fit.get("early_stopping_patience", ""),
        "objective_metric_name": objective_metric_name,
        "diagnostic_metric_source": diagnostic_metric_source,
        "diagnostic_train_metric": _round(diagnostic_train_metric),
        "diagnostic_val_metric": _round(diagnostic_val_metric),
        "diagnostic_test_metric": _round(diagnostic_test_metric),
        "diagnostic_train_val_gap": _round(diagnostic_train_val_gap),
        "diagnostic_test_val_gap": _round(diagnostic_test_val_gap),
        "best_train_accuracy": _round(best_train_acc),
        "best_val_accuracy": _round(best_val_acc),
        "final_train_accuracy": _round(final_train_acc),
        "final_val_accuracy": _round(final_val_acc),
        "best_train_loss": _round(best_train_loss),
        "best_val_loss": _round(best_val_loss),
        "final_train_loss": _round(final_train_loss),
        "final_val_loss": _round(final_val_loss),
        "accuracy_train_val_gap_at_best": _round(accuracy_train_val_gap_at_best),
        "train_val_gap_final": _round(train_val_gap_final),
        "val_acc_drop_after_best": _round(val_acc_drop_after_best),
        "val_loss_rise_after_best": _round(val_loss_rise_after_best),
        "train_acc_tail_delta": _round(train_acc_tail_delta),
        "val_acc_tail_delta": _round(val_acc_tail_delta),
        "val_loss_tail_delta": _round(val_loss_tail_delta),
        "eval_train_metric": _round(train_metric),
        "eval_val_metric": _round(val_metric),
        "eval_test_metric": _round(test_metric),
        "test_val_gap": _round(diagnostic_test_val_gap),
        "overfit_gap": _flag(
            diagnostic_train_val_gap is not None and diagnostic_train_val_gap >= 20.0
        ),
        "post_best_degradation": _flag(
            (val_acc_drop_after_best is not None and val_acc_drop_after_best >= 2.0)
            or (val_loss_rise_after_best is not None and val_loss_rise_after_best > 0.05)
        ),
        "underfit_like": _flag(
            diagnostic_train_metric is not None
            and diagnostic_val_metric is not None
            and diagnostic_train_metric < 65.0
            and diagnostic_val_metric < 65.0
            and (diagnostic_train_val_gap is None or diagnostic_train_val_gap < 10.0)
            and (best_epoch_fraction is None or best_epoch_fraction >= 0.75)
        ),
        "late_best_epoch": _flag(best_epoch_fraction is not None and best_epoch_fraction >= 0.8),
        "unstable_val": _flag(is_unstable(history)),
    }
    return SummaryRecord(path=path, dataset=dataset, probe=probe, row=row)


def checkpoint_keys(checkpoint: str) -> set[str]:
    raw = str(checkpoint).strip()
    if not raw:
        return set()
    keys = {raw}
    try:
        path = Path(raw)
        keys.add(str(path))
        if "artifacts" in path.parts:
            idx = path.parts.index("artifacts")
            keys.add(str(Path(*path.parts[idx:])))
    except (TypeError, ValueError):
        pass
    return keys


def load_eval_metrics(
    train_summary_path: Path,
    *,
    dataset: str,
    probe: str,
    eval_index: EvalIndex,
    checkpoint: str,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    train_dir = train_summary_path.parent
    run_dir = train_dir.parent
    run_root = find_run_root(train_summary_path, dataset, probe)
    rel_eval_parent = None
    run_name = ""
    if run_root is not None:
        run_name = run_root.name
        try:
            rel_eval_parent = train_dir.parent.relative_to(run_root)
        except ValueError:
            rel_eval_parent = None
    for split in ("train", "val", "test"):
        candidates = [
            run_dir / "eval" / split / "probe_eval_summary.json",
            run_dir / split / "probe_eval_summary.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            raw_metrics = data.get("metrics")
            if isinstance(raw_metrics, dict):
                metrics[split] = raw_metrics
                break
        if split not in metrics and checkpoint:
            indexed_metrics = None
            for key in checkpoint_keys(checkpoint):
                indexed_metrics = eval_index.get(("__checkpoint__", key, split))
                if indexed_metrics is not None:
                    break
            if indexed_metrics is not None:
                metrics[split] = indexed_metrics
        if split not in metrics and run_name and rel_eval_parent is not None:
            for result_root in RESULT_ROOTS:
                candidate = (
                    result_root
                    / run_name
                    / rel_eval_parent
                    / "eval"
                    / split
                    / "probe_eval_summary.json"
                )
                if not candidate.exists():
                    continue
                try:
                    data = json.loads(candidate.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                raw_metrics = data.get("metrics")
                if isinstance(raw_metrics, dict):
                    metrics[split] = raw_metrics
                    break
    return metrics


def find_run_root(path: Path, dataset: str, probe: str) -> Path | None:
    marker = f"{dataset}_probe_{probe}_"
    for parent in [path, *path.parents]:
        if parent.name.startswith(marker):
            return parent
    return None


def _metric_from_eval(metrics: dict[str, Any] | None, metric_name: str) -> float | None:
    if not metrics:
        return None
    for key in (metric_name, "accuracy", "voe_accuracy", "roc_auc", "pair_consistency"):
        value = _to_float(metrics.get(key))
        if value is not None:
            return value
    return None


def infer_run_name(path: Path, dataset: str, probe: str) -> str:
    marker = f"{dataset}_probe_{probe}_"
    for part in path.parts:
        if part.startswith(marker):
            return part
    parts = path.parts
    for index, part in enumerate(parts):
        if part != dataset:
            continue
        if index + 1 >= len(parts):
            break
        candidate = parts[index + 1]
        if candidate.startswith("old") and index + 2 < len(parts):
            return parts[index + 2]
        return candidate
    return path.parent.name


def infer_model_name(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).strip()
    probe = str(row.get("probe", "")).strip()
    run_name = str(row.get("run_name", "")).strip()
    prefix = f"{dataset}_probe_{probe}_"
    model = run_name[len(prefix) :] if run_name.startswith(prefix) else run_name
    if re.fullmatch(r"trial_[0-9]+", model):
        summary_path = Path(str(row.get("summary_path", "")))
        model = infer_run_name(summary_path, dataset, probe)
        model = model[len(prefix) :] if model.startswith(prefix) else model
    if "ltx" in model.lower():
        return "ltx_video"
    model = re.sub(r"_[0-9]{8}T[0-9]{6}Z.*$", "", model)
    model = re.sub(r"_[0-9]{8}$", "", model)
    model = re.sub(r"_lr_matrix$", "", model)
    model = re.sub(r"_retry_missing.*$", "", model)
    return model


def format_hyperpars(summary: dict[str, Any]) -> str:
    hparams = summary.get("probe_hparams")
    if not isinstance(hparams, dict):
        return ""
    skip = {
        "checkpoint_path",
        "deterministic",
        "device",
        "eval_output_dir",
        "eval_output_subdir",
        "feature_view",
        "layer",
        "name",
        "num_classes",
        "output_dir",
        "output_subdir",
        "wandb",
    }
    preferred = [
        "lr",
        "weight_decay",
        "batch_size",
        "eval_batch_size",
        "epochs",
        "num_heads",
        "num_self_attn_blocks",
        "mlp_ratio",
        "dropout",
        "hidden_dim",
    ]
    keys = [key for key in preferred if key in hparams and key not in skip]
    keys.extend(sorted(key for key in hparams if key not in skip and key not in keys))
    return ";".join(f"{key}={hparams[key]}" for key in keys)


def is_unstable(history: list[Any]) -> bool:
    vals = [
        _to_float(row.get("val_accuracy"))
        for row in history
        if isinstance(row, dict) and _to_float(row.get("val_accuracy")) is not None
    ]
    if len(vals) < 5:
        return False
    tail = vals[-5:]
    return max(tail) - min(tail) >= 5.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "probe",
        "run_name",
        "summary_path",
        "checkpoint",
        "is_selected_best_config",
        "layer",
        "feature_view",
        "n_train",
        "n_val",
        "lr",
        "weight_decay",
        "batch_size",
        "epochs_configured",
        "hyperpars",
        "n_epochs_ran",
        "best_epoch",
        "best_epoch_fraction",
        "early_stopped",
        "early_stopping_patience",
        "objective_metric_name",
        "diagnostic_metric_source",
        "diagnostic_train_metric",
        "diagnostic_val_metric",
        "diagnostic_test_metric",
        "diagnostic_train_val_gap",
        "diagnostic_test_val_gap",
        "best_train_accuracy",
        "best_val_accuracy",
        "final_train_accuracy",
        "final_val_accuracy",
        "best_train_loss",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "accuracy_train_val_gap_at_best",
        "train_val_gap_final",
        "val_acc_drop_after_best",
        "val_loss_rise_after_best",
        "train_acc_tail_delta",
        "val_acc_tail_delta",
        "val_loss_tail_delta",
        "eval_train_metric",
        "eval_val_metric",
        "eval_test_metric",
        "test_val_gap",
        "overfit_gap",
        "post_best_degradation",
        "underfit_like",
        "late_best_epoch",
        "unstable_val",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_rows_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["model", "layer", "hyperpars", "overfit_gap", "underfit_like"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": infer_model_name(row),
                    "layer": row.get("layer", ""),
                    "hyperpars": row.get("hyperpars", ""),
                    "overfit_gap": row.get("overfit_gap", ""),
                    "underfit_like": row.get("underfit_like", ""),
                }
            )


def write_best_config_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    selected_rows = [
        row for row in rows if row.get("is_selected_best_config") == "true"
    ]
    if selected_rows:
        rows = selected_rows
    for row in rows:
        key = (infer_model_name(row), str(row.get("layer", "")))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (_sort_layer(item[1]), item[0])):
        candidates = grouped[key]
        selected.append(
            max(
                candidates,
                key=lambda row: (
                    _to_float(row.get("diagnostic_val_metric")) is not None,
                    _to_float(row.get("diagnostic_val_metric")) or float("-inf"),
                    _to_float(row.get("diagnostic_test_metric")) or float("-inf"),
                    str(row.get("summary_path", "")),
                ),
            )
        )
    write_selected_rows_compact_csv(path, selected)


def _history_row_for_epoch(history: list[Any], epoch: int | None) -> dict[str, Any] | None:
    if epoch is None:
        return None
    for row in history:
        if not isinstance(row, dict):
            continue
        if _to_int(row.get("epoch")) == epoch:
            return row
    index = epoch - 1
    if 0 <= index < len(history) and isinstance(history[index], dict):
        return history[index]
    return None


def _hparam(summary: dict[str, Any], key: str) -> Any:
    hparams = summary.get("probe_hparams")
    if isinstance(hparams, dict):
        return hparams.get(key, "")
    return ""


def _subtract(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _round(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 6)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _sort_layer(value: Any) -> tuple[int, str]:
    if isinstance(value, int):
        return (value, "")
    try:
        return (int(str(value)), "")
    except ValueError:
        return (10**9, str(value))


if __name__ == "__main__":
    main()
