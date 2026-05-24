#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


BASE = Path("artifacts/analysis/probe_fit_diagnostics")
PROBE_ROOTS = (
    Path("/scratch-shared/scur0511/probe4physics/artifacts/probes"),
    Path("artifacts/probes"),
    Path("artifacts/derived_probe_roots"),
)
DATASETS = ("mvp", "intphys2")
PROBES = ("linear", "mlp", "temporal_attn")
OBJECTIVE_METRICS = {
    "mvp": "pair_consistency",
    "intphys2": "voe_accuracy",
}
LTX_NOISE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
LTX_DEPTH_LAYERS = (12, 24, 36, 48)
LAST_LAYER_BY_MODEL = {
    "jepa_v1_vith16_384": "32",
    "jepa_v2_vitg_384": "40",
    "jepa_v2_1_vitG_384": "48",
    "videomae_vit_huge_16_224": "32",
    "videomae_v2_vit_giant_16_224": "40",
    "ltx_video": "40",
}


def main() -> None:
    best_checkpoints = build_best_checkpoint_set()
    full_errors: list[str] = []
    compact_errors: list[str] = []
    full_rows_checked = 0
    compact_rows_checked = 0
    selected_counts: dict[tuple[str, str, str], int] = {}

    for dataset in DATASETS:
        for probe in PROBES:
            full_path = BASE / f"{dataset}_{probe}_fit_diagnostics.csv"
            compact_path = BASE / f"{dataset}_{probe}_fit_diagnostics_compact.csv"
            if not full_path.exists():
                full_errors.append(f"missing full file: {full_path}")
                continue
            if not compact_path.exists():
                compact_errors.append(f"missing compact file: {compact_path}")
                continue

            full_rows = list(csv.DictReader(full_path.open()))
            compact_rows = list(csv.DictReader(compact_path.open()))
            selected_by_group: dict[tuple[str, str], dict[str, Any]] = {}

            for line_no, row in enumerate(full_rows, start=2):
                source_path = Path(row["summary_path"])
                if not source_path.exists():
                    full_errors.append(f"{full_path}:{line_no} source missing: {source_path}")
                    continue
                source = json.loads(source_path.read_text())
                fit = source.get("fit")
                history = fit.get("history") if isinstance(fit, dict) else None
                if not isinstance(history, list) or not history:
                    full_errors.append(f"{full_path}:{line_no} missing fit.history")
                    continue

                best_epoch = to_int(fit.get("best_epoch"))
                best_row = history_row(history, best_epoch) or {}
                final_row = history[-1] if isinstance(history[-1], dict) else {}
                best_train = to_float(best_row.get("train_accuracy"))
                best_val = first_not_none(
                    to_float(fit.get("best_val_accuracy")),
                    to_float(best_row.get("val_accuracy")),
                )
                final_train = to_float(final_row.get("train_accuracy"))
                final_val = to_float(final_row.get("val_accuracy"))
                best_val_loss = first_not_none(
                    to_float(fit.get("best_val_loss")),
                    to_float(best_row.get("val_loss")),
                )
                final_val_loss = to_float(final_row.get("val_loss"))
                checkpoint = str(source.get("checkpoint", "")).strip()
                selected = bool(checkpoint and checkpoint_keys(checkpoint) & best_checkpoints)
                selected_str = "true" if selected else "false"
                key = (dataset, probe, selected_str)
                selected_counts[key] = selected_counts.get(key, 0) + 1

                checks = [
                    ("dataset", row["dataset"] == dataset),
                    ("probe", row["probe"] == probe),
                    ("objective_metric_name", row["objective_metric_name"] == OBJECTIVE_METRICS[dataset]),
                    ("checkpoint", row["checkpoint"] == checkpoint),
                    ("is_selected_best_config", row["is_selected_best_config"] == selected_str),
                    ("layer", str(row["layer"]) == str(source.get("layer", ""))),
                    (
                        "n_epochs_ran",
                        str(row["n_epochs_ran"]) == str(to_int(fit.get("n_epochs")) or len(history)),
                    ),
                    (
                        "best_epoch",
                        str(row["best_epoch"]) == (str(best_epoch) if best_epoch is not None else ""),
                    ),
                    ("best_train_accuracy", float_equal(row["best_train_accuracy"], best_train)),
                    ("best_val_accuracy", float_equal(row["best_val_accuracy"], best_val)),
                    ("final_train_accuracy", float_equal(row["final_train_accuracy"], final_train)),
                    ("final_val_accuracy", float_equal(row["final_val_accuracy"], final_val)),
                    (
                        "accuracy_train_val_gap_at_best",
                        float_equal(row["accuracy_train_val_gap_at_best"], subtract(best_train, best_val)),
                    ),
                    (
                        "val_acc_drop_after_best",
                        float_equal(row["val_acc_drop_after_best"], subtract(best_val, final_val)),
                    ),
                    (
                        "val_loss_rise_after_best",
                        float_equal(row["val_loss_rise_after_best"], subtract(final_val_loss, best_val_loss)),
                    ),
                ]
                for name, ok in checks:
                    if not ok:
                        full_errors.append(f"{full_path}:{line_no} mismatch {name}")

                if selected:
                    group_key = (infer_model(row), compact_layer_label(row))
                    previous = selected_by_group.get(group_key)
                    if previous is None or rank_row(row) > rank_row(previous):
                        selected_by_group[group_key] = row
                full_rows_checked += 1

            expected_compact = []
            for group_key in sorted(selected_by_group):
                selected_row = selected_by_group[group_key]
                expected_compact.append(
                    {
                        "model": infer_model(selected_row),
                        "layer": compact_layer_label(selected_row),
                        "hyperpars": selected_row["hyperpars"],
                        "overfit_gap": selected_row["overfit_gap"],
                        "underfit_like": selected_row["underfit_like"],
                    }
                )
            compact_sorted = sorted(compact_rows, key=lambda row: (row["model"], row["layer"]))
            if compact_rows and list(compact_rows[0].keys()) != [
                "model",
                "layer",
                "hyperpars",
                "overfit_gap",
                "underfit_like",
            ]:
                compact_errors.append(f"{compact_path} has wrong header")
            if len({(row["model"], row["layer"]) for row in compact_rows}) != len(compact_rows):
                compact_errors.append(f"{compact_path} has duplicate model/layer rows")
            if compact_sorted != expected_compact:
                compact_errors.append(
                    f"{compact_path} does not match selected best rows "
                    f"(expected {len(expected_compact)}, got {len(compact_rows)})"
                )
            compact_rows_checked += len(compact_rows)

    print(f"full_rows_checked {full_rows_checked}")
    print(f"compact_rows_checked {compact_rows_checked}")
    print("selected_counts")
    for key in sorted(selected_counts):
        print(f"{key} {selected_counts[key]}")
    print(f"full_errors {len(full_errors)}")
    for error in full_errors[:20]:
        print(f"FULL_ERROR {error}")
    print(f"compact_errors {len(compact_errors)}")
    for error in compact_errors[:20]:
        print(f"COMPACT_ERROR {error}")
    raise SystemExit(1 if full_errors or compact_errors else 0)


def run_dirs(dataset_root: Path, dataset: str) -> list[Path]:
    out: set[Path] = set()
    for probe in PROBES:
        for pattern in [
            f"{dataset}_probe_{probe}_*",
            f"*_{probe}_*",
            f"old*/{dataset}_probe_{probe}_*",
            f"old*/*_{probe}_*",
        ]:
            out.update(path for path in dataset_root.glob(pattern) if path.is_dir())
    return sorted(out)



def iter_train_eval_summaries(run_dir: Path) -> list[Path]:
    summaries = list(run_dir.rglob("train_eval_summary.json"))
    final_paths = {path.resolve() for path in summaries}
    for partial in run_dir.rglob("train_eval_summary.partial.json"):
        final = partial.with_name("train_eval_summary.json")
        if final.resolve() not in final_paths and not final.exists():
            summaries.append(partial)
    return sorted(summaries)

def build_best_checkpoint_set() -> set[str]:
    best: set[str] = set()
    for root in PROBE_ROOTS:
        if not root.exists():
            continue
        for dataset in DATASETS:
            for dataset_root in iter_dataset_search_roots(root, dataset):
                for run_dir in run_dirs(dataset_root, dataset):
                    for summary_path in iter_train_eval_summaries(run_dir):
                        try:
                            summary = json.loads(summary_path.read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        layers = summary.get("layers")
                        if not isinstance(layers, list):
                            continue
                        for layer in layers:
                            if isinstance(layer, dict):
                                for checkpoint in selected_checkpoint_candidates(layer):
                                    best.update(checkpoint_keys(checkpoint))
    return best


def iter_dataset_search_roots(root: Path, dataset: str) -> list[Path]:
    candidates: list[Path] = []
    if root.name == dataset and root.exists():
        candidates.append(root)
    dataset_root = root / dataset
    if dataset_root.exists():
        candidates.append(dataset_root)
    if root.exists() and any(root.glob(f"{dataset}_probe_*")):
        candidates.append(root)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def selected_checkpoint_candidates(layer_summary: dict[str, Any]) -> set[str]:
    checkpoints: set[str] = set()
    for key in ("checkpoint", "best_checkpoint"):
        value = str(layer_summary.get(key, "")).strip()
        if value:
            checkpoints.add(value)

    train_summary = layer_summary.get("train")
    if isinstance(train_summary, dict):
        for key in ("checkpoint", "best_checkpoint"):
            value = str(train_summary.get(key, "")).strip()
            if value:
                checkpoints.add(value)
        best_trial_number = to_int(train_summary.get("best_trial_number"))
        trials = train_summary.get("trials")
        if best_trial_number is not None and isinstance(trials, list):
            for trial in trials:
                if not isinstance(trial, dict):
                    continue
                if to_int(trial.get("trial_number")) != best_trial_number:
                    continue
                checkpoint = str(trial.get("checkpoint", "")).strip()
                if checkpoint:
                    checkpoints.add(checkpoint)

    eval_summary = layer_summary.get("eval")
    if isinstance(eval_summary, dict):
        checkpoint = str(eval_summary.get("checkpoint", "")).strip()
        if checkpoint:
            checkpoints.add(checkpoint)
        split_evals = eval_summary.get("split_evals")
        if isinstance(split_evals, dict):
            for split_summary in split_evals.values():
                if not isinstance(split_summary, dict):
                    continue
                checkpoint = str(split_summary.get("checkpoint", "")).strip()
                if checkpoint:
                    checkpoints.add(checkpoint)

    return checkpoints


def checkpoint_keys(checkpoint: Any) -> set[str]:
    raw = str(checkpoint).strip()
    if not raw:
        return set()
    path = Path(raw)
    keys = {raw, str(path)}
    if "artifacts" in path.parts:
        idx = path.parts.index("artifacts")
        keys.add(str(Path(*path.parts[idx:])))
    return keys


def history_row(history: list[Any], epoch: int | None) -> dict[str, Any] | None:
    if epoch is None:
        return None
    for row in history:
        if isinstance(row, dict) and to_int(row.get("epoch")) == epoch:
            return row
    idx = epoch - 1
    if 0 <= idx < len(history) and isinstance(history[idx], dict):
        return history[idx]
    return None


def infer_model(row: dict[str, Any]) -> str:
    dataset = str(row["dataset"])
    probe = str(row["probe"])
    prefix = f"{dataset}_probe_{probe}_"
    model = str(row["run_name"])
    if model.startswith(prefix):
        model = model[len(prefix) :]
    if re.fullmatch(r"trial_[0-9]+", model):
        model = infer_run_name_from_path(Path(str(row.get("summary_path", ""))), dataset, probe)
        if model.startswith(prefix):
            model = model[len(prefix) :]
    if "ltx" in model.lower():
        return "ltx_video"
    model = re.sub(r"_[0-9]{8}T[0-9]{6}Z.*$", "", model)
    model = re.sub(r"_[0-9]{8}$", "", model)
    model = re.sub(r"_lr_matrix$", "", model)
    model = re.sub(r"_retry_missing.*$", "", model)
    return model


def infer_run_name_from_path(path: Path, dataset: str, probe: str) -> str:
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



def compact_layer_label(row: dict[str, Any]) -> str:
    model = infer_model(row)
    raw_layer = str(row.get("layer", "")).strip()
    if raw_layer.lower() == "last":
        raw_layer = LAST_LAYER_BY_MODEL.get(model, raw_layer)

    if model == "ltx_video":
        slot = to_int(raw_layer)
        if slot is not None and 1 <= slot <= len(LTX_NOISE_LEVELS) * len(LTX_DEPTH_LAYERS):
            index = slot - 1
            noise = LTX_NOISE_LEVELS[index // len(LTX_DEPTH_LAYERS)]
            depth = LTX_DEPTH_LAYERS[index % len(LTX_DEPTH_LAYERS)]
            return f"noise_{noise:.1f}_block_{depth}"

    return raw_layer

def rank_row(row: dict[str, Any]) -> tuple[bool, float, float, str]:
    val = to_float(row.get("diagnostic_val_metric"))
    test = to_float(row.get("diagnostic_test_metric"))
    return (
        val is not None,
        val if val is not None else float("-inf"),
        test if test is not None else float("-inf"),
        str(row.get("summary_path", "")),
    )


def subtract(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def float_equal(actual: str, expected: float | None, tol: float = 1e-6) -> bool:
    if expected is None:
        return actual == ""
    try:
        actual_value = float(actual)
    except ValueError:
        return False
    if math.isnan(expected):
        return math.isnan(actual_value)
    return abs(actual_value - round(expected, 6)) <= tol


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    value_float = to_float(value)
    if value_float is None or math.isnan(value_float):
        return None
    return int(value_float)


if __name__ == "__main__":
    main()
