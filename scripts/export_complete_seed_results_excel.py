from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from export_seed_results_excel import write_seed_workbook


DEFAULT_SUMMARY = Path("results/seed_runs/complete_seed_results_so_far_summary.csv")
DEFAULT_LONG = Path("results/seed_runs/complete_seed_results_so_far_long.csv")
DEFAULT_TARGET_CONFIGS = Path("results/verified_layerwise_probe_configs.csv")
DEFAULT_OUTPUT = Path("results/seed_runs/complete_seed_results_so_far_grouped.xlsx")

EXPERIMENT_LABELS = {
    "main": "Main",
    "same_L": "SameL",
    "backbone_sweep": "Backbone",
    "ltx": "LTX",
}

EXPERIMENT_ORDER = ["main", "same_L", "backbone_sweep", "ltx"]
PROBE_ORDER = ["Linear", "MLP", "Attentive"]
DATASET_ORDER = ["IntPhys2", "MVP"]

SUMMARY_DISPLAY_FIELDS = [
    "dataset",
    "experiment",
    "model",
    "backbone",
    "probe",
    "layer",
    "layer_label",
    "n_seeds",
    "seeds",
    "test_primary_mean",
    "test_primary_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "val_primary_mean",
    "val_primary_std",
    "val_accuracy_mean",
    "val_accuracy_std",
    "train_primary_mean",
    "train_primary_std",
    "train_accuracy_mean",
    "train_accuracy_std",
    "config_id",
]

LONG_DISPLAY_FIELDS = [
    "run_set",
    "dataset",
    "experiment",
    "model",
    "backbone",
    "probe",
    "layer",
    "layer_label",
    "seed",
    "test_primary",
    "test_accuracy",
    "val_primary",
    "val_accuracy",
    "train_primary",
    "train_accuracy",
    "n_epochs",
    "early_stopped",
    "config_id",
    "run_id",
]

COVERAGE_FIELDS = [
    "experiment",
    "dataset",
    "probe",
    "target_configs",
    "present_configs",
    "missing_configs",
    "seed_rows",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the combined completed seed results into a grouped XLSX workbook."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--long", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--target-configs", type=Path, default=DEFAULT_TARGET_CONFIGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary_rows = _read_csv(args.summary)
    long_rows = _read_csv(args.long)
    target_rows = _read_csv(args.target_configs)

    sheets: list[tuple[str, list[str], list[dict[str, str]]]] = []
    sheets.append(("Overview", ["key", "value"], _overview_rows(summary_rows, long_rows, target_rows, args)))
    sheets.append(("Coverage", COVERAGE_FIELDS, _coverage_rows(summary_rows, long_rows, target_rows)))

    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in summary_rows if row.get("dataset") == dataset]
        if dataset_rows:
            sheets.append((dataset, SUMMARY_DISPLAY_FIELDS + ["run_set"], _sort_summary(dataset_rows)))

    for experiment, probe in _ordered_summary_sheets(summary_rows):
        run_rows = [
            row
            for row in summary_rows
            if row.get("experiment") == experiment and row.get("probe") == probe
        ]
        sheets.append((_sheet_label(experiment, probe), SUMMARY_DISPLAY_FIELDS + ["run_set"], _sort_summary(run_rows)))

    sheets.append(("All Summary", SUMMARY_DISPLAY_FIELDS + ["run_set"], _sort_summary(summary_rows)))
    sheets.append(("All Long", LONG_DISPLAY_FIELDS, _sort_long(long_rows)))
    sheets.append(("README", ["key", "value"], _readme_rows(args)))

    write_seed_workbook(args.output, sheets)
    print(f"wrote {args.output}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _overview_rows(
    summary_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    complete = sum(1 for row in summary_rows if row.get("complete", "").lower() == "true")
    return [
        {"key": "generated_utc", "value": datetime.now(timezone.utc).replace(microsecond=0).isoformat()},
        {"key": "summary_csv", "value": str(args.summary)},
        {"key": "long_csv", "value": str(args.long)},
        {"key": "target_configs_csv", "value": str(args.target_configs)},
        {"key": "target_config_rows", "value": str(len(target_rows))},
        {"key": "summary_rows", "value": str(len(summary_rows))},
        {"key": "complete_summary_rows", "value": str(complete)},
        {"key": "missing_target_config_rows", "value": str(max(0, len(target_rows) - complete))},
        {"key": "long_seed_rows", "value": str(len(long_rows))},
        {"key": "expected_seeds", "value": "42,101,102"},
    ]


def _coverage_rows(
    summary_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    target_counts = Counter(_coverage_bucket(row) for row in target_rows)
    summary_counts = Counter(_coverage_bucket(row) for row in summary_rows)
    long_counts = Counter(_coverage_bucket(row) for row in long_rows)
    rows = []
    for experiment, dataset, probe in sorted(target_counts, key=_coverage_key):
        target_configs = target_counts[(experiment, dataset, probe)]
        present_configs = summary_counts[(experiment, dataset, probe)]
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "probe": probe,
                "target_configs": str(target_configs),
                "present_configs": str(present_configs),
                "missing_configs": str(target_configs - present_configs),
                "seed_rows": str(long_counts[(experiment, dataset, probe)]),
            }
        )
    return rows


def _coverage_bucket(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("experiment", ""), row.get("dataset", ""), row.get("probe", ""))


def _ordered_summary_sheets(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    seen = {(row.get("experiment", ""), row.get("probe", "")) for row in rows}
    ordered = [
        (experiment, probe)
        for experiment in EXPERIMENT_ORDER
        for probe in PROBE_ORDER
        if (experiment, probe) in seen
    ]
    return ordered + sorted(seen - set(ordered))


def _sheet_label(experiment: str, probe: str) -> str:
    return f"{EXPERIMENT_LABELS.get(experiment, experiment)} {probe}".strip()


def _sort_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _experiment_rank(row),
            _dataset_rank(row),
            row.get("model", ""),
            row.get("backbone", ""),
            _probe_rank(row),
            _layer(row),
        ),
    )


def _sort_long(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _experiment_rank(row),
            _dataset_rank(row),
            row.get("model", ""),
            row.get("backbone", ""),
            _probe_rank(row),
            _layer(row),
            _seed(row),
        ),
    )


def _coverage_key(item: tuple[str, str, str]) -> tuple[int, int, int]:
    experiment, dataset, probe = item
    return (
        _experiment_rank({"experiment": experiment}),
        _dataset_rank({"dataset": dataset}),
        _probe_rank({"probe": probe}),
    )


def _experiment_rank(row: dict[str, str]) -> int:
    experiment = row.get("experiment", "")
    try:
        return EXPERIMENT_ORDER.index(experiment)
    except ValueError:
        return len(EXPERIMENT_ORDER)


def _dataset_rank(row: dict[str, str]) -> int:
    dataset = row.get("dataset", "")
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER)


def _probe_rank(row: dict[str, str]) -> int:
    try:
        return PROBE_ORDER.index(row.get("probe", ""))
    except ValueError:
        return len(PROBE_ORDER)


def _layer(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("layer", "")))
    except ValueError:
        return 10**9


def _seed(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("seed", "")))
    except ValueError:
        return 10**9


def _readme_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"key": "layout", "value": "Overview and Coverage first, then one sheet per dataset, one sheet per experiment/probe, then full summary and long seed rows."},
        {"key": "primary_metric", "value": "IntPhys2 primary is VOE accuracy; MVP primary is pair consistency."},
        {"key": "variance", "value": "Standard deviations are computed across seeds 42, 101, and 102."},
        {"key": "coverage", "value": "Coverage compares present completed config rows against results/verified_layerwise_probe_configs.csv."},
        {"key": "summary_csv", "value": str(args.summary)},
        {"key": "long_csv", "value": str(args.long)},
        {"key": "target_configs_csv", "value": str(args.target_configs)},
    ]


if __name__ == "__main__":
    main()
