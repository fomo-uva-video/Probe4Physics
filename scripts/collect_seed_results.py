from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile


DEFAULT_MANIFEST = Path("results/seed_runs/seed_manifest_main_linear_mlp_v1.csv")
DEFAULT_SOURCE = Path("results/verified_best_probe_configs.csv")
DEFAULT_LONG_OUTPUT = Path("results/seed_runs/seed_results_long.csv")
DEFAULT_SUMMARY_OUTPUT = Path("results/seed_runs/seed_summary.csv")
DEFAULT_EXPECTED_SEEDS = (42, 101, 102)

NULL_VALUES = {"", "NULL", "null", "None", "none", "NA", "N/A"}
DATASET_TO_HYDRA = {"IntPhys2": "intphys2", "MVP": "mvp"}
DATASET_PRIMARY = {"intphys2": "voe_accuracy", "mvp": "pair_consistency"}

LONG_FIELDS = [
    "config_id",
    "run_id",
    "dataset",
    "dataset_hydra",
    "experiment",
    "model",
    "backbone",
    "probe",
    "probe_hydra",
    "layer",
    "layer_label",
    "seed",
    "source",
    "status",
    "objective_metric_name",
    "primary_metric_name",
    "train_primary",
    "train_accuracy",
    "val_primary",
    "val_accuracy",
    "test_primary",
    "test_accuracy",
    "n_epochs",
    "expected_epochs",
    "early_stopped",
    "expected_early_stopping",
    "artifact_train_eval_summary",
    "artifact_probe_eval_dir",
    "source_csv_path",
    "source_csv_line_number",
    "source_excel_workbook",
    "source_excel_sheet",
    "source_excel_range",
    "notes",
]

SUMMARY_FIELDS = [
    "config_id",
    "dataset",
    "dataset_hydra",
    "experiment",
    "model",
    "backbone",
    "probe",
    "probe_hydra",
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
    "complete",
    "missing_seeds",
    "notes",
]


@dataclass(frozen=True)
class CellRange:
    start_col: int
    start_row: int
    end_col: int
    end_row: int


class XlsxReader:
    """Small XLSX value reader for the result DBs, avoiding an openpyxl dependency."""

    NS = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip = ZipFile(path)
        self._shared_strings = self._load_shared_strings()
        self._sheet_paths = self._load_sheet_paths()

    def close(self) -> None:
        self._zip.close()

    def read_sheet(self, sheet_name: str) -> dict[str, str]:
        sheet_path = self._sheet_paths.get(sheet_name)
        if sheet_path is None:
            raise KeyError(f"Sheet {sheet_name!r} not found in {self.path}")
        root = ET.fromstring(self._zip.read(sheet_path))
        values: dict[str, str] = {}
        for cell in root.findall(".//m:sheetData/m:row/m:c", self.NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            values[ref] = self._cell_value(cell)
        return values

    def _load_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self._zip.namelist():
            return []
        root = ET.fromstring(self._zip.read("xl/sharedStrings.xml"))
        strings: list[str] = []
        for item in root.findall("m:si", self.NS):
            strings.append("".join(text.text or "" for text in item.findall(".//m:t", self.NS)))
        return strings

    def _load_sheet_paths(self) -> dict[str, str]:
        workbook = ET.fromstring(self._zip.read("xl/workbook.xml"))
        rels = ET.fromstring(self._zip.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", self.NS)
        }
        sheet_paths: dict[str, str] = {}
        rel_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sheet in workbook.findall("m:sheets/m:sheet", self.NS):
            name = sheet.attrib["name"]
            target = rel_targets[sheet.attrib[rel_key]]
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = "xl/" + target
            sheet_paths[name] = target
        return sheet_paths

    def _cell_value(self, cell: ET.Element) -> str:
        kind = cell.attrib.get("t")
        value = cell.find("m:v", self.NS)
        if kind == "s" and value is not None:
            return self._shared_strings[int(value.text or "0")]
        if kind == "inlineStr":
            return "".join(text.text or "" for text in cell.findall(".//m:t", self.NS))
        return "" if value is None else (value.text or "")


class XlsxCache:
    def __init__(self) -> None:
        self._readers: dict[Path, XlsxReader] = {}
        self._sheets: dict[tuple[Path, str], dict[str, str]] = {}

    def read_sheet(self, path: Path, sheet_name: str) -> dict[str, str]:
        path = path.resolve()
        key = (path, sheet_name)
        if key not in self._sheets:
            if path not in self._readers:
                self._readers[path] = XlsxReader(path)
            self._sheets[key] = self._readers[path].read_sheet(sheet_name)
        return self._sheets[key]

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect seed rerun metrics into long and summary CSV files.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source config CSV. Defaults to the source_csv_path recorded in the manifest, then verified_best_probe_configs.csv.",
    )
    parser.add_argument("--long-output", type=Path, default=DEFAULT_LONG_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--expected-seeds",
        default=",".join(str(seed) for seed in DEFAULT_EXPECTED_SEEDS),
        help="Comma-separated seeds expected in the final aggregate, including the source seed.",
    )
    parser.add_argument(
        "--no-source-seed",
        action="store_true",
        help="Do not add seed 42 rows from the source Excel/CSV table.",
    )
    args = parser.parse_args()

    manifest_rows = _read_csv(args.manifest)
    source_path = args.source or _infer_source_path(manifest_rows, args.manifest) or DEFAULT_SOURCE
    source_rows = _read_csv(source_path)
    for index, row in enumerate(source_rows):
        row["_source_csv_line_number"] = str(index + 2)
    source_by_config = _source_rows_by_config(source_rows)
    expected_seeds = [int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()]

    xlsx_cache = XlsxCache()
    long_rows: list[dict[str, str]] = []
    try:
        if not args.no_source_seed:
            for config_id in _ordered_config_ids(manifest_rows):
                source_row = source_by_config.get(config_id)
                if source_row is None:
                    long_rows.append(_missing_source_row(config_id, source_path))
                else:
                    long_rows.append(_source_seed_row(source_row, source_path, xlsx_cache))
        for row in manifest_rows:
            long_rows.append(_artifact_seed_row(row, args.manifest))
    finally:
        xlsx_cache.close()

    summary_rows = _summary_rows(long_rows, expected_seeds=expected_seeds)
    args.long_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.long_output, LONG_FIELDS, long_rows)
    _write_csv(args.summary_output, SUMMARY_FIELDS, summary_rows)

    status_counts: dict[str, int] = {}
    for row in long_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    complete_configs = sum(1 for row in summary_rows if row["complete"] == "true")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "source": str(source_path),
                "long_output": str(args.long_output),
                "summary_output": str(args.summary_output),
                "long_rows": len(long_rows),
                "summary_rows": len(summary_rows),
                "complete_configs": complete_configs,
                "status_counts": status_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _infer_source_path(manifest_rows: list[dict[str, str]], manifest_path: Path) -> Path | None:
    raw_sources = {_value(row.get("source_csv_path")) for row in manifest_rows if _value(row.get("source_csv_path"))}
    if not raw_sources:
        return None
    if len(raw_sources) > 1:
        raise SystemExit(f"Manifest references multiple source CSVs; pass --source explicitly: {sorted(raw_sources)}")
    return _resolve_manifest_relative_path(next(iter(raw_sources)), manifest_path)


def _resolve_manifest_relative_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return manifest_path.parent.parent.parent / path


def _source_rows_by_config(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_config: dict[str, dict[str, str]] = {}
    for row in rows:
        config_id = _value(row.get("config_id"))
        if config_id and config_id not in by_config:
            by_config[config_id] = row
    return by_config


def _ordered_config_ids(rows: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        config_id = _value(row.get("config_id"))
        if config_id and config_id not in seen:
            ordered.append(config_id)
            seen.add(config_id)
    return ordered


def _source_seed_row(source_row: dict[str, str], source_path: Path, xlsx_cache: XlsxCache) -> dict[str, str]:
    dataset_hydra = _dataset_hydra(source_row)
    primary = DATASET_PRIMARY.get(dataset_hydra, "")
    base = _base_from_source(source_row, source_path)
    base.update(
        {
            "run_id": f"{_value(source_row.get('config_id'))}__seed_42",
            "seed": "42",
            "source": "source_csv",
            "objective_metric_name": primary,
            "primary_metric_name": primary,
            "expected_early_stopping": _value(source_row.get("early_stopping_enabled")),
            "expected_epochs": _value(source_row.get("epochs")),
            "source_excel_workbook": _value(source_row.get("excel_workbook")),
            "source_excel_sheet": _value(source_row.get("excel_sheet")),
            "source_excel_range": _value(source_row.get("excel_range")),
        }
    )

    csv_metrics = _read_source_csv_metrics(source_row)
    base.update(csv_metrics)
    required = ["train_primary", "train_accuracy", "val_primary", "val_accuracy", "test_primary", "test_accuracy"]
    if all(base[field] for field in required):
        base.update({"status": "complete", "notes": "seed 42 read from source CSV metrics"})
        return base

    try:
        values = _read_source_excel_metrics(source_row, source_path, xlsx_cache)
    except Exception as exc:  # keep the collector fail-soft for incomplete source tables
        base.update(
            {
                "status": "source_partial",
                "notes": f"source CSV metrics incomplete and Excel fallback failed: {exc}",
            }
        )
        return base

    base.update(
        {
            "source": "sheet",
            "status": "complete",
            "train_primary": _number_string(values.get("Train Primary Metric")),
            "train_accuracy": _number_string(values.get("Train Accuracy")),
            "val_primary": _number_string(values.get("Val Primary Metric")),
            "val_accuracy": _number_string(values.get("Val Accuracy")),
            "test_primary": _number_string(values.get("Test Primary Metric")),
            "test_accuracy": _number_string(values.get("Test Accuracy")),
            "notes": "seed 42 read from Excel DB",
        }
    )
    if any(not base[field] for field in required):
        base["status"] = "source_partial"
        base["notes"] = "source Excel row did not contain all six metrics"
    return base


def _read_source_csv_metrics(source_row: dict[str, str]) -> dict[str, str]:
    return {
        "train_primary": _number_string(source_row.get("train_primary_metric")),
        "train_accuracy": _number_string(source_row.get("train_accuracy")),
        "val_primary": _number_string(source_row.get("val_primary_metric")),
        "val_accuracy": _number_string(source_row.get("val_accuracy")),
        "test_primary": _number_string(source_row.get("test_primary_metric")),
        "test_accuracy": _number_string(source_row.get("test_accuracy")),
    }


def _read_source_excel_metrics(
    source_row: dict[str, str],
    source_path: Path,
    xlsx_cache: XlsxCache,
) -> dict[str, str]:
    workbook = _value(source_row.get("excel_workbook"))
    sheet = _value(source_row.get("excel_sheet"))
    cell_range = _value(source_row.get("excel_range"))
    if not workbook or not sheet or not cell_range:
        raise ValueError("missing excel_workbook, excel_sheet, or excel_range")
    workbook_path = Path(workbook)
    if not workbook_path.is_absolute():
        workbook_path = source_path.parent / workbook_path
    values = xlsx_cache.read_sheet(workbook_path, sheet)
    rng = _parse_range(cell_range)
    if rng.start_row != rng.end_row:
        raise ValueError(f"expected one-row source range, got {cell_range}")
    metrics: dict[str, str] = {}
    for col in range(rng.start_col, rng.end_col + 1):
        header = values.get(f"{_col_name(col)}2", "").strip()
        if not header:
            continue
        metrics[header] = values.get(f"{_col_name(col)}{rng.start_row}", "")
    return metrics


def _artifact_seed_row(row: dict[str, str], manifest_path: Path) -> dict[str, str]:
    base = _base_from_manifest(row, manifest_path)
    train_eval_path = Path(row["probe_output_dir"]) / row["probe_output_subdir"] / "train_eval_summary.json"
    base["artifact_train_eval_summary"] = str(train_eval_path)
    if not train_eval_path.exists():
        base.update({"status": "missing_artifact", "notes": f"missing {train_eval_path}"})
        return base

    try:
        data = json.loads(train_eval_path.read_text(encoding="utf-8"))
        layer = _select_layer(data, row["layer"])
        fit = layer.get("train", {}).get("fit", {})
        eval_summary = layer.get("eval", {})
        metrics_by_split = eval_summary.get("metrics_by_split", {})
        primary = str(data.get("objective_metric_name") or DATASET_PRIMARY.get(row["dataset_hydra"], ""))
        base.update(
            {
                "objective_metric_name": primary,
                "primary_metric_name": primary,
                "train_primary": _metric(metrics_by_split, "train", primary),
                "train_accuracy": _metric(metrics_by_split, "train", "accuracy"),
                "val_primary": _metric(metrics_by_split, "val", primary),
                "val_accuracy": _metric(metrics_by_split, "val", "accuracy"),
                "test_primary": _metric(metrics_by_split, "test", primary),
                "test_accuracy": _metric(metrics_by_split, "test", "accuracy"),
                "n_epochs": _number_string(fit.get("n_epochs")),
                "early_stopped": _bool_string(fit.get("early_stopped")),
                "artifact_probe_eval_dir": str(eval_summary.get("probe_eval_dir", "")),
            }
        )
        notes = _artifact_validation_notes(row, data, layer, fit, primary)
        base["status"] = "complete" if not notes else "artifact_mismatch"
        base["notes"] = "; ".join(notes) if notes else "artifact metrics read from train_eval_summary.json"
    except Exception as exc:
        base.update({"status": "artifact_error", "notes": f"could not parse artifact: {exc}"})
    return base


def _artifact_validation_notes(
    row: dict[str, str], data: dict[str, Any], layer: dict[str, Any], fit: dict[str, Any], primary: str
) -> list[str]:
    notes: list[str] = []
    if _value(data.get("dataset")) and _value(data.get("dataset")) != row["dataset_hydra"]:
        notes.append(f"dataset mismatch artifact={data.get('dataset')} manifest={row['dataset_hydra']}")
    if _value(data.get("probe_name")) and _value(data.get("probe_name")) != row["probe_hydra"]:
        notes.append(f"probe mismatch artifact={data.get('probe_name')} manifest={row['probe_hydra']}")
    if _value(layer.get("layer")) != _value(row.get("layer")):
        notes.append(f"layer mismatch artifact={layer.get('layer')} manifest={row['layer']}")
    expected_primary = DATASET_PRIMARY.get(row["dataset_hydra"])
    if expected_primary and primary != expected_primary:
        notes.append(f"primary metric mismatch artifact={primary} expected={expected_primary}")

    label_control = layer.get("train", {}).get("label_control", {})
    mode = _value(label_control.get("mode")) or "original"
    if mode != "original":
        notes.append(f"label_control.mode={mode}")

    expected_epochs = _int_or_none(row.get("epochs"))
    n_epochs = _int_or_none(fit.get("n_epochs"))
    early_stopping_expected = _as_bool(row.get("early_stopping_enabled"))
    early_stopped = _as_bool(fit.get("early_stopped"))
    if early_stopping_expected is False:
        if early_stopped:
            notes.append("artifact early_stopped=true but manifest expects early stopping off")
        if expected_epochs is not None and n_epochs is not None and n_epochs != expected_epochs:
            notes.append(f"n_epochs={n_epochs} but manifest expects {expected_epochs}")
    elif early_stopping_expected is True:
        if expected_epochs is not None and n_epochs is not None and n_epochs > expected_epochs:
            notes.append(f"n_epochs={n_epochs} exceeds manifest epochs={expected_epochs}")
    return notes


def _select_layer(data: dict[str, Any], expected_layer: str) -> dict[str, Any]:
    layers = data.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise ValueError("artifact has no layers list")
    for layer in layers:
        if _value(layer.get("layer")) == _value(expected_layer):
            return layer
    raise ValueError(f"layer {expected_layer} not found in artifact")


def _summary_rows(long_rows: list[dict[str, str]], *, expected_seeds: list[int]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in long_rows:
        config_id = row["config_id"]
        groups.setdefault(config_id, []).append(row)

    rows: list[dict[str, str]] = []
    for config_id in sorted(groups):
        group = groups[config_id]
        base = next(row for row in group if row.get("dataset") or row.get("dataset_hydra"))
        complete_rows = [row for row in group if row["status"] == "complete"]
        complete_seed_set = {_int_or_none(row["seed"]) for row in complete_rows}
        missing = [str(seed) for seed in expected_seeds if seed not in complete_seed_set]
        noncomplete = [row for row in group if row["status"] != "complete"]
        notes = "; ".join(
            f"seed {row['seed']} {row['status']}: {row['notes']}" for row in noncomplete if row.get("notes")
        )
        rows.append(
            {
                "config_id": config_id,
                "dataset": base.get("dataset", ""),
                "dataset_hydra": base.get("dataset_hydra", ""),
                "experiment": base.get("experiment", ""),
                "model": base.get("model", ""),
                "backbone": base.get("backbone", ""),
                "probe": base.get("probe", ""),
                "probe_hydra": base.get("probe_hydra", ""),
                "layer": base.get("layer", ""),
                "layer_label": base.get("layer_label", ""),
                "n_seeds": str(len(complete_rows)),
                "seeds": ",".join(str(row["seed"]) for row in complete_rows),
                "test_primary_mean": _mean(complete_rows, "test_primary"),
                "test_primary_std": _std(complete_rows, "test_primary"),
                "test_accuracy_mean": _mean(complete_rows, "test_accuracy"),
                "test_accuracy_std": _std(complete_rows, "test_accuracy"),
                "val_primary_mean": _mean(complete_rows, "val_primary"),
                "val_primary_std": _std(complete_rows, "val_primary"),
                "val_accuracy_mean": _mean(complete_rows, "val_accuracy"),
                "val_accuracy_std": _std(complete_rows, "val_accuracy"),
                "train_primary_mean": _mean(complete_rows, "train_primary"),
                "train_primary_std": _std(complete_rows, "train_primary"),
                "train_accuracy_mean": _mean(complete_rows, "train_accuracy"),
                "train_accuracy_std": _std(complete_rows, "train_accuracy"),
                "complete": "true" if not missing else "false",
                "missing_seeds": ",".join(missing),
                "notes": notes,
            }
        )
    return rows


def _base_from_source(source_row: dict[str, str], source_path: Path) -> dict[str, str]:
    dataset_hydra = _dataset_hydra(source_row)
    return _empty_long_row(
        {
            "config_id": _value(source_row.get("config_id")),
            "dataset": _value(source_row.get("dataset")),
            "dataset_hydra": dataset_hydra,
            "experiment": _value(source_row.get("experiment")),
            "model": _value(source_row.get("model")),
            "backbone": _value(source_row.get("backbone")),
            "probe": _value(source_row.get("probe")),
            "probe_hydra": _probe_hydra(source_row),
            "layer": _source_layer(source_row),
            "layer_label": _source_layer_label(source_row),
            "source_csv_path": str(source_path),
            "source_csv_line_number": _value(source_row.get("_source_csv_line_number")),
        }
    )


def _base_from_manifest(row: dict[str, str], manifest_path: Path) -> dict[str, str]:
    source_path = _value(row.get("source_csv_path")) or str(manifest_path)
    return _empty_long_row(
        {
            "config_id": row.get("config_id", ""),
            "run_id": row.get("run_id", ""),
            "dataset": row.get("dataset", ""),
            "dataset_hydra": row.get("dataset_hydra", ""),
            "experiment": row.get("experiment", ""),
            "model": row.get("model", ""),
            "backbone": row.get("backbone", ""),
            "probe": row.get("probe", ""),
            "probe_hydra": row.get("probe_hydra", ""),
            "layer": row.get("layer", ""),
            "layer_label": row.get("layer_label", ""),
            "seed": row.get("seed", ""),
            "source": "artifact",
            "expected_epochs": row.get("epochs", ""),
            "expected_early_stopping": row.get("early_stopping_enabled", ""),
            "source_csv_path": source_path,
            "source_csv_line_number": row.get("source_csv_line_number", ""),
        }
    )


def _missing_source_row(config_id: str, source_path: Path) -> dict[str, str]:
    return _empty_long_row(
        {
            "config_id": config_id,
            "run_id": f"{config_id}__seed_42",
            "seed": "42",
            "source": "source_csv",
            "status": "missing_source",
            "source_csv_path": str(source_path),
            "notes": f"config_id not found in {source_path}",
        }
    )


def _empty_long_row(values: dict[str, str] | None = None) -> dict[str, str]:
    row = {field: "" for field in LONG_FIELDS}
    if values:
        row.update({key: str(value) for key, value in values.items() if key in row})
    return row


def _dataset_hydra(row: dict[str, str]) -> str:
    explicit = _value(row.get("dataset_hydra"))
    if explicit:
        return explicit
    return DATASET_TO_HYDRA.get(_value(row.get("dataset")), "")


def _probe_hydra(row: dict[str, str]) -> str:
    explicit = _value(row.get("probe_hydra")) or _value(row.get("probe_name"))
    if explicit:
        return explicit
    probe = _value(row.get("probe"))
    return {"Linear": "linear", "MLP": "mlp", "Attentive": "temporal_attn"}.get(probe, probe.lower())


def _source_layer(row: dict[str, str]) -> str:
    layer = (
        _value(row.get("selected_layer_id"))
        or _value(row.get("probe_layer"))
        or _value(row.get("excel_layer"))
    )
    if layer:
        return layer
    config = _parse_json_dict(row.get("best_config_json", ""))
    layer_value = config.get("layer")
    return "" if layer_value is None else str(layer_value)


def _source_layer_label(row: dict[str, str]) -> str:
    return (
        _value(row.get("selected_layer_label"))
        or _value(row.get("layer_label"))
        or _value(row.get("depth_layer_id"))
    )


def _metric(metrics_by_split: dict[str, Any], split: str, metric: str) -> str:
    return _number_string(metrics_by_split.get(split, {}).get(metric))


def _mean(rows: list[dict[str, str]], field: str) -> str:
    values = _float_values(rows, field)
    return _number_string(statistics.mean(values)) if values else ""


def _std(rows: list[dict[str, str]], field: str) -> str:
    values = _float_values(rows, field)
    return _number_string(statistics.stdev(values)) if len(values) > 1 else ""


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_range(raw: str) -> CellRange:
    match = re.fullmatch(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", raw.strip())
    if not match:
        raise ValueError(f"unsupported range {raw!r}")
    start_col, start_row, end_col, end_row = match.groups()
    return CellRange(_col_index(start_col), int(start_row), _col_index(end_col), int(end_row))


def _col_index(name: str) -> int:
    value = 0
    for char in name:
        value = value * 26 + ord(char) - 64
    return value


def _col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    value = _value(raw)
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _value(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text in NULL_VALUES else text


def _number_string(raw: Any) -> str:
    value = _float_or_none(raw)
    if value is None:
        return ""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _float_or_none(raw: Any) -> float | None:
    text = _value(raw)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _int_or_none(raw: Any) -> int | None:
    value = _float_or_none(raw)
    return None if value is None else int(value)


def _as_bool(raw: Any) -> bool | None:
    text = _value(raw).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _bool_string(raw: Any) -> str:
    value = _as_bool(raw)
    if value is None:
        return ""
    return "true" if value else "false"


if __name__ == "__main__":
    main()
