from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_PRIMARY = {"intphys2": "voe_accuracy", "mvp": "pair_consistency"}
NULL_VALUES = {"", "NULL", "null", "None", "none", "NA", "N/A", "nan"}

DEFAULT_SEED_MERGE_KEYS = ["dataset", "experiment", "model_label", "backbone", "probe", "layer_key"]
DEFAULT_SEED_STAT_COLUMNS = [
    "n_seeds",
    "seeds",
    "seed_source",
    "test_primary_mean",
    "test_primary_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "val_primary_mean",
    "val_primary_std",
    "val_accuracy_mean",
    "val_accuracy_std",
]

READY_MANIFEST_SUMMARY_CSV = "ready_manifest_seed_summary.csv"

READY_MANIFESTS = (
    "seed_manifest_layerwise_main_attentive_mvp_v1.csv",
    "seed_manifest_mvp_jepa_v1_attentive_layerwise_v1.csv",
    "seed_manifest_mvp_jepa_v2_attentive_layerwise_v1.csv",
    "seed_manifest_mvp_jepa_v2_1_attentive_layerwise_v1.csv",
    "seed_manifest_intphys2_ltx2b_linear_mlp.csv",
    "seed_manifest_intphys2_ltx2b_attentive.csv",
    "seed_manifest_intphys2_ltx13b_linear_mlp.csv",
    "seed_manifest_intphys2_ltx13b_attentive.csv",
    "seed_manifest_mvp_ltx2b_linear_mlp.csv",
    "seed_manifest_mvp_ltx2b_attentive.csv",
    "seed_manifest_mvp_ltx13b_linear_mlp.csv",
)


def load_ready_seed_summary(
    results_dir: Path | str,
    *,
    mode: str = "attentive_continuation",
    seed_merge_keys: list[str] | None = None,
    seed_stat_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load all ready 3-seed summaries used by the large-ablation notebooks.

    The exported complete-seed CSV predates the LTX seed reruns, so this loader
    augments it with complete manifest/artifact rows. A manifest config is used
    only when seed 42 plus seeds 101 and 102 are all available.
    """

    results_dir = Path(results_dir)
    seed_merge_keys = seed_merge_keys or DEFAULT_SEED_MERGE_KEYS
    seed_stat_columns = seed_stat_columns or DEFAULT_SEED_STAT_COLUMNS

    if mode not in {"original", "attentive_continuation"}:
        raise ValueError(f"Unknown seed variance mode: {mode}")

    frames = [_load_original_seed_summary(results_dir)]
    if mode == "attentive_continuation":
        frames.append(_load_attentive_continuation_summary(results_dir))
    frames.append(_load_ready_manifest_summary(results_dir))

    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return _empty_seed_summary(seed_merge_keys, seed_stat_columns)

    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    combined["_seed_priority"] = combined["seed_source"].map(
        {"original": 0, "attentive_continuation": 1, "ready_manifest": 2}
    ).fillna(0)
    combined = _dedupe_seed_summary(combined, seed_merge_keys, seed_stat_columns + ["_seed_priority"])
    combined = combined.sort_values(seed_merge_keys + ["_seed_priority"]).drop_duplicates(seed_merge_keys, keep="last")
    return combined[seed_merge_keys + seed_stat_columns].reset_index(drop=True)


def _load_original_seed_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "seed_runs" / "complete_seed_results_so_far_summary.csv"
    if not path.exists():
        return _empty_seed_summary()
    seed = pd.read_csv(path)
    seed = seed[seed["complete"].astype(str).str.lower().eq("true")].copy()
    seed["dataset"] = seed["dataset_hydra"].map(dataset_key)
    seed["model_label"] = [_model_label(model, backbone) for model, backbone in zip(seed["model"], seed["backbone"])]
    seed["probe"] = seed["probe_hydra"].map(canonical_probe)
    seed["layer"] = pd.to_numeric(seed["layer"], errors="coerce")
    seed["seed_source"] = "original"
    return _dedupe_seed_summary(seed)


def _load_attentive_continuation_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "seed_runs" / "attentive_final_continuation_scientific_audit.csv"
    if not path.exists():
        return _empty_seed_summary()
    cont = pd.read_csv(path)
    cont = cont[pd.to_numeric(cont["n_final_seeds"], errors="coerce").eq(3)].copy()
    cont["dataset"] = cont["dataset"].map(dataset_key)
    cont["model_label"] = [_model_label(model, backbone) for model, backbone in zip(cont["model"], cont["backbone"])]
    cont["probe"] = cont["probe"].map(canonical_probe)
    cont["layer"] = pd.to_numeric(cont["layer"], errors="coerce")
    cont["layer_label"] = ""
    cont["n_seeds"] = cont["n_final_seeds"]
    cont["seeds"] = cont["final_seeds"]
    cont["test_primary_mean"] = cont["final_test_primary_mean"]
    cont["test_primary_std"] = cont["final_test_primary_std"]
    cont["test_accuracy_mean"] = cont["final_test_accuracy_mean"]
    cont["test_accuracy_std"] = cont["final_test_accuracy_std"]
    cont["val_primary_mean"] = np.nan
    cont["val_primary_std"] = np.nan
    cont["val_accuracy_mean"] = np.nan
    cont["val_accuracy_std"] = np.nan
    cont["seed_source"] = "attentive_continuation"
    return _dedupe_seed_summary(cont)


def build_ready_manifest_summary(results_dir: Path | str) -> pd.DataFrame:
    return _scan_ready_manifest_summary(Path(results_dir))


def _load_ready_manifest_summary(results_dir: Path) -> pd.DataFrame:
    cache = results_dir / "seed_runs" / READY_MANIFEST_SUMMARY_CSV
    if cache.exists():
        cached = pd.read_csv(cache)
        cached["seed_source"] = "ready_manifest"
        return _dedupe_seed_summary(cached)
    return _scan_ready_manifest_summary(results_dir)


def _scan_ready_manifest_summary(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_root = results_dir / "seed_runs"
    repo_root = results_dir.parent

    for name in READY_MANIFESTS:
        manifest_path = seed_root / name
        if not manifest_path.exists():
            continue
        manifest = pd.read_csv(manifest_path).fillna("")
        source_by_config = _source_rows_by_config(manifest, manifest_path, repo_root)
        for config_id, group in manifest.groupby("config_id", sort=False):
            seed_metrics = {}
            source_row = source_by_config.get(str(config_id))
            if source_row is not None:
                source_metrics = _source_metrics(source_row)
                if _has_test_metrics(source_metrics):
                    seed_metrics["42"] = source_metrics
            for _, manifest_row in group.iterrows():
                artifact_metrics = _artifact_metrics(manifest_row.to_dict(), repo_root)
                if _has_test_metrics(artifact_metrics):
                    seed_metrics[str(manifest_row["seed"])] = artifact_metrics
            if {"42", "101", "102"} - set(seed_metrics):
                continue
            base = group.iloc[0].to_dict()
            rows.append(_summary_from_seed_metrics(base, seed_metrics))

    if not rows:
        return _empty_seed_summary()
    return _dedupe_seed_summary(pd.DataFrame(rows))


def _source_rows_by_config(manifest: pd.DataFrame, manifest_path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    source_paths = {
        _resolve_path(path, repo_root)
        for path in manifest.get("source_csv_path", pd.Series(dtype=str)).astype(str)
        if _value(path)
    }
    if not source_paths:
        return {}
    if len(source_paths) > 1:
        raise ValueError(f"{manifest_path} references multiple source CSV files: {sorted(map(str, source_paths))}")
    source_path = next(iter(source_paths))
    if not source_path.exists():
        return {}
    source = pd.read_csv(source_path).fillna("")
    return {str(row["config_id"]): row.to_dict() for _, row in source.iterrows() if _value(row.get("config_id"))}


def _source_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "train_primary": _float_or_nan(row.get("train_primary_metric")),
        "train_accuracy": _float_or_nan(row.get("train_accuracy")),
        "val_primary": _float_or_nan(row.get("val_primary_metric")),
        "val_accuracy": _float_or_nan(row.get("val_accuracy")),
        "test_primary": _float_or_nan(row.get("test_primary_metric")),
        "test_accuracy": _float_or_nan(row.get("test_accuracy")),
    }


def _artifact_metrics(row: dict[str, Any], repo_root: Path) -> dict[str, float]:
    train_eval = _find_train_eval_summary(row, repo_root)
    if train_eval is not None:
        data = json.loads(train_eval.read_text(encoding="utf-8"))
        layer = _select_layer(data, row.get("layer"))
        primary = str(data.get("objective_metric_name") or DATASET_PRIMARY.get(dataset_key(row.get("dataset_hydra")), ""))
        return _metrics_from_split_map(layer.get("eval", {}).get("metrics_by_split", {}), primary)

    eval_summary = _find_eval_summary(row, repo_root)
    if eval_summary is None:
        return {}
    data = json.loads(eval_summary.read_text(encoding="utf-8"))
    primary = str(data.get("objective_metric_name") or DATASET_PRIMARY.get(dataset_key(row.get("dataset_hydra")), ""))
    metrics_by_split = data.get("metrics_by_split")
    if not isinstance(metrics_by_split, dict):
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        metrics_by_split = {"test": metrics}
    return _metrics_from_split_map(metrics_by_split, primary)


def _find_train_eval_summary(row: dict[str, Any], repo_root: Path) -> Path | None:
    output_dir = _resolve_path(row.get("probe_output_dir"), repo_root) / str(row.get("probe_output_subdir", ""))
    candidates = [output_dir / "train_eval_summary.json", *sorted(output_dir.glob("*/train_eval_summary.json"))]
    return next((path for path in candidates if path.exists()), None)


def _find_eval_summary(row: dict[str, Any], repo_root: Path) -> Path | None:
    output_dir = _resolve_path(row.get("eval_output_dir"), repo_root) / str(row.get("eval_output_subdir", ""))
    candidates = [output_dir / "eval_summary.json", *sorted(output_dir.glob("**/eval_summary.json"))]
    return next((path for path in candidates if path.exists()), None)


def _metrics_from_split_map(metrics_by_split: dict[str, Any], primary: str) -> dict[str, float]:
    return {
        "train_primary": _metric(metrics_by_split, "train", primary),
        "train_accuracy": _metric(metrics_by_split, "train", "accuracy"),
        "val_primary": _metric(metrics_by_split, "val", primary),
        "val_accuracy": _metric(metrics_by_split, "val", "accuracy"),
        "test_primary": _metric(metrics_by_split, "test", primary),
        "test_accuracy": _metric(metrics_by_split, "test", "accuracy"),
    }


def _summary_from_seed_metrics(base: dict[str, Any], seed_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    ordered_seeds = ["42", "101", "102"]
    summary = {
        "dataset": dataset_key(base.get("dataset_hydra") or base.get("dataset")),
        "experiment": _value(base.get("experiment")),
        "model_label": _model_label(base.get("model"), base.get("backbone")),
        "backbone": _value(base.get("backbone")),
        "probe": canonical_probe(base.get("probe_hydra") or base.get("probe")),
        "layer": pd.to_numeric(pd.Series([base.get("layer")]), errors="coerce").iloc[0],
        "layer_label": _value(base.get("layer_label")),
        "n_seeds": 3,
        "seeds": ",".join(ordered_seeds),
        "seed_source": "ready_manifest",
    }
    for field in [
        "test_primary",
        "test_accuracy",
        "val_primary",
        "val_accuracy",
        "train_primary",
        "train_accuracy",
    ]:
        values = [_float_or_nan(seed_metrics[seed].get(field)) for seed in ordered_seeds]
        clean = [value for value in values if not pd.isna(value)]
        summary[f"{field}_mean"] = statistics.mean(clean) if clean else np.nan
        summary[f"{field}_std"] = statistics.stdev(clean) if len(clean) > 1 else np.nan
    return summary


def _empty_seed_summary(
    seed_merge_keys: list[str] | None = None,
    seed_stat_columns: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(columns=(seed_merge_keys or DEFAULT_SEED_MERGE_KEYS) + (seed_stat_columns or DEFAULT_SEED_STAT_COLUMNS))


def _dedupe_seed_summary(
    df: pd.DataFrame,
    seed_merge_keys: list[str] | None = None,
    seed_stat_columns: list[str] | None = None,
) -> pd.DataFrame:
    seed_merge_keys = seed_merge_keys or DEFAULT_SEED_MERGE_KEYS
    seed_stat_columns = seed_stat_columns or DEFAULT_SEED_STAT_COLUMNS
    if df.empty:
        return _empty_seed_summary(seed_merge_keys, seed_stat_columns)

    keyed_frames = []
    if "layer_key" in df.columns:
        keyed = df.copy()
        keyed["layer_key"] = keyed["layer_key"].astype(str)
        keyed = keyed[keyed["layer_key"].ne("")]
        if not keyed.empty:
            keyed_frames.append(keyed)
    for key_func in (lambda row: numeric_layer_key(row.get("layer")), lambda row: label_layer_key(row.get("layer_label"))):
        keyed = df.copy()
        keyed["layer_key"] = keyed.apply(key_func, axis=1)
        keyed = keyed[keyed["layer_key"].astype(str).ne("")]
        if not keyed.empty:
            keyed_frames.append(keyed)
    if not keyed_frames:
        return _empty_seed_summary(seed_merge_keys, seed_stat_columns)

    out = pd.concat(keyed_frames, ignore_index=True, sort=False)
    for col in [col for col in seed_stat_columns if col.endswith("_mean") or col.endswith("_std")]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "_seed_priority" not in out.columns:
        out["_seed_priority"] = 0
    columns = seed_merge_keys + [col for col in seed_stat_columns if col not in seed_merge_keys]
    keep_columns = [col for col in columns if col in out.columns]
    sort_columns = seed_merge_keys + ["_seed_priority", "seed_source"]
    return out.sort_values(sort_columns).drop_duplicates(seed_merge_keys, keep="last")[keep_columns].reset_index(drop=True)


def dataset_key(value: Any) -> str:
    text = _value(value).lower()
    if text in {"intphys2", "intphys"}:
        return "intphys2"
    if text == "mvp":
        return "mvp"
    return text


def canonical_probe(value: Any) -> str | None:
    text = _value(value).lower()
    if text == "linear":
        return "linear"
    if text == "mlp":
        return "mlp"
    if text in {"attentive", "temporalattn", "temporal_attn", "temporal attentive"}:
        return "temporal_attn"
    return text or None


def numeric_layer_key(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"id:{number:g}"


def label_layer_key(value: Any) -> str:
    label = normalized_layer_label(value)
    return f"label:{label}" if label else ""


def normalized_layer_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _model_label(model: Any, backbone: Any) -> str:
    model_text = _value(model)
    backbone_text = _value(backbone)
    if model_text == "LTX-Video" and backbone_text in {"LTX-2B", "LTX-13B"}:
        return backbone_text
    return model_text


def _resolve_path(value: Any, repo_root: Path) -> Path:
    path = Path(_value(value))
    return path if path.is_absolute() else repo_root / path


def _select_layer(data: dict[str, Any], expected_layer: Any) -> dict[str, Any]:
    expected = _value(expected_layer)
    layers = data.get("layers", [])
    if not isinstance(layers, list):
        return {}
    for layer in layers:
        if _value(layer.get("layer")) == expected:
            return layer
    return layers[0] if layers else {}


def _metric(metrics_by_split: dict[str, Any], split: str, metric: str) -> float:
    metrics = metrics_by_split.get(split, {})
    if not isinstance(metrics, dict):
        return np.nan
    return _float_or_nan(metrics.get(metric))


def _has_test_metrics(metrics: dict[str, float]) -> bool:
    return not pd.isna(metrics.get("test_primary")) and not pd.isna(metrics.get("test_accuracy"))


def _float_or_nan(value: Any) -> float:
    text = _value(value)
    if not text:
        return np.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in NULL_VALUES else text

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the cached ready-manifest seed summary used by notebooks.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output = args.output or results_dir / "seed_runs" / READY_MANIFEST_SUMMARY_CSV
    summary = build_ready_manifest_summary(results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print({"output": str(output), "rows": len(summary)})


if __name__ == "__main__":
    main()

