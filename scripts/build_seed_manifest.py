from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("results/verified_best_probe_configs.csv")
DEFAULT_OUTPUT = Path("results/seed_runs/seed_manifest_main_linear_mlp_v1.csv")
DEFAULT_BLOCKED_OUTPUT = Path("results/seed_runs/seed_manifest_main_linear_mlp_blocked_v1.csv")
DEFAULT_SEEDS = (101, 102)

NULL_VALUES = {"", "NULL", "null", "None", "none", "NA", "N/A"}

DATASET_TO_HYDRA = {
    "IntPhys2": "intphys2",
    "MVP": "mvp",
}

PROBE_TO_HYDRA = {
    "Linear": "linear",
    "MLP": "mlp",
    "Attentive": "temporal_attn",
}

BACKBONE_TO_HYDRA = {
    ("V-JEPA", "ViT-H/16"): ("jepa_v1", "vith16_384"),
    ("V-JEPA 2", "ViT-G/16"): ("jepa_v2", "vitg_384"),
    ("V-JEPA 2.1", "ViT-Gigantic/16"): ("jepa_v2_1", "vitG_384"),
    ("VideoMAE", "ViT-H/16"): ("videomae", "vit_huge_16_224"),
    ("VideoMAE-v2", "ViT-G/16"): ("videomae_v2", "vit_giant_16_224"),
}

MANIFEST_FIELDS = [
    "run_id",
    "config_id",
    "run_command",
    "dataset",
    "dataset_hydra",
    "experiment",
    "model",
    "backbone",
    "probe",
    "probe_hydra",
    "seed",
    "original_seed",
    "backbone_name",
    "backbone_variant",
    "layer",
    "selected_slot",
    "layer_label",
    "feature_view",
    "lr",
    "weight_decay",
    "batch_size",
    "eval_batch_size",
    "epochs",
    "early_stopping_enabled",
    "early_stopping_patience",
    "mlp_hidden_dims",
    "mlp_dropout",
    "temporal_num_heads",
    "temporal_num_self_attn_blocks",
    "temporal_mlp_ratio",
    "temporal_dropout",
    "probe_device",
    "probe_output_dir",
    "probe_output_subdir",
    "eval_output_dir",
    "eval_output_subdir",
    "wandb_group",
    "wandb_name",
    "status",
    "blocked_reason",
    "hydra_overrides_json",
    "source_csv_path",
    "source_csv_row_index",
    "source_csv_line_number",
    "source_row_sha256",
    "source_config_status",
    "source_config_json",
    "source_evidence_path",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic seed-rerun manifest.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blocked-output", type=Path, default=DEFAULT_BLOCKED_OUTPUT)
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--probes", default="Linear,MLP")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()

    requested_probes = {item.strip() for item in args.probes.split(",") if item.strip()}
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed")

    source_rows = _read_csv(args.source)
    _annotate_source_rows(source_rows, args.source)
    selected = [
        row
        for row in source_rows
        if str(row.get("experiment", "")).strip() == args.experiment
        and str(row.get("probe", "")).strip() in requested_probes
    ]

    manifest_rows: list[dict[str, str]] = []
    blocked_rows: list[dict[str, str]] = []
    for row in selected:
        blocked_reason = _blocked_reason(row)
        if blocked_reason:
            blocked_rows.append(_blocked_row(row, blocked_reason))
            continue
        for seed in seeds:
            manifest_rows.append(_manifest_row(row, seed=seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output, manifest_rows, MANIFEST_FIELDS)
    _write_csv(args.blocked_output, blocked_rows, MANIFEST_FIELDS)

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "blocked_output": str(args.blocked_output),
        "experiment": args.experiment,
        "probes": sorted(requested_probes),
        "seeds": seeds,
        "source_rows_selected": len(selected),
        "runnable_configs": len({row["config_id"] for row in manifest_rows}),
        "manifest_rows": len(manifest_rows),
        "blocked_configs": len(blocked_rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _annotate_source_rows(rows: list[dict[str, str]], path: Path) -> None:
    for index, row in enumerate(rows):
        row["_source_csv_path"] = str(path)
        row["_source_csv_row_index"] = str(index)
        row["_source_csv_line_number"] = str(index + 2)
        row["_source_row_sha256"] = _source_row_hash(row)


def _source_row_hash(row: dict[str, str]) -> str:
    payload = {
        str(key): str(value)
        for key, value in row.items()
        if not str(key).startswith("_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _blocked_reason(row: dict[str, str]) -> str:
    status = str(row.get("config_status", "")).strip()
    if status == "MISSING":
        return "source config_status=MISSING"
    if status not in {"VERIFIED_FULL", "VERIFIED_ATTENTIVE_LAYER_LR"}:
        return f"unsupported source config_status={status!r}"
    try:
        _resolve_dataset(row)
        _resolve_probe(row)
        _resolve_backbone(row)
        _resolve_layer(row)
    except ValueError as exc:
        return str(exc)
    return ""


def _blocked_row(row: dict[str, str], reason: str) -> dict[str, str]:
    base = {field: "" for field in MANIFEST_FIELDS}
    config_id = _config_id(row)
    base.update(
        {
            "run_id": f"{config_id}__blocked",
            "config_id": config_id,
            "run_command": "",
            "dataset": str(row.get("dataset", "")).strip(),
            "experiment": str(row.get("experiment", "")).strip(),
            "model": str(row.get("model", "")).strip(),
            "backbone": str(row.get("backbone", "")).strip(),
            "probe": str(row.get("probe", "")).strip(),
            "status": "blocked",
            "blocked_reason": reason,
            "hydra_overrides_json": "",
            "source_csv_path": str(row.get("_source_csv_path", "")),
            "source_csv_row_index": str(row.get("_source_csv_row_index", "")),
            "source_csv_line_number": str(row.get("_source_csv_line_number", "")),
            "source_row_sha256": str(row.get("_source_row_sha256", "")),
            "source_config_status": str(row.get("config_status", "")).strip(),
            "source_config_json": str(row.get("best_config_json", "")).strip(),
            "source_evidence_path": str(row.get("evidence_path", "")).strip(),
        }
    )
    return base


def _manifest_row(row: dict[str, str], *, seed: int) -> dict[str, str]:
    dataset_hydra = _resolve_dataset(row)
    probe_hydra = _resolve_probe(row)
    backbone_name, backbone_variant = _resolve_backbone(row)
    layer = _resolve_layer(row)
    config_id = _config_id(row)
    run_id = f"{config_id}__seed_{int(seed)}"
    output_subdir = f"seed_runs_v1/{config_id}/seed_{int(seed)}"
    source_config = _parse_json_dict(row.get("best_config_json", ""))

    feature_view = _value(row.get("feature_view")) or str(source_config.get("feature_view", "")) or "pooled"
    batch_size = _value(row.get("batch_size")) or _json_value(source_config, "batch_size") or "128"
    eval_batch_size = _value(row.get("eval_batch_size")) or _json_value(source_config, "eval_batch_size") or "1024"
    epochs = _value(row.get("epochs")) or _json_value(source_config, "epochs") or "100"
    weight_decay = _value(row.get("weight_decay")) or _json_value(source_config, "weight_decay") or "0.0"
    lr = _value(row.get("lr")) or _json_value(source_config, "lr")
    if not lr:
        raise ValueError(f"Missing lr for {config_id}")

    early_stopping = source_config.get("early_stopping", {})
    if not isinstance(early_stopping, dict):
        early_stopping = {}
    early_stopping_enabled = (
        _value(row.get("early_stopping_enabled"))
        or _bool_to_string(early_stopping.get("enabled"))
        or "false"
    )
    early_stopping_patience = (
        _value(row.get("early_stopping_patience"))
        or _json_value(early_stopping, "patience")
        or "5"
    )
    if probe_hydra == "mlp":
        # The recovered CSV records the current config shape, but the old MLP
        # main jobs selected checkpoints after the full epoch budget. Keep seed
        # reruns matched to that runtime behavior.
        early_stopping_enabled = "false"

    mlp_hidden_dims = _value(row.get("mlp_hidden_dims"))
    if not mlp_hidden_dims and probe_hydra == "mlp":
        mlp_hidden_dims = _json_compact(source_config.get("hidden_dims")) or "[512]"
    mlp_dropout = _value(row.get("mlp_dropout"))
    if not mlp_dropout and probe_hydra == "mlp":
        mlp_dropout = _json_value(source_config, "dropout") or "0.0"

    manifest_row = {
        "run_id": run_id,
        "config_id": config_id,
        "run_command": f"train_eval.probe.{dataset_hydra}",
        "dataset": str(row.get("dataset", "")).strip(),
        "dataset_hydra": dataset_hydra,
        "experiment": str(row.get("experiment", "")).strip(),
        "model": str(row.get("model", "")).strip(),
        "backbone": str(row.get("backbone", "")).strip(),
        "probe": str(row.get("probe", "")).strip(),
        "probe_hydra": probe_hydra,
        "seed": str(int(seed)),
        "original_seed": "42",
        "backbone_name": backbone_name,
        "backbone_variant": backbone_variant,
        "layer": str(layer),
        "selected_slot": _value(row.get("selected_slot")),
        "layer_label": str(row.get("selected_layer_label", "")).strip(),
        "feature_view": feature_view,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopping_patience": early_stopping_patience,
        "mlp_hidden_dims": mlp_hidden_dims,
        "mlp_dropout": mlp_dropout,
        "temporal_num_heads": _value(row.get("temporal_num_heads")),
        "temporal_num_self_attn_blocks": _value(row.get("temporal_num_self_attn_blocks")),
        "temporal_mlp_ratio": _value(row.get("temporal_mlp_ratio")),
        "temporal_dropout": _value(row.get("temporal_dropout")),
        "probe_device": "cpu",
        "probe_output_dir": _probe_output_dir(dataset_hydra),
        "probe_output_subdir": output_subdir,
        "eval_output_dir": _eval_output_dir(dataset_hydra),
        "eval_output_subdir": output_subdir,
        "wandb_group": f"seed_runs_v1_{config_id}",
        "wandb_name": run_id,
        "status": "pending",
        "blocked_reason": "",
        "hydra_overrides_json": "",
        "source_csv_path": str(row.get("_source_csv_path", "")),
        "source_csv_row_index": str(row.get("_source_csv_row_index", "")),
        "source_csv_line_number": str(row.get("_source_csv_line_number", "")),
        "source_row_sha256": str(row.get("_source_row_sha256", "")),
        "source_config_status": str(row.get("config_status", "")).strip(),
        "source_config_json": str(row.get("best_config_json", "")).strip(),
        "source_evidence_path": str(row.get("evidence_path", "")).strip(),
    }
    manifest_row["hydra_overrides_json"] = _json_compact(_hydra_overrides(manifest_row))
    return manifest_row


def _hydra_overrides(row: dict[str, str]) -> list[str]:
    overrides = [
        f"seed={row['seed']}",
        "split.seed=42",
        f"backbone.name={row['backbone_name']}",
        f"+backbone.kwargs.variant={row['backbone_variant']}",
        f"probe.name={row['probe_hydra']}",
        f"probe.layer={row['layer']}",
        "probe.layers=[]",
        f"probe.feature_view={row['feature_view']}",
        f"probe.device={row['probe_device']}",
        f"probe.lr={row['lr']}",
        f"probe.weight_decay={row['weight_decay']}",
        f"probe.batch_size={row['batch_size']}",
        f"probe.eval_batch_size={row['eval_batch_size']}",
        f"probe.epochs={row['epochs']}",
        f"probe.early_stopping.enabled={row['early_stopping_enabled']}",
        f"probe.early_stopping.patience={row['early_stopping_patience']}",
        "probe.optuna.enabled=false",
        f"probe.output_dir={row['probe_output_dir']}",
        f"probe.output_subdir={row['probe_output_subdir']}",
        f"probe.eval_output_dir={row['eval_output_dir']}",
        f"probe.eval_output_subdir={row['eval_output_subdir']}",
        f"probe.wandb.group={row['wandb_group']}",
        f"probe.wandb.name={row['wandb_name']}",
    ]
    if row["probe_hydra"] == "mlp":
        overrides.extend(
            [
                f"probe.mlp.hidden_dims={row['mlp_hidden_dims']}",
                f"probe.mlp.dropout={row['mlp_dropout']}",
            ]
        )
    return overrides


def _config_id(row: dict[str, str]) -> str:
    config_id = str(row.get("config_id", "")).strip()
    if config_id:
        return config_id
    raise ValueError(f"Missing config_id for row: {row}")


def _resolve_dataset(row: dict[str, str]) -> str:
    raw = str(row.get("dataset", "")).strip()
    try:
        return DATASET_TO_HYDRA[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset label {raw!r}") from exc


def _resolve_probe(row: dict[str, str]) -> str:
    raw = str(row.get("probe", "")).strip()
    try:
        return PROBE_TO_HYDRA[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported probe label {raw!r}") from exc


def _resolve_backbone(row: dict[str, str]) -> tuple[str, str]:
    key = (str(row.get("model", "")).strip(), str(row.get("backbone", "")).strip())
    try:
        return BACKBONE_TO_HYDRA[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported model/backbone mapping: {key!r}") from exc


def _resolve_layer(row: dict[str, str]) -> str:
    layer = _value(row.get("selected_layer_id"))
    if layer:
        return layer
    config = _parse_json_dict(row.get("best_config_json", ""))
    layer_value = config.get("layer")
    if layer_value is not None:
        return str(layer_value)
    raise ValueError(f"Missing selected layer for {_config_id(row)}")


def _probe_output_dir(dataset_hydra: str) -> str:
    if dataset_hydra == "intphys2":
        return "artifacts/probes/intphys2"
    if dataset_hydra == "mvp":
        return "artifacts/probes/mvp"
    raise ValueError(f"Unsupported dataset_hydra={dataset_hydra!r}")


def _eval_output_dir(dataset_hydra: str) -> str:
    if dataset_hydra == "intphys2":
        return "artifacts/results/intphys2"
    if dataset_hydra == "mvp":
        return "artifacts/results/mvp"
    raise ValueError(f"Unsupported dataset_hydra={dataset_hydra!r}")


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    value = _value(raw)
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got: {value}")
    return parsed


def _value(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text in NULL_VALUES else text


def _json_value(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_to_string(value)
    return str(value)


def _json_compact(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"))


def _bool_to_string(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


if __name__ == "__main__":
    main()
