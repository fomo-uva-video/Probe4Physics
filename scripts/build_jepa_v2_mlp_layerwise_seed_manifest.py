from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "seed_runs"
SOURCE_PATH = OUT_DIR / "jepa_v2_mlp_layerwise_seed_source_v1.csv"
MANIFEST_PATH = OUT_DIR / "seed_manifest_jepa_v2_mlp_layerwise_v1.csv"

SEEDS = (42, 101, 102)
GROUP_SUBDIR = "seed_runs_jepa_v2_mlp_layerwise_v1"

LOGS = {
    "intphys2": ROOT
    / "jobs/train/intphys2/mlp/output/training/intphys2/mlp/intphys2_jepa_v2_mlp_layers_22594076.out",
    "mvp": ROOT / "jobs/train/mvp/mlp/output/training/mvp/mlp/mvp_jepa_v2_mlp_layers_22594132.out",
}

DATASET_META = {
    "intphys2": {
        "dataset": "IntPhys2",
        "run_command": "train_eval.probe.intphys2",
        "eval_output_dir": "artifacts/results/intphys2",
        "probe_output_dir": "artifacts/probes/intphys2",
    },
    "mvp": {
        "dataset": "MVP",
        "run_command": "train_eval.probe.mvp",
        "eval_output_dir": "artifacts/results/mvp",
        "probe_output_dir": "artifacts/probes/mvp",
    },
}

SOURCE_FIELDS = [
    "config_id",
    "dataset_hydra",
    "dataset",
    "model",
    "backbone",
    "probe",
    "layer",
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
    "best_trial",
    "n_trials",
    "source_evidence_path",
    "source_feature_signature",
    "source_config_json",
]

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
    source_rows = _build_source_rows()
    manifest_rows = _build_manifest_rows(source_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(SOURCE_PATH, SOURCE_FIELDS, source_rows)
    _write_csv(MANIFEST_PATH, MANIFEST_FIELDS, manifest_rows)
    print(f"wrote {SOURCE_PATH.relative_to(ROOT)} ({len(source_rows)} rows)")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)} ({len(manifest_rows)} rows)")


def _build_source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_hydra, log_path in LOGS.items():
        data = _load_json_from_log(log_path)
        meta = DATASET_META[dataset_hydra]
        for layer in data["layers"]:
            train = layer["train"]
            params = _normalize_params(train["best_params"])
            layer_id = int(layer["layer"])
            config_json = {
                "batch_size": params["batch_size"],
                "dropout": params["dropout"],
                "early_stopping": {"enabled": False, "patience": 5},
                "epochs": params["epochs"],
                "feature_view": "pooled",
                "hidden_dims": params["hidden_dims"],
                "layer": layer_id,
                "lr": params["lr"],
                "slot": None,
                "weight_decay": params["weight_decay"],
            }
            rows.append(
                {
                    "config_id": f"{dataset_hydra}__jepa_v2__mlp__layer_{layer_id}",
                    "dataset_hydra": dataset_hydra,
                    "dataset": meta["dataset"],
                    "model": "V-JEPA 2",
                    "backbone": "ViT-G/16",
                    "probe": "MLP",
                    "layer": str(layer_id),
                    "feature_view": "pooled",
                    "lr": str(params["lr"]),
                    "weight_decay": str(params["weight_decay"]),
                    "batch_size": str(params["batch_size"]),
                    "eval_batch_size": "1024",
                    "epochs": str(params["epochs"]),
                    "early_stopping_enabled": "false",
                    "early_stopping_patience": "5",
                    "mlp_hidden_dims": _format_hydra_list(params["hidden_dims"]),
                    "mlp_dropout": str(params["dropout"]),
                    "best_trial": str(train["best_trial_number"]),
                    "n_trials": str(train["n_trials"]),
                    "source_evidence_path": str(log_path.relative_to(ROOT)),
                    "source_feature_signature": str(layer["eval"]["feature_signature"]),
                    "source_config_json": _json_compact(config_json),
                }
            )
    return rows


def _build_manifest_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_index, source_row in enumerate(source_rows):
        source_hash = _source_row_hash(source_row)
        for seed in SEEDS:
            rows.append(_manifest_row(source_row, source_index, source_hash, seed))
    return rows


def _manifest_row(
    source_row: dict[str, str],
    source_index: int,
    source_hash: str,
    seed: int,
) -> dict[str, str]:
    dataset_hydra = source_row["dataset_hydra"]
    meta = DATASET_META[dataset_hydra]
    config_id = source_row["config_id"]
    run_id = f"{config_id}__seed_{seed}"
    output_subdir = f"{GROUP_SUBDIR}/{config_id}/seed_{seed}"
    row = {
        "run_id": run_id,
        "config_id": config_id,
        "run_command": meta["run_command"],
        "dataset": source_row["dataset"],
        "dataset_hydra": dataset_hydra,
        "experiment": "main_layerwise_seed_pilot",
        "model": "V-JEPA 2",
        "backbone": "ViT-G/16",
        "probe": "MLP",
        "probe_hydra": "mlp",
        "seed": str(seed),
        "original_seed": "42",
        "backbone_name": "jepa_v2",
        "backbone_variant": "vitg_384",
        "layer": source_row["layer"],
        "selected_slot": "",
        "layer_label": source_row["layer"],
        "feature_view": source_row["feature_view"],
        "lr": source_row["lr"],
        "weight_decay": source_row["weight_decay"],
        "batch_size": source_row["batch_size"],
        "eval_batch_size": source_row["eval_batch_size"],
        "epochs": source_row["epochs"],
        "early_stopping_enabled": source_row["early_stopping_enabled"],
        "early_stopping_patience": source_row["early_stopping_patience"],
        "mlp_hidden_dims": source_row["mlp_hidden_dims"],
        "mlp_dropout": source_row["mlp_dropout"],
        "temporal_num_heads": "",
        "temporal_num_self_attn_blocks": "",
        "temporal_mlp_ratio": "",
        "temporal_dropout": "",
        "probe_device": "cpu",
        "probe_output_dir": meta["probe_output_dir"],
        "probe_output_subdir": output_subdir,
        "eval_output_dir": meta["eval_output_dir"],
        "eval_output_subdir": output_subdir,
        "wandb_group": f"{GROUP_SUBDIR}_{config_id}",
        "wandb_name": run_id,
        "status": "pending",
        "blocked_reason": "",
        "hydra_overrides_json": "",
        "source_csv_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_csv_row_index": str(source_index),
        "source_csv_line_number": str(source_index + 2),
        "source_row_sha256": source_hash,
        "source_config_status": "RECOVERED_LAYERWISE_LOG",
        "source_config_json": source_row["source_config_json"],
        "source_evidence_path": source_row["source_evidence_path"],
    }
    row["hydra_overrides_json"] = _json_compact(_hydra_overrides(row))
    return row


def _hydra_overrides(row: dict[str, str]) -> list[str]:
    return [
        f"seed={row['seed']}",
        "split.seed=42",
        f"backbone.name={row['backbone_name']}",
        f"+backbone.kwargs.variant={row['backbone_variant']}",
        "probe.name=mlp",
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
        f"probe.mlp.hidden_dims={row['mlp_hidden_dims']}",
        f"probe.mlp.dropout={row['mlp_dropout']}",
    ]


def _normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    hidden = raw["hidden_dims"]
    if isinstance(hidden, str):
        hidden_dims = [int(item) for item in hidden.replace("x", ",").split(",") if item]
    else:
        hidden_dims = [int(item) for item in hidden]
    return {
        "batch_size": int(raw["batch_size"]),
        "dropout": float(raw["dropout"]),
        "epochs": int(raw["epochs"]),
        "hidden_dims": hidden_dims,
        "lr": float(raw["lr"]),
        "weight_decay": float(raw["weight_decay"]),
    }


def _load_json_from_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in {path}")
    return json.loads(text[start : end + 1])


def _source_row_hash(row: dict[str, str]) -> str:
    payload = {str(key): str(value) for key, value in row.items() if not str(key).startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _format_hydra_list(values: list[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
