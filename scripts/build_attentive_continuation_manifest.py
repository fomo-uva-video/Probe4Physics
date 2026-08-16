from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_MANIFESTS = (
    Path("results/seed_runs/seed_manifest_layerwise_main_attentive_intphys2_v1.csv"),
    Path("results/seed_runs/seed_manifest_layerwise_same_l_attentive_v1.csv"),
    Path("results/seed_runs/seed_manifest_layerwise_backbone_sweep_attentive_v1.csv"),
)
DEFAULT_OUTPUT = Path("results/seed_runs/attentive_continuation_manifest_v1.csv")
DEFAULT_SKIPPED_OUTPUT = Path("results/seed_runs/attentive_continuation_manifest_skipped_v1.csv")
DEFAULT_RUN_GROUP = "seed_runs_attentive_continued_to_max_epoch_v1"
BACKUP_MARKER = "artifacts/backups/attentive_pre_full_epoch_20260813"


EXTRA_FIELDS = [
    "source_manifest_path",
    "source_manifest_row_index",
    "source_manifest_line_number",
    "source_manifest_row_sha256",
    "source_train_dir",
    "source_train_summary",
    "source_train_summary_sha256",
    "source_checkpoint",
    "source_checkpoint_size_bytes",
    "old_completed_epochs",
    "old_best_epoch",
    "old_early_stopped",
    "target_total_epochs",
    "continuation_epochs",
    "continuation_protocol",
    "continuation_output_dir",
    "continuation_eval_output_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a guarded manifest for continuing attentive seed runs to their target epoch."
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        type=Path,
        default=None,
        help="Existing attentive seed manifest to scan. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skipped-output", type=Path, default=DEFAULT_SKIPPED_OUTPUT)
    parser.add_argument("--run-group", default=DEFAULT_RUN_GROUP)
    args = parser.parse_args()

    source_manifests = tuple(args.source_manifest or DEFAULT_SOURCE_MANIFESTS)
    runnable: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for source_manifest in source_manifests:
        rows = _read_csv(source_manifest)
        if not rows:
            continue
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) + EXTRA_FIELDS
        for index, row in enumerate(rows):
            annotated = dict(row)
            annotated["_source_manifest_path"] = str(source_manifest)
            annotated["_source_manifest_row_index"] = str(index)
            annotated["_source_manifest_line_number"] = str(index + 2)
            annotated["_source_manifest_row_sha256"] = _row_hash(row)

            built, reason = _build_row(annotated, run_group=args.run_group)
            if reason:
                skipped.append(_skipped_row(annotated, reason, fieldnames))
                continue
            runnable.append(built)

    if fieldnames is None:
        fieldnames = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output, runnable, fieldnames)
    _write_csv(args.skipped_output, skipped, fieldnames + ["skip_reason"])

    total_remaining = sum(int(row["continuation_epochs"]) for row in runnable)
    summary = {
        "source_manifests": [str(path) for path in source_manifests],
        "output": str(args.output),
        "skipped_output": str(args.skipped_output),
        "run_group": args.run_group,
        "pending_rows": len(runnable),
        "skipped_rows": len(skipped),
        "total_remaining_epochs": total_remaining,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_row(row: dict[str, str], *, run_group: str) -> tuple[dict[str, str], str]:
    if row.get("status", "").strip() != "pending":
        return {}, "source manifest row is not pending"
    if row.get("blocked_reason", "").strip():
        return {}, f"source manifest row is blocked: {row['blocked_reason']}"
    if row.get("probe_hydra", "").strip() != "temporal_attn":
        return {}, "source manifest row is not temporal_attn"

    try:
        layer = int(row["layer"])
        target_total_epochs = int(float(row["epochs"]))
    except (KeyError, ValueError) as exc:
        return {}, f"invalid source layer/epochs: {exc}"

    train_dir = _source_train_dir(row, layer=layer)
    summary_path = train_dir / "train_summary.json"
    if not summary_path.exists():
        return {}, f"missing train_summary.json: {summary_path}"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid train_summary.json: {exc}"

    probe_name = str(summary.get("probe_name", "")).strip()
    if probe_name != "temporal_attn":
        return {}, f"summary probe_name is not temporal_attn: {probe_name!r}"
    if int(summary.get("layer", -1)) != layer:
        return {}, f"summary layer {summary.get('layer')!r} does not match manifest layer {layer}"

    old_completed_epochs = _completed_epochs(summary)
    if old_completed_epochs <= 0:
        return {}, "summary has no completed epochs"
    continuation_epochs = target_total_epochs - old_completed_epochs
    if continuation_epochs <= 0:
        return {}, f"already reached target: {old_completed_epochs}/{target_total_epochs}"

    checkpoint = _checkpoint_path(summary, train_dir=train_dir)
    if _contains_backup_marker(checkpoint):
        return {}, f"source checkpoint points inside backup: {checkpoint}"
    if not checkpoint.exists():
        return {}, f"missing source checkpoint: {checkpoint}"

    output_subdir = f"{run_group}/{row['config_id']}/seed_{row['seed']}"
    output_dir = Path(row["probe_output_dir"]) / output_subdir
    eval_output_dir = Path(row["eval_output_dir"]) / output_subdir
    if _contains_backup_marker(output_dir) or _contains_backup_marker(eval_output_dir):
        return {}, "continuation output points inside backup"

    overrides = _continuation_overrides(
        _manifest_overrides(row),
        row=row,
        run_group=run_group,
        output_subdir=output_subdir,
        continuation_epochs=continuation_epochs,
        checkpoint=checkpoint,
    )

    out = dict(row)
    out["run_id"] = f"{row['run_id']}__continue_to_epoch_{target_total_epochs}"
    out["epochs"] = str(continuation_epochs)
    out["early_stopping_enabled"] = "false"
    out["probe_output_subdir"] = output_subdir
    out["eval_output_subdir"] = output_subdir
    out["wandb_group"] = f"{run_group}_{row['config_id']}"
    out["wandb_name"] = out["run_id"]
    out["hydra_overrides_json"] = _json_compact(overrides)
    out["source_csv_path"] = str(row["_source_manifest_path"])
    out["source_csv_row_index"] = str(row["_source_manifest_row_index"])
    out["source_csv_line_number"] = str(row["_source_manifest_line_number"])
    out["source_row_sha256"] = str(row["_source_manifest_row_sha256"])
    out["source_config_status"] = "continuation_from_official_seed_manifest"
    out["source_config_json"] = ""
    out["source_evidence_path"] = str(summary_path)
    out.update(
        {
            "source_manifest_path": str(row["_source_manifest_path"]),
            "source_manifest_row_index": str(row["_source_manifest_row_index"]),
            "source_manifest_line_number": str(row["_source_manifest_line_number"]),
            "source_manifest_row_sha256": str(row["_source_manifest_row_sha256"]),
            "source_train_dir": str(train_dir),
            "source_train_summary": str(summary_path),
            "source_train_summary_sha256": _file_hash(summary_path),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_size_bytes": str(checkpoint.stat().st_size),
            "old_completed_epochs": str(old_completed_epochs),
            "old_best_epoch": str(summary.get("fit", {}).get("best_epoch", "")),
            "old_early_stopped": _bool_to_string(summary.get("fit", {}).get("early_stopped")),
            "target_total_epochs": str(target_total_epochs),
            "continuation_epochs": str(continuation_epochs),
            "continuation_protocol": "resume_probe_last_disable_early_stopping_to_target_epoch",
            "continuation_output_dir": str(output_dir),
            "continuation_eval_output_dir": str(eval_output_dir),
        }
    )
    return out, ""


def _source_train_dir(row: dict[str, str], *, layer: int) -> Path:
    return (
        Path(row["probe_output_dir"])
        / row["probe_output_subdir"]
        / f"layer_{layer}"
        / "train"
    )


def _completed_epochs(summary: dict[str, Any]) -> int:
    history = summary.get("fit", {}).get("history", [])
    if not isinstance(history, list) or not history:
        return 0
    epochs: list[int] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            epochs.append(int(float(item["epoch"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(epochs, default=0)


def _checkpoint_path(summary: dict[str, Any], *, train_dir: Path) -> Path:
    raw = str(summary.get("checkpoint_last") or summary.get("checkpoint") or "").strip()
    path = Path(raw) if raw else train_dir / "probe_last.pt"
    if not path.is_absolute():
        path = Path(path)
    return path


def _continuation_overrides(
    overrides: list[str],
    *,
    row: dict[str, str],
    run_group: str,
    output_subdir: str,
    continuation_epochs: int,
    checkpoint: Path,
) -> list[str]:
    replacements = {
        "probe.epochs": str(continuation_epochs),
        "probe.early_stopping.enabled": "false",
        "probe.output_subdir": output_subdir,
        "probe.eval_output_subdir": output_subdir,
        "probe.wandb.group": f"{run_group}_{row['config_id']}",
        "probe.wandb.name": f"{row['run_id']}__continue_to_epoch_{row['epochs']}",
        "probe.init_checkpoint_path": str(checkpoint),
    }
    seen: set[str] = set()
    out: list[str] = []
    for item in overrides:
        key = item.split("=", 1)[0]
        if key in replacements:
            out.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            out.append(item)
    for key, value in replacements.items():
        if key not in seen:
            out.append(f"{key}={value}")
    return out


def _skipped_row(row: dict[str, str], reason: str, fieldnames: list[str]) -> dict[str, str]:
    out = {field: row.get(field, "") for field in fieldnames}
    out.update(
        {
            "source_manifest_path": str(row.get("_source_manifest_path", "")),
            "source_manifest_row_index": str(row.get("_source_manifest_row_index", "")),
            "source_manifest_line_number": str(row.get("_source_manifest_line_number", "")),
            "source_manifest_row_sha256": str(row.get("_source_manifest_row_sha256", "")),
            "skip_reason": reason,
        }
    )
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_overrides(row: dict[str, str]) -> list[str]:
    parsed = json.loads(row.get("hydra_overrides_json", ""))
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid hydra_overrides_json for {row.get('run_id')!r}")
    return parsed


def _row_hash(row: dict[str, str]) -> str:
    payload = {str(key): str(value) for key, value in row.items() if not str(key).startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _bool_to_string(value: Any) -> str:
    return "true" if bool(value) else "false"


def _contains_backup_marker(path: Path) -> bool:
    return BACKUP_MARKER in str(path)


if __name__ == "__main__":
    main()
