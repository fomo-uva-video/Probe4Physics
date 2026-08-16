from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("results/seed_runs/attentive_continuation_manifest_v1.csv")
EXPECTED_RUN_GROUP = "seed_runs_attentive_continued_to_max_epoch_v1"
BACKUP_MARKER = "artifacts/backups/attentive_pre_full_epoch_20260813"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate attentive continuation manifest rows before Slurm submission."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--expected-run-group", default=EXPECTED_RUN_GROUP)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow rows whose continuation train_summary.json already exists.",
    )
    args = parser.parse_args()

    rows = _read_csv(args.manifest)
    selected = rows if args.row_index is None else [_select_row(rows, args.row_index)]
    errors: list[str] = []
    for index, row in enumerate(selected):
        source_index = args.row_index if args.row_index is not None else index
        try:
            validate_row(
                row,
                manifest_path=args.manifest,
                expected_run_group=args.expected_run_group,
                allow_existing_output=args.allow_existing_output,
            )
        except Exception as exc:  # noqa: BLE001 - validation CLI should report all row failures.
            errors.append(f"row {source_index} {row.get('run_id', '')}: {exc}")

    summary = {
        "manifest": str(args.manifest),
        "checked_rows": len(selected),
        "errors": len(errors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        if len(errors) > 50:
            print(f"ERROR: ... {len(errors) - 50} additional errors")
        raise SystemExit(1)


def validate_row(
    row: dict[str, str],
    *,
    manifest_path: Path,
    expected_run_group: str = EXPECTED_RUN_GROUP,
    allow_existing_output: bool = False,
) -> None:
    run_id = row.get("run_id", "").strip()
    if row.get("status", "").strip() != "pending":
        raise ValueError("row status is not pending")
    if row.get("probe_hydra", "").strip() != "temporal_attn":
        raise ValueError("row is not temporal_attn")
    if row.get("early_stopping_enabled", "").strip().lower() != "false":
        raise ValueError("continuation row must disable early stopping")

    old_completed = _positive_int(row, "old_completed_epochs")
    continuation = _positive_int(row, "continuation_epochs")
    target = _positive_int(row, "target_total_epochs")
    if old_completed + continuation != target:
        raise ValueError(
            f"epoch arithmetic mismatch: {old_completed}+{continuation}!={target}"
        )
    if int(float(row.get("epochs", "0"))) != continuation:
        raise ValueError("manifest epochs column does not equal continuation_epochs")

    source_summary = Path(row["source_train_summary"])
    source_checkpoint = Path(row["source_checkpoint"])
    source_train_dir = Path(row["source_train_dir"])
    output_dir = Path(row["continuation_output_dir"])
    eval_output_dir = Path(row["continuation_eval_output_dir"])

    for label, path in (
        ("source_train_summary", source_summary),
        ("source_checkpoint", source_checkpoint),
        ("source_train_dir", source_train_dir),
        ("continuation_output_dir", output_dir),
        ("continuation_eval_output_dir", eval_output_dir),
    ):
        if _contains_backup_marker(path):
            raise ValueError(f"{label} points inside backup: {path}")

    if not source_summary.exists():
        raise FileNotFoundError(source_summary)
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    if str(source_checkpoint_size := source_checkpoint.stat().st_size) != row["source_checkpoint_size_bytes"]:
        raise ValueError(
            f"source checkpoint size changed: manifest={row['source_checkpoint_size_bytes']} current={source_checkpoint_size}"
        )
    if _file_hash(source_summary) != row["source_train_summary_sha256"]:
        raise ValueError("source train_summary hash changed")
    if str(output_dir) == str(source_train_dir) or str(eval_output_dir) == str(source_train_dir):
        raise ValueError("continuation output aliases the source train dir")

    if not row["probe_output_subdir"].startswith(f"{expected_run_group}/"):
        raise ValueError("probe_output_subdir is not in the continuation namespace")
    if not row["eval_output_subdir"].startswith(f"{expected_run_group}/"):
        raise ValueError("eval_output_subdir is not in the continuation namespace")
    completed_output = output_dir / f"layer_{row['layer']}" / "train" / "train_summary.json"
    if completed_output.exists() and not allow_existing_output:
        raise ValueError(f"continuation output already complete: {completed_output}")

    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if summary.get("probe_name") != "temporal_attn":
        raise ValueError("source summary is not temporal_attn")
    if int(summary.get("layer", -1)) != int(row["layer"]):
        raise ValueError("source summary layer does not match manifest layer")
    if _completed_epochs(summary) != old_completed:
        raise ValueError("source summary completed epochs changed")

    overrides = _manifest_overrides(row)
    required = {
        "probe.name=temporal_attn",
        f"seed={row['seed']}",
        f"probe.layer={row['layer']}",
        "probe.device=cuda",
        "probe.feature_view=tokens",
        "feature_cache.include_tokens=true",
        f"probe.epochs={continuation}",
        "probe.early_stopping.enabled=false",
        f"probe.init_checkpoint_path={source_checkpoint}",
        f"probe.output_subdir={row['probe_output_subdir']}",
        f"probe.eval_output_subdir={row['eval_output_subdir']}",
    }
    missing = sorted(required - set(overrides))
    if missing:
        raise ValueError(f"missing required Hydra overrides: {missing}")

    _validate_source_manifest_hash(row, manifest_path=manifest_path)
    if not run_id.endswith(f"continue_to_epoch_{target}"):
        raise ValueError("run_id does not encode the continuation target epoch")


def _select_row(rows: list[dict[str, str]], index: int) -> dict[str, str]:
    if index < 0 or index >= len(rows):
        raise ValueError(f"row-index {index} out of range for {len(rows)} rows")
    return rows[index]


def _positive_int(row: dict[str, str], key: str) -> int:
    value = int(float(row.get(key, "")))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _validate_source_manifest_hash(row: dict[str, str], *, manifest_path: Path) -> None:
    source_path = Path(row["source_manifest_path"])
    if not source_path.is_absolute():
        source_path = manifest_path.parent.parent.parent / source_path
    source_rows = _read_csv(source_path)
    index = int(row["source_manifest_row_index"])
    source_row = _select_row(source_rows, index)
    if _row_hash(source_row) != row["source_manifest_row_sha256"]:
        raise ValueError("source manifest row hash changed")
    if row.get("source_row_sha256", "") != row["source_manifest_row_sha256"]:
        raise ValueError("source_row_sha256 does not match source_manifest_row_sha256")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_overrides(row: dict[str, str]) -> list[str]:
    parsed = json.loads(row.get("hydra_overrides_json", ""))
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid hydra_overrides_json for {row.get('run_id')!r}")
    return parsed


def _completed_epochs(summary: dict[str, Any]) -> int:
    history = summary.get("fit", {}).get("history", [])
    if not isinstance(history, list):
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


def _contains_backup_marker(path: Path) -> bool:
    return BACKUP_MARKER in str(path)


if __name__ == "__main__":
    main()
