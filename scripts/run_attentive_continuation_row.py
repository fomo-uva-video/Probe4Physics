from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from validate_attentive_continuation_manifest import validate_row


DEFAULT_MANIFEST = Path("results/seed_runs/attentive_continuation_manifest_v1.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one attentive continuation manifest row.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Manifest row index. Defaults to SLURM_ARRAY_TASK_ID, then 0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides appended last.")
    args, unknown_overrides = parser.parse_known_args()
    extra_overrides = [*args.overrides, *unknown_overrides]

    task_index = _task_index(args.task_index)
    rows = _read_csv(args.manifest)
    if task_index < 0 or task_index >= len(rows):
        raise SystemExit(f"Invalid task index {task_index}; manifest has {len(rows)} rows.")
    row = rows[task_index]
    validate_row(row, manifest_path=args.manifest)
    command = _build_command(row, extra_overrides=extra_overrides)

    _print_provenance(row, task_index=task_index, command=command, dry_run=args.dry_run)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def _task_index(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    raw = __import__("os").environ.get("SLURM_ARRAY_TASK_ID", "0")
    return int(raw)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_command(row: dict[str, str], *, extra_overrides: list[str]) -> list[str]:
    run_command = row.get("run_command", "").strip()
    if not run_command:
        raise ValueError(f"Missing run_command for {row.get('run_id')!r}")
    overrides = json.loads(row["hydra_overrides_json"])
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise ValueError(f"Invalid hydra_overrides_json for {row.get('run_id')!r}")
    return [sys.executable, "run.py", run_command, *overrides, *extra_overrides]


def _print_provenance(
    row: dict[str, str],
    *,
    task_index: int,
    command: list[str],
    dry_run: bool,
) -> None:
    payload: dict[str, Any] = {
        "dry_run": dry_run,
        "task_index": task_index,
        "run_id": row.get("run_id", ""),
        "dataset": row.get("dataset_hydra", ""),
        "experiment": row.get("experiment", ""),
        "model": row.get("model", ""),
        "backbone": row.get("backbone", ""),
        "layer": row.get("layer", ""),
        "seed": row.get("seed", ""),
        "source_checkpoint": row.get("source_checkpoint", ""),
        "old_completed_epochs": row.get("old_completed_epochs", ""),
        "continuation_epochs": row.get("continuation_epochs", ""),
        "target_total_epochs": row.get("target_total_epochs", ""),
        "continuation_output_dir": row.get("continuation_output_dir", ""),
        "command": command,
        "command_shell": " ".join(shlex.quote(part) for part in command),
    }
    print("===== ATTENTIVE CONTINUATION ROW =====")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("======================================")


if __name__ == "__main__":
    main()
