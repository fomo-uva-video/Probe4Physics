from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("results/seed_runs/seed_manifest_main_linear_mlp_v1.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one row from a deterministic seed-rerun manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Index within the selected manifest rows. Defaults to SLURM_ARRAY_TASK_ID=0 in the shell wrapper.",
    )
    parser.add_argument(
        "--probe",
        choices=("linear", "mlp", "temporal_attn"),
        default=None,
        help="Filter manifest rows by probe_hydra before applying --task-index.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides appended after the manifest overrides.",
    )
    args, unknown_overrides = parser.parse_known_args()
    extra_overrides = [*args.overrides, *unknown_overrides]

    task_index = 0 if args.task_index is None else args.task_index
    rows = _read_csv(args.manifest)
    selected_rows = _select_rows(rows, probe=args.probe)
    if task_index < 0 or task_index >= len(selected_rows):
        raise SystemExit(
            f"Invalid task index {task_index}; selected manifest has {len(selected_rows)} rows."
        )

    row = selected_rows[task_index]
    _validate_row(row, manifest_path=args.manifest)
    command = _build_command(row, extra_overrides=extra_overrides)

    _print_provenance(row, task_index=task_index, command=command, dry_run=args.dry_run)
    if args.dry_run:
        return

    subprocess.run(command, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_rows(rows: list[dict[str, str]], *, probe: str | None) -> list[dict[str, str]]:
    selected = rows
    if probe is not None:
        selected = [row for row in selected if row.get("probe_hydra", "").strip() == probe]
    return selected


def _validate_row(row: dict[str, str], *, manifest_path: Path) -> None:
    run_id = row.get("run_id", "").strip()
    if row.get("status", "").strip() != "pending":
        raise ValueError(f"Manifest row {run_id!r} is not pending.")
    if row.get("blocked_reason", "").strip():
        raise ValueError(f"Manifest row {run_id!r} is blocked: {row['blocked_reason']}")
    probe_hydra = row.get("probe_hydra", "").strip()
    if probe_hydra not in {"linear", "mlp", "temporal_attn"}:
        raise ValueError(f"Unsupported probe_hydra for seed runner: {row.get('probe_hydra')!r}")

    overrides = _manifest_overrides(row)
    required = {
        "probe.optuna.enabled=false",
        f"seed={row['seed']}",
        f"probe.name={probe_hydra}",
        f"probe.layer={row['layer']}",
    }
    if probe_hydra == "temporal_attn":
        required.update(
            {
                "probe.device=cuda",
                "probe.feature_view=tokens",
                "feature_cache.include_tokens=true",
            }
        )
    else:
        required.add("probe.device=cpu")
    missing = sorted(required - set(overrides))
    if missing:
        raise ValueError(f"Manifest row {run_id!r} is missing required overrides: {missing}")

    _validate_source_hash(row, manifest_path=manifest_path)


def _manifest_overrides(row: dict[str, str]) -> list[str]:
    raw = row.get("hydra_overrides_json", "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid hydra_overrides_json for {row.get('run_id')!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"hydra_overrides_json must be a JSON list of strings for {row.get('run_id')!r}")
    return parsed


def _validate_source_hash(row: dict[str, str], *, manifest_path: Path) -> None:
    source_path = Path(row.get("source_csv_path", "").strip())
    if not source_path.is_absolute():
        source_path = manifest_path.parent.parent.parent / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV for {row.get('run_id')!r} not found: {source_path}")

    source_index = int(row["source_csv_row_index"])
    source_rows = _read_csv(source_path)
    try:
        source_row = source_rows[source_index]
    except IndexError as exc:
        raise ValueError(
            f"source_csv_row_index={source_index} is out of range for {source_path}"
        ) from exc

    digest = _source_row_hash(source_row)
    expected = row.get("source_row_sha256", "").strip()
    if digest != expected:
        raise ValueError(
            f"Source hash mismatch for {row.get('run_id')!r}: manifest={expected}, current={digest}"
        )


def _source_row_hash(row: dict[str, str]) -> str:
    payload = {
        str(key): str(value)
        for key, value in row.items()
        if not str(key).startswith("_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _build_command(row: dict[str, str], *, extra_overrides: list[str]) -> list[str]:
    run_command = row.get("run_command", "").strip()
    if not run_command:
        raise ValueError(f"Missing run_command for {row.get('run_id')!r}")
    return [sys.executable, "run.py", run_command, *_manifest_overrides(row), *extra_overrides]


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
        "config_id": row.get("config_id", ""),
        "dataset": row.get("dataset_hydra", ""),
        "probe": row.get("probe_hydra", ""),
        "seed": row.get("seed", ""),
        "source_csv_line_number": row.get("source_csv_line_number", ""),
        "command": command,
        "command_shell": " ".join(shlex.quote(part) for part in command),
    }
    print("===== SEED MANIFEST ROW =====")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("=============================")


if __name__ == "__main__":
    main()
