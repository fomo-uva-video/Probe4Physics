from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.intphys2.core import (
    IntPhys2Benchmark,
    IntPhys2Prediction,
    asdict_metrics,
)
from benchmarks.intphys2.data import derive_sample_id, load_intphys2_rows, normalize_intphys2_row


class ConfigError(ValueError):
    pass


def run_intphys2_eval(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    seed = int(config["seed"])
    random.seed(seed)

    metadata_file = Path(config["metadata_file"])
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"IntPhys2 metadata file not found: {metadata_file}. "
            "Download the dataset and set metadata_file in configs/intphys2.yaml."
        )

    split_cfg = _split_cfg(config)
    split_dir = Path(split_cfg["dir"])
    split_scenes_path = split_dir / "split_scenes.parquet"
    manifest_path = split_dir / "manifest.json"

    split_name = str(config["split_name"])
    predictor_mode = str(config["predictor"]["mode"])

    _log_eval(
        "Starting eval",
        split_name=split_name,
        split_dir=str(split_dir),
        predictor_mode=predictor_mode,
        metadata_file=str(metadata_file),
    )

    if not split_scenes_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Split artifacts not found. Run init first with: python run.py init.intphys2\n"
            f"Missing: {split_scenes_path} or {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_sha = _sha256_file(metadata_file)
    expected_sha = str(manifest.get("metadata_sha256", ""))
    if metadata_sha != expected_sha:
        raise ConfigError(
            "Metadata hash mismatch with split manifest.\n"
            f"Current metadata: {metadata_file}\n"
            f"Current sha256: {metadata_sha}\n"
            f"Manifest sha256: {expected_sha}\n"
            "Re-run init.intphys2 to regenerate split artifacts for this metadata file."
        )

    split_scene_rows = _load_split_scenes(split_scenes_path)
    scene_rows_for_split = [
        row for row in split_scene_rows if str(row.get("split", "")) == split_name
    ]
    if not scene_rows_for_split:
        raise ConfigError(
            f"No scenes found for split_name='{split_name}' in {split_scenes_path}"
        )

    ordered_sample_ids: list[str] = []
    for row in sorted(scene_rows_for_split, key=lambda item: str(item.get("scene_id", ""))):
        ordered_sample_ids.extend(_parse_sample_ids_json(row.get("sample_ids_json", "[]")))

    _log_eval(
        "Loaded split assignments",
        split_scenes=len(scene_rows_for_split),
        split_samples=len(ordered_sample_ids),
    )

    all_rows = load_intphys2_rows(metadata_file)
    norm_rows = [normalize_intphys2_row(row) for row in all_rows]
    row_by_sample: dict[str, dict[str, Any]] = {}
    target_ids = set(ordered_sample_ids)
    for norm_row in norm_rows:
        sample_id = derive_sample_id(norm_row)
        if sample_id in target_ids and sample_id not in row_by_sample:
            row_by_sample[sample_id] = norm_row

    missing = [sid for sid in ordered_sample_ids if sid not in row_by_sample]
    if missing:
        raise ConfigError(
            "Split references sample_ids missing from metadata file. "
            f"First missing ids: {missing[:5]}"
        )

    selected_rows = [row_by_sample[sid] for sid in ordered_sample_ids]

    benchmark = IntPhys2Benchmark()
    samples = benchmark.load_samples(selected_rows, split=split_name)

    predictions = _predict(samples, config)
    _log_eval("Generated predictions", n_predictions=len(predictions))

    metrics = benchmark.evaluate(samples, predictions)
    log_fields: dict[str, Any] = {
        "accuracy": f"{metrics.accuracy:.4f}",
        "voe_accuracy": f"{metrics.voe_accuracy:.4f}",
        "n_scenes": metrics.n_scenes,
        "n_samples": metrics.n_samples,
    }
    if metrics.roc_auc is not None:
        log_fields["roc_auc"] = f"{metrics.roc_auc:.6f}"
    _log_eval("Scoring complete", **log_fields)

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(output_dir / "metrics.json", metrics)
    _write_predictions(output_dir / "predictions.csv", samples, predictions)
    _write_summary(output_dir / "summary.md", metrics, config)
    _write_config_snapshot(output_dir / "run_config.snapshot.yaml", config)
    _write_provenance(output_dir / "provenance.json", config, split_dir, manifest_path)
    _log_eval("Artifacts written", output_dir=str(output_dir))

    return {
        "output_dir": str(output_dir),
        "split_dir": str(split_dir),
        "metrics": asdict_metrics(metrics),
    }


def _log_eval(message: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        print(f"[eval.intphys2] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[eval.intphys2] {message}", file=sys.stderr)


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "metadata_file",
        "split_name",
        "seed",
        "predictor",
        "output_dir",
        "output_subdir",
        "split",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config keys: {missing}")


def _split_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise ConfigError("split config must be a dictionary")
    return {"dir": str(raw.get("dir", "data/splits/intphys2"))}


def _predict(samples, config: dict[str, Any]) -> list[IntPhys2Prediction]:
    mode = str(config["predictor"]["mode"])

    if mode == "oracle":
        return [
            IntPhys2Prediction(sample_id=sample.sample_id, pred_idx=sample.plausibility)
            for sample in samples
        ]

    if mode == "random":
        rng = random.Random(int(config["seed"]))
        return [
            IntPhys2Prediction(sample_id=sample.sample_id, pred_idx=rng.randint(0, 1))
            for sample in samples
        ]

    if mode == "from_file":
        prediction_file = config["predictor"].get("prediction_file", "")
        if not prediction_file:
            raise ConfigError("predictor.prediction_file is required for mode=from_file")
        pred_by_id, score_by_id = _load_prediction_file(prediction_file)
        missing = [sample.sample_id for sample in samples if sample.sample_id not in pred_by_id]
        if missing:
            raise ConfigError(
                "Prediction file missing sample_ids. "
                f"First missing ids: {missing[:5]}"
            )
        return [
            IntPhys2Prediction(
                sample_id=sample.sample_id,
                pred_idx=int(pred_by_id[sample.sample_id]),
                score=score_by_id.get(sample.sample_id),
            )
            for sample in samples
        ]

    raise ConfigError(f"Unsupported predictor mode: {mode}")


def _load_prediction_file(path_str: str) -> tuple[dict[str, int], dict[str, float]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    pred_by_id: dict[str, int] = {}
    score_by_id: dict[str, float] = {}

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # Support two formats:
            # {"sample_id": pred_idx} or {"sample_id": {"pred_idx": ..., "score": ...}}
            for k, v in payload.items():
                if isinstance(v, dict):
                    pred_by_id[str(k)] = int(v["pred_idx"])
                    if "score" in v:
                        score_by_id[str(k)] = float(v["score"])
                else:
                    pred_by_id[str(k)] = int(v)
        else:
            raise ConfigError("JSON prediction file must be an object: {sample_id: pred_idx}")
        return pred_by_id, score_by_id

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = str(row["sample_id"])
                pred_by_id[sid] = int(row["pred_idx"])
                if "score" in row and row["score"].strip():
                    score_by_id[sid] = float(row["score"])
        return pred_by_id, score_by_id

    raise ConfigError("Prediction file must be .json or .csv")


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["output_dir"])
    subdir = str(config.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"intphys2_eval_{timestamp}"


def _write_metrics(path: Path, metrics) -> None:
    path.write_text(
        json.dumps(asdict_metrics(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_predictions(path: Path, samples, predictions) -> None:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "scene_id",
                "condition",
                "plausibility",
                "pred_idx",
                "score",
                "is_correct",
                "video_ref",
            ],
        )
        writer.writeheader()
        for pred in sorted(predictions, key=lambda p: p.sample_id):
            sample = sample_by_id[pred.sample_id]
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "scene_id": sample.scene_id,
                    "condition": sample.condition,
                    "plausibility": sample.plausibility,
                    "pred_idx": pred.pred_idx,
                    "score": "" if pred.score is None else f"{pred.score:.6f}",
                    "is_correct": int(sample.plausibility == pred.pred_idx),
                    "video_ref": sample.video_ref,
                }
            )


def _write_summary(path: Path, metrics, config: dict[str, Any]) -> None:
    lines = [
        "# IntPhys2 Evaluation Summary",
        "",
        f"- Split: `{config['split_name']}`",
        f"- Seed: `{config['seed']}`",
        f"- Samples: `{metrics.n_samples}`",
        f"- Scenes: `{metrics.n_scenes}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy (%) | {metrics.accuracy:.4f} |",
    ]
    if metrics.roc_auc is not None:
        lines.append(f"| ROC AUC | {metrics.roc_auc:.6f} |")
    lines += [
        f"| VOE Accuracy (%) | {metrics.voe_accuracy:.4f} |",
    ]
    if metrics.accuracy_by_condition:
        lines += ["", "### Per-Condition Accuracy", ""]
        for cond, acc in sorted(metrics.accuracy_by_condition.items()):
            lines.append(f"- {cond}: {acc:.4f}%")
    if metrics.voe_by_condition:
        lines += ["", "### Per-Condition VOE", ""]
        for cond, voe in sorted(metrics.voe_by_condition.items()):
            lines.append(f"- {cond}: {voe:.4f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config_snapshot(path: Path, config: dict[str, Any]) -> None:
    _dump_yaml(path, config)


def _write_provenance(
    path: Path, config: dict[str, Any], split_dir: Path, manifest_path: Path
) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "seed": int(config["seed"]),
        "split_name": str(config["split_name"]),
        "split_dir": str(split_dir),
        "manifest": str(manifest_path),
        "git": {
            "repo_sha": _safe_git_rev_parse(Path(__file__).resolve().parents[2]),
        },
        "packages": _package_versions(["PyYAML", "hydra-core", "omegaconf", "pandas", "pyarrow"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_split_scenes(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading split_scenes.parquet requires pandas. "
            "Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _parse_sample_ids_json(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list):
        raise ConfigError(f"sample_ids_json must decode to a list, got: {parsed!r}")
    return [str(item) for item in parsed]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_git_rev_parse(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as metadata
    except Exception:
        return versions

    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config snapshots. "
            "Install it with: python -m pip install pyyaml"
        ) from exc

    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
