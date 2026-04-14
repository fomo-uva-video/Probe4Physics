from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.ssv2.core import SSv2Benchmark, SSv2Prediction, asdict_metrics
from benchmarks.ssv2.data import sha256_file


class ConfigError(ValueError):
    pass


def run_ssv2_eval(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    seed = int(config["seed"])
    random.seed(seed)

    split_cfg = _split_cfg(config)
    split_dir = Path(split_cfg["dir"])
    split_clips_path = split_dir / "split_clips.parquet"
    manifest_path = split_dir / "manifest.json"

    split_name = str(config["split_name"])
    predictor_mode = str(config["predictor"]["mode"])

    _log_eval(
        "Starting eval",
        split_name=split_name,
        split_dir=str(split_dir),
        predictor_mode=predictor_mode,
    )

    if not split_clips_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Split artifacts not found. Run init first with: python run.py init.ssv2\n"
            f"Missing: {split_clips_path} or {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_annotation_hashes(config, manifest)

    split_clip_rows = _load_split_clips(split_clips_path)
    rows_for_split = [
        row for row in split_clip_rows if str(row.get("split", "")) == split_name
    ]
    if not rows_for_split:
        raise ConfigError(
            f"No samples found for split_name='{split_name}' in {split_clips_path}"
        )

    _log_eval("Loaded split samples", n_samples=len(rows_for_split))

    benchmark = SSv2Benchmark()
    samples = benchmark.load_samples(rows_for_split, split=split_name)

    n_classes = manifest.get("stats", {}).get("n_classes", 174)
    predictions = _predict(samples, config, n_classes=int(n_classes))
    _log_eval("Generated predictions", n_predictions=len(predictions))

    metrics = benchmark.evaluate(samples, predictions)
    _log_eval(
        "Scoring complete",
        top1=f"{metrics.top1_accuracy:.4f}",
        top5=f"{metrics.top5_accuracy:.4f}",
        n_samples=metrics.n_samples,
        n_classes=metrics.n_classes,
    )

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
        print(f"[eval.ssv2] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[eval.ssv2] {message}", file=sys.stderr)


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "train_annotation_file",
        "val_annotation_file",
        "labels_file",
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
    return {"dir": str(raw.get("dir", "data/splits/ssv2"))}


def _verify_annotation_hashes(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    stored = manifest.get("annotation_sha256", {})
    if not isinstance(stored, dict):
        return

    split_name = str(config.get("split_name", ""))
    checks: list[tuple[str, Path, str]] = [
        ("labels", Path(str(config["labels_file"])), stored.get("labels", "")),
    ]
    if split_name == "train":
        checks.append(("train", Path(str(config["train_annotation_file"])), stored.get("train", "")))
    elif split_name == "val":
        checks.append(("val", Path(str(config["val_annotation_file"])), stored.get("val", "")))

    for name, path, expected in checks:
        if not expected or not path.exists():
            continue
        actual = sha256_file(path)
        if actual != expected:
            raise ConfigError(
                f"SSv2 {name} annotation hash mismatch with split manifest.\n"
                f"Current file: {path}\n"
                f"Current sha256: {actual}\n"
                f"Manifest sha256: {expected}\n"
                "Re-run init.ssv2 to regenerate split artifacts."
            )


def _predict(samples, config: dict[str, Any], n_classes: int) -> list[SSv2Prediction]:
    mode = str(config["predictor"]["mode"])

    if mode == "oracle":
        return [
            SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx)
            for s in samples
        ]

    if mode == "random":
        rng = random.Random(int(config["seed"]))
        return [
            SSv2Prediction(sample_id=s.sample_id, pred_idx=rng.randint(0, n_classes - 1))
            for s in samples
        ]

    if mode == "from_file":
        prediction_file = config["predictor"].get("prediction_file", "")
        if not prediction_file:
            raise ConfigError("predictor.prediction_file is required for mode=from_file")
        pred_by_id, scores_by_id = _load_prediction_file(prediction_file)
        missing = [s.sample_id for s in samples if s.sample_id not in pred_by_id]
        if missing:
            raise ConfigError(
                "Prediction file missing sample_ids. "
                f"First missing ids: {missing[:5]}"
            )
        return [
            SSv2Prediction(
                sample_id=s.sample_id,
                pred_idx=int(pred_by_id[s.sample_id]),
                scores=scores_by_id.get(s.sample_id),
            )
            for s in samples
        ]

    raise ConfigError(f"Unsupported predictor mode: {mode}")


def _load_prediction_file(
    path_str: str,
) -> tuple[dict[str, int], dict[str, tuple[float, ...]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    pred_by_id: dict[str, int] = {}
    scores_by_id: dict[str, tuple[float, ...]] = {}

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigError("JSON prediction file must be an object: {sample_id: pred_idx}")
        for k, v in payload.items():
            if isinstance(v, dict):
                pred_by_id[str(k)] = int(v["pred_idx"])
                if "scores" in v:
                    scores_by_id[str(k)] = tuple(float(x) for x in v["scores"])
            else:
                pred_by_id[str(k)] = int(v)
        return pred_by_id, scores_by_id

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = str(row["sample_id"])
                pred_by_id[sid] = int(row["pred_idx"])
        return pred_by_id, scores_by_id

    raise ConfigError("Prediction file must be .json or .csv")


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["output_dir"])
    subdir = str(config.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"ssv2_eval_{timestamp}"


def _write_metrics(path: Path, metrics) -> None:
    path.write_text(
        json.dumps(asdict_metrics(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_predictions(path: Path, samples, predictions) -> None:
    sample_by_id = {s.sample_id: s for s in samples}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "label_idx",
                "label_name",
                "template",
                "pred_idx",
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
                    "label_idx": sample.label_idx,
                    "label_name": sample.label_name,
                    "template": sample.template,
                    "pred_idx": pred.pred_idx,
                    "is_correct": int(sample.label_idx == pred.pred_idx),
                    "video_ref": sample.video_ref,
                }
            )


def _write_summary(path: Path, metrics, config: dict[str, Any]) -> None:
    lines = [
        "# SSv2 Evaluation Summary",
        "",
        f"- Split: `{config['split_name']}`",
        f"- Seed: `{config['seed']}`",
        f"- Samples: `{metrics.n_samples}`",
        f"- Classes in split: `{metrics.n_classes}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Top-1 Accuracy (%) | {metrics.top1_accuracy:.4f} |",
        f"| Top-5 Accuracy (%) | {metrics.top5_accuracy:.4f} |",
    ]
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


def _load_split_clips(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading split_clips.parquet requires pandas. "
            "Install it with: python -m pip install pandas"
        ) from exc

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


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
