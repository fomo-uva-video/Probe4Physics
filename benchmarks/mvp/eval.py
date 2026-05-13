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

from benchmarks.mvp.core import MVPBenchmark, MVPPrediction, asdict_metrics
from benchmarks.mvp.data import (
    DEFAULT_MVP_SUBSETS,
    ensure_mvp_annotation_file,
    load_mvp_rows,
    resolve_video_path,
)
from benchmarks.mvp.selection import derive_sample_id


class ConfigError(ValueError):
    pass


def run_mvp_eval(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    seed = int(config["seed"])
    random.seed(seed)

    annotations_cfg = _annotations_cfg(config)
    annotation_file = ensure_mvp_annotation_file(
        annotation_file=config["annotation_file"],
        split_hint="full",
        auto_download=bool(annotations_cfg["auto_download"]),
        dataset_id=str(annotations_cfg["dataset_id"]),
        subsets=list(annotations_cfg["subsets"]),
        hf_cache_dir=str(annotations_cfg["hf_cache_dir"]),
        target="mvp",
    )

    split_cfg = _split_cfg(config)
    split_dir = _resolve_split_dir(split_cfg)
    split_pairs_path = split_dir / "split_pairs.parquet"
    manifest_path = split_dir / "manifest.json"
    split_name = str(config["split_name"])
    predictor_mode = str(config["predictor"]["mode"])

    _log_eval(
        "Starting eval",
        split_name=split_name,
        split_dir=str(split_dir),
        predictor_mode=predictor_mode,
        annotation_file=str(annotation_file),
    )

    if not split_pairs_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Split artifacts not found. Run init first with: python run.py init.mvp\n"
            f"Missing: {split_pairs_path} or {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_sha = _sha256_file(Path(annotation_file))
    expected_sha = str(manifest.get("annotation_sha256", ""))
    if annotation_sha != expected_sha:
        raise ConfigError(
            "Annotation hash mismatch with split manifest.\n"
            f"Current annotation: {annotation_file}\n"
            f"Current sha256: {annotation_sha}\n"
            f"Manifest sha256: {expected_sha}\n"
            "Re-run init.mvp to regenerate split artifacts for this annotation file."
        )

    split_rows = _load_split_pairs(split_pairs_path)
    pair_rows = [row for row in split_rows if str(row.get("split", "")) == split_name]
    if not pair_rows:
        raise ConfigError(f"No pairs found for split_name='{split_name}' in {split_pairs_path}")

    ordered_sample_ids: list[str] = []
    for row in sorted(pair_rows, key=lambda item: str(item.get("pair_id", ""))):
        ordered_sample_ids.extend(_parse_sample_ids_json(row.get("sample_ids_json", "[]")))
    _log_eval(
        "Loaded split assignments",
        split_pairs=len(pair_rows),
        split_samples=len(ordered_sample_ids),
    )

    all_rows = load_mvp_rows(annotation_file)
    row_by_sample: dict[str, dict[str, Any]] = {}
    target_ids = set(ordered_sample_ids)
    for row in all_rows:
        sample_id = derive_sample_id(row)
        if sample_id in target_ids and sample_id not in row_by_sample:
            row_by_sample[sample_id] = row

    missing = [sample_id for sample_id in ordered_sample_ids if sample_id not in row_by_sample]
    if missing:
        raise ConfigError(
            "Split references sample_ids missing from annotation file. "
            f"First missing ids: {missing[:5]}"
        )

    selected_rows = [row_by_sample[sample_id] for sample_id in ordered_sample_ids]

    benchmark = MVPBenchmark(official_repo_root=config["official_repo_root"])
    samples = benchmark.load_samples(selected_rows, split=split_name)
    label_override = _label_override(config)
    if label_override is not None:
        samples = [_apply_label_override(sample, label_override) for sample in samples]
    n_pairs_before_trim = len({sample.pair_id for sample in samples})
    n_samples_before_trim = len(samples)
    samples = _trim_to_max_pairs(samples, int(config["max_pairs"]))
    if not samples:
        raise ConfigError("No samples available for evaluation after max_pairs filter.")
    _log_eval(
        "Prepared samples",
        n_pairs_before_trim=n_pairs_before_trim,
        n_samples_before_trim=n_samples_before_trim,
        n_pairs_after_trim=len({sample.pair_id for sample in samples}),
        n_samples_after_trim=len(samples),
        max_pairs=int(config["max_pairs"]),
    )

    resolved_samples = []
    for sample in samples:
        resolved_video = resolve_video_path(
            video_ref=sample.video_a_ref,
            videos_root=config["videos_root"],
            cache_dir=config["cache_dir"],
            materialize_missing=bool(config["materialize_missing"]),
            timeout_seconds=int(config["download_timeout_seconds"]),
        )
        resolved_samples.append(
            sample.__class__(
                sample_id=sample.sample_id,
                pair_id=sample.pair_id,
                question=sample.question,
                choices=sample.choices,
                answer_idx=sample.answer_idx,
                plausibility_label=sample.plausibility_label,
                yes_choice_idx=sample.yes_choice_idx,
                no_choice_idx=sample.no_choice_idx,
                video_a_ref=resolved_video,
                video_b_ref=sample.video_b_ref,
                split=sample.split,
            )
        )

    predictions = _predict(resolved_samples, config)
    _log_eval("Generated predictions", n_predictions=len(predictions))
    metrics = benchmark.evaluate(resolved_samples, predictions)
    _log_eval(
        "Scoring complete",
        accuracy=f"{metrics.accuracy:.4f}",
        pair_consistency=f"{metrics.pair_consistency:.4f}",
        label_override="none" if label_override is None else label_override,
        n_pairs=metrics.n_pairs,
        n_samples=metrics.n_samples,
    )

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(output_dir / "metrics.json", metrics)
    _write_predictions(output_dir / "predictions.csv", resolved_samples, predictions)
    _write_summary(output_dir / "summary.md", metrics, config)
    _write_config_snapshot(output_dir / "run_config.snapshot.yaml", config)
    _write_provenance(output_dir / "provenance.json", config, split_dir, manifest_path)
    _log_eval("Artifacts written", output_dir=str(output_dir))

    return {
        "output_dir": str(output_dir),
        "split_dir": str(split_dir),
        "metrics": asdict_metrics(metrics),
        "label_override": label_override,
    }


def _log_eval(message: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        print(f"[eval.mvp] {message} | {payload}", file=sys.stderr)
    else:
        print(f"[eval.mvp] {message}", file=sys.stderr)


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "annotation_file",
        "official_repo_root",
        "videos_root",
        "cache_dir",
        "split_name",
        "seed",
        "max_pairs",
        "predictor",
        "materialize_missing",
        "download_timeout_seconds",
        "output_dir",
        "output_subdir",
        "split",
    ]

    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config keys: {missing}")


def _label_override(config: dict[str, Any]) -> int | None:
    raw = config.get("label_override", None)
    if raw is None or str(raw).strip() == "":
        return None
    value = int(raw)
    if value not in (0, 1):
        raise ConfigError(f"label_override must be 0 or 1, got {value}")
    return value


def _apply_label_override(sample, label_value: int):
    if sample.yes_choice_idx is None or sample.no_choice_idx is None:
        raise ConfigError(
            "label_override requires MVP samples with yes/no plausibility semantics."
        )
    answer_idx = sample.yes_choice_idx if int(label_value) == 1 else sample.no_choice_idx
    return sample.__class__(
        sample_id=sample.sample_id,
        pair_id=sample.pair_id,
        question=sample.question,
        choices=sample.choices,
        answer_idx=int(answer_idx),
        plausibility_label=int(label_value),
        yes_choice_idx=sample.yes_choice_idx,
        no_choice_idx=sample.no_choice_idx,
        video_a_ref=sample.video_a_ref,
        video_b_ref=sample.video_b_ref,
        split=sample.split,
    )


def _annotations_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("annotations", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("annotations must be a dictionary")

    subsets = raw.get("subsets", DEFAULT_MVP_SUBSETS)
    if not isinstance(subsets, (list, tuple)) or not subsets:
        raise ConfigError("annotations.subsets must be a non-empty list")

    subset_values = [str(item) for item in subsets if str(item).strip()]
    if not subset_values:
        raise ConfigError("annotations.subsets resolved to empty list")

    return {
        "auto_download": bool(raw.get("auto_download", True)),
        "dataset_id": str(raw.get("dataset_id", "facebook/minimal_video_pairs")),
        "hf_cache_dir": str(raw.get("hf_cache_dir", ".cache/hf_datasets")),
        "subsets": subset_values,
    }


def _split_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("split", {})
    if not isinstance(raw, dict):
        raise ConfigError("split config must be a dictionary")

    return {
        "dir": str(raw.get("dir", "data/splits/mvp/full_60_20_20")),
    }


def _resolve_split_dir(split_cfg: dict[str, Any]) -> Path:
    return Path(split_cfg["dir"])


def _load_split_pairs(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading split_pairs.parquet requires pandas. Install it with: python -m pip install pandas"
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


def _trim_to_max_pairs(samples, max_pairs: int):
    if max_pairs <= 0:
        return samples

    allowed_pairs = []
    seen = set()
    for sample in samples:
        if sample.pair_id in seen:
            continue
        seen.add(sample.pair_id)
        allowed_pairs.append(sample.pair_id)
        if len(allowed_pairs) >= max_pairs:
            break

    allowed = set(allowed_pairs)
    return [sample for sample in samples if sample.pair_id in allowed]


def _predict(samples, config: dict[str, Any]) -> list[MVPPrediction]:
    mode = str(config["predictor"]["mode"])

    if mode == "oracle":
        return [
            MVPPrediction(sample_id=sample.sample_id, pred_idx=sample.answer_idx)
            for sample in samples
        ]

    if mode == "random":
        rng = random.Random(int(config["seed"]))
        return [
            MVPPrediction(sample_id=sample.sample_id, pred_idx=rng.randint(0, 1))
            for sample in samples
        ]

    if mode == "from_file":
        prediction_file = config["predictor"].get("prediction_file", "")
        if not prediction_file:
            raise ConfigError("predictor.prediction_file is required for mode=from_file")
        pred_by_id = _load_prediction_file(prediction_file)
        missing = [sample.sample_id for sample in samples if sample.sample_id not in pred_by_id]
        if missing:
            raise ConfigError(
                "Prediction file missing sample_ids. "
                f"First missing ids: {missing[:5]}"
            )
        return [
            MVPPrediction(sample_id=sample.sample_id, pred_idx=int(pred_by_id[sample.sample_id]))
            for sample in samples
        ]

    raise ConfigError(f"Unsupported predictor mode: {mode}")


def _load_prediction_file(path_str: str) -> dict[str, int]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigError("JSON prediction file must be an object: {sample_id: pred_idx}")
        return {str(k): int(v) for k, v in payload.items()}

    if path.suffix.lower() == ".csv":
        result: dict[str, int] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                result[str(row["sample_id"])] = int(row["pred_idx"])
        return result

    raise ConfigError("Prediction file must be .json or .csv")


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["output_dir"])
    subdir = str(config.get("output_subdir", "")).strip()
    if subdir:
        return root / subdir

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"mvp_eval_{timestamp}"


def _write_metrics(path: Path, metrics) -> None:
    path.write_text(json.dumps(asdict_metrics(metrics), indent=2, sort_keys=True), encoding="utf-8")


def _write_predictions(path: Path, samples, predictions) -> None:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "pair_id",
                "question",
                "choices",
                "answer_idx",
                "pred_idx",
                "is_correct",
                "video_ref",
            ],
        )
        writer.writeheader()
        for pred in sorted(predictions, key=lambda item: item.sample_id):
            sample = sample_by_id[pred.sample_id]
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "pair_id": sample.pair_id,
                    "question": sample.question,
                    "choices": json.dumps(list(sample.choices), ensure_ascii=True),
                    "answer_idx": sample.answer_idx,
                    "pred_idx": pred.pred_idx,
                    "is_correct": int(sample.answer_idx == pred.pred_idx),
                    "video_ref": sample.video_a_ref,
                }
            )


def _write_summary(path: Path, metrics, config: dict[str, Any]) -> None:
    lines = [
        "# MVP Evaluation Summary",
        "",
        f"- Split: `{config['split_name']}`",
        f"- Seed: `{config['seed']}`",
        f"- Label override: `{config.get('label_override', 'none')}`",
        f"- Samples: `{metrics.n_samples}`",
        f"- Pairs: `{metrics.n_pairs}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy (%) | {metrics.accuracy:.4f} |",
        f"| Pair Consistency (%) | {metrics.pair_consistency:.4f} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config_snapshot(path: Path, config: dict[str, Any]) -> None:
    _dump_yaml(path, config)


def _write_provenance(path: Path, config: dict[str, Any], split_dir: Path, manifest_path: Path) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "seed": int(config["seed"]),
        "split_name": str(config["split_name"]),
        "label_override": config.get("label_override", None),
        "split_dir": str(split_dir),
        "manifest": str(manifest_path),
        "git": {
            "repo_sha": _safe_git_rev_parse(Path(__file__).resolve().parents[1]),
            "submodule_sha": _safe_git_rev_parse(Path(config["official_repo_root"])),
        },
        "packages": _package_versions(["PyYAML", "hydra-core", "omegaconf", "pandas", "pyarrow"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            "PyYAML is required for config snapshots. Install it with: python -m pip install pyyaml"
        ) from exc

    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
