from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.mvp import MVPBenchmark, MVPPrediction, asdict_metrics
from evaluation.mvp_data import (
    DEFAULT_MVP_SUBSETS,
    ensure_mvp_annotation_file,
    load_mvp_rows,
    resolve_video_path,
)


class ConfigError(ValueError):
    pass


def run_mvp_eval(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)

    seed = int(config["seed"])
    random.seed(seed)

    annotations_cfg = _annotations_cfg(config)
    annotation_file = ensure_mvp_annotation_file(
        annotation_file=config["annotation_file"],
        split_hint=str(config["split"]),
        auto_download=bool(annotations_cfg["auto_download"]),
        dataset_id=str(annotations_cfg["dataset_id"]),
        subsets=list(annotations_cfg["subsets"]),
        hf_cache_dir=str(annotations_cfg["hf_cache_dir"]),
        target=str(annotations_cfg["target"]),
    )

    benchmark = MVPBenchmark(official_repo_root=config["official_repo_root"])
    rows = load_mvp_rows(annotation_file)

    kept_rows, dropped_rows, selection_report = _apply_selection(rows, config["selection"])

    samples = benchmark.load_samples(kept_rows, split=config["split"])
    samples = _trim_to_max_pairs(samples, int(config["max_pairs"]))
    if not samples:
        raise ConfigError(
            "No samples available after selection/max_pairs. "
            "Adjust selection filters or max_pairs."
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
                video_a_ref=resolved_video,
                video_b_ref=sample.video_b_ref,
                split=sample.split,
            )
        )

    predictions = _predict(resolved_samples, config)
    metrics = benchmark.evaluate(resolved_samples, predictions)

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(output_dir / "metrics.json", metrics)
    _write_predictions(output_dir / "predictions.csv", resolved_samples, predictions)
    _write_summary(output_dir / "summary.md", metrics, config)
    _write_config_snapshot(output_dir / "run_config.snapshot.yaml", config)
    _write_provenance(output_dir / "provenance.json", config)

    if bool(config["selection"]["artifacts"]["enabled"]):
        _write_selection_artifacts(output_dir, kept_rows, dropped_rows, selection_report)

    return {
        "output_dir": str(output_dir),
        "metrics": asdict_metrics(metrics),
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "annotation_file",
        "official_repo_root",
        "videos_root",
        "cache_dir",
        "split",
        "seed",
        "max_pairs",
        "predictor",
        "materialize_missing",
        "download_timeout_seconds",
        "output_dir",
        "output_subdir",
        "selection",
    ]

    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config keys: {missing}")

    selection = config["selection"]
    if not isinstance(selection, dict):
        raise ConfigError("selection must be a dictionary")

    if not bool(selection.get("drop_incomplete_pairs", True)):
        raise ConfigError("selection.drop_incomplete_pairs=false is not allowed")


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
        "target": str(raw.get("target", "auto")),
        "subsets": subset_values,
    }


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
        f"- Split: `{config['split']}`",
        f"- Seed: `{config['seed']}`",
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


def _write_provenance(path: Path, config: dict[str, Any]) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "seed": int(config["seed"]),
        "git": {
            "repo_sha": _safe_git_rev_parse(Path(__file__).resolve().parents[1]),
            "submodule_sha": _safe_git_rev_parse(Path(config["official_repo_root"])),
        },
        "packages": _package_versions(["PyYAML", "hydra-core", "omegaconf", "pandas"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            "PyYAML is required for config snapshots. Install it with: pip install pyyaml"
        ) from exc

    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _apply_selection(
    rows: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = [_build_record(index, row) for index, row in enumerate(rows)]

    if bool(selection_cfg.get("enabled", True)):
        records = _filter_subset(records, selection_cfg)
        records = _filter_plausibility(records, selection_cfg)
        records = _filter_binary_yes_no(records, selection_cfg)
        records = _filter_include_exclude(records, selection_cfg)

    records = _drop_incomplete_pairs(records)

    kept = [record for record in records if record["drop_reason"] is None]
    dropped = [record for record in records if record["drop_reason"] is not None]

    kept.sort(key=lambda item: (item["pair_id"], item["sample_id"], item["index"]))
    dropped.sort(key=lambda item: (item["drop_reason"], item["pair_id"], item["sample_id"], item["index"]))

    kept_rows: list[dict[str, Any]] = []
    for record in kept:
        row_copy = dict(record["row"])
        row_copy["drop_reason"] = ""
        kept_rows.append(row_copy)

    dropped_rows: list[dict[str, Any]] = []
    for record in dropped:
        row_copy = dict(record["row"])
        row_copy["drop_reason"] = str(record["drop_reason"])
        dropped_rows.append(row_copy)

    report = _selection_report(records, kept, dropped)
    return kept_rows, dropped_rows, report


def _build_record(index: int, row: dict[str, Any]) -> dict[str, Any]:
    sample_id = _derive_sample_id(row)
    pair_id = _derive_pair_id(row, sample_id)
    question = str(row.get("question", "")).strip()
    choices = _extract_choices(row)
    answer = row.get("answer", "")
    subset = _derive_subset(row)
    source = str(row.get("source", "")).strip()
    video_ref = _derive_video_ref(row)

    return {
        "index": index,
        "row": row,
        "sample_id": sample_id,
        "pair_id": pair_id,
        "question": question,
        "choices": choices,
        "answer": answer,
        "subset": subset,
        "source": source,
        "video_ref": video_ref,
        "drop_reason": None,
    }


def _filter_subset(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    target = _normalize_text(str(cfg.get("subset", "all")))
    if target in {"", "all"}:
        return records

    for record in records:
        if record["drop_reason"] is not None:
            continue
        subset_value = _normalize_text(record["subset"])
        if not subset_value or target not in subset_value:
            record["drop_reason"] = "subset_mismatch"
    return records


def _filter_plausibility(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg.get("plausibility_only", False)):
        return records

    keywords = [
        _normalize_text(keyword)
        for keyword in cfg.get("plausibility_keywords", [])
        if str(keyword).strip()
    ]

    for record in records:
        if record["drop_reason"] is not None:
            continue
        if not _is_plausibility_question(record["question"], keywords):
            record["drop_reason"] = "not_plausibility_question"
    return records


def _filter_binary_yes_no(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg.get("require_binary_yes_no", True)):
        return records

    yes_aliases = {_normalize_text(x) for x in cfg.get("yes_aliases", ["yes", "plausible", "possible"])}
    no_aliases = {_normalize_text(x) for x in cfg.get("no_aliases", ["no", "implausible", "impossible"])}

    for record in records:
        if record["drop_reason"] is not None:
            continue

        choices = record["choices"]
        if choices is None or len(choices) != 2:
            record["drop_reason"] = "invalid_choices"
            continue

        p0 = _choice_polarity(choices[0], yes_aliases, no_aliases)
        p1 = _choice_polarity(choices[1], yes_aliases, no_aliases)
        if {p0, p1} != {"yes", "no"}:
            record["drop_reason"] = "not_binary_yes_no"
    return records


def _filter_include_exclude(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    include_pair_ids = {str(item).strip() for item in cfg.get("include_pair_ids", []) if str(item).strip()}
    exclude_pair_ids = {str(item).strip() for item in cfg.get("exclude_pair_ids", []) if str(item).strip()}
    include_question_contains = [
        _normalize_text(str(item))
        for item in cfg.get("include_question_contains", [])
        if str(item).strip()
    ]

    for record in records:
        if record["drop_reason"] is not None:
            continue

        if include_pair_ids and record["pair_id"] not in include_pair_ids:
            record["drop_reason"] = "pair_not_in_include_list"
            continue

        if exclude_pair_ids and record["pair_id"] in exclude_pair_ids:
            record["drop_reason"] = "pair_in_exclude_list"
            continue

        if include_question_contains:
            q = _normalize_text(record["question"])
            if not any(token in q for token in include_question_contains):
                record["drop_reason"] = "question_not_in_include_list"

    return records


def _drop_incomplete_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [record for record in records if record["drop_reason"] is None]
    pair_counts: dict[str, int] = defaultdict(int)
    for record in active:
        pair_counts[record["pair_id"]] += 1

    for record in active:
        count = pair_counts[record["pair_id"]]
        if count != 2:
            record["drop_reason"] = f"incomplete_pair_{count}"

    return records


def _selection_report(
    all_records: list[dict[str, Any]],
    kept_records: list[dict[str, Any]],
    dropped_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dropped_reason_counter = Counter(record["drop_reason"] for record in dropped_records)
    dropped_question_counter = Counter(record["question"] for record in dropped_records if record["question"])

    kept_answer_counter = Counter(_answer_bucket(record.get("answer")) for record in kept_records)

    all_pair_counts = Counter(record["pair_id"] for record in all_records)
    kept_pair_counts = Counter(record["pair_id"] for record in kept_records)

    return {
        "total_rows": len(all_records),
        "kept_rows": len(kept_records),
        "dropped_rows": len(dropped_records),
        "dropped_by_reason": dict(sorted(dropped_reason_counter.items())),
        "top_dropped_questions": [
            {"question": question, "count": count}
            for question, count in dropped_question_counter.most_common(20)
        ],
        "kept_answer_distribution": dict(sorted(kept_answer_counter.items())),
        "pair_integrity": {
            "pairs_total_before": len(all_pair_counts),
            "pairs_complete_before": sum(1 for count in all_pair_counts.values() if count == 2),
            "pairs_incomplete_before": sum(1 for count in all_pair_counts.values() if count != 2),
            "pairs_total_kept": len(kept_pair_counts),
            "pairs_complete_kept": sum(1 for count in kept_pair_counts.values() if count == 2),
            "pairs_incomplete_dropped": len(
                {
                    record["pair_id"]
                    for record in dropped_records
                    if str(record["drop_reason"]).startswith("incomplete_pair_")
                }
            ),
        },
    }


def _write_selection_artifacts(
    output_dir: Path,
    kept_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    kept_path = output_dir / "selection_kept.csv"
    dropped_path = output_dir / "selection_dropped.csv"
    report_path = output_dir / "selection_report.json"

    _write_selection_rows(kept_path, kept_rows, drop_reason="")
    _write_selection_rows(dropped_path, dropped_rows, drop_reason="unknown")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _write_selection_rows(path: Path, rows: list[dict[str, Any]], drop_reason: str) -> None:
    fieldnames = [
        "sample_id",
        "pair_id",
        "question",
        "choices",
        "answer",
        "subset",
        "source",
        "video_ref",
        "drop_reason",
        "row_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sample_id = _derive_sample_id(row)
            pair_id = _derive_pair_id(row, sample_id)
            choices = _extract_choices(row)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "question": str(row.get("question", "")).strip(),
                    "choices": json.dumps(list(choices) if choices is not None else [], ensure_ascii=True),
                    "answer": str(row.get("answer", "")),
                    "subset": _derive_subset(row),
                    "source": str(row.get("source", "")).strip(),
                    "video_ref": _derive_video_ref(row),
                    "drop_reason": str(row.get("drop_reason", drop_reason)),
                    "row_json": json.dumps(row, sort_keys=True, ensure_ascii=True),
                }
            )


def _derive_sample_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "video_id"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()

    stable_payload = json.dumps(row, sort_keys=True, default=str)
    digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()
    return f"sample_{digest[:16]}"


def _derive_pair_id(row: dict[str, Any], sample_id: str) -> str:
    if "pair_id" in row and str(row["pair_id"]).strip():
        return str(row["pair_id"]).strip()

    if "video_id" in row and str(row["video_id"]).strip():
        video_id = str(row["video_id"]).strip()
        chunks = video_id.split("_")
        if len(chunks) > 1:
            return "_".join(chunks[:-1])

    chunks = sample_id.split("_")
    if len(chunks) > 1:
        return "_".join(chunks[:-1])
    return sample_id


def _derive_subset(row: dict[str, Any]) -> str:
    for key in ("subset", "dataset_name", "task", "category", "source_subset"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _derive_video_ref(row: dict[str, Any]) -> str:
    for key in ("video_path", "video", "video_ref"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _extract_choices(row: dict[str, Any]) -> tuple[str, str] | None:
    raw = row.get("candidates", row.get("choices"))
    if raw is None:
        return None

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None

    return str(parsed[0]), str(parsed[1])


def _is_plausibility_question(question: str, keywords: list[str]) -> bool:
    q = _normalize_text(question)
    return any(keyword in q for keyword in keywords)


def _normalize_text(text: Any) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _choice_polarity(choice: str, yes_aliases: set[str], no_aliases: set[str]) -> str | None:
    normalized = _normalize_text(choice)

    if normalized in yes_aliases or normalized.startswith("yes "):
        return "yes"
    if normalized in no_aliases or normalized.startswith("no "):
        return "no"
    return None


def _answer_bucket(answer: Any) -> str:
    a = _normalize_text(answer)
    if a.startswith("yes"):
        return "yes"
    if a.startswith("no"):
        return "no"
    if a in {"plausible", "possible"}:
        return "yes_like"
    if a in {"implausible", "impossible"}:
        return "no_like"
    return "other"
