from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def apply_mvp_selection(
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
    dropped.sort(
        key=lambda item: (item["drop_reason"], item["pair_id"], item["sample_id"], item["index"])
    )

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


def write_selection_artifacts(
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


def normalize_question_template(question: str) -> str:
    return normalize_text(question)


def derive_sample_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "video_id"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()

    stable_payload = json.dumps(row, sort_keys=True, default=str)
    digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()
    return f"sample_{digest[:16]}"


def derive_pair_id(row: dict[str, Any], sample_id: str | None = None) -> str:
    sid = sample_id or derive_sample_id(row)

    if "pair_id" in row and str(row["pair_id"]).strip():
        return str(row["pair_id"]).strip()

    if "video_id" in row and str(row["video_id"]).strip():
        video_id = str(row["video_id"]).strip()
        chunks = video_id.split("_")
        if len(chunks) > 1:
            return "_".join(chunks[:-1])

    chunks = sid.split("_")
    if len(chunks) > 1:
        return "_".join(chunks[:-1])
    return sid


def derive_subset(row: dict[str, Any]) -> str:
    for key in ("subset", "dataset_name", "task", "category", "source_subset"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def derive_video_ref(row: dict[str, Any]) -> str:
    for key in ("video_path", "video", "video_ref"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def extract_choices(row: dict[str, Any]) -> tuple[str, str] | None:
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


def normalize_text(text: Any) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _build_record(index: int, row: dict[str, Any]) -> dict[str, Any]:
    sample_id = derive_sample_id(row)
    pair_id = derive_pair_id(row, sample_id)
    question = str(row.get("question", "")).strip()
    choices = extract_choices(row)
    answer = row.get("answer", "")
    subset = derive_subset(row)
    source = str(row.get("source", "")).strip()
    video_ref = derive_video_ref(row)

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
    target = normalize_text(str(cfg.get("subset", "all")))
    if target in {"", "all"}:
        return records

    for record in records:
        if record["drop_reason"] is not None:
            continue
        subset_value = normalize_text(record["subset"])
        if not subset_value or target not in subset_value:
            record["drop_reason"] = "subset_mismatch"
    return records


def _filter_plausibility(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg.get("plausibility_only", False)):
        return records

    keywords = [
        normalize_text(keyword)
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

    yes_aliases = {normalize_text(x) for x in cfg.get("yes_aliases", ["yes", "plausible", "possible"])}
    no_aliases = {normalize_text(x) for x in cfg.get("no_aliases", ["no", "implausible", "impossible"])}

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
        normalize_text(str(item))
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
            q = normalize_text(record["question"])
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
            sample_id = derive_sample_id(row)
            pair_id = derive_pair_id(row, sample_id)
            choices = extract_choices(row)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "question": str(row.get("question", "")).strip(),
                    "choices": json.dumps(list(choices) if choices is not None else [], ensure_ascii=True),
                    "answer": str(row.get("answer", "")),
                    "subset": derive_subset(row),
                    "source": str(row.get("source", "")).strip(),
                    "video_ref": derive_video_ref(row),
                    "drop_reason": str(row.get("drop_reason", drop_reason)),
                    "row_json": json.dumps(row, sort_keys=True, ensure_ascii=True),
                }
            )


def _is_plausibility_question(question: str, keywords: list[str]) -> bool:
    q = normalize_text(question)
    return any(keyword in q for keyword in keywords)


def _choice_polarity(choice: str, yes_aliases: set[str], no_aliases: set[str]) -> str | None:
    normalized = normalize_text(choice)

    if normalized in yes_aliases or normalized.startswith("yes "):
        return "yes"
    if normalized in no_aliases or normalized.startswith("no "):
        return "no"
    return None


def _answer_bucket(answer: Any) -> str:
    a = normalize_text(answer)
    if a.startswith("yes"):
        return "yes"
    if a.startswith("no"):
        return "no"
    if a in {"plausible", "possible"}:
        return "yes_like"
    if a in {"implausible", "impossible"}:
        return "no_like"
    return "other"
