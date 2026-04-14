from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_units(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(group_key, "")).strip()
        if not value:
            raise ValueError(f"Missing group key '{group_key}' in row: {row}")
        grouped[value].append(row)

    units: list[dict[str, Any]] = []
    for unit_id in sorted(grouped):
        unit_rows = sorted(
            grouped[unit_id],
            key=lambda row: (
                str(row.get("sample_id", "")),
                str(row.get("video_id", "")),
            ),
        )
        units.append({"unit_id": unit_id, "rows": unit_rows})
    return units


def build_strata(
    units: list[dict[str, Any]],
    keys: list[str] | tuple[str, ...],
    rare_strata_min_units: int,
    other_label: str,
) -> dict[str, str]:
    raw_by_unit: dict[str, str] = {}

    for unit in units:
        values: list[str] = []
        for key in keys:
            uniq = {
                _normalize_token(row.get(key, ""))
                for row in unit["rows"]
                if _normalize_token(row.get(key, ""))
            }
            if not uniq:
                values.append("unknown")
            elif len(uniq) == 1:
                values.append(next(iter(uniq)))
            else:
                values.append("|".join(sorted(uniq)))
        raw_by_unit[unit["unit_id"]] = "::".join(values)

    counts = Counter(raw_by_unit.values())
    mapped: dict[str, str] = {}
    for unit_id, raw in raw_by_unit.items():
        mapped[unit_id] = raw if counts[raw] >= rare_strata_min_units else other_label
    return mapped


def split_units(
    units: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
    stratified: bool = True,
) -> dict[str, str]:
    _validate_ratios(ratios)
    if not units:
        return {}

    unit_ids = [str(unit["unit_id"]) for unit in units]
    strata = [str(unit.get("stratum", "unknown")) for unit in units]

    if not stratified:
        strata = ["all"] * len(unit_ids)

    train_ratio = float(ratios["train"])
    val_ratio = float(ratios["val"])
    test_ratio = float(ratios["test"])
    holdout_ratio = val_ratio + test_ratio

    if len(unit_ids) < 3:
        raise ValueError(
            "Need at least 3 units to create train/val/test with scikit-learn stratified split."
        )

    train_ids, holdout_ids = _stratified_two_way_split(
        unit_ids=unit_ids,
        strata=strata,
        left_ratio=train_ratio,
        seed=seed,
        stage_name="train_vs_holdout",
    )

    holdout_strata_map = dict(zip(unit_ids, strata))
    holdout_strata = [holdout_strata_map[unit_id] for unit_id in holdout_ids]

    val_ratio_in_holdout = val_ratio / holdout_ratio
    val_ids, test_ids = _stratified_two_way_split(
        unit_ids=holdout_ids,
        strata=holdout_strata,
        left_ratio=val_ratio_in_holdout,
        seed=seed + 1,
        stage_name="val_vs_test",
    )

    mapping: dict[str, str] = {}
    for unit_id in train_ids:
        mapping[unit_id] = "train"
    for unit_id in val_ids:
        mapping[unit_id] = "val"
    for unit_id in test_ids:
        mapping[unit_id] = "test"

    if len(mapping) != len(unit_ids):
        missing = sorted(set(unit_ids) - set(mapping))
        raise RuntimeError(f"Split assignment missing units: {missing[:5]}")
    return mapping


def materialize_sample_assignments(
    rows: list[dict[str, Any]],
    unit_split_map: dict[str, str],
    group_key: str,
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row in rows:
        unit_id = str(row.get(group_key, "")).strip()
        split_name = unit_split_map.get(unit_id)
        if split_name is None:
            continue
        item = dict(row)
        item["split"] = split_name
        materialized.append(item)

    materialized.sort(
        key=lambda row: (
            str(row.get("split", "")),
            str(row.get(group_key, "")),
            str(row.get("sample_id", "")),
        )
    )
    return materialized


def _stratified_two_way_split(
    unit_ids: list[str],
    strata: list[str],
    left_ratio: float,
    seed: int,
    stage_name: str,
) -> tuple[list[str], list[str]]:
    try:
        import numpy as np
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for split generation. Install with: "
            "python -m pip install scikit-learn"
        ) from exc

    if len(unit_ids) != len(strata):
        raise ValueError("unit_ids and strata length mismatch")

    test_ratio = 1.0 - left_ratio
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError(f"Invalid ratio for {stage_name}: left_ratio={left_ratio}")

    # Deterministic canonical order before sklearn shuffling.
    ordered = sorted(zip(unit_ids, strata), key=lambda item: item[0])
    ordered_ids = [item[0] for item in ordered]
    ordered_strata = [item[1] for item in ordered]

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        left_idx, right_idx = next(splitter.split(np.zeros(len(ordered_ids)), ordered_strata))
    except ValueError as exc:
        raise ValueError(
            f"Stratified split failed at stage '{stage_name}'. "
            "Increase split.rare_strata_min_units or adjust ratios/seed. "
            f"Original error: {exc}"
        ) from exc

    left_ids = [ordered_ids[i] for i in left_idx.tolist()]
    right_ids = [ordered_ids[i] for i in right_idx.tolist()]
    return left_ids, right_ids


def _validate_ratios(ratios: dict[str, float]) -> None:
    required = {"train", "val", "test"}
    if set(ratios) != required:
        raise ValueError(f"ratios must contain exactly {sorted(required)}")

    values = [float(ratios[k]) for k in ("train", "val", "test")]
    if any(value <= 0 for value in values):
        raise ValueError("train/val/test ratios must all be > 0")

    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())
