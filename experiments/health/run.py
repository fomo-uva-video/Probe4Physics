from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmarks.intphys2.data import load_intphys2_rows, normalize_intphys2_row
from benchmarks.mvp.data import load_mvp_rows
from benchmarks.mvp.selection import derive_sample_id, derive_video_ref
from benchmarks.ssv2.data import load_ssv2_annotations
from models.jepa_v1_adapter import resolve_relative_depth_layers as resolve_jepa_v1_relative_depth_layers
from models.jepa_v2_adapter import resolve_relative_depth_layers as resolve_jepa_v2_relative_depth_layers
from models.jepa_v2_1_adapter import resolve_relative_depth_layers as resolve_jepa_v2_1_relative_depth_layers
from models import get_registered_adapters
from models.ltx_video_adapter import resolve_probe_layer_ids as resolve_ltx_probe_layer_ids
from models.videomae_adapter import resolve_relative_depth_layers as resolve_videomae_relative_depth_layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

BACKBONE_ORDER = [
    "jepa_v1",
    "jepa_v2",
    "jepa_v2_1",
    "videomae",
    "videomae_v2",
    "ltx_video",
]

# Largest variants requested for health checks.
LARGEST_VARIANTS = {
    "jepa_v1": "vith16_384",
    "jepa_v2": "vitg_384",
    "jepa_v2_1": "vitG_384",
    "videomae": "vit_huge_16_224",
    "videomae_v2": "vit_giant_16_224",
    "ltx_video": "ltxv_13b_0_9_8_distilled",
}


def run_health(config: dict[str, Any]) -> dict[str, Any]:
    synthetic_forward = bool(config.get("synthetic_forward", False))
    strict_exit = bool(config.get("strict_exit", False))
    device = str(config.get("device", "cpu")).strip() or "cpu"
    backbone_cfg = _load_yaml(CONFIGS_DIR / "backbones.yaml")
    mvp_cfg = _load_yaml(CONFIGS_DIR / "mvp.yaml")
    intphys2_cfg = _load_yaml(CONFIGS_DIR / "intphys2.yaml")
    ssv2_cfg = _load_yaml(CONFIGS_DIR / "ssv2.yaml")

    backbones = [
        _check_backbone(name, backbone_cfg, synthetic_forward=synthetic_forward, device=device)
        for name in BACKBONE_ORDER
    ]
    datasets = [
        _check_mvp(mvp_cfg),
        _check_intphys2(intphys2_cfg),
        _check_ssv2(ssv2_cfg),
    ]

    checks: list[dict[str, Any]] = []
    for item in backbones + datasets:
        checks.extend(item.get("checks", []))

    total = len(checks)
    passed = sum(1 for check in checks if check.get("status") == "pass")
    failed = sum(1 for check in checks if check.get("status") == "fail")
    ok = failed == 0

    report = {
        "ok": ok,
        "exit_code": 0 if ok or not strict_exit else 1,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
        },
        "mode": {
            "synthetic_forward": synthetic_forward,
            "device": device,
            "strict_exit": strict_exit,
        },
        "tested_backbones": [
            {
                "name": item.get("name"),
                "variant": item.get("variant"),
                "status": item.get("status"),
            }
            for item in backbones
        ],
        "tested_datasets": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
            }
            for item in datasets
        ],
        "backbones": backbones,
        "datasets": datasets,
    }
    report["human_report"] = _format_human_report(report)
    return report


def run_health_layers(config: dict[str, Any]) -> dict[str, Any]:
    """Validate layer mapping for each configured backbone variant.

    This command is static and config-driven: it does not instantiate models or
    run forwards. It reports:
    - Original backbone block ids (1-based and 0-based)
    - Selected extraction layer ids resolved from config
    - Benchmark layer requests and whether they are valid subsets
    """

    strict_exit = bool(config.get("strict_exit", False))
    backbone_cfg = _load_yaml(CONFIGS_DIR / "backbones.yaml")
    mvp_cfg = _load_yaml(CONFIGS_DIR / "mvp.yaml")
    intphys2_cfg = _load_yaml(CONFIGS_DIR / "intphys2.yaml")
    ssv2_cfg = _load_yaml(CONFIGS_DIR / "ssv2.yaml")

    variants: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    variant_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for backbone_name in BACKBONE_ORDER:
        section = backbone_cfg.get(backbone_name)
        if not isinstance(section, dict):
            checks.append(
                _check_result(
                    f"{backbone_name}.section_exists",
                    False,
                    f"Missing '{backbone_name}' section in configs/backbones.yaml",
                )
            )
            continue

        raw_variants = section.get("variants")
        if not isinstance(raw_variants, dict) or not raw_variants:
            checks.append(
                _check_result(
                    f"{backbone_name}.variants_present",
                    False,
                    f"'{backbone_name}.variants' must be a non-empty mapping.",
                )
            )
            continue

        for variant_name in sorted(raw_variants):
            variant_payload = _check_backbone_variant_layer_mapping(
                backbone_name,
                variant_name,
                section,
                raw_variants.get(variant_name),
            )
            variants.append(variant_payload)
            checks.extend(variant_payload.get("checks", []))
            variant_lookup[(backbone_name, variant_name)] = variant_payload

    benchmark_layer_requests = _resolve_benchmark_layer_requests(
        backbone_cfg,
        variant_lookup,
        mvp_cfg,
        intphys2_cfg,
        ssv2_cfg,
    )
    checks.extend(item["check"] for item in benchmark_layer_requests)

    total = len(checks)
    passed = sum(1 for check in checks if check.get("status") == "pass")
    failed = total - passed
    ok = failed == 0

    report = {
        "ok": ok,
        "exit_code": 0 if ok or not strict_exit else 1,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "variant_mappings": len(variants),
            "benchmark_requests": len(benchmark_layer_requests),
        },
        "mode": {"strict_exit": strict_exit},
        "variants": variants,
        "benchmark_layer_requests": benchmark_layer_requests,
    }
    report["human_report"] = _format_layer_human_report(report)
    return report


def run_health_features(config: dict[str, Any]) -> dict[str, Any]:
    """Validate cached feature artifacts without instantiating backbones.

    The check intentionally stays CPU- and filesystem-only. It scans configured
    feature roots, reads each cache manifest and index, and memory-maps tensor
    payloads when supported so shape checks do not require GPU or model loads.
    """

    strict_exit = bool(config.get("strict_exit", False))
    raw_feature_cfg = config.get("features", {})
    feature_cfg = raw_feature_cfg if isinstance(raw_feature_cfg, dict) else {}
    check_tensors = bool(feature_cfg.get("check_tensors", True))
    check_video_paths = bool(feature_cfg.get("check_video_paths", False))
    allow_full_tensor_load = bool(feature_cfg.get("allow_full_tensor_load", False))
    require_all_backbones = bool(feature_cfg.get("require_all_backbones", True))
    expected_backbones = _feature_expected_backbones(feature_cfg)

    datasets = [
        _check_feature_dataset(
            spec,
            expected_backbones=expected_backbones,
            require_all_backbones=require_all_backbones,
            check_tensors=check_tensors,
            check_video_paths=check_video_paths,
            allow_full_tensor_load=allow_full_tensor_load,
        )
        for spec in _feature_dataset_specs(feature_cfg)
    ]

    checks: list[dict[str, Any]] = []
    cache_count = 0
    found_backbones: set[str] = set()
    for dataset in datasets:
        checks.extend(dataset.get("checks", []))
        for cache in dataset.get("caches", []):
            cache_count += 1
            checks.extend(cache.get("checks", []))
            backbone = str(cache.get("backbone", "")).strip()
            if backbone:
                found_backbones.add(backbone)

    total = len(checks)
    passed = sum(1 for check in checks if check.get("status") == "pass")
    failed = total - passed
    ok = failed == 0

    report = {
        "ok": ok,
        "exit_code": 0 if ok or not strict_exit else 1,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "datasets": len(datasets),
            "caches": cache_count,
            "backbones": len(found_backbones),
        },
        "mode": {
            "strict_exit": strict_exit,
            "check_tensors": check_tensors,
            "check_video_paths": check_video_paths,
            "allow_full_tensor_load": allow_full_tensor_load,
            "require_all_backbones": require_all_backbones,
            "expected_backbones": expected_backbones,
        },
        "datasets": datasets,
    }
    report["human_report"] = _format_feature_human_report(report)
    return report


def _feature_dataset_specs(feature_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_datasets = feature_cfg.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raw_datasets = {
            "mvp": {
                "dir": "/scratch-shared/scur0511/probe4physics/artifacts/features/mvp",
                "config": "mvp",
                "required": True,
            },
            "intphys2": {
                "dir": "/scratch-shared/scur0511/probe4physics/artifacts/features/intphys2",
                "config": "intphys2",
                "required": True,
            },
        }

    ordered_names = [name for name in ("mvp", "intphys2", "ssv2") if name in raw_datasets]
    ordered_names.extend(sorted(str(name) for name in raw_datasets if str(name) not in ordered_names))

    specs: list[dict[str, Any]] = []
    for name in ordered_names:
        raw_spec = raw_datasets.get(name, {})
        if isinstance(raw_spec, str):
            payload: dict[str, Any] = {"dir": raw_spec}
        elif isinstance(raw_spec, dict):
            payload = dict(raw_spec)
        else:
            payload = {}
        payload["name"] = str(name)
        payload["dir"] = str(payload.get("dir", "")).strip()
        payload["config"] = str(payload.get("config", name)).strip() or str(name)
        payload["required"] = bool(payload.get("required", True))
        specs.append(payload)
    return specs


def _feature_expected_backbones(feature_cfg: dict[str, Any]) -> list[str]:
    raw = feature_cfg.get("expected_backbones", BACKBONE_ORDER)
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raw = [raw]
    expected: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name and name not in expected:
            expected.append(name)
    return expected


def _check_feature_dataset(
    spec: dict[str, Any],
    *,
    expected_backbones: list[str],
    require_all_backbones: bool,
    check_tensors: bool,
    check_video_paths: bool,
    allow_full_tensor_load: bool,
) -> dict[str, Any]:
    name = str(spec.get("name", "")).strip()
    root = _as_project_path(str(spec.get("dir", "")))
    config_name = str(spec.get("config", name)).strip() or name
    required = bool(spec.get("required", True))
    checks: list[dict[str, Any]] = []
    caches: list[dict[str, Any]] = []

    checks.append(_check_result("feature_root_configured", bool(str(spec.get("dir", "")).strip()), str(root)))
    if root.exists():
        checks.append(_check_result("feature_root_exists", root.is_dir(), str(root)))
    else:
        checks.append(
            _check_result(
                "feature_root_exists",
                not required,
                f"{root} ({'required' if required else 'optional'})",
            )
        )

    dataset_cfg_path = CONFIGS_DIR / (config_name if config_name.endswith(".yaml") else f"{config_name}.yaml")
    if dataset_cfg_path.exists():
        try:
            dataset_cfg = _load_yaml(dataset_cfg_path)
            declared = str((dataset_cfg.get("feature_cache", {}) or {}).get("dir", "")).strip()
            declared_root = _as_project_path(declared) if declared else Path("")
            checks.append(
                _check_result(
                    "training_config_points_to_feature_root",
                    bool(declared) and declared_root == root,
                    f"{dataset_cfg_path.name}: feature_cache.dir={declared_root if declared else '<missing>'}",
                )
            )
        except Exception as exc:
            checks.append(_check_result("training_config_points_to_feature_root", False, f"Could not load {dataset_cfg_path}: {exc}"))
    else:
        checks.append(_check_result("training_config_points_to_feature_root", False, f"Missing config file: {dataset_cfg_path}"))

    manifest_paths = sorted(root.glob("*/*/*/manifest.json")) if root.is_dir() else []
    checks.append(_check_result("cache_manifests_found", bool(manifest_paths) or not required, f"count={len(manifest_paths)}"))

    for manifest_path in manifest_paths:
        caches.append(
            _check_feature_cache_dir(
                dataset_name=name,
                cache_dir=manifest_path.parent,
                expected_backbones=expected_backbones,
                check_tensors=check_tensors,
                check_video_paths=check_video_paths,
                allow_full_tensor_load=allow_full_tensor_load,
            )
        )

    found_backbones = sorted({str(cache.get("backbone", "")) for cache in caches if str(cache.get("backbone", ""))})
    if require_all_backbones and expected_backbones:
        missing = [backbone for backbone in expected_backbones if backbone not in found_backbones]
        checks.append(
            _check_result(
                "expected_backbones_present",
                not missing,
                f"found={found_backbones}, missing={missing}",
            )
        )
    else:
        checks.append(
            _check_result(
                "expected_backbones_present",
                True,
                f"not enforced; found={found_backbones}",
            )
        )

    failed = any(check.get("status") == "fail" for check in checks) or any(
        cache.get("status") == "fail" for cache in caches
    )
    return {
        "kind": "feature_dataset",
        "name": name,
        "root": str(root),
        "status": "fail" if failed else "pass",
        "checks": checks,
        "caches": caches,
        "found_backbones": found_backbones,
    }


def _check_feature_cache_dir(
    *,
    dataset_name: str,
    cache_dir: Path,
    expected_backbones: list[str],
    check_tensors: bool,
    check_video_paths: bool,
    allow_full_tensor_load: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_path = cache_dir / "manifest.json"
    index_frame: Any | None = None
    manifest: dict[str, Any] = {}
    n_rows = 0

    backbone_dir = cache_dir.parents[1].name if len(cache_dir.parents) > 1 else ""
    split_key = cache_dir.parent.name
    backbone = _match_feature_backbone(backbone_dir, expected_backbones)

    checks.append(_check_result("manifest_exists", manifest_path.exists(), str(manifest_path)))
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
            manifest = {}

    kind = str(manifest.get("kind", ""))
    checks.append(
        _check_result(
            "manifest_kind_matches_dataset",
            bool(kind) and dataset_name in kind,
            f"kind={kind or '<missing>'}",
        )
    )

    files = manifest.get("files", {}) if isinstance(manifest.get("files", {}), dict) else {}
    index_name = str(files.get("index", "index.parquet") or "index.parquet")
    index_path = cache_dir / index_name
    checks.append(_check_result("index_exists", index_path.exists(), str(index_path)))
    if index_path.exists():
        try:
            import pandas as pd

            index_frame = pd.read_parquet(index_path)
            n_rows = int(len(index_frame))
            checks.append(_check_result("index_readable", True, f"rows={n_rows}"))
        except Exception as exc:
            checks.append(_check_result("index_readable", False, f"Failed to read index parquet: {exc}"))

    features_meta = manifest.get("features", {}) if isinstance(manifest.get("features", {}), dict) else {}
    manifest_samples = _safe_int(features_meta.get("n_samples"), default=-1)
    if index_frame is not None:
        checks.extend(
            _check_feature_index(
                dataset_name,
                index_frame,
                manifest_samples,
                features_meta,
                _manifest_split_names(manifest),
            )
        )
        if check_video_paths:
            checks.append(_check_feature_video_paths(index_frame))
        else:
            checks.append(_check_result("video_paths_exist", True, "Skipped by features.check_video_paths=false"))

    if check_tensors:
        checks.extend(
            _check_feature_tensor_payloads(
                cache_dir,
                files,
                features_meta,
                n_rows,
                allow_full_tensor_load=allow_full_tensor_load,
            )
        )
    else:
        checks.append(_check_result("tensor_shapes_valid", True, "Skipped by features.check_tensors=false"))

    failed = any(check.get("status") == "fail" for check in checks)
    return {
        "dataset": dataset_name,
        "cache_dir": str(cache_dir),
        "backbone_dir": backbone_dir,
        "backbone": backbone,
        "split_key": split_key,
        "signature": str(manifest.get("signature", cache_dir.name)),
        "n_samples": n_rows,
        "status": "fail" if failed else "pass",
        "checks": checks,
    }


def _check_feature_index(
    dataset_name: str,
    index_frame: Any,
    manifest_samples: int,
    features_meta: dict[str, Any],
    manifest_splits: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    n_rows = int(len(index_frame))
    checks.append(_check_result("index_nonempty", n_rows > 0, f"rows={n_rows}"))
    if manifest_samples >= 0:
        checks.append(
            _check_result(
                "index_rows_match_manifest",
                n_rows == manifest_samples,
                f"index_rows={n_rows}, manifest_n_samples={manifest_samples}",
            )
        )

    required_columns = {
        "mvp": ["feature_index", "sample_id", "split", "video_path", "plausibility_label"],
        "intphys2": ["feature_index", "sample_id", "split", "video_path", "plausibility"],
        "ssv2": ["feature_index", "sample_id", "split", "video_path", "label_idx"],
    }.get(dataset_name, ["feature_index", "sample_id", "split", "video_path"])
    missing_columns = [column for column in required_columns if column not in index_frame.columns]
    checks.append(_check_result("index_required_columns", not missing_columns, f"missing={missing_columns}"))

    if "feature_index" in index_frame.columns:
        try:
            values = sorted(int(value) for value in index_frame["feature_index"].tolist())
            expected = list(range(n_rows))
            checks.append(
                _check_result(
                    "feature_index_contiguous",
                    values == expected,
                    f"min={values[0] if values else '<none>'}, max={values[-1] if values else '<none>'}, rows={n_rows}",
                )
            )
        except Exception as exc:
            checks.append(_check_result("feature_index_contiguous", False, f"Could not parse feature_index: {exc}"))

    if "split" in index_frame.columns:
        splits = sorted(str(value) for value in index_frame["split"].dropna().astype(str).unique().tolist())
        split_ok = bool(splits) and (not manifest_splits or set(splits).issubset(set(manifest_splits)))
        checks.append(_check_result("index_splits_valid", split_ok, f"index={splits}, manifest={manifest_splits}"))

    manifest_hash = str(features_meta.get("sample_ids_sha256", "")).strip()
    if manifest_hash and "sample_id" in index_frame.columns and "feature_index" in index_frame.columns:
        try:
            ordered = index_frame.sort_values("feature_index")
            actual_hash = _sha256_lines(str(value) for value in ordered["sample_id"].tolist())
            checks.append(
                _check_result(
                    "sample_ids_match_manifest_hash",
                    actual_hash == manifest_hash,
                    f"actual={actual_hash}, manifest={manifest_hash}",
                )
            )
        except Exception as exc:
            checks.append(_check_result("sample_ids_match_manifest_hash", False, f"Could not hash sample ids: {exc}"))
    else:
        checks.append(
            _check_result(
                "sample_ids_match_manifest_hash",
                True,
                "No sample_ids_sha256 in manifest or sample_id/feature_index missing; skipped.",
            )
        )

    return checks


def _manifest_split_names(manifest: dict[str, Any]) -> list[str]:
    split_meta = manifest.get("split", {}) if isinstance(manifest.get("split", {}), dict) else {}
    raw = split_meta.get("names", [])
    if not isinstance(raw, list):
        raw = [raw]
    return [str(value) for value in raw if str(value).strip()]


def _check_feature_video_paths(index_frame: Any) -> dict[str, Any]:
    if "video_path" not in index_frame.columns:
        return _check_result("video_paths_exist", False, "index has no video_path column")

    missing_examples: list[str] = []
    missing_count = 0
    for raw in index_frame["video_path"].tolist():
        value = str(raw or "").strip()
        if not value:
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append("<empty>")
            continue
        path = _as_project_path(value)
        if not path.exists():
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append(str(path))
    return _check_result(
        "video_paths_exist",
        missing_count == 0,
        f"checked={len(index_frame)}, missing={missing_count}, examples={missing_examples}",
    )


def _check_feature_tensor_payloads(
    cache_dir: Path,
    files: dict[str, Any],
    features_meta: dict[str, Any],
    n_rows: int,
    *,
    allow_full_tensor_load: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, expected_ndim in (("pooled", 2), ("tokens", 3)):
        include_key = f"include_{key}"
        included = bool(features_meta.get(include_key, False))
        filename = str(files.get(key, "") or "").strip()
        if not filename:
            checks.append(
                _check_result(
                    f"{key}_tensor_present",
                    not included,
                    "not requested" if not included else f"manifest marks {include_key}=true but no file is listed",
                )
            )
            continue

        tensor_path = cache_dir / filename
        checks.append(_check_result(f"{key}_tensor_present", tensor_path.exists(), str(tensor_path)))
        if not tensor_path.exists():
            continue

        try:
            payload = _torch_load_feature_payload(tensor_path, allow_full_load=allow_full_tensor_load)
        except Exception as exc:
            checks.append(_check_result(f"{key}_tensor_readable", False, f"torch.load failed: {exc}"))
            continue
        checks.append(_check_result(f"{key}_tensor_readable", True, "loaded metadata without full tensor materialization"))
        checks.extend(_check_single_tensor_payload(key, payload, n_rows, expected_ndim))
    return checks


def _check_single_tensor_payload(
    key: str,
    payload: Any,
    n_rows: int,
    expected_ndim: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return [_check_result(f"{key}_tensor_payload_valid", False, f"expected dict, got {type(payload)!r}")]

    raw_layers = payload.get("selected_layers", [])
    try:
        selected_layers = [int(value) for value in raw_layers]
    except Exception:
        selected_layers = []
    by_layer = payload.get("by_layer", {})
    checks.append(
        _check_result(
            f"{key}_selected_layers_present",
            bool(selected_layers) and isinstance(by_layer, dict),
            f"selected_layers={selected_layers}",
        )
    )
    if not selected_layers or not isinstance(by_layer, dict):
        return checks

    bad_layers: list[str] = []
    for layer in selected_layers:
        tensor = _lookup_layer_tensor(by_layer, layer)
        if tensor is None:
            bad_layers.append(f"{layer}:missing")
            continue
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        if len(shape) != expected_ndim:
            bad_layers.append(f"{layer}:ndim={len(shape)} shape={shape}")
            continue
        if shape[0] != n_rows:
            bad_layers.append(f"{layer}:rows={shape[0]} expected={n_rows}")
    checks.append(
        _check_result(
            f"{key}_tensor_shapes_match_index",
            not bad_layers,
            f"layers={selected_layers}, rows={n_rows}, problems={bad_layers}",
        )
    )
    return checks


def _torch_load_feature_payload(path: Path, *, allow_full_load: bool) -> Any:
    import torch

    fake_error: Exception | None = None
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        fake_error = exc

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError as exc:
        if allow_full_load:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        raise RuntimeError(
            "Could not inspect tensor metadata with FakeTensorMode or mmap=True, "
            "and full tensor loading is disabled. Re-run with "
            "features.allow_full_tensor_load=true only if the cache fits in RAM. "
            f"fake_error={fake_error}; mmap_error={exc}"
        ) from exc


def _lookup_layer_tensor(by_layer: dict[Any, Any], layer: int) -> Any | None:
    if layer in by_layer:
        return by_layer[layer]
    layer_str = str(layer)
    if layer_str in by_layer:
        return by_layer[layer_str]
    return None


def _match_feature_backbone(backbone_dir: str, expected_backbones: list[str]) -> str:
    for name in sorted(expected_backbones, key=len, reverse=True):
        if backbone_dir == name or backbone_dir.startswith(f"{name}_"):
            return name
    return backbone_dir


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _sha256_lines(lines: Any) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _format_feature_human_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ok = bool(report.get("ok", False))
    summary = report.get("summary", {})
    mode = report.get("mode", {})
    lines.append("Probe4Physics Feature Cache Health Check")
    lines.append("")
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'}")
    lines.append(
        "Mode: CPU metadata/index/tensor-shape checks "
        f"(check_tensors={mode.get('check_tensors')}, "
        f"check_video_paths={mode.get('check_video_paths')}, "
        f"allow_full_tensor_load={mode.get('allow_full_tensor_load')})"
    )
    lines.append(
        "Checks: "
        f"total={summary.get('total_checks', 0)} "
        f"passed={summary.get('passed', 0)} "
        f"failed={summary.get('failed', 0)}"
    )
    lines.append(
        "Coverage: "
        f"datasets={summary.get('datasets', 0)} "
        f"caches={summary.get('caches', 0)} "
        f"backbones={summary.get('backbones', 0)}"
    )
    lines.append("")
    lines.append("Datasets")
    for dataset in report.get("datasets", []):
        status = str(dataset.get("status", "unknown")).upper()
        lines.append(
            f"- {dataset.get('name', '')}: {status} "
            f"root={dataset.get('root', '')} "
            f"caches={len(dataset.get('caches', []))} "
            f"backbones={dataset.get('found_backbones', [])}"
        )
        for cache in dataset.get("caches", []):
            cache_status = str(cache.get("status", "unknown")).upper()
            lines.append(
                f"  - {cache.get('backbone_dir', '')}/{cache.get('split_key', '')}/{cache.get('signature', '')}: "
                f"{cache_status} rows={cache.get('n_samples', 0)}"
            )

    failing = _collect_feature_failing_checks(report)
    lines.append("")
    lines.append("Failing checks")
    if not failing:
        lines.append("- none")
    else:
        for row in failing:
            lines.append(f"- {row['scope']} [{row['check']}]: {row['detail']}")
    return "\n".join(lines)


def _collect_feature_failing_checks(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in report.get("datasets", []):
        dataset_scope = str(dataset.get("name", ""))
        for check in dataset.get("checks", []):
            if str(check.get("status")) == "fail":
                rows.append(
                    {
                        "scope": dataset_scope,
                        "check": str(check.get("name", "")),
                        "detail": str(check.get("detail", "")),
                    }
                )
        for cache in dataset.get("caches", []):
            cache_scope = (
                f"{dataset_scope}/{cache.get('backbone_dir', '')}/"
                f"{cache.get('split_key', '')}/{cache.get('signature', '')}"
            )
            for check in cache.get("checks", []):
                if str(check.get("status")) != "fail":
                    continue
                rows.append(
                    {
                        "scope": cache_scope,
                        "check": str(check.get("name", "")),
                        "detail": str(check.get("detail", "")),
                    }
                )
    return rows


def _check_backbone_variant_layer_mapping(
    backbone_name: str,
    variant_name: str,
    section: dict[str, Any],
    variant_cfg: Any,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    model_name = ""
    depth = 0
    relative_depths: list[float] = []
    selected_layers: list[int] = []
    depth_map: dict[str, int] = {}

    if not isinstance(variant_cfg, dict):
        checks.append(
            _check_result(
                "variant_config_mapping",
                False,
                f"Variant '{variant_name}' must resolve to a mapping.",
            )
        )
        return {
            "name": backbone_name,
            "variant": variant_name,
            "model_name": model_name,
            "depth": depth,
            "requested_relative_depths": relative_depths,
            "backbone_layer_ids_1_based": [],
            "backbone_layer_ids_0_based": [],
            "selected_layers_1_based": selected_layers,
            "selected_layers_0_based": [],
            "status": "fail",
            "checks": checks,
        }

    model_name = str(variant_cfg.get("model_name", "")).strip()
    checks.append(
        _check_result(
            "variant_model_name_present",
            bool(model_name),
            f"model_name='{model_name}'" if model_name else "model_name is missing.",
        )
    )

    raw_depth_map = section.get("model_block_depths")
    if isinstance(raw_depth_map, dict) and raw_depth_map:
        try:
            depth_map = {str(key): int(value) for key, value in raw_depth_map.items()}
            checks.append(_check_result("model_block_depths_valid", True, "Depth map parsed."))
        except Exception as exc:
            checks.append(_check_result("model_block_depths_valid", False, f"Invalid depth map: {exc}"))
    else:
        checks.append(_check_result("model_block_depths_valid", False, "model_block_depths is missing or empty."))

    if model_name and model_name in depth_map:
        depth = int(depth_map[model_name])
        checks.append(_check_result("model_depth_known", depth > 0, f"depth={depth}"))
    else:
        known = ", ".join(sorted(depth_map)) if depth_map else "<none>"
        checks.append(
            _check_result(
                "model_depth_known",
                False,
                f"Model '{model_name}' not found in model_block_depths. Known: {known}",
            )
        )

    raw_relative_depths = section.get("default_relative_depths")
    if isinstance(raw_relative_depths, list) and raw_relative_depths:
        try:
            relative_depths = [float(value) for value in raw_relative_depths]
            checks.append(
                _check_result(
                    "default_relative_depths_valid",
                    True,
                    f"relative_depths={relative_depths}",
                )
            )
        except Exception as exc:
            checks.append(
                _check_result(
                    "default_relative_depths_valid",
                    False,
                    f"Could not cast default_relative_depths to float: {exc}",
                )
            )
    else:
        checks.append(
            _check_result(
                "default_relative_depths_valid",
                False,
                "default_relative_depths is missing or empty.",
            )
        )

    noise_levels: list[float] = []
    if backbone_name == "ltx_video":
        raw_noise_levels = section.get("default_noise_levels")
        if isinstance(raw_noise_levels, list) and raw_noise_levels:
            try:
                noise_levels = [float(value) for value in raw_noise_levels]
                checks.append(
                    _check_result(
                        "default_noise_levels_valid",
                        True,
                        f"noise_levels={noise_levels}",
                    )
                )
            except Exception as exc:
                checks.append(
                    _check_result(
                        "default_noise_levels_valid",
                        False,
                        f"Could not cast default_noise_levels to float: {exc}",
                    )
                )
        else:
            checks.append(
                _check_result(
                    "default_noise_levels_valid",
                    False,
                    "default_noise_levels is missing or empty.",
                )
            )

    if model_name and depth > 0 and depth_map:
        try:
            selected_layers = _resolve_selected_layers_for_variant(
                backbone_name=backbone_name,
                model_name=model_name,
                relative_depths=relative_depths,
                model_block_depths=depth_map,
                noise_levels=noise_levels,
            )
            checks.append(
                _check_result(
                    "selected_layers_resolved",
                    bool(selected_layers),
                    f"selected_layers={selected_layers}",
                )
            )
        except Exception as exc:
            checks.append(_check_result("selected_layers_resolved", False, str(exc)))

    selected_layer_upper_bound = len(selected_layers) if backbone_name == "ltx_video" else depth
    in_bounds = bool(
        selected_layer_upper_bound > 0 and all(1 <= layer <= selected_layer_upper_bound for layer in selected_layers)
    )
    checks.append(
        _check_result(
            "selected_layers_in_bounds",
            in_bounds,
            f"upper_bound={selected_layer_upper_bound}, selected_layers={selected_layers}",
        )
    )

    strictly_increasing = all(left < right for left, right in zip(selected_layers, selected_layers[1:]))
    checks.append(
        _check_result(
            "selected_layers_strictly_increasing",
            strictly_increasing,
            f"selected_layers={selected_layers}",
        )
    )

    original_ids_1 = list(range(1, depth + 1)) if depth > 0 else []
    original_ids_0 = [layer - 1 for layer in original_ids_1]
    selected_ids_0 = [layer - 1 for layer in selected_layers]
    failed = any(check.get("status") == "fail" for check in checks)
    return {
        "name": backbone_name,
        "variant": variant_name,
        "model_name": model_name,
        "depth": depth,
        "requested_relative_depths": relative_depths,
        "backbone_layer_ids_1_based": original_ids_1,
        "backbone_layer_ids_0_based": original_ids_0,
        "selected_layers_1_based": selected_layers,
        "selected_layers_0_based": selected_ids_0,
        "status": "fail" if failed else "pass",
        "checks": checks,
    }


def _resolve_selected_layers_for_variant(
    *,
    backbone_name: str,
    model_name: str,
    relative_depths: list[float],
    model_block_depths: dict[str, int],
    noise_levels: list[float] | None = None,
) -> list[int]:
    if backbone_name == "jepa_v1":
        return list(
            resolve_jepa_v1_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
                model_block_depths=model_block_depths,
            )
        )
    if backbone_name == "jepa_v2":
        return list(
            resolve_jepa_v2_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
                model_block_depths=model_block_depths,
            )
        )
    if backbone_name == "jepa_v2_1":
        return list(
            resolve_jepa_v2_1_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
                model_block_depths=model_block_depths,
            )
        )
    if backbone_name == "videomae":
        return list(
            resolve_videomae_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
                model_block_depths=model_block_depths,
                backbone_key="videomae",
            )
        )
    if backbone_name == "videomae_v2":
        return list(
            resolve_videomae_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
                model_block_depths=model_block_depths,
                backbone_key="videomae_v2",
            )
        )
    if backbone_name == "ltx_video":
        return list(
            resolve_ltx_probe_layer_ids(
                model_name,
                relative_depths=relative_depths,
                noise_levels=noise_levels,
                model_block_depths=model_block_depths,
            )
        )
    raise ValueError(f"Unsupported backbone for layer-resolution health check: '{backbone_name}'")


def _resolve_benchmark_layer_requests(
    backbone_cfg: dict[str, Any],
    variant_lookup: dict[tuple[str, str], dict[str, Any]],
    mvp_cfg: dict[str, Any],
    intphys2_cfg: dict[str, Any],
    ssv2_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    benchmark_cfgs = {
        "mvp": mvp_cfg,
        "intphys2": intphys2_cfg,
        "ssv2": ssv2_cfg,
    }
    for benchmark_name, benchmark_cfg in benchmark_cfgs.items():
        records.append(
            _resolve_single_benchmark_layer_request(
                benchmark_name=benchmark_name,
                benchmark_cfg=benchmark_cfg,
                backbone_cfg=backbone_cfg,
                variant_lookup=variant_lookup,
            )
        )
    return records


def _resolve_single_benchmark_layer_request(
    *,
    benchmark_name: str,
    benchmark_cfg: dict[str, Any],
    backbone_cfg: dict[str, Any],
    variant_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    raw_backbone = benchmark_cfg.get("backbone")
    backbone_section = raw_backbone if isinstance(raw_backbone, dict) else {}
    backbone_name = str(backbone_section.get("name", "")).strip()
    backbone_kwargs = backbone_section.get("kwargs")
    kwargs = backbone_kwargs if isinstance(backbone_kwargs, dict) else {}

    feature_section = benchmark_cfg.get("feature_cache")
    feature_cache = feature_section if isinstance(feature_section, dict) else {}
    raw_layer_ids = feature_cache.get("layer_ids", [])
    if raw_layer_ids is None:
        raw_layer_ids = []

    try:
        requested_layer_ids = [int(value) for value in raw_layer_ids]
    except Exception as exc:
        check = _check_result(
            f"{benchmark_name}.requested_layers_valid",
            False,
            f"feature_cache.layer_ids must be integer-like values: {exc}",
        )
        return {
            "benchmark": benchmark_name,
            "backbone": backbone_name,
            "variant": "",
            "requested_layer_ids": [],
            "available_selected_layers": [],
            "effective_layer_ids": [],
            "missing_requested_layers": [],
            "check": check,
        }

    section = backbone_cfg.get(backbone_name) if isinstance(backbone_cfg.get(backbone_name), dict) else {}
    requested_variant = str(kwargs.get("variant", "")).strip()
    variant = requested_variant or str(section.get("default_variant", "")).strip()
    variant_payload = variant_lookup.get((backbone_name, variant), {})
    available_selected = list(variant_payload.get("selected_layers_1_based", []) or [])

    if not backbone_name or not variant:
        check = _check_result(
            f"{benchmark_name}.requested_layers_valid",
            False,
            f"Could not resolve backbone/variant from benchmark config (backbone='{backbone_name}', variant='{variant}').",
        )
        return {
            "benchmark": benchmark_name,
            "backbone": backbone_name,
            "variant": variant,
            "requested_layer_ids": requested_layer_ids,
            "available_selected_layers": available_selected,
            "effective_layer_ids": [],
            "missing_requested_layers": requested_layer_ids,
            "check": check,
        }

    if not available_selected:
        check = _check_result(
            f"{benchmark_name}.requested_layers_valid",
            False,
            f"No selected layers resolved for backbone='{backbone_name}', variant='{variant}'.",
        )
        return {
            "benchmark": benchmark_name,
            "backbone": backbone_name,
            "variant": variant,
            "requested_layer_ids": requested_layer_ids,
            "available_selected_layers": available_selected,
            "effective_layer_ids": [],
            "missing_requested_layers": requested_layer_ids,
            "check": check,
        }

    if requested_layer_ids:
        missing = sorted(set(requested_layer_ids) - set(available_selected))
        ok = len(missing) == 0
        effective = requested_layer_ids if ok else []
        detail = (
            f"requested={requested_layer_ids}, available={available_selected}, missing={missing}"
            if not ok
            else f"requested={requested_layer_ids}, available={available_selected}"
        )
        check = _check_result(f"{benchmark_name}.requested_layers_valid", ok, detail)
        return {
            "benchmark": benchmark_name,
            "backbone": backbone_name,
            "variant": variant,
            "requested_layer_ids": requested_layer_ids,
            "available_selected_layers": available_selected,
            "effective_layer_ids": effective,
            "missing_requested_layers": missing,
            "check": check,
        }

    check = _check_result(
        f"{benchmark_name}.requested_layers_valid",
        True,
        f"requested=<default>, effective={available_selected}",
    )
    return {
        "benchmark": benchmark_name,
        "backbone": backbone_name,
        "variant": variant,
        "requested_layer_ids": requested_layer_ids,
        "available_selected_layers": available_selected,
        "effective_layer_ids": available_selected,
        "missing_requested_layers": [],
        "check": check,
    }


def _check_backbone(
    name: str,
    backbone_cfg: dict[str, Any],
    *,
    synthetic_forward: bool,
    device: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    section = backbone_cfg.get(name)
    variant = LARGEST_VARIANTS.get(name, "")
    registered_adapters = set(get_registered_adapters())
    checks.append(
        _check_result(
            "adapter_registered",
            name in registered_adapters,
            f"Adapter '{name}' is {'registered' if name in registered_adapters else 'missing from registry'}.",
        )
    )
    if not isinstance(section, dict):
        checks.append(_check_result("config_section_exists", False, f"Missing '{name}' in backbones.yaml"))
        return _bundle("backbone", name, variant, checks)

    variants = section.get("variants", {})
    if not isinstance(variants, dict) or variant not in variants:
        checks.append(
            _check_result(
                "variant_wired",
                False,
                f"Variant '{variant}' not found under '{name}.variants'.",
            )
        )
        return _bundle("backbone", name, variant, checks)

    checks.append(_check_result("variant_wired", True, f"Configured variant '{variant}' found."))
    variant_cfg = variants[variant]

    resolved_checkpoint: Path | None = None
    if "checkpoints_dir" in section and isinstance(variant_cfg, dict) and "checkpoint_filename" in variant_cfg:
        checkpoints_dir = _as_project_path(str(section.get("checkpoints_dir", "")))
        checkpoint_filename = str(variant_cfg.get("checkpoint_filename", "")).strip()
        resolved_checkpoint = checkpoints_dir / checkpoint_filename
        checks.append(
            _check_result(
                "checkpoint_present",
                resolved_checkpoint.exists(),
                str(resolved_checkpoint),
            )
        )
    else:
        checks.append(_check_result("checkpoint_present", True, "Not required (HF-backed backbone)."))

    if synthetic_forward:
        probe_kwargs: dict[str, Any] = {"variant": variant, "device": device}
        if resolved_checkpoint is not None:
            probe_kwargs["checkpoint_path"] = str(resolved_checkpoint)

        probe_ok, probe_detail = _run_backbone_probe_subprocess(name, probe_kwargs)
        checks.append(_check_result("synthetic_forward", probe_ok, probe_detail))
    else:
        checks.append(
            _check_result(
                "synthetic_forward",
                True,
                "Skipped in lightweight mode. Set synthetic_forward=true to run backbone smoke forwards.",
            )
        )
    return _bundle("backbone", name, variant, checks)


def _run_backbone_probe_subprocess(name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
    script = f"""
import json
import traceback
import torch
from models import create_adapter

name = {name!r}
kwargs = {json.dumps(kwargs)}
runtime = {{
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
}}
try:
    if kwargs.get("device") == "cuda" and not runtime["cuda_available"]:
        raise RuntimeError(
            "Requested device=cuda but torch.cuda.is_available() is False"
        )
    adapter = create_adapter(name, **kwargs)
    frames = int(getattr(adapter, "frames_per_clip", 16))
    crop = int(getattr(adapter, "crop_size", 224))
    clips = torch.zeros((1, 3, frames, crop, crop), dtype=torch.float32)
    selected = getattr(adapter, "selected_layers", ())
    layer_ids = [int(selected[0])] if isinstance(selected, (list, tuple)) and selected else None
    with torch.no_grad():
        features = adapter.extract(clips, layer_ids=layer_ids)
    payload = {{
        "ok": True,
        "frames_per_clip": frames,
        "crop_size": crop,
        "selected_layers": list(getattr(features, "selected_layers", ()) or ()),
        "runtime": runtime,
    }}
    print(json.dumps(payload))
except Exception as exc:
    payload = {{
        "ok": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "runtime": runtime,
    }}
    print(json.dumps(payload))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as exc:
        return False, f"Failed to execute probe subprocess: {exc}"

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "<empty stderr>"
        return False, f"Subprocess returned {proc.returncode}: {stderr}"

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return False, "No output from probe subprocess."

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False, f"Probe output is not JSON: {lines[-1][:300]}"

    if bool(payload.get("ok", False)):
        frames = payload.get("frames_per_clip")
        crop = payload.get("crop_size")
        runtime = payload.get("runtime", {})
        return True, (
            f"Probe succeeded (frames={frames}, crop={crop}, "
            f"torch={runtime.get('torch_version')}, "
            f"cuda_available={runtime.get('cuda_available')}, "
            f"device_count={runtime.get('cuda_device_count')})."
        )

    runtime = payload.get("runtime", {})
    return False, (
        f"{payload.get('error', 'Unknown probe error')} "
        f"[torch={runtime.get('torch_version')}, "
        f"torch_cuda={runtime.get('torch_cuda_version')}, "
        f"cuda_available={runtime.get('cuda_available')}, "
        f"device_count={runtime.get('cuda_device_count')}]"
    )


def _check_mvp(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    annotation_file = _as_project_path(str(config.get("annotation_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/mvp/full_60_20_20")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("annotation_file_exists", annotation_file.exists(), str(annotation_file)))
    split_pairs = split_dir / "split_pairs.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_pairs_exists", split_pairs.exists(), str(split_pairs)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if annotation_file.exists() and split_pairs.exists():
        try:
            import pandas as pd

            frame = pd.read_parquet(split_pairs)
            required_sample_ids: set[str] = set()
            for raw in frame.get("sample_ids_json", []):
                try:
                    required_sample_ids.update(str(item) for item in json.loads(str(raw)))
                except Exception:
                    continue

            rows = load_mvp_rows(annotation_file)
            sample_to_ref: dict[str, str] = {}
            for row in rows:
                sample_to_ref[derive_sample_id(row)] = derive_video_ref(row)

            present = 0
            missing = 0
            missing_refs: list[str] = []
            for sample_id in sorted(required_sample_ids):
                ref = sample_to_ref.get(sample_id, "")
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate or sample_id)

            stats["video_stats"] = {
                "required_samples": len(required_sample_ids),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Missing annotation or split_pairs file."))

    return _bundle("dataset", "mvp", None, checks, stats=stats)


def _check_intphys2(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    metadata_file = _as_project_path(str(config.get("metadata_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/intphys2")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("metadata_file_exists", metadata_file.exists(), str(metadata_file)))
    split_scenes = split_dir / "split_scenes.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_scenes_exists", split_scenes.exists(), str(split_scenes)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if metadata_file.exists():
        try:
            rows = [normalize_intphys2_row(row) for row in load_intphys2_rows(metadata_file)]
            present = 0
            missing = 0
            missing_refs: list[str] = []
            for row in rows:
                ref = str(row.get("video_path", "")).strip()
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(rows),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Metadata file not found."))

    return _bundle("dataset", "intphys2", None, checks, stats=stats)


def _check_ssv2(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    train_file = _as_project_path(str(config.get("train_annotation_file", "")))
    val_file = _as_project_path(str(config.get("val_annotation_file", "")))
    labels_file = _as_project_path(str(config.get("labels_file", "")))
    split_dir = _as_project_path(str((config.get("split", {}) or {}).get("dir", "data/splits/ssv2")))
    videos_root = _as_project_path(str(config.get("videos_root", "")))

    checks.append(_check_result("train_annotation_exists", train_file.exists(), str(train_file)))
    checks.append(_check_result("val_annotation_exists", val_file.exists(), str(val_file)))
    checks.append(_check_result("labels_exists", labels_file.exists(), str(labels_file)))

    split_clips = split_dir / "split_clips.parquet"
    split_manifest = split_dir / "manifest.json"
    checks.append(_check_result("split_clips_exists", split_clips.exists(), str(split_clips)))
    checks.append(_check_result("split_manifest_exists", split_manifest.exists(), str(split_manifest)))

    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            stats["manifest_stats"] = manifest.get("stats", {})
            checks.append(_check_result("manifest_readable", True, "Parsed manifest.json"))
        except Exception as exc:
            checks.append(_check_result("manifest_readable", False, f"Failed to parse manifest: {exc}"))
    else:
        checks.append(_check_result("manifest_readable", False, "Manifest file not found."))

    if split_clips.exists():
        try:
            import pandas as pd

            frame = pd.read_parquet(split_clips)
            refs: list[str] = []
            if "video_ref" in frame.columns:
                refs = [str(item) for item in frame["video_ref"].tolist()]
            elif "sample_id" in frame.columns:
                refs = [f"{item}.webm" for item in frame["sample_id"].tolist()]

            present = 0
            missing = 0
            missing_refs: list[str] = []
            for ref in refs:
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(refs),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    elif train_file.exists() and val_file.exists():
        try:
            train_rows = load_ssv2_annotations(train_file)
            val_rows = load_ssv2_annotations(val_file)
            refs = [f"{row['id']}.webm" for row in train_rows + val_rows if "id" in row]
            present = 0
            missing = 0
            missing_refs: list[str] = []
            for ref in refs:
                exists, candidate = _video_exists(ref, videos_root)
                if exists:
                    present += 1
                else:
                    missing += 1
                    if len(missing_refs) < 20:
                        missing_refs.append(candidate)
            stats["video_stats"] = {
                "required_samples": len(refs),
                "present_videos": present,
                "missing_videos": missing,
                "missing_examples": missing_refs,
            }
            checks.append(_check_result("videos_downloaded", missing == 0, f"missing={missing}"))
        except Exception as exc:
            checks.append(_check_result("videos_downloaded", False, f"Video check failed: {exc}"))
    else:
        checks.append(_check_result("videos_downloaded", False, "Missing split_clips and annotations."))

    return _bundle("dataset", "ssv2", None, checks, stats=stats)


def _video_exists(video_ref: str, videos_root: Path) -> tuple[bool, str]:
    ref = str(video_ref or "").strip()
    if not ref:
        return False, "<empty video ref>"

    direct = Path(ref)
    normalized = ref.lstrip("/\\")
    rooted_relative = videos_root / normalized
    local = videos_root / ref

    if rooted_relative.exists():
        return True, str(rooted_relative)
    if direct.is_absolute() and direct.exists():
        return True, str(direct)
    if local.exists():
        return True, str(local)
    return False, str(rooted_relative)


def _check_result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "detail": detail,
    }


def _bundle(
    kind: str,
    name: str,
    variant: str | None,
    checks: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed = any(check.get("status") == "fail" for check in checks)
    payload: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "status": "fail" if failed else "pass",
        "checks": checks,
    }
    if variant is not None:
        payload["variant"] = variant
    if stats:
        payload["stats"] = stats
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(payload)!r}")
    return payload


def _as_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _format_layer_human_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ok = bool(report.get("ok", False))
    summary = report.get("summary", {})
    lines.append("Probe4Physics Layer Health Check")
    lines.append("")
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'}")
    lines.append(
        "Checks: "
        f"total={summary.get('total_checks', 0)} "
        f"passed={summary.get('passed', 0)} "
        f"failed={summary.get('failed', 0)}"
    )
    lines.append(
        "Mappings: "
        f"variants={summary.get('variant_mappings', 0)} "
        f"benchmark_requests={summary.get('benchmark_requests', 0)}"
    )
    lines.append("")
    lines.append("Backbone Variant Layer Mapping")
    for item in report.get("variants", []):
        status = str(item.get("status", "unknown")).upper()
        name = str(item.get("name", ""))
        variant = str(item.get("variant", ""))
        model_name = str(item.get("model_name", ""))
        depth = int(item.get("depth", 0) or 0)
        lines.append(f"- {name} ({variant}) model={model_name} depth={depth}: {status}")
        lines.append(f"  original_1_based={item.get('backbone_layer_ids_1_based', [])}")
        lines.append(f"  selected_1_based={item.get('selected_layers_1_based', [])}")
        lines.append(f"  selected_0_based={item.get('selected_layers_0_based', [])}")

    lines.append("")
    lines.append("Benchmark Layer Requests")
    for item in report.get("benchmark_layer_requests", []):
        check = item.get("check", {})
        status = str(check.get("status", "unknown")).upper()
        benchmark = str(item.get("benchmark", ""))
        backbone = str(item.get("backbone", ""))
        variant = str(item.get("variant", ""))
        lines.append(f"- {benchmark}: {status} ({backbone}, {variant})")
        lines.append(f"  requested={item.get('requested_layer_ids', [])}")
        lines.append(f"  available_selected={item.get('available_selected_layers', [])}")
        lines.append(f"  effective={item.get('effective_layer_ids', [])}")
        missing = item.get("missing_requested_layers", [])
        if missing:
            lines.append(f"  missing={missing}")

    failing = _collect_layer_failing_checks(report)
    lines.append("")
    lines.append("Failing checks")
    if not failing:
        lines.append("- none")
    else:
        for row in failing:
            lines.append(f"- {row['scope']} [{row['check']}]: {row['detail']}")

    return "\n".join(lines)


def _collect_layer_failing_checks(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in report.get("variants", []):
        scope = f"{item.get('name', '')} ({item.get('variant', '')})"
        for check in item.get("checks", []):
            if str(check.get("status")) != "fail":
                continue
            rows.append(
                {
                    "scope": scope,
                    "check": str(check.get("name", "")),
                    "detail": str(check.get("detail", "")),
                }
            )

    for item in report.get("benchmark_layer_requests", []):
        check = item.get("check", {})
        if str(check.get("status")) != "fail":
            continue
        scope = f"{item.get('benchmark', '')} ({item.get('backbone', '')}/{item.get('variant', '')})"
        rows.append(
            {
                "scope": scope,
                "check": str(check.get("name", "")),
                "detail": str(check.get("detail", "")),
            }
        )
    return rows


def _format_human_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ok = bool(report.get("ok", False))
    summary = report.get("summary", {})
    mode = report.get("mode", {})
    lines.append("Probe4Physics Health Check")
    lines.append("")
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'}")
    lines.append(
        "Mode: "
        + (
            f"deep synthetic-forward smoke (device={mode.get('device', 'cpu')})"
            if bool(mode.get("synthetic_forward", False))
            else "lightweight static checks"
        )
    )
    lines.append(
        "Checks: "
        f"total={summary.get('total_checks', 0)} "
        f"passed={summary.get('passed', 0)} "
        f"failed={summary.get('failed', 0)}"
    )
    lines.append("")
    lines.append("Backbones")
    for item in report.get("tested_backbones", []):
        status = str(item.get("status", "unknown")).upper()
        name = str(item.get("name", ""))
        variant = str(item.get("variant", ""))
        lines.append(f"- {name} ({variant}): {status}")

    lines.append("")
    lines.append("Datasets")
    for item in report.get("tested_datasets", []):
        status = str(item.get("status", "unknown")).upper()
        name = str(item.get("name", ""))
        lines.append(f"- {name}: {status}")

    failing = _collect_failing_checks(report)
    lines.append("")
    lines.append("Failing checks")
    if not failing:
        lines.append("- none")
    else:
        for row in failing:
            lines.append(
                f"- {row['scope']} [{row['check']}]: {row['detail']}"
            )

    return "\n".join(lines)


def _collect_failing_checks(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section in ("backbones", "datasets"):
        for item in report.get(section, []):
            name = str(item.get("name", ""))
            variant = str(item.get("variant", "")).strip()
            scope = f"{name} ({variant})" if variant else name
            for check in item.get("checks", []):
                if str(check.get("status")) != "fail":
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "check": str(check.get("name", "")),
                        "detail": str(check.get("detail", "")),
                    }
                )
    return rows
