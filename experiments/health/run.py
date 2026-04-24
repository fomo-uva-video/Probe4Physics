from __future__ import annotations

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
from models.jepa_v2_1_adapter import _hierarchical_selected_layers
from models.ltx_video_adapter import resolve_relative_depth_layers as resolve_ltx_relative_depth_layers
from models import get_registered_adapters
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

    if model_name and depth > 0 and depth_map:
        try:
            selected_layers = _resolve_selected_layers_for_variant(
                backbone_name=backbone_name,
                model_name=model_name,
                relative_depths=relative_depths,
                model_block_depths=depth_map,
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

    in_bounds = bool(depth > 0 and all(1 <= layer <= depth for layer in selected_layers))
    checks.append(
        _check_result(
            "selected_layers_in_bounds",
            in_bounds,
            f"depth={depth}, selected_layers={selected_layers}",
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
        return list(_hierarchical_selected_layers(model_name, model_block_depths))
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
            resolve_ltx_relative_depth_layers(
                model_name,
                relative_depths=relative_depths,
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
