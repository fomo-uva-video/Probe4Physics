from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from .ltx_video_adapter import resolve_noise_levels, resolve_probe_layer_ids, resolve_probe_layer_specs
from .preprocessing import imagenet_preprocessing_metadata, ltx_diffusion_preprocessing_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"
_HIERARCHICAL_LAYERS: dict[int, tuple[int, ...]] = {
    12: (2, 5, 8, 11),
    24: (5, 11, 17, 23),
    40: (9, 19, 29, 39),
    48: (11, 23, 37, 47),
}


def resolve_backbone_cache_metadata(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve lightweight backbone identity used in feature-cache signatures."""

    key = str(name).strip()
    path = Path(str(kwargs.get("config_path", DEFAULT_BACKBONES_CONFIG_PATH))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        return {"name": key, "config_path": str(path)}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    section = payload.get(key)
    if not isinstance(section, dict):
        return {"name": key, "config_path": str(path)}

    variants = section.get("variants", {})
    chosen = str(kwargs.get("variant", "") or section.get("default_variant", "")).strip()
    variant_cfg = variants.get(chosen, {}) if isinstance(variants, dict) else {}
    if not isinstance(variant_cfg, dict):
        variant_cfg = {}

    model_name = str(kwargs.get("model_name", "") or variant_cfg.get("model_name", ""))
    relative_depths = kwargs.get("relative_depths")
    if relative_depths is None:
        relative_depths = section.get("default_relative_depths", [])
    noise_levels = kwargs.get("noise_levels")
    if noise_levels is None:
        noise_levels = section.get("default_noise_levels", [])

    metadata: dict[str, Any] = {
        "name": key,
        "config_path": str(path),
        "variant": chosen,
        "model_name": model_name,
        "frames_per_clip": _int_setting(kwargs, variant_cfg, "frames_per_clip"),
        "crop_size": _int_setting(kwargs, variant_cfg, "crop_size"),
        "patch_size": _int_setting(kwargs, variant_cfg, "patch_size"),
        "tubelet_size": _int_setting(kwargs, variant_cfg, "tubelet_size"),
        "selected_layers": _resolve_selected_layers(
            key,
            model_name=model_name,
            relative_depths=relative_depths,
            model_block_depths=section.get("model_block_depths", {}),
            noise_levels=noise_levels,
        ),
        "preprocessing": _resolve_preprocessing_metadata(key, kwargs),
    }
    if "patch_size_t" in kwargs or "patch_size_t" in variant_cfg:
        metadata["patch_size_t"] = _int_setting(kwargs, variant_cfg, "patch_size_t")
    return metadata


def resolve_backbone_layer_label(name: str, kwargs: dict[str, Any], layer: int | str) -> str:
    if isinstance(layer, str):
        return str(layer)

    resolved_layer = int(layer)
    key = str(name).strip()
    if key != "ltx_video":
        return str(resolved_layer)

    path = Path(str(kwargs.get("config_path", DEFAULT_BACKBONES_CONFIG_PATH))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        return str(resolved_layer)

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    section = payload.get(key)
    if not isinstance(section, dict):
        return str(resolved_layer)

    variants = section.get("variants", {})
    chosen = str(kwargs.get("variant", "") or section.get("default_variant", "")).strip()
    variant_cfg = variants.get(chosen, {}) if isinstance(variants, dict) else {}
    if not isinstance(variant_cfg, dict):
        variant_cfg = {}

    model_name = str(kwargs.get("model_name", "") or variant_cfg.get("model_name", ""))
    relative_depths = kwargs.get("relative_depths")
    if relative_depths is None:
        relative_depths = section.get("default_relative_depths", [])
    noise_levels = kwargs.get("noise_levels")
    if noise_levels is None:
        noise_levels = section.get("default_noise_levels", [])
    model_block_depths = section.get("model_block_depths", {})
    if not model_name or not isinstance(model_block_depths, dict):
        return str(resolved_layer)

    try:
        specs = resolve_probe_layer_specs(
            model_name,
            relative_depths=relative_depths,
            noise_levels=noise_levels,
            model_block_depths={str(k): int(v) for k, v in model_block_depths.items()},
            config_path=path,
        )
    except Exception:
        return str(resolved_layer)

    for spec in specs:
        if int(spec.probe_layer_id) == resolved_layer:
            return f"noise_{spec.noise_fraction:.1f}_block_{int(spec.depth_layer_id)}"
    return str(resolved_layer)


def _int_setting(kwargs: dict[str, Any], variant_cfg: dict[str, Any], key: str) -> int | None:
    value = kwargs.get(key, variant_cfg.get(key))
    if value is None:
        return None
    return int(value)


def _resolve_selected_layers(
    backbone_name: str,
    *,
    model_name: str,
    relative_depths: Any,
    model_block_depths: Any,
    noise_levels: Any = None,
) -> list[int]:
    if not model_name or not isinstance(model_block_depths, dict):
        return []

    depths = {str(k): int(v) for k, v in model_block_depths.items()}
    if model_name not in depths:
        return []

    depth = depths[model_name]
    if backbone_name == "jepa_v2_1":
        return [idx + 1 for idx in _HIERARCHICAL_LAYERS.get(depth, ())]

    if backbone_name == "ltx_video":
        return list(
            resolve_probe_layer_ids(
                model_name,
                relative_depths=relative_depths,
                noise_levels=noise_levels,
                model_block_depths=depths,
            )
        )

    if not isinstance(relative_depths, Sequence) or isinstance(relative_depths, (str, bytes)):
        return []

    resolved: list[int] = []
    for item in relative_depths:
        value = float(item)
        if not (0.0 < value <= 1.0):
            continue
        block_id = int(round(depth * value))
        block_id = max(1, min(depth, block_id))
        if block_id not in resolved:
            resolved.append(block_id)
    return resolved


def _resolve_preprocessing_metadata(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    if name in {"jepa_v1", "jepa_v2", "jepa_v2_1", "videomae", "videomae_v2"}:
        return imagenet_preprocessing_metadata(family=name)
    if name == "ltx_video":
        return ltx_diffusion_preprocessing_metadata(
            normalize_input=bool(kwargs.get("normalize_input", True)),
            noise_levels=resolve_noise_levels(
                kwargs.get("noise_levels"),
                config_path=kwargs.get("config_path", DEFAULT_BACKBONES_CONFIG_PATH),
            ),
            prompt_mode="empty_string",
            noise_policy="fixed_reference_noise",
        )
    return {}
