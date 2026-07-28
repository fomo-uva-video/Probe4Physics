from __future__ import annotations

"""Parameter-count helpers for configured backbone variants.

The exact counter works on any instantiated ``torch.nn.Module`` but avoids a
top-level torch import so this module can also build architecture-count tables
from ``configs/backbones.yaml`` in lightweight environments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"

VIT_BACKBONE_KEYS = ("jepa_v1", "jepa_v2", "jepa_v2_1", "videomae", "videomae_v2")


@dataclass(frozen=True)
class ViTArchitectureSpec:
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float


@dataclass(frozen=True)
class ParameterGroupCount:
    group: str
    total: int
    trainable: int
    frozen: int
    tensors: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "tensors": self.tensors,
        }


VIT_ARCHITECTURES: dict[str, ViTArchitectureSpec] = {
    "vit_base": ViTArchitectureSpec(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "vit_large": ViTArchitectureSpec(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    "vit_huge": ViTArchitectureSpec(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0),
    "vit_giant": ViTArchitectureSpec(embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11),
    "vit_giant_xformers": ViTArchitectureSpec(embed_dim=1408, depth=40, num_heads=22, mlp_ratio=48 / 11),
    "vit_gigantic": ViTArchitectureSpec(embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64 / 13),
    "vit_gigantic_xformers": ViTArchitectureSpec(
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mlp_ratio=64 / 13,
    ),
}


def count_module_parameters(
    module: Any,
    *,
    group_depth: int = 1,
    include_buffers: bool = False,
) -> dict[str, Any]:
    """Count parameters in an instantiated module, grouped by name prefix.

    Args:
        module: Any object implementing the ``torch.nn.Module``-style
            ``named_parameters`` method.
        group_depth: Number of dotted name components to keep as the group key.
            ``group_depth=1`` groups ``blocks.0.attn.qkv.weight`` under
            ``blocks``; ``group_depth=3`` groups it under ``blocks.0.attn``.
        include_buffers: Include ``named_buffers`` in a separate ``buffers``
            total. Buffers are not added to parameter totals.

    Returns:
        A serializable dict with total/trainable/frozen counts and per-group
        rows. Shared parameters follow PyTorch's default ``named_parameters``
        behavior, which counts duplicate references once.
    """

    if group_depth < 1:
        raise ValueError(f"group_depth must be >= 1, got {group_depth!r}.")
    if not hasattr(module, "named_parameters"):
        raise TypeError("module must provide a named_parameters() method.")

    groups: dict[str, ParameterGroupCount] = {}
    total = 0
    trainable = 0
    tensor_count = 0
    for name, parameter in module.named_parameters(recurse=True):
        count = int(parameter.numel())
        is_trainable = bool(getattr(parameter, "requires_grad", False))
        group = _parameter_group_name(name, group_depth=group_depth)
        groups[group] = _add_group_count(
            groups.get(group),
            group=group,
            count=count,
            trainable=is_trainable,
        )
        total += count
        trainable += count if is_trainable else 0
        tensor_count += 1

    buffer_total = 0
    buffer_tensors = 0
    if include_buffers and hasattr(module, "named_buffers"):
        for _, buffer in module.named_buffers(recurse=True):
            buffer_total += int(buffer.numel())
            buffer_tensors += 1

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "tensors": tensor_count,
        "groups": [
            group.as_dict()
            for group in sorted(groups.values(), key=lambda item: (-item.total, item.group))
        ],
        "buffers": {
            "total": buffer_total,
            "tensors": buffer_tensors,
        },
    }


def count_adapter_parameters(
    adapter: Any,
    *,
    group_depth: int = 1,
    include_buffers: bool = False,
) -> dict[str, Any]:
    """Count the primary module held by one of this repo's backbone adapters."""

    module = _resolve_adapter_module(adapter)
    result = count_module_parameters(
        module,
        group_depth=group_depth,
        include_buffers=include_buffers,
    )
    result["module_attribute"] = _resolve_adapter_module_name(adapter)
    return result


def build_vit_parameter_table(
    *,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
    backbones: Sequence[str] | None = None,
    variants: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build a parameter-count table for configured ViT-family variants.

    The table covers the ViT-style encoder backbones in ``configs/backbones.yaml``
    and intentionally excludes LTX-Video, whose transformer architecture is not
    a plain ViT encoder.
    """

    config = _load_backbones_config(config_path)
    requested_backbones = tuple(backbones or VIT_BACKBONE_KEYS)
    rows: list[dict[str, Any]] = []

    for backbone in requested_backbones:
        section = config.get(backbone)
        if not isinstance(section, dict):
            continue
        raw_variants = section.get("variants")
        if not isinstance(raw_variants, dict):
            continue
        selected_variants = tuple(variants.get(backbone, ())) if variants else ()
        variant_names = selected_variants or tuple(raw_variants.keys())
        default_variant = str(section.get("default_variant", "")).strip()

        for variant_name in variant_names:
            variant_cfg = raw_variants.get(variant_name)
            if not isinstance(variant_cfg, dict):
                continue
            model_name = str(variant_cfg.get("model_name", "")).strip()
            if model_name not in VIT_ARCHITECTURES:
                continue
            rows.append(
                _build_variant_parameter_row(
                    backbone=backbone,
                    variant=str(variant_name),
                    variant_cfg=variant_cfg,
                    model_name=model_name,
                    is_default_variant=str(variant_name) == default_variant,
                )
            )

    return rows


def select_vit_size_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select ViT-L rows plus one largest configured ViT row per backbone."""

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    by_backbone: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        backbone = str(row.get("backbone", ""))
        by_backbone.setdefault(backbone, []).append(row)
        if str(row.get("model_name", "")) == "vit_large":
            selected[(backbone, str(row.get("variant", "")))] = dict(row)

    for backbone, backbone_rows in by_backbone.items():
        largest = _select_largest_variant(backbone_rows)
        selected[(backbone, str(largest.get("variant", "")))] = dict(largest)

    return sorted(
        selected.values(),
        key=lambda row: (
            str(row.get("backbone", "")),
            int(row.get("depth", 0)),
            int(row.get("embed_dim", 0)),
            str(row.get("variant", "")),
        ),
    )


def _build_variant_parameter_row(
    *,
    backbone: str,
    variant: str,
    variant_cfg: Mapping[str, Any],
    model_name: str,
    is_default_variant: bool,
) -> dict[str, Any]:
    spec = VIT_ARCHITECTURES[model_name]
    crop_size = int(variant_cfg.get("crop_size", 224))
    patch_size = int(variant_cfg.get("patch_size", 16))
    frames_per_clip = int(variant_cfg.get("frames_per_clip", 16))
    tubelet_size = int(variant_cfg.get("tubelet_size", 2))

    counts = _estimate_vit_parameter_groups(
        backbone=backbone,
        spec=spec,
        crop_size=crop_size,
        patch_size=patch_size,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
    )
    total = sum(counts.values())
    fixed_parameters = counts.get("position_embedding", 0) if backbone == "jepa_v1" else 0

    row = {
        "backbone": backbone,
        "variant": variant,
        "size_label": _vit_size_label(model_name),
        "model_name": model_name,
        "is_default_variant": bool(is_default_variant),
        "crop_size": crop_size,
        "frames_per_clip": frames_per_clip,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "depth": spec.depth,
        "embed_dim": spec.embed_dim,
        "num_heads": spec.num_heads,
        "mlp_ratio": spec.mlp_ratio,
        "patch_embed_params": counts.get("patch_embed", 0),
        "patch_embed_img_params": counts.get("patch_embed_img", 0),
        "position_embedding_params": counts.get("position_embedding", 0),
        "blocks_attention_params": counts.get("blocks_attention", 0),
        "blocks_mlp_params": counts.get("blocks_mlp", 0),
        "blocks_norm_params": counts.get("blocks_norm", 0),
        "final_norm_params": counts.get("final_norm", 0),
        "hierarchical_norm_params": counts.get("hierarchical_norms", 0),
        "modality_embedding_params": counts.get("modality_embedding", 0),
        "total_parameters": total,
        "total_millions": total / 1_000_000,
        "fixed_parameters": fixed_parameters,
        "formula": _formula_label(backbone),
    }
    row["blocks_total_params"] = (
        row["blocks_attention_params"]
        + row["blocks_mlp_params"]
        + row["blocks_norm_params"]
    )
    return row


def _estimate_vit_parameter_groups(
    *,
    backbone: str,
    spec: ViTArchitectureSpec,
    crop_size: int,
    patch_size: int,
    frames_per_clip: int,
    tubelet_size: int,
) -> dict[str, int]:
    hidden_dim = int(spec.embed_dim * spec.mlp_ratio)
    qkv_bias_params = _qkv_bias_parameters(backbone, spec.embed_dim)
    counts = {
        "patch_embed": _conv3d_parameters(
            in_channels=3,
            out_channels=spec.embed_dim,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
        ),
        "blocks_attention": spec.depth
        * (4 * spec.embed_dim * spec.embed_dim + qkv_bias_params + spec.embed_dim),
        "blocks_mlp": spec.depth
        * (2 * spec.embed_dim * hidden_dim + hidden_dim + spec.embed_dim),
        "blocks_norm": spec.depth * (4 * spec.embed_dim),
    }

    if backbone == "jepa_v1":
        counts["position_embedding"] = _num_video_tokens(
            crop_size=crop_size,
            patch_size=patch_size,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
        ) * spec.embed_dim
        counts["final_norm"] = 2 * spec.embed_dim
    elif backbone == "jepa_v2":
        counts["final_norm"] = 2 * spec.embed_dim
    elif backbone == "jepa_v2_1":
        counts["patch_embed_img"] = _conv3d_parameters(
            in_channels=3,
            out_channels=spec.embed_dim,
            tubelet_size=1,
            patch_size=patch_size,
        )
        counts["hierarchical_norms"] = 4 * 2 * spec.embed_dim
        counts["modality_embedding"] = 2 * spec.embed_dim
    elif backbone in {"videomae", "videomae_v2"}:
        counts["final_norm"] = 2 * spec.embed_dim
    else:
        raise ValueError(f"Unsupported ViT backbone '{backbone}'.")

    return counts


def _qkv_bias_parameters(backbone: str, embed_dim: int) -> int:
    if backbone in {"videomae", "videomae_v2"}:
        # HF VideoMAE-style attention uses q/v bias only; key bias is fixed zero.
        return 2 * embed_dim
    return 3 * embed_dim


def _conv3d_parameters(
    *,
    in_channels: int,
    out_channels: int,
    tubelet_size: int,
    patch_size: int,
) -> int:
    return out_channels * in_channels * tubelet_size * patch_size * patch_size + out_channels


def _num_video_tokens(
    *,
    crop_size: int,
    patch_size: int,
    frames_per_clip: int,
    tubelet_size: int,
) -> int:
    return (frames_per_clip // tubelet_size) * (crop_size // patch_size) * (crop_size // patch_size)


def _parameter_group_name(name: str, *, group_depth: int) -> str:
    pieces = [piece for piece in str(name).split(".") if piece]
    if not pieces:
        return "<root>"
    return ".".join(pieces[:group_depth])


def _add_group_count(
    existing: ParameterGroupCount | None,
    *,
    group: str,
    count: int,
    trainable: bool,
) -> ParameterGroupCount:
    if existing is None:
        return ParameterGroupCount(
            group=group,
            total=count,
            trainable=count if trainable else 0,
            frozen=0 if trainable else count,
            tensors=1,
        )
    return ParameterGroupCount(
        group=existing.group,
        total=existing.total + count,
        trainable=existing.trainable + (count if trainable else 0),
        frozen=existing.frozen + (0 if trainable else count),
        tensors=existing.tensors + 1,
    )


def _resolve_adapter_module(adapter: Any) -> Any:
    attr_name = _resolve_adapter_module_name(adapter)
    return getattr(adapter, attr_name)


def _resolve_adapter_module_name(adapter: Any) -> str:
    for attr_name in ("_encoder", "_model", "_transformer"):
        if hasattr(adapter, attr_name):
            return attr_name
    raise TypeError("Could not locate a countable module on adapter.")


def _load_backbones_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Backbone config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Backbone config must be a mapping, got {type(payload)!r}.")
    return payload


def _vit_size_label(model_name: str) -> str:
    labels = {
        "vit_base": "ViT-B",
        "vit_large": "ViT-L",
        "vit_huge": "ViT-H",
        "vit_giant": "ViT-g",
        "vit_giant_xformers": "ViT-g",
        "vit_gigantic": "ViT-G",
        "vit_gigantic_xformers": "ViT-G",
    }
    return labels.get(model_name, model_name)


def _select_largest_variant(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    largest_depth_embed = max(
        (int(row.get("depth", 0)), int(row.get("embed_dim", 0)))
        for row in rows
    )
    candidates = [
        row
        for row in rows
        if (int(row.get("depth", 0)), int(row.get("embed_dim", 0))) == largest_depth_embed
    ]
    return max(
        candidates,
        key=lambda row: (
            bool(row.get("is_default_variant", False)),
            int(row.get("total_parameters", 0)),
            str(row.get("variant", "")),
        ),
    )


def _formula_label(backbone: str) -> str:
    if backbone == "jepa_v1":
        return "upstream_vit_with_frozen_pos_parameter"
    if backbone == "jepa_v2":
        return "upstream_vit_rope"
    if backbone == "jepa_v2_1":
        return "upstream_vit_rope_hierarchical"
    if backbone in {"videomae", "videomae_v2"}:
        return "videomae_encoder_formula_qv_bias"
    return "unknown"


def format_parameter_count(value: int | float) -> str:
    """Format a raw parameter count as a compact million-count string."""

    return f"{float(value) / 1_000_000:.2f}M"


__all__ = [
    "DEFAULT_BACKBONES_CONFIG_PATH",
    "VIT_ARCHITECTURES",
    "VIT_BACKBONE_KEYS",
    "ParameterGroupCount",
    "ViTArchitectureSpec",
    "build_vit_parameter_table",
    "count_adapter_parameters",
    "count_module_parameters",
    "format_parameter_count",
    "select_vit_size_comparison_rows",
]
