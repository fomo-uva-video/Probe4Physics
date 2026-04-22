from __future__ import annotations

"""
Thin wrapper adapters around HuggingFace VideoMAE v1 and VideoMAEv2.

VideoMAE v1: MCG-NJU/videomae-* (via transformers.VideoMAEModel)
VideoMAEv2:  OpenGVLab/VideoMAEv2-* (via AutoModel, trust_remote_code=True)

Both adapters follow the same BackboneFeatures contract established by
jepa_v1_adapter.py. No submodule or sys.path manipulation is needed here
because both models load entirely from HuggingFace hub.
"""

import hashlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import yaml

import torch

from .registry import (
    BackboneFeatures,
    VideoBackboneAdapter,
    register_adapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"
logger = logging.getLogger(__name__)


def _torch_version_tuple() -> tuple[int, int]:
    raw = str(torch.__version__).split("+", 1)[0]
    parts = raw.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor_text = parts[1] if len(parts) > 1 else "0"
    minor_digits = "".join(ch for ch in minor_text if ch.isdigit())
    minor = int(minor_digits) if minor_digits else 0
    return major, minor


def _resolve_auto_model_reference(auto_map: Any) -> str:
    if isinstance(auto_map, dict):
        value = auto_map.get("AutoModel")
    elif isinstance(auto_map, (list, tuple)):
        value = auto_map[0] if auto_map else ""
    else:
        value = ""

    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    return str(value or "").strip()


def _prefer_safetensors_kwargs(hf_cache_dir: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_safetensors": True}
    if hf_cache_dir is not None:
        kwargs["cache_dir"] = hf_cache_dir
    return kwargs


def _normalize_auto_map(config: Any) -> str:
    auto_map = getattr(config, "auto_map", {}) or {}
    class_reference = _resolve_auto_model_reference(auto_map)
    if class_reference and not isinstance(auto_map, dict):
        try:
            config.auto_map = {"AutoModel": class_reference}
        except Exception:
            pass
    return class_reference


def _resolve_cached_snapshot_dir(hf_model_id: str, hf_cache_dir: str | None) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    cached = try_to_load_from_cache(
        repo_id=hf_model_id,
        filename="config.json",
        cache_dir=hf_cache_dir,
    )
    if not isinstance(cached, str):
        return None

    path = Path(cached)
    if not path.exists():
        return None
    return path.parent


def _load_remote_code_modules(snapshot_dir: Path) -> tuple[Any, Any]:
    package_name = f"videomaev2_cache_{hashlib.sha1(str(snapshot_dir).encode('utf-8')).hexdigest()[:12]}"
    existing_cfg = sys.modules.get(f"{package_name}.modeling_config")
    existing_model = sys.modules.get(f"{package_name}.modeling_videomaev2")
    if existing_cfg is not None and existing_model is not None:
        return existing_cfg, existing_model

    package_module = types.ModuleType(package_name)
    package_module.__path__ = [str(snapshot_dir)]
    sys.modules[package_name] = package_module

    loaded: dict[str, Any] = {}
    for module_name in ("modeling_config", "modeling_videomaev2"):
        module_path = snapshot_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{module_name}"] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module

    return loaded["modeling_config"], loaded["modeling_videomaev2"]


def _sample_meta_tensor_names(
    model: torch.nn.Module,
    *,
    kind: str,
    limit: int = 12,
) -> list[str]:
    if kind == "parameter":
        iterator = model.named_parameters()
    elif kind == "buffer":
        iterator = model.named_buffers()
    else:
        raise ValueError(f"Unsupported tensor kind: {kind}")

    names: list[str] = []
    for name, tensor in iterator:
        if getattr(tensor, "is_meta", False):
            names.append(str(name))
            if len(names) >= limit:
                break
    return names


def _count_meta_tensors(model: torch.nn.Module, *, kind: str) -> int:
    if kind == "parameter":
        iterator = model.parameters()
    elif kind == "buffer":
        iterator = model.buffers()
    else:
        raise ValueError(f"Unsupported tensor kind: {kind}")

    return sum(1 for tensor in iterator if getattr(tensor, "is_meta", False))


def _iter_module_tensor_attributes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module, str, torch.Tensor]]:
    found: list[tuple[str, torch.nn.Module, str, torch.Tensor]] = []
    for module_name, module in model.named_modules():
        prefix = module_name.strip()
        for attr_name, value in vars(module).items():
            if not isinstance(value, torch.Tensor):
                continue
            if attr_name in module._parameters or attr_name in module._buffers:
                continue
            fq_name = f"{prefix}.{attr_name}" if prefix else attr_name
            found.append((fq_name, module, attr_name, value))
    return found


def _sample_meta_attribute_names(
    model: torch.nn.Module,
    *,
    limit: int = 12,
) -> list[str]:
    names: list[str] = []
    for fq_name, _, _, tensor in _iter_module_tensor_attributes(model):
        if getattr(tensor, "is_meta", False):
            names.append(fq_name)
            if len(names) >= limit:
                break
    return names


def _count_meta_tensor_attributes(model: torch.nn.Module) -> int:
    return sum(
        1
        for _, _, _, tensor in _iter_module_tensor_attributes(model)
        if getattr(tensor, "is_meta", False)
    )


def _format_load_context(context: dict[str, Any] | None) -> str:
    if not context:
        return "load_context=<none>"
    parts = [f"{key}={value}" for key, value in sorted(context.items())]
    return " ".join(parts)


def _build_meta_tensor_debug_message(
    *,
    adapter_name: str,
    hf_model_id: str,
    variant: str,
    device: torch.device,
    load_context: dict[str, Any] | None,
    model: torch.nn.Module,
    stage: str,
) -> str:
    meta_param_count = _count_meta_tensors(model, kind="parameter")
    meta_buffer_count = _count_meta_tensors(model, kind="buffer")
    meta_attr_count = _count_meta_tensor_attributes(model)
    meta_param_names = _sample_meta_tensor_names(model, kind="parameter")
    meta_buffer_names = _sample_meta_tensor_names(model, kind="buffer")
    meta_attr_names = _sample_meta_attribute_names(model)
    return (
        f"{adapter_name} failed during {stage}: model still contains meta tensors. "
        f"hf_model_id={hf_model_id} variant={variant} device={device} "
        f"{_format_load_context(load_context)} "
        f"meta_parameter_count={meta_param_count} "
        f"meta_buffer_count={meta_buffer_count} "
        f"meta_attribute_count={meta_attr_count} "
        f"meta_parameter_examples={meta_param_names} "
        f"meta_buffer_examples={meta_buffer_names} "
        f"meta_attribute_examples={meta_attr_names}"
    )


def _materialize_videomaev2_plain_tensor_attrs(
    model: torch.nn.Module,
    *,
    config: Any,
    model_module: Any,
) -> list[str]:
    repaired: list[str] = []
    if model_module is None or not hasattr(model_module, "get_sinusoid_encoding_table"):
        return repaired

    inner = getattr(model, "model", None)
    if inner is None:
        return repaired

    pos_embed = getattr(inner, "pos_embed", None)
    if isinstance(pos_embed, torch.Tensor) and getattr(pos_embed, "is_meta", False):
        model_cfg = getattr(config, "model_config", {}) or {}
        num_patches = int(getattr(getattr(inner, "patch_embed", None), "num_patches", 0))
        embed_dim = int(getattr(inner, "embed_dim", 0))
        if num_patches <= 0 or embed_dim <= 0:
            raise RuntimeError(
                "Could not materialize VideoMAEv2 pos_embed from config. "
                f"num_patches={num_patches}, embed_dim={embed_dim}"
            )

        rebuilt = model_module.get_sinusoid_encoding_table(num_patches, embed_dim)
        setattr(inner, "pos_embed", rebuilt)
        repaired.append("model.pos_embed")

        logger.warning(
            "Materialized VideoMAEv2 plain tensor attribute from helper: %s "
            "(num_patches=%s embed_dim=%s model_config_num_frames=%s)",
            "model.pos_embed",
            num_patches,
            embed_dim,
            model_cfg.get("num_frames"),
        )

    return repaired


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_videomae_config(
    backbone_key: str,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> dict[str, Any]:
    """Load a VideoMAE config section from `configs/backbones.yaml`.

    Args:
        backbone_key: Either ``"videomae"`` or ``"videomae_v2"``.
        config_path: Path to the global backbones YAML config.

    Returns:
        The config dict for the requested backbone section.
    """

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone config file not found: {path}. "
            "Expected global config at configs/backbones.yaml."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    cfg = payload.get(backbone_key)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"configs/backbones.yaml must define a '{backbone_key}' object."
        )

    return cfg


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------

def _resolve_videomae_variant(
    cfg: dict[str, Any],
    *,
    variant: str | None,
    hf_cache_dir: str | Path | None,
) -> tuple[str, dict[str, Any], str]:
    """Resolve the variant name, variant config dict, and HuggingFace model ID.

    Args:
        cfg: Top-level backbone config (e.g. cfg["videomae"]).
        variant: Optional explicit variant name. Falls back to
            ``cfg["default_variant"]``.
        hf_cache_dir: Ignored here; returned for caller convenience.

    Returns:
        ``(chosen_variant, variant_cfg, hf_model_id)``
    """

    variants = cfg.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("Backbone config variants must be a non-empty mapping.")

    chosen = str(variant).strip() if variant is not None else ""
    if not chosen:
        chosen = str(cfg.get("default_variant", "")).strip()
    if not chosen:
        raise ValueError("default_variant is required for zero-config adapter creation.")

    if chosen not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown variant '{chosen}'. Known variants: {known}"
        )

    variant_cfg = variants[chosen]
    if not isinstance(variant_cfg, dict):
        raise ValueError(f"Variant config for '{chosen}' must be a mapping.")

    hf_model_id = str(variant_cfg.get("hf_model_id", "")).strip()
    if not hf_model_id:
        raise ValueError(
            f"Variant '{chosen}' is missing hf_model_id in backbones.yaml."
        )

    return chosen, variant_cfg, hf_model_id


# ---------------------------------------------------------------------------
# Shared layer resolution (same math as jepa_v1_adapter)
# ---------------------------------------------------------------------------

def resolve_relative_depth_layers(
    model_name: str,
    relative_depths: Sequence[float] | None = None,
    *,
    model_block_depths: dict[str, int] | None = None,
    backbone_key: str = "videomae",
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    """Map relative layer depths (e.g. 0.25, 0.5 …) to 1-based block ids.

    Args:
        model_name: Architecture key, e.g. ``"vit_base"``.
        relative_depths: Fractions in ``(0, 1]``. Defaults to
            ``cfg.default_relative_depths``.
        model_block_depths: Override mapping ``{model_name: n_blocks}``.
            Defaults to ``cfg.model_block_depths``.
        backbone_key: Config section to read when defaults are needed.
        config_path: Path to the global backbones YAML.

    Returns:
        Tuple of 1-based block ids in ascending order, deduplicated.
    """

    if relative_depths is None or model_block_depths is None:
        cfg = _load_videomae_config(backbone_key, config_path)

        raw_relative_depths = cfg.get("default_relative_depths")
        if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
            raise ValueError(
                f"{backbone_key}.default_relative_depths must be a non-empty list."
            )
        cfg_relative_depths = tuple(float(v) for v in raw_relative_depths)

        raw_model_depths = cfg.get("model_block_depths")
        if not isinstance(raw_model_depths, dict) or not raw_model_depths:
            raise ValueError(
                f"{backbone_key}.model_block_depths must be a non-empty mapping."
            )
        cfg_model_block_depths = {str(k): int(v) for k, v in raw_model_depths.items()}

        if relative_depths is None:
            relative_depths = cfg_relative_depths
        if model_block_depths is None:
            model_block_depths = cfg_model_block_depths

    if model_name not in model_block_depths:
        known = ", ".join(sorted(model_block_depths))
        raise ValueError(
            f"Unsupported model_name='{model_name}'. Known: {known}"
        )

    if not relative_depths:
        raise ValueError("relative_depths cannot be empty.")

    depth = model_block_depths[model_name]
    resolved: list[int] = []
    for value in relative_depths:
        current = float(value)
        if not (0.0 < current <= 1.0):
            raise ValueError(
                f"Invalid relative depth: {value}. Expected values in (0, 1]."
            )
        block_id = int(round(depth * current))
        block_id = max(1, min(depth, block_id))
        if block_id not in resolved:
            resolved.append(block_id)

    return tuple(resolved)


# ---------------------------------------------------------------------------
# Shared base adapter
# ---------------------------------------------------------------------------

class _VideoMAEBaseAdapter(VideoBackboneAdapter):
    """Shared init + extract logic for VideoMAE v1 and v2.

    Concrete subclasses must implement:
      - ``_load_hf_model(hf_model_id, hf_cache_dir)``  → ``torch.nn.Module``
      - ``_forward(clips)``  → ``list[torch.Tensor]``
        Each tensor in the list has shape ``[B, N, D]`` and corresponds to
        ``self.selected_layers`` in order.
    """

    # Sub-classes set this to identify themselves in backbones.yaml.
    _BACKBONE_KEY: str = "videomae"

    def __init__(
        self,
        *,
        variant: str | None = None,
        hf_cache_dir: str | Path | None = None,
        model_name: str | None = None,
        relative_depths: Sequence[float] | None = None,
        config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
        device: str | torch.device = "cpu",
        crop_size: int | None = None,
        patch_size: int | None = None,
        frames_per_clip: int | None = None,
        tubelet_size: int | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        self._load_debug_context: dict[str, Any] = {}

        cfg = _load_videomae_config(self._BACKBONE_KEY, self.config_path)
        self.variant, variant_cfg, self.hf_model_id = _resolve_videomae_variant(
            cfg, variant=variant, hf_cache_dir=hf_cache_dir
        )

        self.model_name = str(model_name or variant_cfg.get("model_name", "vit_base"))
        self.crop_size = int(
            crop_size if crop_size is not None else variant_cfg.get("crop_size", 224)
        )
        self.patch_size = int(
            patch_size if patch_size is not None else variant_cfg.get("patch_size", 16)
        )
        self.frames_per_clip = int(
            frames_per_clip
            if frames_per_clip is not None
            else variant_cfg.get("frames_per_clip", 16)
        )
        self.tubelet_size = int(
            tubelet_size
            if tubelet_size is not None
            else variant_cfg.get("tubelet_size", 2)
        )

        raw_relative_depths = cfg.get("default_relative_depths")
        if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
            raise ValueError(
                f"{self._BACKBONE_KEY}.default_relative_depths must be a non-empty list."
            )
        config_relative_depths = tuple(float(v) for v in raw_relative_depths)

        raw_model_depths = cfg.get("model_block_depths")
        if not isinstance(raw_model_depths, dict) or not raw_model_depths:
            raise ValueError(
                f"{self._BACKBONE_KEY}.model_block_depths must be a non-empty mapping."
            )
        model_block_depths = {str(k): int(v) for k, v in raw_model_depths.items()}

        if relative_depths is None:
            relative_depths = config_relative_depths

        # User-visible layer ids are 1-based to match transformer block numbering.
        self.selected_layers = resolve_relative_depth_layers(
            self.model_name,
            relative_depths,
            model_block_depths=model_block_depths,
            backbone_key=self._BACKBONE_KEY,
            config_path=self.config_path,
        )

        # Load and freeze the HuggingFace model (implemented by subclass).
        try:
            self._model = self._load_hf_model(self.hf_model_id, self.hf_cache_dir)
        except Exception as exc:
            context = _format_load_context(self._load_debug_context)
            raise RuntimeError(
                f"{self.__class__.__name__} failed to load HuggingFace model. "
                f"hf_model_id={self.hf_model_id} variant={self.variant} device={self.device} "
                f"{context}"
            ) from exc

        if any(getattr(param, "is_meta", False) for param in self._model.parameters()) or any(
            getattr(buf, "is_meta", False) for buf in self._model.buffers()
        ):
            raise RuntimeError(
                _build_meta_tensor_debug_message(
                    adapter_name=self.__class__.__name__,
                    hf_model_id=self.hf_model_id,
                    variant=self.variant,
                    device=self.device,
                    load_context=self._load_debug_context,
                    model=self._model,
                    stage="post_load_pre_to_device",
                )
            )

        try:
            self._model.to(self.device)
        except Exception as exc:
            message = (
                _build_meta_tensor_debug_message(
                    adapter_name=self.__class__.__name__,
                    hf_model_id=self.hf_model_id,
                    variant=self.variant,
                    device=self.device,
                    load_context=self._load_debug_context,
                    model=self._model,
                    stage="model.to(device)",
                )
                if "meta tensor" in str(exc).lower()
                else (
                    f"{self.__class__.__name__} failed during model.to(device). "
                    f"hf_model_id={self.hf_model_id} variant={self.variant} "
                    f"device={self.device} {_format_load_context(self._load_debug_context)}"
                )
            )
            raise RuntimeError(message) from exc
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def _load_hf_model(
        self, hf_model_id: str, hf_cache_dir: str | None
    ) -> torch.nn.Module:
        raise NotImplementedError

    def _forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        """Run the model and return hidden states for selected layers.

        Args:
            clips: Preprocessed clips, shape ``[B, C, T, H, W]``, already on
                ``self.device``.

        Returns:
            List of token tensors ``[B, N, D]``, one per layer in
            ``self.selected_layers`` order.
        """
        raise NotImplementedError

    def _resolve_requested_layers(
        self, layer_ids: Sequence[int] | None
    ) -> tuple[int, ...]:
        """Validate optional layer subset against configured layers."""

        if layer_ids is None:
            return self.selected_layers

        requested = tuple(int(layer) for layer in layer_ids)
        if not requested:
            raise ValueError("layer_ids cannot be empty when provided.")

        unknown = sorted(set(requested) - set(self.selected_layers))
        if unknown:
            raise ValueError(
                f"Requested layer_ids {unknown} are not available. "
                f"Available: {self.selected_layers}"
            )
        return requested

    def extract(
        self,
        clips: torch.Tensor,
        layer_ids: Sequence[int] | None = None,
    ) -> BackboneFeatures:
        """Run frozen forward pass and return token + pooled features by layer.

        Args:
            clips: Preprocessed clip tensor, shape ``[B, C, T, H, W]``.
            layer_ids: Optional subset of ``self.selected_layers`` to return.

        Returns:
            :class:`BackboneFeatures` with ``tokens_by_layer``,
            ``pooled_by_layer``, ``selected_layers``, and ``metadata``.
        """

        if not isinstance(clips, torch.Tensor):
            raise TypeError(
                f"clips must be a torch.Tensor, got {type(clips)!r}"
            )
        if clips.ndim != 5:
            raise ValueError(
                f"Expected clips shape [B, C, T, H, W], got {tuple(clips.shape)}"
            )

        requested_layers = self._resolve_requested_layers(layer_ids)
        clips = clips.to(self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                all_layer_tokens = self._forward(clips)
        except Exception as exc:
            message = (
                _build_meta_tensor_debug_message(
                    adapter_name=self.__class__.__name__,
                    hf_model_id=self.hf_model_id,
                    variant=self.variant,
                    device=self.device,
                    load_context=self._load_debug_context,
                    model=self._model,
                    stage="forward",
                )
                if "meta tensor" in str(exc).lower()
                else (
                    f"{self.__class__.__name__} forward failed. "
                    f"hf_model_id={self.hf_model_id} variant={self.variant} "
                    f"device={self.device} requested_layers={requested_layers} "
                    f"{_format_load_context(self._load_debug_context)}"
                )
            )
            raise RuntimeError(message) from exc

        if len(all_layer_tokens) != len(self.selected_layers):
            raise RuntimeError(
                "Model returned unexpected number of layer outputs: "
                f"{len(all_layer_tokens)} vs expected {len(self.selected_layers)}."
            )

        all_tokens: dict[int, torch.Tensor] = {
            layer: tokens
            for layer, tokens in zip(self.selected_layers, all_layer_tokens)
        }

        # Optional filtering: let probes request a subset without re-forwarding.
        tokens_by_layer = {layer: all_tokens[layer] for layer in requested_layers}
        # Mean pooling on token dimension creates per-clip features [B, D].
        pooled_by_layer = {
            layer: token_values.mean(dim=1)
            for layer, token_values in tokens_by_layer.items()
        }

        metadata = {
            "hf_model_id": self.hf_model_id,
            "config_path": str(self.config_path),
            "variant": self.variant,
            "model_name": self.model_name,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
        }
        return BackboneFeatures(
            tokens_by_layer=tokens_by_layer,
            pooled_by_layer=pooled_by_layer,
            selected_layers=requested_layers,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# VideoMAE v1 concrete adapter
# ---------------------------------------------------------------------------

class VideoMAEAdapter(_VideoMAEBaseAdapter):
    """Frozen-feature extractor for VideoMAE v1 (MCG-NJU/videomae-*).

    Uses ``transformers.VideoMAEModel``. The HuggingFace model expects input
    shape ``[B, T, C, H, W]``, so the adapter permutes from the project
    contract ``[B, C, T, H, W]`` before forwarding.
    """

    _BACKBONE_KEY = "videomae"

    def _load_hf_model(
        self, hf_model_id: str, hf_cache_dir: str | None
    ) -> torch.nn.Module:
        self._load_debug_context = {
            "backbone_key": self._BACKBONE_KEY,
            "hf_model_id": hf_model_id,
            "hf_cache_dir": hf_cache_dir or "<default>",
            "load_strategy": "videomae_v1_from_pretrained",
        }
        try:
            from transformers import VideoMAEModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for VideoMAE v1. "
                "Install with: python -m pip install transformers"
            ) from exc

        kwargs = _prefer_safetensors_kwargs(hf_cache_dir)
        try:
            return VideoMAEModel.from_pretrained(hf_model_id, **kwargs)
        except Exception as exc:
            if "serious vulnerability issue in `torch.load`" not in str(exc):
                raise

            major, minor = _torch_version_tuple()
            raise RuntimeError(
                "VideoMAE requires either safetensors weights on the Hub or torch>=2.6 "
                f"for secure checkpoint loading. Current torch={major}.{minor}."
            ) from exc

    def _forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        # HuggingFace VideoMAEModel expects [B, T, C, H, W].
        pixel_values = clips.permute(0, 2, 1, 3, 4)

        outputs = self._model(pixel_values, output_hidden_states=True)

        # hidden_states is a tuple of length (num_blocks + 1):
        #   index 0 = patch embedding output
        #   indices 1..num_blocks = transformer block outputs (1-based)
        hidden_states = outputs.hidden_states

        # selected_layers contains 1-based block ids → index directly.
        return [hidden_states[layer_id] for layer_id in self.selected_layers]


# ---------------------------------------------------------------------------
# VideoMAEv2 concrete adapter
# ---------------------------------------------------------------------------

class VideoMAEV2Adapter(_VideoMAEBaseAdapter):
    """Frozen-feature extractor for VideoMAEv2 (OpenGVLab/VideoMAEv2-*).

    Uses ``AutoModel`` with ``trust_remote_code=True`` because VideoMAEv2
    registers a custom architecture on the HuggingFace hub. The model
    already expects ``[B, C, T, H, W]``, matching the project contract,
    so no permutation is required.
    """

    _BACKBONE_KEY = "videomae_v2"

    def _load_hf_model(
        self, hf_model_id: str, hf_cache_dir: str | None
    ) -> torch.nn.Module:
        self._load_debug_context = {
            "backbone_key": self._BACKBONE_KEY,
            "hf_model_id": hf_model_id,
            "hf_cache_dir": hf_cache_dir or "<default>",
            "load_strategy": "videomae_v2_init",
        }
        try:
            from transformers import AutoConfig, AutoModel
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError as exc:
            raise ImportError(
                "transformers is required for VideoMAEv2. "
                "Install with: python -m pip install transformers"
            ) from exc

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            **_prefer_safetensors_kwargs(hf_cache_dir),
        }

        snapshot_dir = _resolve_cached_snapshot_dir(hf_model_id, hf_cache_dir)
        if snapshot_dir is not None:
            self._load_debug_context = {
                **self._load_debug_context,
                "load_strategy": "cached_snapshot_remote_code",
                "snapshot_dir": str(snapshot_dir),
            }
            logger.info(
                "Loading VideoMAEv2 from cached snapshot: hf_model_id=%s snapshot_dir=%s",
                hf_model_id,
                str(snapshot_dir),
            )
            cfg_module, model_module = _load_remote_code_modules(snapshot_dir)
            config = cfg_module.VideoMAEv2Config.from_pretrained(
                str(snapshot_dir),
                local_files_only=True,
            )
            model_cls = model_module.VideoMAEv2
            if not hasattr(model_cls, "_tied_weights_keys"):
                model_cls._tied_weights_keys = []
            if not hasattr(model_cls, "all_tied_weights_keys"):
                model_cls.all_tied_weights_keys = {}
            model = model_cls.from_pretrained(
                str(snapshot_dir),
                config=config,
                local_files_only=True,
                use_safetensors=True,
            )
            repaired = _materialize_videomaev2_plain_tensor_attrs(
                model,
                config=config,
                model_module=model_module,
            )
            if repaired:
                self._load_debug_context = {
                    **self._load_debug_context,
                    "materialized_plain_attrs": ",".join(repaired),
                }
            return model

        config = AutoConfig.from_pretrained(hf_model_id, **kwargs)
        class_reference = _normalize_auto_map(config)
        self._load_debug_context = {
            **self._load_debug_context,
            "load_strategy": "automodel_from_pretrained",
            "auto_model_reference": class_reference or "<missing>",
        }
        try:
            model = AutoModel.from_pretrained(hf_model_id, config=config, **kwargs)
            repaired = _materialize_videomaev2_plain_tensor_attrs(
                model,
                config=config,
                model_module=sys.modules.get(getattr(model.__class__, "__module__", ""), None),
            )
            if repaired:
                self._load_debug_context = {
                    **self._load_debug_context,
                    "materialized_plain_attrs": ",".join(repaired),
                }
            return model
        except Exception as exc:
            # Some recent transformers versions expect custom remote-code models
            # to expose tied-weight metadata attributes that older hub repos omit.
            text = str(exc)
            if (
                "all_tied_weights_keys" not in text
                and "'list' object has no attribute 'keys'" not in text
            ):
                raise

            if not class_reference:
                raise RuntimeError(
                    "VideoMAEv2 config is missing auto_map['AutoModel']; "
                    "cannot load the custom remote-code model fallback."
                ) from exc

            model_cls = get_class_from_dynamic_module(
                class_reference,
                hf_model_id,
                **kwargs,
            )
            self._load_debug_context = {
                **self._load_debug_context,
                "load_strategy": "dynamic_module_fallback",
                "dynamic_model_class": getattr(model_cls, "__name__", "<unknown>"),
            }
            if not hasattr(model_cls, "_tied_weights_keys"):
                model_cls._tied_weights_keys = []
            if not hasattr(model_cls, "all_tied_weights_keys"):
                model_cls.all_tied_weights_keys = {}
            model = model_cls.from_pretrained(hf_model_id, config=config, **kwargs)
            model_module = sys.modules.get(getattr(model_cls, "__module__", ""), None)
            repaired = _materialize_videomaev2_plain_tensor_attrs(
                model,
                config=config,
                model_module=model_module,
            )
            if repaired:
                self._load_debug_context = {
                    **self._load_debug_context,
                    "materialized_plain_attrs": ",".join(repaired),
                }
            return model

    def _get_blocks(self) -> torch.nn.ModuleList:
        """Navigate to the transformer block list inside the HF wrapper."""

        # HF wraps the custom model: self._model.model.blocks
        inner = getattr(self._model, "model", self._model)
        blocks = getattr(inner, "blocks", None)
        if blocks is None or not isinstance(blocks, torch.nn.ModuleList):
            raise RuntimeError(
                "Could not locate transformer blocks in VideoMAEv2 model. "
                "Expected attribute path: model.model.blocks"
            )
        return blocks

    def _forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        # VideoMAEv2 expects [B, C, T, H, W] — no permute needed.
        # The custom forward() does not support output_hidden_states, so we
        # capture intermediate outputs via registered forward hooks instead.
        blocks = self._get_blocks()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        for layer_id in self.selected_layers:
            block = blocks[layer_id - 1]  # selected_layers are 1-based

            def _make_hook(lid: int):
                def _hook(
                    module: torch.nn.Module,
                    input: tuple,
                    output: torch.Tensor | tuple,
                ) -> None:
                    # Some block implementations return (tensor, ...) tuples.
                    captured[lid] = output[0] if isinstance(output, tuple) else output

                return _hook

            handles.append(block.register_forward_hook(_make_hook(layer_id)))

        try:
            self._model(clips)
        finally:
            for handle in handles:
                handle.remove()

        return [captured[layer_id] for layer_id in self.selected_layers]


# ---------------------------------------------------------------------------
# Factory functions + registry
# ---------------------------------------------------------------------------

def create_videomae_adapter(**kwargs: Any) -> VideoMAEAdapter:
    """Factory function used by the central adapter registry."""

    return VideoMAEAdapter(**kwargs)


def create_videomae_v2_adapter(**kwargs: Any) -> VideoMAEV2Adapter:
    """Factory function used by the central adapter registry."""

    return VideoMAEV2Adapter(**kwargs)


# Register at import-time so callers can use `create_adapter("videomae", ...)`.
register_adapter("videomae", create_videomae_adapter, replace=True)
register_adapter("videomae_v2", create_videomae_v2_adapter, replace=True)
