from __future__ import annotations

"""
Thin wrapper adapters around HuggingFace VideoMAE v1 and VideoMAEv2.

VideoMAE v1: MCG-NJU/videomae-* (via transformers.VideoMAEModel)
VideoMAEv2:  OpenGVLab/VideoMAEv2-* (via AutoModel, trust_remote_code=True)

Both adapters follow the same BackboneFeatures contract established by
jepa_v1_adapter.py. No submodule or sys.path manipulation is needed here
because both models load entirely from HuggingFace hub.
"""

import yaml
from pathlib import Path
from typing import Any, Sequence

import torch

from .registry import (
    BackboneFeatures,
    VideoBackboneAdapter,
    register_adapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"


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
        self._model = self._load_hf_model(self.hf_model_id, self.hf_cache_dir)
        self._model.to(self.device)
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

        with torch.no_grad():
            all_layer_tokens = self._forward(clips)

        if len(all_layer_tokens) != len(self.selected_layers):
            raise RuntimeError(
                "Model returned unexpected number of layer outputs: "
                f"{len(all_layer_tokens)} vs expected {len(self.selected_layers)}."
            )

        all_tokens: dict[int, torch.Tensor] = {
            layer: tokens
            for layer, tokens in zip(self.selected_layers, all_layer_tokens, strict=True)
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
        try:
            from transformers import VideoMAEModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for VideoMAE v1. "
                "Install with: python -m pip install transformers"
            ) from exc

        kwargs: dict[str, Any] = {}
        if hf_cache_dir is not None:
            kwargs["cache_dir"] = hf_cache_dir

        return VideoMAEModel.from_pretrained(hf_model_id, **kwargs)

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
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for VideoMAEv2. "
                "Install with: python -m pip install transformers"
            ) from exc

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_cache_dir is not None:
            kwargs["cache_dir"] = hf_cache_dir

        config = AutoConfig.from_pretrained(hf_model_id, **kwargs)
        return AutoModel.from_pretrained(hf_model_id, config=config, **kwargs)

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
