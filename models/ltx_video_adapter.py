from __future__ import annotations

"""
LTX-Video adapter using deterministic VAE-encoder stage activations.

This adapter intentionally extracts features from the LTX VAE encoder rather than
running diffusion denoising steps, so extraction stays deterministic and aligns
with this repository's frozen-feature probing pipeline.
"""

from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .registry import BackboneFeatures, VideoBackboneAdapter, register_adapter
from .preprocessing import ltx_preprocessing_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"


def _load_ltx_video_config(
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the ``ltx_video`` section from ``configs/backbones.yaml``."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone config file not found: {path}. "
            "Expected global config at configs/backbones.yaml."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    cfg = payload.get("ltx_video")
    if not isinstance(cfg, dict):
        raise ValueError("configs/backbones.yaml must define an 'ltx_video' object.")

    return cfg


def _resolve_variant_bundle(
    cfg: dict[str, Any],
    *,
    variant: str | None,
) -> tuple[str, dict[str, Any], str]:
    """Resolve preset variant and Hugging Face model id."""

    variants = cfg.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("ltx_video.variants must be a non-empty mapping.")

    chosen = str(variant).strip() if variant is not None else ""
    if not chosen:
        chosen = str(cfg.get("default_variant", "")).strip()
    if not chosen:
        raise ValueError("ltx_video.default_variant is required for zero-config adapter creation.")

    if chosen not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(f"Unknown ltx_video variant '{chosen}'. Known variants: {known}")

    variant_cfg = variants[chosen]
    if not isinstance(variant_cfg, dict):
        raise ValueError(f"Variant config for '{chosen}' must be a mapping.")

    hf_model_id = str(variant_cfg.get("hf_model_id", "")).strip()
    if not hf_model_id:
        raise ValueError(
            f"ltx_video variant '{chosen}' is missing hf_model_id in backbones.yaml."
        )

    return chosen, variant_cfg, hf_model_id


def _resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    name = str(dtype_name or "float32").strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported torch_dtype '{dtype_name}'. Known values: {known}")
    return mapping[name]


def resolve_relative_depth_layers(
    model_name: str,
    relative_depths: Sequence[float] | None = None,
    *,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    """Map relative probe depths (e.g. 0.25, 0.5, ...) to 1-based stage ids."""

    if relative_depths is None or model_block_depths is None:
        cfg = _load_ltx_video_config(config_path)

        if relative_depths is None:
            raw_rel = cfg.get("default_relative_depths")
            if not isinstance(raw_rel, list) or not raw_rel:
                raise ValueError("ltx_video.default_relative_depths must be a non-empty list.")
            relative_depths = tuple(float(v) for v in raw_rel)

        if model_block_depths is None:
            raw_depths = cfg.get("model_block_depths")
            if not isinstance(raw_depths, dict) or not raw_depths:
                raise ValueError("ltx_video.model_block_depths must be a non-empty mapping.")
            model_block_depths = {str(k): int(v) for k, v in raw_depths.items()}

    if model_name not in model_block_depths:
        known = ", ".join(sorted(model_block_depths))
        raise ValueError(f"Unsupported model_name='{model_name}'. Known: {known}")

    if not relative_depths:
        raise ValueError("relative_depths cannot be empty.")

    depth = int(model_block_depths[model_name])
    resolved: list[int] = []
    for value in relative_depths:
        current = float(value)
        if not (0.0 < current <= 1.0):
            raise ValueError(f"Invalid relative depth {value!r}. Expected values in (0, 1].")

        layer_id = int(round(depth * current))
        layer_id = max(1, min(depth, layer_id))
        if layer_id not in resolved:
            resolved.append(layer_id)

    return tuple(resolved)


class LTXVideoAdapter(VideoBackboneAdapter):
    """Frozen-feature extractor for LTX-Video VAE encoder stages.

    Inputs to ``extract`` are expected to be clip tensors in shape ``[B, C, T, H, W]``
    with values in ``[0, 1]``. By default, clips are mapped to ``[-1, 1]`` before
    feeding the VAE encoder.
    """

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
        frames_per_clip: int | None = None,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
        vae_subfolder: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        normalize_input: bool = True,
        enable_vae_tiling: bool = False,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        self.normalize_input = bool(normalize_input)
        self.enable_vae_tiling = bool(enable_vae_tiling)

        cfg = _load_ltx_video_config(self.config_path)
        self.variant, variant_cfg, self.hf_model_id = _resolve_variant_bundle(
            cfg,
            variant=variant,
        )

        self.model_name = str(model_name or variant_cfg.get("model_name", "ltx_vae_5"))
        self.crop_size = int(crop_size if crop_size is not None else variant_cfg.get("crop_size", 224))
        self.frames_per_clip = int(
            frames_per_clip if frames_per_clip is not None else variant_cfg.get("frames_per_clip", 16)
        )
        self.patch_size = int(patch_size if patch_size is not None else variant_cfg.get("patch_size", 4))
        self.patch_size_t = int(
            patch_size_t if patch_size_t is not None else variant_cfg.get("patch_size_t", 1)
        )
        self.vae_subfolder = str(vae_subfolder or variant_cfg.get("vae_subfolder", "vae"))

        dtype_from_cfg = variant_cfg.get("torch_dtype", "float32")
        self.model_dtype = _resolve_torch_dtype(torch_dtype if torch_dtype is not None else dtype_from_cfg)

        raw_model_depths = cfg.get("model_block_depths")
        if not isinstance(raw_model_depths, dict) or not raw_model_depths:
            raise ValueError("ltx_video.model_block_depths must be a non-empty mapping.")
        model_block_depths = {str(k): int(v) for k, v in raw_model_depths.items()}

        raw_rel = cfg.get("default_relative_depths")
        if not isinstance(raw_rel, list) or not raw_rel:
            raise ValueError("ltx_video.default_relative_depths must be a non-empty list.")
        default_relative_depths = tuple(float(v) for v in raw_rel)

        self.selected_layers = resolve_relative_depth_layers(
            self.model_name,
            relative_depths=relative_depths if relative_depths is not None else default_relative_depths,
            model_block_depths=model_block_depths,
            config_path=self.config_path,
        )

        self._vae = self._load_vae()
        self._encoder_stages = self._get_encoder_stages()
        self._temporal_downsample_strides = self._collect_temporal_downsample_strides()

        max_selected = max(self.selected_layers)
        if max_selected > len(self._encoder_stages):
            raise ValueError(
                "Configured selected_layers exceed available LTX VAE encoder stages: "
                f"max_selected={max_selected}, available={len(self._encoder_stages)}. "
                "Adjust ltx_video.model_block_depths or relative depths in backbones.yaml."
            )

    def _load_vae(self) -> torch.nn.Module:
        try:
            from diffusers import AutoencoderKLLTXVideo
        except ImportError as exc:
            raise ImportError(
                "diffusers with AutoencoderKLLTXVideo support is required for LTX-Video. "
                "Install or update with: python -m pip install -U diffusers transformers accelerate safetensors"
            ) from exc

        kwargs: dict[str, Any] = {"torch_dtype": self.model_dtype}
        if self.hf_cache_dir is not None:
            kwargs["cache_dir"] = self.hf_cache_dir

        try:
            vae = AutoencoderKLLTXVideo.from_pretrained(
                self.hf_model_id,
                subfolder=self.vae_subfolder,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on HF/network/license
            raise RuntimeError(
                "Failed to load LTX-Video VAE from HuggingFace. "
                f"model_id='{self.hf_model_id}', subfolder='{self.vae_subfolder}'. "
                "Ensure network access, accepted model license, and valid HF authentication if required."
            ) from exc

        vae.to(self.device)
        vae.eval()
        for parameter in vae.parameters():
            parameter.requires_grad = False

        if self.enable_vae_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        return vae

    def _get_encoder_stages(self) -> list[torch.nn.Module]:
        encoder = getattr(self._vae, "encoder", None)
        if encoder is None:
            raise RuntimeError("Loaded LTX VAE does not expose an encoder module.")

        down_blocks = getattr(encoder, "down_blocks", None)
        mid_block = getattr(encoder, "mid_block", None)
        if not isinstance(down_blocks, torch.nn.ModuleList) or mid_block is None:
            raise RuntimeError(
                "Could not locate LTX VAE encoder stages. Expected attributes: encoder.down_blocks and encoder.mid_block."
            )

        return [*list(down_blocks), mid_block]

    def _resolve_requested_layers(self, layer_ids: Sequence[int] | None) -> tuple[int, ...]:
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

    def _collect_temporal_downsample_strides(self) -> tuple[int, ...]:
        """Inspect encoder downsamplers and collect temporal strides > 1."""

        encoder = getattr(self._vae, "encoder", None)
        down_blocks = getattr(encoder, "down_blocks", None)
        if not isinstance(down_blocks, torch.nn.ModuleList):
            return ()

        strides: list[int] = []
        for block in down_blocks:
            downsamplers = getattr(block, "downsamplers", None)
            if not isinstance(downsamplers, (list, tuple, torch.nn.ModuleList)):
                continue
            for downsampler in downsamplers:
                raw_stride = getattr(downsampler, "stride", 1)
                if isinstance(raw_stride, int):
                    temporal_stride = int(raw_stride)
                elif isinstance(raw_stride, (tuple, list)) and raw_stride:
                    temporal_stride = int(raw_stride[0])
                else:
                    temporal_stride = 1

                if temporal_stride > 1:
                    strides.append(temporal_stride)

        return tuple(strides)

    @staticmethod
    def _is_valid_temporal_length(num_frames: int, strides: Sequence[int]) -> bool:
        current = int(num_frames)
        if current <= 0:
            return False

        for stride in strides:
            current += int(stride) - 1
            if current % int(stride) != 0:
                return False
            current //= int(stride)
        return True

    def _align_temporal_length(self, clips: torch.Tensor) -> torch.Tensor:
        """Pad by repeating the last frame so LTX temporal downsamplers can unflatten safely."""

        if not self._temporal_downsample_strides:
            return clips

        num_frames = int(clips.shape[2])
        if self._is_valid_temporal_length(num_frames, self._temporal_downsample_strides):
            return clips

        max_extra = 1
        for stride in self._temporal_downsample_strides:
            max_extra *= int(stride)

        target_frames: int | None = None
        for extra in range(1, max_extra + 1):
            candidate = num_frames + extra
            if self._is_valid_temporal_length(candidate, self._temporal_downsample_strides):
                target_frames = candidate
                break

        if target_frames is None:
            raise RuntimeError(
                "Unable to align clip temporal length for LTX VAE downsampling. "
                f"num_frames={num_frames}, temporal_strides={self._temporal_downsample_strides}"
            )

        extra_frames = target_frames - num_frames
        last_frame = clips[:, :, -1:, :, :].expand(-1, -1, extra_frames, -1, -1)
        return torch.cat([clips, last_frame], dim=2)

    def preprocessing_metadata(self) -> dict[str, Any]:
        """Return the raw-clip preprocessing contract used before forward."""

        return ltx_preprocessing_metadata(normalize_input=self.normalize_input)

    @staticmethod
    def _stage_to_tokens(stage_output: torch.Tensor) -> torch.Tensor:
        """Convert a stage activation to canonical token layout [B, N, D]."""

        if stage_output.ndim != 5:
            raise RuntimeError(
                "Expected LTX VAE stage output with shape [B, C, T, H, W], "
                f"got shape {tuple(stage_output.shape)}"
            )

        batch, channels, frames, height, width = stage_output.shape
        _ = frames, height, width
        return stage_output.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)

    def extract(
        self,
        clips: torch.Tensor,
        layer_ids: Sequence[int] | None = None,
    ) -> BackboneFeatures:
        if not isinstance(clips, torch.Tensor):
            raise TypeError(f"clips must be a torch.Tensor, got {type(clips)!r}")
        if clips.ndim != 5:
            raise ValueError(f"Expected clips shape [B, C, T, H, W], got {tuple(clips.shape)}")

        requested_layers = self._resolve_requested_layers(layer_ids)

        # Benchmark decoders output [0, 1] float clips. LTX VAE generally expects [-1, 1].
        inputs = clips.to(self.device, dtype=torch.float32)
        inputs = self._align_temporal_length(inputs)
        if self.normalize_input:
            inputs = inputs * 2.0 - 1.0

        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        for layer_id in self.selected_layers:
            stage_module = self._encoder_stages[layer_id - 1]

            def _make_hook(lid: int):
                def _hook(
                    module: torch.nn.Module,
                    input: tuple[Any, ...],
                    output: torch.Tensor | tuple[Any, ...],
                ) -> None:
                    _ = module, input
                    value = output[0] if isinstance(output, tuple) else output
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError(
                            f"Unexpected non-tensor output from LTX VAE stage {lid}: {type(value)!r}"
                        )
                    captured[lid] = value

                return _hook

            handles.append(stage_module.register_forward_hook(_make_hook(layer_id)))

        try:
            with torch.no_grad():
                self._vae.encode(inputs)
        finally:
            for handle in handles:
                handle.remove()

        missing_layers = [layer for layer in self.selected_layers if layer not in captured]
        if missing_layers:
            raise RuntimeError(
                "Failed to capture all requested LTX VAE stage activations. "
                f"Missing layers: {missing_layers}"
            )

        tokens_by_layer: dict[int, torch.Tensor] = {}
        pooled_by_layer: dict[int, torch.Tensor] = {}
        for layer in requested_layers:
            tokens = self._stage_to_tokens(captured[layer])
            tokens_by_layer[layer] = tokens
            pooled_by_layer[layer] = tokens.mean(dim=1)

        metadata = {
            "hf_model_id": self.hf_model_id,
            "config_path": str(self.config_path),
            "variant": self.variant,
            "model_name": self.model_name,
            "extract_source": "vae_encoder_stages",
            "vae_subfolder": self.vae_subfolder,
            "torch_dtype": str(self.model_dtype).replace("torch.", ""),
            "normalize_input": self.normalize_input,
            "patch_size": self.patch_size,
            "patch_size_t": self.patch_size_t,
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
            "spatial_compression_ratio": int(getattr(self._vae, "spatial_compression_ratio", 1)),
            "temporal_compression_ratio": int(getattr(self._vae, "temporal_compression_ratio", 1)),
            "temporal_downsample_strides": [int(v) for v in self._temporal_downsample_strides],
            "preprocessing": self.preprocessing_metadata(),
        }

        return BackboneFeatures(
            tokens_by_layer=tokens_by_layer,
            pooled_by_layer=pooled_by_layer,
            selected_layers=requested_layers,
            metadata=metadata,
        )


def create_ltx_video_adapter(**kwargs: Any) -> LTXVideoAdapter:
    """Factory function used by the central adapter registry."""

    return LTXVideoAdapter(**kwargs)


register_adapter("ltx_video", create_ltx_video_adapter, replace=True)
