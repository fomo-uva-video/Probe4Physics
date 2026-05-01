from __future__ import annotations

"""
LTX-Video adapter using diffusion-transformer hidden states at fixed noise levels.

The proposal for this repository treats diffusion models differently from plain
encoder backbones: probe both multiple denoising regimes and multiple backbone
depths. This adapter therefore:

1. encodes raw clips into normalized LTX VAE latents,
2. injects deterministic reference noise at configured sigma levels, and
3. captures hidden states from selected LTX transformer blocks.

The canonical adapter schema only supports integer layer ids, so the adapter
flattens the `(noise_level, transformer_depth)` grid into ordered probe-slot ids.
Metadata exposes the exact mapping for reproducibility.
"""

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .preprocessing import (
    ltx_diffusion_preprocessing_metadata,
    normalize_rgb_minus_one_one,
)
from .registry import BackboneFeatures, VideoBackboneAdapter, register_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"
DEFAULT_NOISE_LEVELS = (0.9, 0.5, 0.1)
DEFAULT_NOISE_SEED = 0
DEFAULT_FRAME_RATE = 25.0


@dataclass(frozen=True)
class LTXProbeLayerSpec:
    probe_layer_id: int
    noise_level_index: int
    noise_fraction: float
    noise_label: str
    depth_layer_id: int


@dataclass(frozen=True)
class _ResolvedLTXAdapterConfig:
    variant: str
    hf_model_id: str
    model_name: str
    crop_size: int
    frames_per_clip: int
    patch_size: int
    patch_size_t: int
    vae_subfolder: str
    model_dtype: torch.dtype
    noise_levels: tuple[float, ...]
    probe_layer_specs: tuple[LTXProbeLayerSpec, ...]


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


def _resolve_model_block_depths(cfg: dict[str, Any]) -> dict[str, int]:
    raw_model_depths = cfg.get("model_block_depths")
    if not isinstance(raw_model_depths, dict) or not raw_model_depths:
        raise ValueError("ltx_video.model_block_depths must be a non-empty mapping.")
    return {str(key): int(value) for key, value in raw_model_depths.items()}


def _resolve_default_relative_depths(cfg: dict[str, Any]) -> tuple[float, ...]:
    raw_relative_depths = cfg.get("default_relative_depths")
    if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
        raise ValueError("ltx_video.default_relative_depths must be a non-empty list.")
    return tuple(float(value) for value in raw_relative_depths)


def _resolve_adapter_config(
    cfg: dict[str, Any],
    *,
    variant: str | None,
    model_name: str | None,
    relative_depths: Sequence[float] | None,
    noise_levels: Sequence[float] | None,
    crop_size: int | None,
    frames_per_clip: int | None,
    patch_size: int | None,
    patch_size_t: int | None,
    vae_subfolder: str | None,
    torch_dtype: str | torch.dtype | None,
    config_path: str | Path,
) -> _ResolvedLTXAdapterConfig:
    chosen_variant, variant_cfg, hf_model_id = _resolve_variant_bundle(cfg, variant=variant)
    model_block_depths = _resolve_model_block_depths(cfg)
    default_relative_depths = _resolve_default_relative_depths(cfg)
    selected_noise_levels = resolve_noise_levels(
        noise_levels if noise_levels is not None else cfg.get("default_noise_levels", list(DEFAULT_NOISE_LEVELS)),
        config_path=config_path,
    )
    selected_relative_depths = (
        tuple(float(value) for value in relative_depths)
        if relative_depths is not None
        else default_relative_depths
    )
    resolved_model_name = str(model_name or variant_cfg.get("model_name", "ltx_transformer_48"))
    dtype_from_cfg = variant_cfg.get("torch_dtype", "float32")

    return _ResolvedLTXAdapterConfig(
        variant=chosen_variant,
        hf_model_id=hf_model_id,
        model_name=resolved_model_name,
        crop_size=int(crop_size if crop_size is not None else variant_cfg.get("crop_size", 224)),
        frames_per_clip=int(
            frames_per_clip if frames_per_clip is not None else variant_cfg.get("frames_per_clip", 16)
        ),
        patch_size=int(patch_size if patch_size is not None else variant_cfg.get("patch_size", 1)),
        patch_size_t=int(
            patch_size_t if patch_size_t is not None else variant_cfg.get("patch_size_t", 1)
        ),
        vae_subfolder=str(vae_subfolder or variant_cfg.get("vae_subfolder", "vae")),
        model_dtype=_resolve_torch_dtype(
            torch_dtype if torch_dtype is not None else dtype_from_cfg
        ),
        noise_levels=selected_noise_levels,
        probe_layer_specs=resolve_probe_layer_specs(
            resolved_model_name,
            relative_depths=selected_relative_depths,
            noise_levels=selected_noise_levels,
            model_block_depths=model_block_depths,
            config_path=config_path,
        ),
    )


def resolve_relative_depth_layers(
    model_name: str,
    relative_depths: Sequence[float] | None = None,
    *,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    """Map relative probe depths (e.g. 0.25, 0.5, ...) to 1-based transformer block ids."""

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


def resolve_noise_levels(
    noise_levels: Sequence[float] | None = None,
    *,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[float, ...]:
    """Resolve configured diffusion noise levels as fractions in ``(0, 1]``."""

    if noise_levels is None:
        cfg = _load_ltx_video_config(config_path)
        raw_levels = cfg.get("default_noise_levels", list(DEFAULT_NOISE_LEVELS))
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("ltx_video.default_noise_levels must be a non-empty list.")
        noise_levels = tuple(float(v) for v in raw_levels)

    resolved: list[float] = []
    for value in noise_levels:
        current = float(value)
        if not (0.0 < current <= 1.0):
            raise ValueError(f"Invalid noise level {value!r}. Expected values in (0, 1].")
        if current not in resolved:
            resolved.append(current)
    if not resolved:
        raise ValueError("noise_levels cannot be empty.")
    return tuple(resolved)


def resolve_probe_layer_specs(
    model_name: str,
    *,
    relative_depths: Sequence[float] | None = None,
    noise_levels: Sequence[float] | None = None,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[LTXProbeLayerSpec, ...]:
    """Resolve the flattened proposal grid of ``(noise_level, depth)`` probe slots."""

    depth_layers = resolve_relative_depth_layers(
        model_name,
        relative_depths=relative_depths,
        model_block_depths=model_block_depths,
        config_path=config_path,
    )
    resolved_noise_levels = resolve_noise_levels(noise_levels, config_path=config_path)

    specs: list[LTXProbeLayerSpec] = []
    probe_layer_id = 1
    for noise_index, noise_fraction in enumerate(resolved_noise_levels):
        specs_for_noise = len(resolved_noise_levels)
        if specs_for_noise == 3:
            label = ("high", "mid", "low")[noise_index]
        else:
            label = f"noise_{noise_index + 1}"
        for depth_layer_id in depth_layers:
            specs.append(
                LTXProbeLayerSpec(
                    probe_layer_id=probe_layer_id,
                    noise_level_index=noise_index,
                    noise_fraction=float(noise_fraction),
                    noise_label=label,
                    depth_layer_id=int(depth_layer_id),
                )
            )
            probe_layer_id += 1
    return tuple(specs)


def resolve_probe_layer_ids(
    model_name: str,
    *,
    relative_depths: Sequence[float] | None = None,
    noise_levels: Sequence[float] | None = None,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    return tuple(
        spec.probe_layer_id
        for spec in resolve_probe_layer_specs(
            model_name,
            relative_depths=relative_depths,
            noise_levels=noise_levels,
            model_block_depths=model_block_depths,
            config_path=config_path,
        )
    )


def _ensure_ltx_tokenizer_runtime_support() -> None:
    has_sentencepiece = importlib.util.find_spec("sentencepiece") is not None
    has_tiktoken = importlib.util.find_spec("tiktoken") is not None
    if has_sentencepiece or has_tiktoken:
        return
    raise RuntimeError(
        "LTX-Video tokenizer dependencies are missing. Install 'sentencepiece' or "
        "'tiktoken' before loading the LTX pipeline."
    )


class LTXVideoAdapter(VideoBackboneAdapter):
    """Frozen-feature extractor for LTX diffusion transformer blocks."""

    def __init__(
        self,
        *,
        variant: str | None = None,
        hf_cache_dir: str | Path | None = None,
        model_name: str | None = None,
        relative_depths: Sequence[float] | None = None,
        noise_levels: Sequence[float] | None = None,
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
        noise_seed: int = DEFAULT_NOISE_SEED,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        self.normalize_input = bool(normalize_input)
        self.enable_vae_tiling = bool(enable_vae_tiling)
        self.noise_seed = int(noise_seed)

        cfg = _load_ltx_video_config(self.config_path)
        resolved = _resolve_adapter_config(
            cfg,
            variant=variant,
            config_path=self.config_path,
            model_name=model_name,
            relative_depths=relative_depths,
            noise_levels=noise_levels,
            crop_size=crop_size,
            frames_per_clip=frames_per_clip,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            vae_subfolder=vae_subfolder,
            torch_dtype=torch_dtype,
        )
        self.variant = resolved.variant
        self.hf_model_id = resolved.hf_model_id
        self.model_name = resolved.model_name
        self.crop_size = resolved.crop_size
        self.frames_per_clip = resolved.frames_per_clip
        self.patch_size = resolved.patch_size
        self.patch_size_t = resolved.patch_size_t
        self.vae_subfolder = resolved.vae_subfolder
        self.model_dtype = resolved.model_dtype
        self.noise_levels = resolved.noise_levels
        self._probe_layer_specs = resolved.probe_layer_specs
        self._probe_specs_by_slot = {spec.probe_layer_id: spec for spec in self._probe_layer_specs}
        self.selected_layers = tuple(spec.probe_layer_id for spec in self._probe_layer_specs)
        self._depth_layers = tuple(
            dict.fromkeys(spec.depth_layer_id for spec in self._probe_layer_specs).keys()
        )

        (
            self._vae,
            self._transformer,
            self._scheduler,
            self._tokenizer,
            self._text_encoder,
        ) = self._load_components()
        self._noise_timesteps, self._noise_sigmas = self._resolve_noise_schedule(self.noise_levels)
        self._transformer_blocks = self._get_transformer_blocks()
        self._temporal_downsample_strides = self._collect_temporal_downsample_strides()
        self._prompt_embeds, self._prompt_attention_mask = self._encode_prompt()

        max_selected_depth = max(self._depth_layers)
        if max_selected_depth > len(self._transformer_blocks):
            raise ValueError(
                "Configured selected transformer depths exceed available LTX transformer blocks: "
                f"max_selected={max_selected_depth}, available={len(self._transformer_blocks)}. "
                "Adjust ltx_video.model_block_depths or relative depths in backbones.yaml."
            )

    def _load_components(
        self,
    ) -> tuple[torch.nn.Module, torch.nn.Module, Any, Any, torch.nn.Module]:
        _ensure_ltx_tokenizer_runtime_support()
        try:
            from diffusers import LTXPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers with LTXPipeline support is required for LTX-Video. "
                "Install or update with: python -m pip install -U diffusers transformers accelerate safetensors"
            ) from exc

        kwargs: dict[str, Any] = {"torch_dtype": self.model_dtype}
        if self.hf_cache_dir is not None:
            kwargs["cache_dir"] = self.hf_cache_dir

        try:
            pipeline = LTXPipeline.from_pretrained(self.hf_model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on HF/network/license
            raise RuntimeError(
                "Failed to load LTX-Video pipeline from HuggingFace. "
                f"model_id='{self.hf_model_id}'. "
                "Ensure network access, accepted model license, and valid HF authentication if required."
            ) from exc

        pipeline.to(self.device)

        components = (
            getattr(pipeline, "vae", None),
            getattr(pipeline, "transformer", None),
            getattr(pipeline, "scheduler", None),
            getattr(pipeline, "tokenizer", None),
            getattr(pipeline, "text_encoder", None),
        )
        if any(component is None for component in components):
            raise RuntimeError(
                "Loaded LTX pipeline is missing one or more required components: "
                "vae, transformer, scheduler, tokenizer, text_encoder."
            )

        vae, transformer, scheduler, tokenizer, text_encoder = components
        modules = (vae, transformer, text_encoder)
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        if self.enable_vae_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        return vae, transformer, scheduler, tokenizer, text_encoder

    def _get_transformer_blocks(self) -> list[torch.nn.Module]:
        blocks = getattr(self._transformer, "transformer_blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            raise RuntimeError(
                "Loaded LTX transformer does not expose transformer_blocks ModuleList."
            )
        return list(blocks)

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

    def _encode_prompt(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = getattr(self._tokenizer, "model_max_length", None)
        tokenizer_kwargs: dict[str, Any] = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        if isinstance(max_length, int) and max_length > 0:
            tokenizer_kwargs["max_length"] = max_length

        tokens = self._tokenizer([""], **tokenizer_kwargs)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self._text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("LTX text encoder did not return last_hidden_state.")

        return hidden_states, attention_mask

    @staticmethod
    def _pack_latents(latents: torch.Tensor, *, patch_size: int, patch_size_t: int) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    def _expand_prompt_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = self._prompt_embeds.to(device=self.device, dtype=self.model_dtype)
        prompt_mask = self._prompt_attention_mask.to(device=self.device)
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1).contiguous()
        prompt_mask = prompt_mask.expand(batch_size, -1).contiguous()
        return prompt_embeds, prompt_mask

    def _resolve_transformer_patch_sizes(self) -> tuple[int, int]:
        transformer_cfg = getattr(self._transformer, "config", None)
        patch_size = int(getattr(transformer_cfg, "patch_size", self.patch_size))
        patch_size_t = int(getattr(transformer_cfg, "patch_size_t", self.patch_size_t))
        return patch_size, patch_size_t

    def _rope_interpolation_scale(self) -> tuple[float, float, float]:
        temporal_ratio = float(getattr(self._vae, "temporal_compression_ratio", 8))
        spatial_ratio = float(getattr(self._vae, "spatial_compression_ratio", 32))
        return (
            temporal_ratio / DEFAULT_FRAME_RATE,
            spatial_ratio,
            spatial_ratio,
        )

    def _resolve_noise_schedule(
        self,
        noise_levels: Sequence[float],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        scheduler_sigmas = getattr(self._scheduler, "sigmas", None)
        scheduler_timesteps = getattr(self._scheduler, "timesteps", None)
        if (
            isinstance(scheduler_sigmas, torch.Tensor)
            and isinstance(scheduler_timesteps, torch.Tensor)
            and scheduler_sigmas.ndim == 1
            and scheduler_timesteps.ndim == 1
            and scheduler_sigmas.numel() > 0
            and scheduler_sigmas.numel() == scheduler_timesteps.numel()
        ):
            sigmas = scheduler_sigmas.detach().to(device="cpu", dtype=torch.float32).flatten()
            timesteps = scheduler_timesteps.detach().to(device="cpu", dtype=torch.float32).flatten()
            resolved_timesteps: list[float] = []
            resolved_sigmas: list[float] = []
            for noise_fraction in noise_levels:
                target_sigma = torch.tensor(float(noise_fraction), dtype=sigmas.dtype)
                nearest_index = int(torch.argmin(torch.abs(sigmas - target_sigma)).item())
                resolved_timesteps.append(float(timesteps[nearest_index].item()))
                resolved_sigmas.append(float(sigmas[nearest_index].item()))
            return tuple(resolved_timesteps), tuple(resolved_sigmas)

        scheduler_cfg = getattr(self._scheduler, "config", None)
        total = int(getattr(scheduler_cfg, "num_train_timesteps", 1000))
        upper = max(total, 1)
        resolved_timesteps = []
        resolved_sigmas = []
        for noise_fraction in noise_levels:
            timestep = round(float(noise_fraction) * upper)
            timestep = max(1, min(upper, timestep))
            resolved_timesteps.append(float(timestep))
            resolved_sigmas.append(float(noise_fraction))
        return tuple(resolved_timesteps), tuple(resolved_sigmas)

    def preprocessing_metadata(self) -> dict[str, Any]:
        """Return the raw-clip preprocessing contract used before forward."""

        return ltx_diffusion_preprocessing_metadata(
            normalize_input=self.normalize_input,
            noise_levels=self.noise_levels,
            prompt_mode="empty_string",
            noise_policy="fixed_reference_noise",
        )

    def _encode_clips_to_latents(self, clips: torch.Tensor) -> torch.Tensor:
        inputs = clips.to(self.device, dtype=torch.float32)
        inputs = self._align_temporal_length(inputs)
        if self.normalize_input:
            inputs = normalize_rgb_minus_one_one(inputs)
        inputs = inputs.to(dtype=self.model_dtype)

        with torch.no_grad():
            encoded = self._vae.encode(inputs)

        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None or not hasattr(latent_dist, "mode"):
            raise RuntimeError("LTX VAE encode() did not return a latent distribution with mode().")

        latents = latent_dist.mode()
        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(f"LTX VAE latent mode() returned non-tensor value: {type(latents)!r}")
        return self._normalize_ltx_latents(latents)

    def _normalize_ltx_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = getattr(self._vae, "latents_mean", None)
        latents_std = getattr(self._vae, "latents_std", None)
        if not isinstance(latents_mean, torch.Tensor) or not isinstance(latents_std, torch.Tensor):
            raise RuntimeError(
                "Loaded LTX VAE is missing latents_mean/latents_std required for transformer conditioning."
            )

        vae_cfg = getattr(self._vae, "config", None)
        scaling_factor = float(getattr(vae_cfg, "scaling_factor", 1.0))
        mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) * scaling_factor / std

    def _reference_noise(self, latents: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.noise_seed)
        base_noise = torch.randn(
            (1, *latents.shape[1:]),
            generator=generator,
            dtype=torch.float32,
        )
        base_noise = base_noise.to(device=latents.device, dtype=latents.dtype)
        return base_noise.expand(latents.shape[0], -1, -1, -1, -1).contiguous()

    def _scale_noise(
        self,
        latents: torch.Tensor,
        *,
        noise: torch.Tensor,
        timestep_value: float,
    ) -> torch.Tensor:
        timestep = torch.full(
            (latents.shape[0],),
            float(timestep_value),
            device=latents.device,
            dtype=latents.dtype,
        )
        scaled = self._scheduler.scale_noise(latents, timestep, noise)
        if not isinstance(scaled, torch.Tensor):
            raise RuntimeError(
                f"LTX scheduler scale_noise() returned non-tensor value: {type(scaled)!r}"
            )
        return scaled

    def _capture_transformer_layers(
        self,
        noisy_latents: torch.Tensor,
        *,
        depth_layers: Sequence[int],
        timestep_value: float,
    ) -> dict[int, torch.Tensor]:
        if noisy_latents.ndim != 5:
            raise RuntimeError(
                "Expected LTX latent tensor with shape [B, C, T, H, W], "
                f"got shape {tuple(noisy_latents.shape)}"
            )
        patch_size, patch_size_t = self._resolve_transformer_patch_sizes()
        hidden_states = self._pack_latents(
            noisy_latents,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
        ).to(dtype=self.model_dtype)
        batch_size = int(hidden_states.shape[0])
        prompt_embeds, prompt_mask = self._expand_prompt_batch(batch_size)
        rope_interpolation_scale = self._rope_interpolation_scale()

        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        for depth_layer in depth_layers:
            block = self._transformer_blocks[depth_layer - 1]

            def _make_hook(layer_id: int):
                def _hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    output: torch.Tensor | tuple[Any, ...],
                ) -> None:
                    _ = module, inputs
                    value = output[0] if isinstance(output, tuple) else output
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError(
                            f"Unexpected non-tensor output from LTX transformer block {layer_id}: {type(value)!r}"
                        )
                    captured[layer_id] = value

                return _hook

            handles.append(block.register_forward_hook(_make_hook(depth_layer)))

        timestep = torch.full(
            (batch_size,),
            float(timestep_value),
            device=self.device,
            dtype=hidden_states.dtype,
        )

        try:
            with torch.no_grad():
                self._transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_mask,
                    num_frames=int(noisy_latents.shape[2]),
                    height=int(noisy_latents.shape[3]),
                    width=int(noisy_latents.shape[4]),
                    rope_interpolation_scale=rope_interpolation_scale,
                    return_dict=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        missing_layers = [layer for layer in depth_layers if layer not in captured]
        if missing_layers:
            raise RuntimeError(
                "Failed to capture all requested LTX transformer block activations. "
                f"Missing layers: {missing_layers}"
            )
        return captured

    def _build_layer_spec_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            str(spec.probe_layer_id): {
                "noise_level_index": int(spec.noise_level_index),
                "noise_fraction": float(spec.noise_fraction),
                "noise_label": spec.noise_label,
                "depth_layer_id": int(spec.depth_layer_id),
            }
            for spec in self._probe_layer_specs
        }

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "hf_model_id": self.hf_model_id,
            "config_path": str(self.config_path),
            "variant": self.variant,
            "model_name": self.model_name,
            "extract_source": "diffusion_transformer_blocks",
            "vae_subfolder": self.vae_subfolder,
            "torch_dtype": str(self.model_dtype).replace("torch.", ""),
            "normalize_input": self.normalize_input,
            "noise_seed": self.noise_seed,
            "noise_levels": [float(v) for v in self.noise_levels],
            "noise_sigmas": [float(v) for v in self._noise_sigmas],
            "noise_timesteps": [float(v) for v in self._noise_timesteps],
            "patch_size": self.patch_size,
            "patch_size_t": self.patch_size_t,
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
            "spatial_compression_ratio": int(getattr(self._vae, "spatial_compression_ratio", 1)),
            "temporal_compression_ratio": int(getattr(self._vae, "temporal_compression_ratio", 1)),
            "temporal_downsample_strides": [int(v) for v in self._temporal_downsample_strides],
            "transformer_block_count": len(self._transformer_blocks),
            "layer_spec_by_id": self._build_layer_spec_metadata(),
            "preprocessing": self.preprocessing_metadata(),
        }

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
        requested_specs = [self._probe_specs_by_slot[layer_id] for layer_id in requested_layers]

        latents = self._encode_clips_to_latents(clips)
        reference_noise = self._reference_noise(latents)

        specs_by_noise_index: dict[int, list[LTXProbeLayerSpec]] = defaultdict(list)
        for spec in requested_specs:
            specs_by_noise_index[spec.noise_level_index].append(spec)

        tokens_by_layer: dict[int, torch.Tensor] = {}
        pooled_by_layer: dict[int, torch.Tensor] = {}

        for noise_index, specs in specs_by_noise_index.items():
            timestep_value = self._noise_timesteps[noise_index]
            noisy_latents = self._scale_noise(
                latents,
                noise=reference_noise,
                timestep_value=timestep_value,
            )

            depth_layers = tuple(dict.fromkeys(spec.depth_layer_id for spec in specs).keys())
            captured = self._capture_transformer_layers(
                noisy_latents,
                depth_layers=depth_layers,
                timestep_value=timestep_value,
            )

            for spec in specs:
                tokens = captured[spec.depth_layer_id]
                if tokens.ndim != 3:
                    raise RuntimeError(
                        "Expected LTX transformer hidden states with shape [B, N, D], "
                        f"got shape {tuple(tokens.shape)}"
                    )
                tokens_by_layer[spec.probe_layer_id] = tokens
                pooled_by_layer[spec.probe_layer_id] = tokens.mean(dim=1)

        return BackboneFeatures(
            tokens_by_layer=tokens_by_layer,
            pooled_by_layer=pooled_by_layer,
            selected_layers=requested_layers,
            metadata=self._build_metadata(),
        )


def create_ltx_video_adapter(**kwargs: Any) -> LTXVideoAdapter:
    """Factory function used by the central adapter registry."""

    return LTXVideoAdapter(**kwargs)


register_adapter("ltx_video", create_ltx_video_adapter, replace=True)
