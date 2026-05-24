from __future__ import annotations

"""
Wan video adapter using diffusion-transformer hidden states at fixed noise levels.

Wan is a latent video diffusion backbone, so this adapter mirrors the LTX probing
contract: encode benchmark clips to VAE latents, inject deterministic reference
noise at configured diffusion levels, and capture selected transformer blocks.
The `(noise_level, transformer_depth)` grid is flattened into integer probe-slot
ids because the shared probe runner addresses features by one integer layer id.
"""

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .preprocessing import normalize_rgb_minus_one_one, wan_diffusion_preprocessing_metadata
from .registry import BackboneFeatures, VideoBackboneAdapter, register_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"
DEFAULT_NOISE_LEVELS = (0.9, 0.5, 0.1)
DEFAULT_NOISE_SEED = 0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_MAX_SEQUENCE_LENGTH = 226


@dataclass(frozen=True)
class WanProbeLayerSpec:
    probe_layer_id: int
    noise_level_index: int
    noise_fraction: float
    noise_label: str
    depth_layer_id: int


@dataclass(frozen=True)
class _ResolvedWanAdapterConfig:
    variant: str
    hf_model_id: str
    model_name: str
    crop_size: int
    frames_per_clip: int
    model_dtype: torch.dtype
    noise_levels: tuple[float, ...]
    probe_layer_specs: tuple[WanProbeLayerSpec, ...]
    num_inference_steps: int
    max_sequence_length: int


def _load_wan_video_config(
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone config file not found: {path}. "
            "Expected global config at configs/backbones.yaml."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    cfg = payload.get("wan_video")
    if not isinstance(cfg, dict):
        raise ValueError("configs/backbones.yaml must define a 'wan_video' object.")
    return cfg


def _resolve_variant_bundle(
    cfg: dict[str, Any],
    *,
    variant: str | None,
) -> tuple[str, dict[str, Any], str]:
    variants = cfg.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("wan_video.variants must be a non-empty mapping.")

    chosen = str(variant).strip() if variant is not None else ""
    if not chosen:
        chosen = str(cfg.get("default_variant", "")).strip()
    if not chosen:
        raise ValueError("wan_video.default_variant is required for zero-config adapter creation.")

    if chosen not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(f"Unknown wan_video variant '{chosen}'. Known variants: {known}")

    variant_cfg = variants[chosen]
    if not isinstance(variant_cfg, dict):
        raise ValueError(f"Variant config for '{chosen}' must be a mapping.")

    hf_model_id = str(variant_cfg.get("hf_model_id", "")).strip()
    if not hf_model_id:
        raise ValueError(f"wan_video variant '{chosen}' is missing hf_model_id in backbones.yaml.")

    return chosen, variant_cfg, hf_model_id


def _resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    name = str(dtype_name or "bfloat16").strip().lower()
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
        raise ValueError("wan_video.model_block_depths must be a non-empty mapping.")
    return {str(key): int(value) for key, value in raw_model_depths.items()}


def _resolve_default_relative_depths(cfg: dict[str, Any]) -> tuple[float, ...]:
    raw_relative_depths = cfg.get("default_relative_depths")
    if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
        raise ValueError("wan_video.default_relative_depths must be a non-empty list.")
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
    torch_dtype: str | torch.dtype | None,
    num_inference_steps: int | None,
    max_sequence_length: int | None,
    config_path: str | Path,
) -> _ResolvedWanAdapterConfig:
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
    resolved_model_name = str(model_name or variant_cfg.get("model_name", "wan2_1_t2v_14b"))
    dtype_from_cfg = variant_cfg.get("torch_dtype", "bfloat16")

    return _ResolvedWanAdapterConfig(
        variant=chosen_variant,
        hf_model_id=hf_model_id,
        model_name=resolved_model_name,
        crop_size=int(crop_size if crop_size is not None else variant_cfg.get("crop_size", 224)),
        frames_per_clip=int(
            frames_per_clip if frames_per_clip is not None else variant_cfg.get("frames_per_clip", 17)
        ),
        model_dtype=_resolve_torch_dtype(torch_dtype if torch_dtype is not None else dtype_from_cfg),
        noise_levels=selected_noise_levels,
        probe_layer_specs=resolve_probe_layer_specs(
            resolved_model_name,
            relative_depths=selected_relative_depths,
            noise_levels=selected_noise_levels,
            model_block_depths=model_block_depths,
            config_path=config_path,
        ),
        num_inference_steps=int(
            num_inference_steps
            if num_inference_steps is not None
            else variant_cfg.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        ),
        max_sequence_length=int(
            max_sequence_length
            if max_sequence_length is not None
            else variant_cfg.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
        ),
    )


def resolve_relative_depth_layers(
    model_name: str,
    relative_depths: Sequence[float] | None = None,
    *,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    """Map relative probe depths to 1-based Wan transformer block ids."""

    if relative_depths is None or model_block_depths is None:
        cfg = _load_wan_video_config(config_path)
        if relative_depths is None:
            relative_depths = _resolve_default_relative_depths(cfg)
        if model_block_depths is None:
            model_block_depths = _resolve_model_block_depths(cfg)

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
    if noise_levels is None:
        cfg = _load_wan_video_config(config_path)
        raw_levels = cfg.get("default_noise_levels", list(DEFAULT_NOISE_LEVELS))
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("wan_video.default_noise_levels must be a non-empty list.")
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
) -> tuple[WanProbeLayerSpec, ...]:
    depth_layers = resolve_relative_depth_layers(
        model_name,
        relative_depths=relative_depths,
        model_block_depths=model_block_depths,
        config_path=config_path,
    )
    resolved_noise_levels = resolve_noise_levels(noise_levels, config_path=config_path)

    specs: list[WanProbeLayerSpec] = []
    probe_layer_id = 1
    for noise_index, noise_fraction in enumerate(resolved_noise_levels):
        label = f"noise_{noise_index + 1}"
        for depth_layer_id in depth_layers:
            specs.append(
                WanProbeLayerSpec(
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


def _ensure_wan_runtime_support() -> None:
    if importlib.util.find_spec("sentencepiece") is None:
        raise RuntimeError(
            "Wan tokenizer dependencies are missing. Install 'sentencepiece' before loading the Wan pipeline."
        )
    try:
        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "diffusers with Wan support is required. Install/update diffusers, transformers, "
            "accelerate, safetensors, and sentencepiece."
        ) from exc


class WanVideoAdapter(VideoBackboneAdapter):
    """Frozen-feature extractor for Wan diffusion transformer blocks."""

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
        torch_dtype: str | torch.dtype | None = None,
        normalize_input: bool = True,
        enable_vae_tiling: bool = False,
        noise_seed: int = DEFAULT_NOISE_SEED,
        num_inference_steps: int | None = None,
        max_sequence_length: int | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        self.normalize_input = bool(normalize_input)
        self.enable_vae_tiling = bool(enable_vae_tiling)
        self.noise_seed = int(noise_seed)

        cfg = _load_wan_video_config(self.config_path)
        resolved = _resolve_adapter_config(
            cfg,
            variant=variant,
            config_path=self.config_path,
            model_name=model_name,
            relative_depths=relative_depths,
            noise_levels=noise_levels,
            crop_size=crop_size,
            frames_per_clip=frames_per_clip,
            torch_dtype=torch_dtype,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
        )
        self.variant = resolved.variant
        self.hf_model_id = resolved.hf_model_id
        self.model_name = resolved.model_name
        self.crop_size = resolved.crop_size
        self.frames_per_clip = resolved.frames_per_clip
        self.model_dtype = resolved.model_dtype
        self.noise_levels = resolved.noise_levels
        self.num_inference_steps = resolved.num_inference_steps
        self.max_sequence_length = resolved.max_sequence_length
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
        self._set_scheduler_timesteps()
        self._noise_timesteps, self._noise_sigmas = self._resolve_noise_schedule(self.noise_levels)
        self._transformer_blocks = self._get_transformer_blocks()
        self._prompt_embeds = self._encode_prompt()

        max_selected_depth = max(self._depth_layers)
        if max_selected_depth > len(self._transformer_blocks):
            raise ValueError(
                "Configured selected transformer depths exceed available Wan transformer blocks: "
                f"max_selected={max_selected_depth}, available={len(self._transformer_blocks)}. "
                "Adjust wan_video.model_block_depths or relative depths in backbones.yaml."
            )

    def _load_components(
        self,
    ) -> tuple[torch.nn.Module, torch.nn.Module, Any, Any, torch.nn.Module]:
        _ensure_wan_runtime_support()
        from diffusers import WanPipeline

        kwargs: dict[str, Any] = {"torch_dtype": self.model_dtype}
        if self.hf_cache_dir is not None:
            kwargs["cache_dir"] = self.hf_cache_dir

        try:
            pipeline = WanPipeline.from_pretrained(self.hf_model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on HF/network/license
            raise RuntimeError(
                "Failed to load Wan pipeline from HuggingFace. "
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
                "Loaded Wan pipeline is missing one or more required components: "
                "vae, transformer, scheduler, tokenizer, text_encoder."
            )

        vae, transformer, scheduler, tokenizer, text_encoder = components
        for module in (vae, transformer, text_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        if self.enable_vae_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        return vae, transformer, scheduler, tokenizer, text_encoder

    def _set_scheduler_timesteps(self) -> None:
        if not hasattr(self._scheduler, "set_timesteps"):
            return
        self._scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    def _get_transformer_blocks(self) -> list[torch.nn.Module]:
        blocks = getattr(self._transformer, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            raise RuntimeError("Loaded Wan transformer does not expose blocks ModuleList.")
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

    def _align_temporal_length(self, clips: torch.Tensor) -> torch.Tensor:
        scale = int(getattr(getattr(self._vae, "config", None), "scale_factor_temporal", 4))
        scale = max(scale, 1)
        num_frames = int(clips.shape[2])
        if num_frames % scale == 1:
            return clips

        target_frames = (num_frames // scale) * scale + 1
        if target_frames < num_frames:
            target_frames += scale
        extra_frames = target_frames - num_frames
        last_frame = clips[:, :, -1:, :, :].expand(-1, -1, extra_frames, -1, -1)
        return torch.cat([clips, last_frame], dim=2)

    def _encode_prompt(self) -> torch.Tensor:
        tokens = self._tokenizer(
            [""],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            outputs = self._text_encoder(input_ids, attention_mask)

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("Wan text encoder did not return last_hidden_state.")

        prompt_embeds = hidden_states.to(device=self.device, dtype=self.model_dtype)
        rows = []
        for row, seq_len in zip(prompt_embeds, seq_lens):
            used = row[: int(seq_len.item())]
            pad_len = self.max_sequence_length - used.shape[0]
            if pad_len > 0:
                used = torch.cat([used, used.new_zeros(pad_len, used.shape[1])], dim=0)
            rows.append(used)
        return torch.stack(rows, dim=0)

    def _expand_prompt_batch(self, batch_size: int) -> torch.Tensor:
        return self._prompt_embeds.expand(batch_size, -1, -1).contiguous()

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
            raise RuntimeError("Wan VAE encode() did not return a latent distribution with mode().")

        latents = latent_dist.mode()
        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(f"Wan VAE latent mode() returned non-tensor value: {type(latents)!r}")
        return self._normalize_wan_latents(latents)

    def _normalize_wan_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_cfg = getattr(self._vae, "config", None)
        latents_mean = getattr(vae_cfg, "latents_mean", None)
        latents_std = getattr(vae_cfg, "latents_std", None)
        z_dim = int(getattr(vae_cfg, "z_dim", latents.shape[1]))
        if latents_mean is None or latents_std is None:
            raise RuntimeError("Loaded Wan VAE config is missing latents_mean/latents_std.")

        mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(latents_std, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        return (latents - mean) / std

    def _reference_noise(self, latents: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.noise_seed)
        base_noise = torch.randn((1, *latents.shape[1:]), generator=generator, dtype=torch.float32)
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
        if not hasattr(self._scheduler, "add_noise"):
            raise RuntimeError(
                "Wan scheduler does not expose add_noise(); "
                f"got scheduler type {type(self._scheduler)!r}."
            )

        scaled = self._scheduler.add_noise(latents, noise, timestep)
        if not isinstance(scaled, torch.Tensor):
            raise RuntimeError(
                f"Wan scheduler add_noise() returned non-tensor value: {type(scaled)!r}"
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
                "Expected Wan latent tensor with shape [B, C, T, H, W], "
                f"got shape {tuple(noisy_latents.shape)}"
            )

        batch_size = int(noisy_latents.shape[0])
        prompt_embeds = self._expand_prompt_batch(batch_size)
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
                            f"Unexpected non-tensor output from Wan transformer block {layer_id}: {type(value)!r}"
                        )
                    captured[layer_id] = value

                return _hook

            handles.append(block.register_forward_hook(_make_hook(depth_layer)))

        timestep = torch.full(
            (batch_size,),
            float(timestep_value),
            device=self.device,
            dtype=noisy_latents.dtype,
        )

        try:
            with torch.no_grad():
                self._transformer(
                    hidden_states=noisy_latents.to(dtype=self.model_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        missing_layers = [layer for layer in depth_layers if layer not in captured]
        if missing_layers:
            raise RuntimeError(
                "Failed to capture all requested Wan transformer block activations. "
                f"Missing layers: {missing_layers}"
            )
        return captured

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

        if isinstance(scheduler_timesteps, torch.Tensor) and scheduler_timesteps.numel() > 0:
            timesteps = scheduler_timesteps.detach().to(device="cpu", dtype=torch.float32).flatten()
            resolved_timesteps = []
            resolved_sigmas = []
            last_index = max(int(timesteps.numel()) - 1, 0)
            for noise_fraction in noise_levels:
                index = int(round((1.0 - float(noise_fraction)) * last_index))
                index = max(0, min(last_index, index))
                resolved_timesteps.append(float(timesteps[index].item()))
                resolved_sigmas.append(float(noise_fraction))
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
        return wan_diffusion_preprocessing_metadata(
            normalize_input=self.normalize_input,
            noise_levels=self.noise_levels,
            prompt_mode="empty_string",
            noise_policy="fixed_reference_noise",
            frame_policy="pad_repeat_last_to_4k_plus_1",
        )

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
        vae_cfg = getattr(self._vae, "config", None)
        transformer_cfg = getattr(self._transformer, "config", None)
        return {
            "hf_model_id": self.hf_model_id,
            "config_path": str(self.config_path),
            "variant": self.variant,
            "model_name": self.model_name,
            "extract_source": "diffusion_transformer_blocks",
            "torch_dtype": str(self.model_dtype).replace("torch.", ""),
            "normalize_input": self.normalize_input,
            "noise_seed": self.noise_seed,
            "noise_levels": [float(v) for v in self.noise_levels],
            "noise_sigmas": [float(v) for v in self._noise_sigmas],
            "noise_timesteps": [float(v) for v in self._noise_timesteps],
            "num_inference_steps": int(self.num_inference_steps),
            "max_sequence_length": int(self.max_sequence_length),
            "patch_size": list(getattr(transformer_cfg, "patch_size", [])),
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
            "spatial_compression_ratio": int(getattr(vae_cfg, "scale_factor_spatial", 1)),
            "temporal_compression_ratio": int(getattr(vae_cfg, "scale_factor_temporal", 1)),
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

        specs_by_noise_index: dict[int, list[WanProbeLayerSpec]] = defaultdict(list)
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
                        "Expected Wan transformer hidden states with shape [B, N, D], "
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


def create_wan_video_adapter(**kwargs: Any) -> WanVideoAdapter:
    return WanVideoAdapter(**kwargs)


register_adapter("wan_video", create_wan_video_adapter, replace=True)
