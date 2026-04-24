from __future__ import annotations

"""
Thin wrapper adapter around official V-JEPA v1 code.
"""

import importlib
import copy
import sys
import yaml
from pathlib import Path
from typing import Any, Sequence
import torch
from .registry import (
    BackboneFeatures,
    VideoBackboneAdapter,
    enforce_single_jepa_namespace,
    register_adapter,
)
from .preprocessing import imagenet_preprocessing_metadata, normalize_rgb_imagenet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"


def _load_jepa_v1_config(config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH) -> dict[str, Any]:
    """Load global JEPAV1 config from `configs/backbones.yaml`."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone config file not found: {path}. "
            "Expected global config at configs/backbones.yaml."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    jepa_cfg = payload.get("jepa_v1")
    if not isinstance(jepa_cfg, dict):
        raise ValueError("configs/backbones.yaml must define a 'jepa_v1' object.")

    return jepa_cfg


def _as_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_variant_bundle(
    jepa_cfg: dict[str, Any],
    *,
    variant: str | None,
    checkpoint_path: str | Path | None,
) -> tuple[str, dict[str, Any], Path]:
    """Resolve plug-and-play variant settings and checkpoint path."""

    variants = jepa_cfg.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("jepa_v1.variants must be a non-empty mapping.")

    checkpoints_dir_raw = jepa_cfg.get("checkpoints_dir", "data/checkpoints/jepa_v1")
    checkpoints_dir = _as_project_path(checkpoints_dir_raw)

    chosen_variant = str(variant).strip() if variant is not None else ""
    if not chosen_variant:
        chosen_variant = str(jepa_cfg.get("default_variant", "")).strip()

    if not chosen_variant:
        raise ValueError("jepa_v1.default_variant is required for zero-config adapter creation.")

    if chosen_variant not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(f"Unknown jepa_v1 variant '{chosen_variant}'. Known variants: {known}")

    variant_cfg = variants[chosen_variant]
    if not isinstance(variant_cfg, dict):
        raise ValueError(f"Variant config for '{chosen_variant}' must be a mapping.")

    if checkpoint_path is None:
        filename = str(variant_cfg.get("checkpoint_filename", "")).strip()
        if not filename:
            raise ValueError(f"Variant '{chosen_variant}' is missing checkpoint_filename.")
        resolved_checkpoint = checkpoints_dir / filename
    else:
        resolved_checkpoint = _as_project_path(checkpoint_path)

    return chosen_variant, variant_cfg, resolved_checkpoint


def resolve_relative_depth_layers(
    model_name: str,
    relative_depths: Sequence[float] | None = None,
    *,
    model_block_depths: dict[str, int] | None = None,
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> tuple[int, ...]:
    """Map relative layer depths (e.g., 0.25, 0.5...) to 1-based block ids."""

    if relative_depths is None or model_block_depths is None:
        jepa_cfg = _load_jepa_v1_config(config_path)
        raw_relative_depths = jepa_cfg.get("default_relative_depths")
        if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
            raise ValueError("jepa_v1.default_relative_depths must be a non-empty list.")
        cfg_relative_depths = tuple(float(value) for value in raw_relative_depths)

        raw_model_depths = jepa_cfg.get("model_block_depths")
        if not isinstance(raw_model_depths, dict) or not raw_model_depths:
            raise ValueError("jepa_v1.model_block_depths must be a non-empty mapping.")
        cfg_model_block_depths = {str(key): int(value) for key, value in raw_model_depths.items()}

        if relative_depths is None:
            relative_depths = cfg_relative_depths
        if model_block_depths is None:
            model_block_depths = cfg_model_block_depths

    if model_name not in model_block_depths:
        known = ", ".join(sorted(model_block_depths))
        raise ValueError(f"Unsupported model_name='{model_name}'. Known: {known}")

    if not relative_depths:
        raise ValueError("relative_depths cannot be empty.")

    depth = model_block_depths[model_name]
    resolved: list[int] = []
    for value in relative_depths:
        current = float(value)
        if not (0.0 < current <= 1.0):
            raise ValueError(f"Invalid relative depth: {value}. Expected values in (0, 1].")
        block_id = int(round(depth * current))
        block_id = max(1, min(depth, block_id))
        if block_id not in resolved:
            resolved.append(block_id)

    return tuple(resolved)


def _import_official_jepa_modules(repo_root: Path) -> Any:
    """Import official V-JEPA v1 ViT module from a specific local repo root."""

    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        # Prepend official repo path so imports resolve against pinned submodule.
        sys.path.insert(0, root_str)

    vit_module = importlib.import_module("src.models.vision_transformer")
    return vit_module


def _load_pretrained_official(
    *,
    encoder: torch.nn.Module,
    pretrained: str | Path,
    checkpoint_key: str = "target_encoder",
) -> torch.nn.Module:
    """Load checkpoint with official V-JEPA v1 fallback and key-cleanup rules.

    This mirrors the official `load_pretrained` behavior while avoiding import of
    the full official eval module (which pulls optional cv2/torchvision I/O deps).
    """

    checkpoint = torch.load(str(pretrained), map_location="cpu", weights_only=False)
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}

    # Keep model tensors when checkpoint tensors have mismatched shapes.
    for key, value in encoder.state_dict().items():
        if key not in pretrained_dict:
            continue
        if pretrained_dict[key].shape != value.shape:
            pretrained_dict[key] = copy.deepcopy(value)

    encoder.load_state_dict(pretrained_dict, strict=False)
    return encoder


class JEPAV1Adapter(VideoBackboneAdapter):
    """Frozen-feature extractor for official V-JEPA v1 checkpoints.

    Inputs to `extract` are raw RGB clip tensors in [0, 1] with shape
    [B, C, T, H, W]. The adapter applies ImageNet normalization internally and returns per-layer tokens and mean-pooled
    clip embeddings through `BackboneFeatures`.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        variant: str | None = None,
        repo_root: str | Path = "third_party/jepa_v1",
        model_name: str | None = None,
        checkpoint_key: str = "target_encoder",
        relative_depths: Sequence[float] | None = None,
        config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
        device: str | torch.device = "cpu",
        crop_size: int | None = None,
        patch_size: int | None = None,
        frames_per_clip: int | None = None,
        tubelet_size: int | None = None,
        use_sdpa: bool = True,
        use_silu: bool = False,
        tight_silu: bool = True,
        uniform_power: bool = False,
    ) -> None:
        # Guard against loading multiple JEPA family repos in one process.
        enforce_single_jepa_namespace("jepa_v1")

        self.repo_root = _as_project_path(repo_root)
        self.checkpoint_key = str(checkpoint_key)
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.use_sdpa = bool(use_sdpa)
        self.use_silu = bool(use_silu)
        self.tight_silu = bool(tight_silu)
        self.uniform_power = bool(uniform_power)

        jepa_cfg = _load_jepa_v1_config(self.config_path)
        self.variant, variant_cfg, resolved_checkpoint_path = _resolve_variant_bundle(
            jepa_cfg,
            variant=variant,
            checkpoint_path=checkpoint_path,
        )

        self.checkpoint_path = resolved_checkpoint_path
        self.model_name = str(model_name or variant_cfg.get("model_name", "vit_large"))
        self.crop_size = int(crop_size if crop_size is not None else variant_cfg.get("crop_size", 224))
        self.patch_size = int(patch_size if patch_size is not None else variant_cfg.get("patch_size", 16))
        self.frames_per_clip = int(
            frames_per_clip if frames_per_clip is not None else variant_cfg.get("frames_per_clip", 16)
        )
        self.tubelet_size = int(
            tubelet_size if tubelet_size is not None else variant_cfg.get("tubelet_size", 2)
        )

        raw_relative_depths = jepa_cfg.get("default_relative_depths")
        if not isinstance(raw_relative_depths, list) or not raw_relative_depths:
            raise ValueError("jepa_v1.default_relative_depths must be a non-empty list.")
        config_relative_depths = tuple(float(value) for value in raw_relative_depths)

        raw_model_depths = jepa_cfg.get("model_block_depths")
        if not isinstance(raw_model_depths, dict) or not raw_model_depths:
            raise ValueError("jepa_v1.model_block_depths must be a non-empty mapping.")
        model_block_depths = {str(key): int(value) for key, value in raw_model_depths.items()}

        if relative_depths is None:
            relative_depths = config_relative_depths

        # User-visible layer ids are 1-based to match transformer block numbering.
        self.selected_layers = resolve_relative_depth_layers(
            self.model_name,
            relative_depths,
            model_block_depths=model_block_depths,
            config_path=self.config_path,
        )
        # Official ViT implementation expects zero-based `out_layers`.
        self._out_layers_zero_indexed = tuple(layer - 1 for layer in self.selected_layers)

        self._validate_repo_layout()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"V-JEPA v1 checkpoint not found: {self.checkpoint_path}")

        vit_module = _import_official_jepa_modules(self.repo_root)

        encoder_ctor = vit_module.__dict__.get(self.model_name)
        if encoder_ctor is None:
            known = sorted(key for key in vit_module.__dict__ if key.startswith("vit_"))
            raise ValueError(
                f"Model '{self.model_name}' is not available in official V-JEPA v1. "
                f"Known constructors: {known}"
                )

        encoder = encoder_ctor(
            img_size=self.crop_size,
            patch_size=self.patch_size,
            num_frames=self.frames_per_clip,
            tubelet_size=self.tubelet_size,
            out_layers=list(self._out_layers_zero_indexed),
            uniform_power=self.uniform_power,
            use_sdpa=self.use_sdpa,
            use_SiLU=self.use_silu,
            tight_SiLU=self.tight_silu,
        )
        encoder.to(self.device)

        # Reuse official checkpoint loading path (key handling + fallback/cleanup).
        try:
            self._encoder = _load_pretrained_official(
                encoder=encoder,
                pretrained=str(self.checkpoint_path),
                checkpoint_key=self.checkpoint_key,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load V-JEPA v1 checkpoint from "
                f"{self.checkpoint_path}. "
                "The checkpoint may be corrupted or partially downloaded. "
                "Please re-download the official file and try again."
            ) from exc
        self._encoder.eval()
        for parameter in self._encoder.parameters():
            parameter.requires_grad = False

    def _validate_repo_layout(self) -> None:
        """Fail fast if official submodule files are missing."""

        if not self.repo_root.exists():
            raise FileNotFoundError(
                "Official V-JEPA v1 repository is missing at "
                f"{self.repo_root}. Run: git submodule update --init --recursive"
            )

        expected_files = [
            self.repo_root / "src" / "models" / "vision_transformer.py",
        ]
        missing = [str(path) for path in expected_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Official V-JEPA v1 repository layout is incomplete. Missing files: "
                + ", ".join(missing)
                )

    def _resolve_requested_layers(self, layer_ids: Sequence[int] | None) -> tuple[int, ...]:
        """Validate optional layer subset request against configured layers."""

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

    def preprocessing_metadata(self) -> dict[str, Any]:
        """Return the raw-clip preprocessing contract used before forward."""

        return imagenet_preprocessing_metadata(family="jepa_v1")

    def extract(
        self,
        clips: torch.Tensor,
        layer_ids: Sequence[int] | None = None,
    ) -> BackboneFeatures:
        """Run frozen forward pass and return token + pooled features by layer."""

        if not isinstance(clips, torch.Tensor):
            raise TypeError(f"clips must be a torch.Tensor, got {type(clips)!r}")
        if clips.ndim != 5:
            raise ValueError(f"Expected clips shape [B, C, T, H, W], got {tuple(clips.shape)}")

        requested_layers = self._resolve_requested_layers(layer_ids)
        clips = clips.to(self.device, dtype=torch.float32)
        clips = normalize_rgb_imagenet(clips)

        with torch.no_grad():
            outputs = self._encoder(clips)

        # Official encoder can return Tensor or List[Tensor] depending on config.
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) != len(self.selected_layers):
            raise RuntimeError(
                "Official encoder returned unexpected number of layer outputs: "
                f"{len(outputs)} vs expected {len(self.selected_layers)}."
            )

        all_tokens = {
            layer: layer_tokens
            for layer, layer_tokens in zip(self.selected_layers, outputs)
        }
        # Optional layer filtering lets probes request a subset without re-forwarding.
        tokens_by_layer = {layer: all_tokens[layer] for layer in requested_layers}
        # Mean pooling on token dimension creates per-clip features [B, D].
        pooled_by_layer = {layer: token_values.mean(dim=1) for layer, token_values in tokens_by_layer.items()}
        metadata = {
            "repo_root": str(self.repo_root),
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "variant": self.variant,
            "model_name": self.model_name,
            "checkpoint_key": self.checkpoint_key,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
            "preprocessing": self.preprocessing_metadata(),
        }
        return BackboneFeatures(
            tokens_by_layer=tokens_by_layer,
            pooled_by_layer=pooled_by_layer,
            selected_layers=requested_layers,
            metadata=metadata,
        )


def create_jepa_v1_adapter(**kwargs: Any) -> JEPAV1Adapter:
    """Factory function used by the central adapter registry."""

    return JEPAV1Adapter(**kwargs)


# Register at import-time so callers can use `create_adapter("jepa_v1", ...)`.
register_adapter("jepa_v1", create_jepa_v1_adapter, replace=True)




