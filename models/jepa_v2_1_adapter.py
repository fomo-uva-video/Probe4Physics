from __future__ import annotations

"""
Thin wrapper adapter around official V-JEPA v2.1 code.

V-JEPA 2.1 lives in ``third_party/vjepa2/app/vjepa_2_1/`` and shares the same
``src.*`` import namespace as V-JEPA 2 (both repos root at
``third_party/vjepa2``).  The two adapters therefore use the same namespace
guard string (``"jepa_v2"``) and can coexist in one process, but neither can
coexist with V-JEPA 1.

**Layer selection for V-JEPA 2.1**

The V-JEPA 2.1 ViT defines a fixed set of *hierarchical layers* at exactly the
25 / 50 / 75 / 100 % positions of the network depth (matching
``default_relative_depths: [0.25, 0.5, 0.75, 1.0]``):

    depth=12  → [3, 6, 9, 12]   (0-based indices [2, 5, 8, 11])
    depth=24  → [6, 12, 18, 24] (0-based indices [5, 11, 17, 23])
    depth=40  → [10, 20, 30, 40](0-based indices [9, 19, 29, 39])
    depth=48  → [12, 24, 38, 48](0-based indices [11, 23, 37, 47])

``out_layers`` passed to the encoder constructor must be a subset of these
indices; arbitrary layer extraction is not supported by the V-JEPA 2.1 ViT.
The adapter always probes all four hierarchical layers; users can filter with
``layer_ids`` in ``extract()``.
"""

import copy
import importlib
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
from .jepa_v2_adapter import resolve_relative_depth_layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbones.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_jepa_v2_1_config(
    config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the ``jepa_v2_1`` section from ``configs/backbones.yaml``."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone config file not found: {path}. "
            "Expected global config at configs/backbones.yaml."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    cfg = payload.get("jepa_v2_1")
    if not isinstance(cfg, dict):
        raise ValueError("configs/backbones.yaml must define a 'jepa_v2_1' object.")
    return cfg


def _as_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_variant_bundle(
    cfg: dict[str, Any],
    *,
    variant: str | None,
    checkpoint_path: str | Path | None,
) -> tuple[str, dict[str, Any], Path]:
    """Pick the requested (or default) variant and resolve the checkpoint path."""

    variants = cfg.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("jepa_v2_1.variants must be a non-empty mapping.")

    checkpoints_dir = _as_project_path(
        cfg.get("checkpoints_dir", "data/checkpoints/jepa_v2_1")
    )

    chosen = str(variant).strip() if variant is not None else ""
    if not chosen:
        chosen = str(cfg.get("default_variant", "")).strip()
    if not chosen:
        raise ValueError(
            "jepa_v2_1.default_variant is required for zero-config adapter creation."
        )
    if chosen not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(f"Unknown jepa_v2_1 variant '{chosen}'. Known variants: {known}")

    variant_cfg = variants[chosen]
    if not isinstance(variant_cfg, dict):
        raise ValueError(f"Variant config for '{chosen}' must be a mapping.")

    if checkpoint_path is None:
        filename = str(variant_cfg.get("checkpoint_filename", "")).strip()
        if not filename:
            raise ValueError(f"jepa_v2_1 variant '{chosen}' is missing checkpoint_filename.")
        resolved_checkpoint = checkpoints_dir / filename
    else:
        resolved_checkpoint = _as_project_path(checkpoint_path)

    return chosen, variant_cfg, resolved_checkpoint


def _clean_checkpoint_keys(state_dict: dict) -> dict:
    """Remove ``module.`` and ``backbone.`` prefixes from checkpoint state-dict keys."""

    cleaned: dict = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------

def _import_vjepa2_1_vit_module(repo_root: Path) -> Any:
    """Import the V-JEPA 2.1 ViT module from ``third_party/vjepa2``."""

    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("app.vjepa_2_1.models.vision_transformer")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class JEPAV2_1Adapter(VideoBackboneAdapter):
    """Frozen-feature extractor for official V-JEPA v2.1 checkpoints.

    Inputs to ``extract`` are raw RGB clip tensors in [0, 1] with shape
    ``[B, C, T, H, W]``.  The adapter applies ImageNet normalization internally
    and returns per-layer tokens and mean-pooled clip embeddings through
    ``BackboneFeatures``.

    Feature extraction is fixed to the four hierarchical layers of the network
    (25 / 50 / 75 / 100 % depth).  These are the only positions at which the
    V-JEPA 2.1 ViT applies layer normalisation during a forward pass without
    masking.  Requesting arbitrary intermediate layers is not supported.

    V-JEPA 2.1 uses RoPE positional encoding (``use_rope=True``) and a
    modality embedding that distinguishes image from video tokens
    (``modality_embedding=True``).  The ``img_temporal_dim_size=1`` and
    ``interpolate_rope=True`` arguments match the official training setup.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        variant: str | None = None,
        repo_root: str | Path = "third_party/vjepa2",
        model_name: str | None = None,
        checkpoint_key: str | None = None,
        config_path: str | Path = DEFAULT_BACKBONES_CONFIG_PATH,
        device: str | torch.device = "cpu",
        crop_size: int | None = None,
        patch_size: int | None = None,
        frames_per_clip: int | None = None,
        tubelet_size: int | None = None,
        use_sdpa: bool = True,
        use_rope: bool = True,
        use_silu: bool = False,
        wide_silu: bool = True,
        uniform_power: bool = False,
        img_temporal_dim_size: int = 1,
        interpolate_rope: bool = True,
        modality_embedding: bool = True,
    ) -> None:
        # V-JEPA 2.1 shares the vjepa2 repo root → same src.* namespace as v2.
        enforce_single_jepa_namespace("jepa_v2")

        self.repo_root = _as_project_path(repo_root)
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.use_sdpa = bool(use_sdpa)
        self.use_rope = bool(use_rope)
        self.use_silu = bool(use_silu)
        self.wide_silu = bool(wide_silu)
        self.uniform_power = bool(uniform_power)
        self.img_temporal_dim_size = int(img_temporal_dim_size)
        self.interpolate_rope = bool(interpolate_rope)
        self.modality_embedding = bool(modality_embedding)

        cfg = _load_jepa_v2_1_config(self.config_path)
        self.variant, variant_cfg, resolved_checkpoint = _resolve_variant_bundle(
            cfg, variant=variant, checkpoint_path=checkpoint_path
        )

        self.checkpoint_path = resolved_checkpoint
        self.model_name = str(
            model_name or variant_cfg.get("model_name", "vit_gigantic_xformers")
        )
        self.checkpoint_key = str(
            checkpoint_key if checkpoint_key is not None
            else variant_cfg.get("checkpoint_key", "target_encoder")
        )
        self.crop_size = int(
            crop_size if crop_size is not None else variant_cfg.get("crop_size", 384)
        )
        self.patch_size = int(
            patch_size if patch_size is not None else variant_cfg.get("patch_size", 16)
        )
        self.frames_per_clip = int(
            frames_per_clip if frames_per_clip is not None
            else variant_cfg.get("frames_per_clip", 16)
        )
        self.tubelet_size = int(
            tubelet_size if tubelet_size is not None else variant_cfg.get("tubelet_size", 2)
        )

        raw_depths = cfg.get("model_block_depths")
        if not isinstance(raw_depths, dict) or not raw_depths:
            raise ValueError("jepa_v2_1.model_block_depths must be a non-empty mapping.")
        model_block_depths = {str(k): int(v) for k, v in raw_depths.items()}

        raw_rel = cfg.get("default_relative_depths")
        if not isinstance(raw_rel, list) or not raw_rel:
            raise ValueError("jepa_v2_1.default_relative_depths must be a non-empty list.")
        config_relative_depths = tuple(float(v) for v in raw_rel)

        # V-JEPA 2.1 is always probed at all four hierarchical layers (25/50/75/100% depth).
        self.selected_layers = resolve_relative_depth_layers(
            self.model_name,
            config_relative_depths,
            model_block_depths=model_block_depths,
            config_path=self.config_path,
        )

        # Convert 1-based user ids back to 0-based indices for the ViT constructor.
        self._out_layers_zero_indexed = tuple(layer - 1 for layer in self.selected_layers)

        self._validate_repo_layout()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"V-JEPA v2.1 checkpoint not found: {self.checkpoint_path}. "
                "Download the official checkpoint and place it under "
                f"{self.checkpoint_path.parent}."
            )

        vit_module = _import_vjepa2_1_vit_module(self.repo_root)

        encoder_ctor = vit_module.__dict__.get(self.model_name)
        if encoder_ctor is None:
            known = sorted(k for k in vit_module.__dict__ if k.startswith("vit_"))
            raise ValueError(
                f"Model '{self.model_name}' is not available in V-JEPA v2.1 "
                f"({self.repo_root / 'app' / 'vjepa_2_1' / 'models' / 'vision_transformer.py'}). "
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
            use_silu=self.use_silu,
            wide_silu=self.wide_silu,
            use_rope=self.use_rope,
            img_temporal_dim_size=self.img_temporal_dim_size,
            interpolate_rope=self.interpolate_rope,
            modality_embedding=self.modality_embedding,
        )
        encoder.to(self.device)

        try:
            checkpoint = torch.load(
                str(self.checkpoint_path), map_location="cpu", weights_only=False
            )
            try:
                pretrained_dict = checkpoint[self.checkpoint_key]
            except KeyError:
                pretrained_dict = checkpoint["encoder"]
            pretrained_dict = _clean_checkpoint_keys(pretrained_dict)

            # Discard shape-mismatched tensors (e.g. pos_embed absent with RoPE).
            for key, value in encoder.state_dict().items():
                if key not in pretrained_dict:
                    continue
                if pretrained_dict[key].shape != value.shape:
                    pretrained_dict[key] = copy.deepcopy(value)

            encoder.load_state_dict(pretrained_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load V-JEPA v2.1 checkpoint from "
                f"{self.checkpoint_path}. "
                "The checkpoint may be corrupted or partially downloaded. "
                "Please re-download the official file and try again."
            ) from exc

        self._encoder = encoder
        self._encoder.eval()
        for param in self._encoder.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_repo_layout(self) -> None:
        """Fail fast when the expected submodule files are missing."""

        if not self.repo_root.exists():
            raise FileNotFoundError(
                "Official V-JEPA v2 repository is missing at "
                f"{self.repo_root}. Run: git submodule update --init --recursive"
            )
        expected = [
            self.repo_root / "src" / "models" / "vision_transformer.py",
            self.repo_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Official V-JEPA v2 repository layout is incomplete. Missing files: "
                + ", ".join(missing)
            )

    def _resolve_requested_layers(
        self, layer_ids: Sequence[int] | None
    ) -> tuple[int, ...]:
        """Validate an optional layer subset against the hierarchical layers."""

        if layer_ids is None:
            return self.selected_layers

        requested = tuple(int(layer) for layer in layer_ids)
        if not requested:
            raise ValueError("layer_ids cannot be empty when provided.")

        unknown = sorted(set(requested) - set(self.selected_layers))
        if unknown:
            raise ValueError(
                f"Requested layer_ids {unknown} are not available. "
                f"V-JEPA 2.1 only supports hierarchical layers: {self.selected_layers}"
            )
        return requested

    def preprocessing_metadata(self) -> dict[str, Any]:
        """Return the raw-clip preprocessing contract used before forward."""

        return imagenet_preprocessing_metadata(family="jepa_v2_1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        clips: torch.Tensor,
        layer_ids: Sequence[int] | None = None,
    ) -> BackboneFeatures:
        """Run frozen forward pass and return token + pooled features by layer.

        Args:
            clips: Raw RGB video tensor in [0, 1] with shape ``[B, C, T, H, W]``.
            layer_ids: Optional 1-based subset of ``selected_layers`` to return.
                       Must be a subset of the four hierarchical layer ids.
                       When ``None``, all four hierarchical layers are returned.

        Returns:
            :class:`BackboneFeatures` with ``tokens_by_layer`` of shape
            ``[B, N, D]`` and ``pooled_by_layer`` of shape ``[B, D]``.
        """

        if not isinstance(clips, torch.Tensor):
            raise TypeError(f"clips must be a torch.Tensor, got {type(clips)!r}")
        if clips.ndim != 5:
            raise ValueError(
                f"Expected clips shape [B, C, T, H, W], got {tuple(clips.shape)}"
            )

        requested_layers = self._resolve_requested_layers(layer_ids)
        clips = clips.to(self.device, dtype=torch.float32)
        clips = normalize_rgb_imagenet(clips)

        with torch.no_grad():
            outputs = self._encoder(clips)

        # Encoder returns List[Tensor] when out_layers is set.
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) != len(self.selected_layers):
            raise RuntimeError(
                "Official encoder returned an unexpected number of layer outputs: "
                f"{len(outputs)} vs expected {len(self.selected_layers)}."
            )

        all_tokens: dict[int, torch.Tensor] = {
            layer: layer_tokens
            for layer, layer_tokens in zip(self.selected_layers, outputs)
        }
        # Optional layer filtering — avoids re-forwarding when only a subset is needed.
        tokens_by_layer = {layer: all_tokens[layer] for layer in requested_layers}
        # Mean pooling over token dimension → per-clip features [B, D].
        pooled_by_layer = {layer: t.mean(dim=1) for layer, t in tokens_by_layer.items()}

        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "variant": self.variant,
            "checkpoint_key": self.checkpoint_key,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "frames_per_clip": self.frames_per_clip,
            "crop_size": self.crop_size,
            "repo_root": str(self.repo_root),
            "preprocessing": self.preprocessing_metadata(),
        }
        return BackboneFeatures(
            tokens_by_layer=tokens_by_layer,
            pooled_by_layer=pooled_by_layer,
            selected_layers=requested_layers,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def create_jepa_v2_1_adapter(**kwargs: Any) -> JEPAV2_1Adapter:
    """Factory function used by the central adapter registry."""

    return JEPAV2_1Adapter(**kwargs)


# Register at import-time so callers can use ``create_adapter("jepa_v2_1", ...)``.
register_adapter("jepa_v2_1", create_jepa_v2_1_adapter, replace=True)
