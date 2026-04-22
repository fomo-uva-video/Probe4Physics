from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from models.videomae_adapter import (
    VideoMAEAdapter,
    VideoMAEV2Adapter,
    _materialize_videomaev2_plain_tensor_attrs,
    resolve_relative_depth_layers,
)
from models.registry import create_adapter, get_registered_adapters


# ---------------------------------------------------------------------------
# Fake HuggingFace model helpers
# ---------------------------------------------------------------------------

class _FakeVideoMAEOutput:
    """Minimal stand-in for HuggingFace model output with hidden_states."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


def _make_fake_hf_model(num_blocks: int, n_tokens: int = 4, d_model: int = 8):
    """Return a fake torch.nn.Module that mimics HuggingFace VideoMAEModel.

    ``hidden_states`` has ``num_blocks + 1`` elements (index 0 = patch
    embedding, indices 1..num_blocks = transformer block outputs).
    Each hidden state has shape ``[B, n_tokens, d_model]``.
    """

    class _FakeHFModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self,
            pixel_values: torch.Tensor,
            output_hidden_states: bool = False,
            **_: Any,
        ) -> _FakeVideoMAEOutput:
            batch = pixel_values.shape[0]
            states = tuple(
                torch.full((batch, n_tokens, d_model), float(i))
                for i in range(num_blocks + 1)
            )
            return _FakeVideoMAEOutput(hidden_states=states)

    return _FakeHFModel()


def _noop_load_hf_model(self: Any, hf_model_id: str, hf_cache_dir: Any) -> torch.nn.Module:
    """Patch for ``_load_hf_model`` (v1) — returns a fake VideoMAEModel-style model."""

    num_blocks = 12  # vit_base default for tests
    return _make_fake_hf_model(num_blocks)


# ---------------------------------------------------------------------------
# Fake VideoMAEv2 model helpers
#
# The real VideoMAEv2 HF model has structure: self._model.model.blocks
# (a ModuleList).  _get_blocks() navigates this path, so test fakes must
# mirror it.  Each block must actually be called during forward() so that
# the registered forward hooks fire correctly.
# ---------------------------------------------------------------------------

class _FakeV2Block(torch.nn.Module):
    """Single transformer block stub — passes hidden state through unchanged."""

    def __init__(self, n_tokens: int = 4, d_model: int = 8) -> None:
        super().__init__()
        self._n_tokens = n_tokens
        self._d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a fixed-shape tensor so hooks always capture [B, N, D].
        b = x.shape[0]
        return torch.zeros(b, self._n_tokens, self._d_model)


class _FakeV2Inner(torch.nn.Module):
    """Inner model with a ``blocks`` ModuleList, called sequentially in forward."""

    def __init__(self, num_blocks: int = 12, n_tokens: int = 4, d_model: int = 8) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_FakeV2Block(n_tokens, d_model) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class _FakeV2HFModel(torch.nn.Module):
    """Mimics the VideoMAEv2 HF wrapper: ``self.model`` → ``_FakeV2Inner``.

    Stores the shape of ``pixel_values`` passed to ``forward`` so tests can
    verify whether the adapter applied an input permutation or not.
    """

    def __init__(self, num_blocks: int = 12, n_tokens: int = 4, d_model: int = 8) -> None:
        super().__init__()
        self.model = _FakeV2Inner(num_blocks, n_tokens, d_model)
        self.last_pixel_values_shape: tuple[int, ...] | None = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.last_pixel_values_shape = tuple(pixel_values.shape)
        return self.model(pixel_values)


class _FakeV2HFModelMetaPosEmbed(torch.nn.Module):
    def __init__(self, num_blocks: int = 12, n_tokens: int = 4, d_model: int = 8) -> None:
        super().__init__()
        self.model = _FakeV2Inner(num_blocks, n_tokens, d_model)
        self.model.embed_dim = d_model
        self.model.patch_embed = types.SimpleNamespace(num_patches=n_tokens)
        self.model.pos_embed = torch.empty((1, n_tokens, d_model), device="meta")


def _noop_load_hf_model_v2(
    self: Any, hf_model_id: str, hf_cache_dir: Any
) -> torch.nn.Module:
    """Patch for ``_load_hf_model`` (v2) — returns a fake model with correct structure."""

    return _FakeV2HFModel(num_blocks=12)


def _load_hf_model_v2_meta_failure(
    self: Any, hf_model_id: str, hf_cache_dir: Any
) -> torch.nn.Module:
    class _MetaModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.meta_weight = torch.nn.Parameter(
                torch.empty(1, device="meta"),
            )

    self._load_debug_context = {
        "backbone_key": "videomae_v2",
        "hf_model_id": hf_model_id,
        "hf_cache_dir": str(hf_cache_dir or "<default>"),
        "load_strategy": "unit_test_meta_failure",
    }
    return _MetaModel()


def _make_minimal_config(tmp: Path, backbone_key: str, model_name: str = "vit_base") -> Path:
    """Write a minimal backbones.yaml with a single variant for testing."""

    block_depths = {"vit_base": 12, "vit_large": 24}
    hf_ids = {
        "videomae": "MCG-NJU/videomae-base",
        "videomae_v2": "OpenGVLab/VideoMAEv2-Base",
    }
    config_path = tmp / "backbones.yaml"
    import yaml

    payload = {
        backbone_key: {
            "default_variant": "vit_base_16_224",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": block_depths,
            "variants": {
                "vit_base_16_224": {
                    "hf_model_id": hf_ids[backbone_key],
                    "model_name": model_name,
                    "frames_per_clip": 16,
                    "crop_size": 224,
                    "patch_size": 16,
                    "tubelet_size": 2,
                }
            },
        }
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Layer mapping tests
# ---------------------------------------------------------------------------

class VideoMAELayerMappingTests(unittest.TestCase):
    """Verify relative-depth-to-block-id mapping for each supported model size."""

    def test_vit_base_default_depths(self) -> None:
        # 12 blocks × [0.25, 0.5, 0.75, 1.0] → [3, 6, 9, 12]
        result = resolve_relative_depth_layers("vit_base")
        self.assertEqual(result, (3, 6, 9, 12))

    def test_vit_large_default_depths(self) -> None:
        # 24 blocks × [0.25, 0.5, 0.75, 1.0] → [6, 12, 18, 24]
        result = resolve_relative_depth_layers("vit_large")
        self.assertEqual(result, (6, 12, 18, 24))

    def test_vit_huge_default_depths(self) -> None:
        # 32 blocks × [0.25, 0.5, 0.75, 1.0] → [8, 16, 24, 32]
        result = resolve_relative_depth_layers("vit_huge")
        self.assertEqual(result, (8, 16, 24, 32))

    def test_custom_relative_depths(self) -> None:
        block_depths = {"vit_base": 12}
        result = resolve_relative_depth_layers(
            "vit_base",
            relative_depths=[1.0],
            model_block_depths=block_depths,
        )
        self.assertEqual(result, (12,))

    def test_unknown_model_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "vit_unknown",
                relative_depths=[0.5],
                model_block_depths={"vit_base": 12},
            )

    def test_invalid_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "vit_base",
                relative_depths=[0.0],  # must be > 0
                model_block_depths={"vit_base": 12},
            )

    def test_videomae_v2_vit_giant_depths(self) -> None:
        # 40 blocks × [0.25, 0.5, 0.75, 1.0] → [10, 20, 30, 40]
        result = resolve_relative_depth_layers(
            "vit_giant",
            relative_depths=[0.25, 0.5, 0.75, 1.0],
            model_block_depths={"vit_giant": 40},
        )
        self.assertEqual(result, (10, 20, 30, 40))


# ---------------------------------------------------------------------------
# VideoMAE v1 adapter tests
# ---------------------------------------------------------------------------

class VideoMAEAdapterTests(unittest.TestCase):
    """Tests for VideoMAEAdapter (v1 — MCG-NJU/videomae-*)."""

    def _make_adapter(self, config_path: Path) -> VideoMAEAdapter:
        with mock.patch.object(VideoMAEAdapter, "_load_hf_model", _noop_load_hf_model):
            return VideoMAEAdapter(config_path=config_path)

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with mock.patch.object(VideoMAEAdapter, "_load_hf_model", _noop_load_hf_model):
                VideoMAEAdapter(config_path=Path("/tmp/does_not_exist_backbones.yaml"))

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            with self.assertRaises(ValueError):
                with mock.patch.object(VideoMAEAdapter, "_load_hf_model", _noop_load_hf_model):
                    VideoMAEAdapter(config_path=config_path, variant="nonexistent_variant")

    def test_extract_shape_contract(self) -> None:
        """tokens [B, N, D] and pooled [B, D] per selected layer."""

        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            adapter = self._make_adapter(config_path)

            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

        self.assertEqual(features.selected_layers, (3, 6, 9, 12))
        for layer_id in features.selected_layers:
            tokens = features.tokens_by_layer[layer_id]
            pooled = features.pooled_by_layer[layer_id]
            self.assertEqual(tokens.ndim, 3, f"layer {layer_id} tokens should be [B, N, D]")
            self.assertEqual(pooled.ndim, 2, f"layer {layer_id} pooled should be [B, D]")
            self.assertEqual(tokens.shape[0], 2)
            self.assertEqual(pooled.shape[0], 2)
            self.assertEqual(tokens.shape[2], pooled.shape[1])

    def test_extract_layer_subset(self) -> None:
        """Requesting a subset of layers should return only those layers."""

        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            adapter = self._make_adapter(config_path)

            clips = torch.randn(1, 3, 16, 224, 224)
            features = adapter.extract(clips, layer_ids=[6, 12])

        self.assertEqual(features.selected_layers, (6, 12))
        self.assertEqual(set(features.tokens_by_layer), {6, 12})
        self.assertNotIn(3, features.tokens_by_layer)
        self.assertNotIn(9, features.tokens_by_layer)

    def test_extract_invalid_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            adapter = self._make_adapter(config_path)

            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 224, 224), layer_ids=[99])

    def test_extract_wrong_input_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            adapter = self._make_adapter(config_path)

            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(2, 3, 224, 224))  # 4-D, not 5-D

    def test_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            adapter = self._make_adapter(config_path)

            features = adapter.extract(torch.randn(1, 3, 16, 224, 224))

        for key in ("model_name", "hf_model_id", "config_path", "patch_size", "frames_per_clip"):
            self.assertIn(key, features.metadata, f"metadata missing key: {key}")

    def test_input_permutation_is_applied(self) -> None:
        """VideoMAEAdapter must permute [B,C,T,H,W] → [B,T,C,H,W] before forward."""

        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")

            received_shapes: list[tuple[int, ...]] = []

            class _ShapeCapturingModel(torch.nn.Module):
                def __init__(inner_self) -> None:
                    super().__init__()

                def forward(inner_self, pixel_values, output_hidden_states=False, **_):
                    received_shapes.append(tuple(pixel_values.shape))
                    b = pixel_values.shape[0]
                    states = tuple(
                        torch.zeros(b, 4, 8) for _ in range(13)  # 12 blocks + 1
                    )
                    return _FakeVideoMAEOutput(hidden_states=states)

            def _capturing_load(self, hf_model_id, hf_cache_dir):
                return _ShapeCapturingModel()

            with mock.patch.object(VideoMAEAdapter, "_load_hf_model", _capturing_load):
                adapter = VideoMAEAdapter(config_path=config_path)

            clips = torch.randn(2, 3, 16, 224, 224)  # [B, C, T, H, W]
            adapter.extract(clips)

        self.assertEqual(len(received_shapes), 1)
        b, t, c, h, w = received_shapes[0]
        self.assertEqual((b, t, c, h, w), (2, 16, 3, 224, 224),
                         "VideoMAE v1 adapter must permute to [B, T, C, H, W]")


# ---------------------------------------------------------------------------
# VideoMAE v2 adapter tests
# ---------------------------------------------------------------------------

class VideoMAEV2AdapterTests(unittest.TestCase):
    """Tests for VideoMAEV2Adapter (OpenGVLab/VideoMAEv2-*)."""

    def _make_adapter(self, config_path: Path) -> VideoMAEV2Adapter:
        with mock.patch.object(VideoMAEV2Adapter, "_load_hf_model", _noop_load_hf_model_v2):
            return VideoMAEV2Adapter(config_path=config_path)

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with mock.patch.object(VideoMAEV2Adapter, "_load_hf_model", _noop_load_hf_model_v2):
                VideoMAEV2Adapter(config_path=Path("/tmp/does_not_exist_backbones.yaml"))

    def test_extract_shape_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")
            adapter = self._make_adapter(config_path)

            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

        self.assertEqual(features.selected_layers, (3, 6, 9, 12))
        for layer_id in features.selected_layers:
            tokens = features.tokens_by_layer[layer_id]
            pooled = features.pooled_by_layer[layer_id]
            self.assertEqual(tokens.ndim, 3)
            self.assertEqual(pooled.ndim, 2)
            self.assertEqual(tokens.shape[0], 2)
            self.assertEqual(pooled.shape[0], 2)

    def test_extract_layer_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")
            adapter = self._make_adapter(config_path)

            clips = torch.randn(1, 3, 16, 224, 224)
            features = adapter.extract(clips, layer_ids=[9])

        self.assertEqual(features.selected_layers, (9,))
        self.assertEqual(set(features.tokens_by_layer), {9})

    def test_no_permutation_applied(self) -> None:
        """VideoMAEv2 must NOT permute input — it already expects [B,C,T,H,W]."""

        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")

            with mock.patch.object(VideoMAEV2Adapter, "_load_hf_model", _noop_load_hf_model_v2):
                adapter = VideoMAEV2Adapter(config_path=config_path)

            clips = torch.randn(2, 3, 16, 224, 224)  # [B, C, T, H, W]
            adapter.extract(clips)

        # _FakeV2HFModel.forward() records the shape it received.
        received = adapter._model.last_pixel_values_shape
        self.assertIsNotNone(received)
        b, c, t, h, w = received
        self.assertEqual((b, c, t, h, w), (2, 3, 16, 224, 224),
                         "VideoMAEv2 adapter must NOT permute — model expects [B, C, T, H, W]")

    def test_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")
            adapter = self._make_adapter(config_path)

            features = adapter.extract(torch.randn(1, 3, 16, 224, 224))

        for key in ("model_name", "hf_model_id", "config_path", "patch_size", "frames_per_clip"):
            self.assertIn(key, features.metadata, f"metadata missing key: {key}")

    def test_meta_tensor_failure_message_includes_load_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")
            with mock.patch.object(
                VideoMAEV2Adapter,
                "_load_hf_model",
                _load_hf_model_v2_meta_failure,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    VideoMAEV2Adapter(config_path=config_path)

        message = str(ctx.exception)
        self.assertIn("VideoMAEV2Adapter failed during post_load_pre_to_device", message)
        self.assertIn("load_strategy=unit_test_meta_failure", message)
        self.assertIn("hf_model_id=OpenGVLab/VideoMAEv2-Base", message)
        self.assertIn("meta_parameter_count=1", message)

    def test_materialize_plain_meta_pos_embed_repairs_tensor(self) -> None:
        model = _FakeV2HFModelMetaPosEmbed()
        config = types.SimpleNamespace(model_config={"num_frames": 16})
        model_module = types.SimpleNamespace(
            get_sinusoid_encoding_table=lambda n_position, d_hid: torch.zeros(1, n_position, d_hid),
        )

        repaired = _materialize_videomaev2_plain_tensor_attrs(
            model,
            config=config,
            model_module=model_module,
        )

        self.assertEqual(repaired, ["model.pos_embed"])
        self.assertFalse(model.model.pos_embed.is_meta)
        self.assertEqual(tuple(model.model.pos_embed.shape), (1, 4, 8))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class VideoMAERegistryTests(unittest.TestCase):
    """Verify both adapters are reachable through the central registry."""

    def test_videomae_is_registered(self) -> None:
        self.assertIn("videomae", get_registered_adapters())

    def test_videomae_v2_is_registered(self) -> None:
        self.assertIn("videomae_v2", get_registered_adapters())

    def test_registry_creates_videomae_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae")
            with mock.patch.object(VideoMAEAdapter, "_load_hf_model", _noop_load_hf_model):
                adapter = create_adapter("videomae", config_path=config_path)
        self.assertIsInstance(adapter, VideoMAEAdapter)

    def test_registry_creates_videomae_v2_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp), "videomae_v2")
            with mock.patch.object(VideoMAEV2Adapter, "_load_hf_model", _noop_load_hf_model_v2):
                adapter = create_adapter("videomae_v2", config_path=config_path)
        self.assertIsInstance(adapter, VideoMAEV2Adapter)


if __name__ == "__main__":
    unittest.main()
