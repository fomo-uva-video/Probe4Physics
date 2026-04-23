from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from models.ltx_video_adapter import LTXVideoAdapter, resolve_relative_depth_layers
from models.registry import create_adapter, get_registered_adapters


class _FakeStage(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SimpleNamespace(
            down_blocks=torch.nn.ModuleList(
                [_FakeStage(1.0), _FakeStage(2.0), _FakeStage(3.0), _FakeStage(4.0)]
            ),
            mid_block=_FakeStage(5.0),
        )
        self.spatial_compression_ratio = 8
        self.temporal_compression_ratio = 8
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor):
        hidden = x
        for block in self.encoder.down_blocks:
            hidden = block(hidden)
        hidden = self.encoder.mid_block(hidden)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: hidden))


def _fake_load_vae(self: LTXVideoAdapter) -> torch.nn.Module:
    return _FakeVAE()


def _make_minimal_config(tmp: Path) -> Path:
    config_path = tmp / "backbones.yaml"
    payload = {
        "ltx_video": {
            "default_variant": "ltxv_13b_0_9_8_dev",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {
                "ltx_vae_5": 5,
            },
            "variants": {
                "ltxv_13b_0_9_8_dev": {
                    "hf_model_id": "Lightricks/LTX-Video-0.9.8-dev",
                    "model_name": "ltx_vae_5",
                    "vae_subfolder": "vae",
                    "frames_per_clip": 16,
                    "crop_size": 224,
                    "patch_size": 4,
                    "patch_size_t": 1,
                    "torch_dtype": "float32",
                }
            },
        }
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


class LTXVideoLayerMappingTests(unittest.TestCase):
    def test_default_depth_mapping(self) -> None:
        result = resolve_relative_depth_layers(
            "ltx_vae_5",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"ltx_vae_5": 5},
        )
        self.assertEqual(result, (1, 2, 4, 5))

    def test_invalid_relative_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "ltx_vae_5",
                [0.0],
                model_block_depths={"ltx_vae_5": 5},
            )


class LTXVideoAdapterTests(unittest.TestCase):
    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                LTXVideoAdapter(config_path=Path("/tmp/missing_ltx_backbones.yaml"))

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with self.assertRaises(ValueError):
                with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                    LTXVideoAdapter(config_path=config_path, variant="nope")

    def test_extract_shape_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                adapter = LTXVideoAdapter(config_path=config_path)

            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

        self.assertEqual(features.selected_layers, (1, 2, 4, 5))
        for layer_id in features.selected_layers:
            tokens = features.tokens_by_layer[layer_id]
            pooled = features.pooled_by_layer[layer_id]
            self.assertEqual(tokens.ndim, 3)
            self.assertEqual(pooled.ndim, 2)
            self.assertEqual(tokens.shape[0], 2)
            self.assertEqual(pooled.shape[0], 2)
            self.assertEqual(tokens.shape[2], pooled.shape[1])

    def test_extract_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                adapter = LTXVideoAdapter(config_path=config_path)

            clips = torch.randn(1, 3, 16, 224, 224)
            features = adapter.extract(clips, layer_ids=[2, 5])

        self.assertEqual(features.selected_layers, (2, 5))
        self.assertEqual(set(features.tokens_by_layer), {2, 5})
        self.assertNotIn(1, features.tokens_by_layer)

    def test_extract_invalid_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                adapter = LTXVideoAdapter(config_path=config_path)

            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 224, 224), layer_ids=[99])

    def test_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                adapter = LTXVideoAdapter(config_path=config_path)

            features = adapter.extract(torch.randn(1, 3, 16, 224, 224))

        required = {
            "model_name",
            "hf_model_id",
            "config_path",
            "variant",
            "extract_source",
            "frames_per_clip",
            "crop_size",
        }
        self.assertTrue(required.issubset(features.metadata.keys()))


class LTXVideoRegistryTests(unittest.TestCase):
    def test_adapter_is_registered(self) -> None:
        self.assertIn("ltx_video", get_registered_adapters())

    def test_registry_factory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_vae", _fake_load_vae):
                adapter = create_adapter("ltx_video", config_path=config_path)

        self.assertIsInstance(adapter, LTXVideoAdapter)


if __name__ == "__main__":
    unittest.main()
