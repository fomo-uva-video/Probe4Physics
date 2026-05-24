from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from models.cache_metadata import resolve_backbone_cache_metadata, resolve_backbone_layer_label
from models.registry import create_adapter, get_registered_adapters
from models.wan_video_adapter import (
    WanVideoAdapter,
    _ensure_wan_runtime_support,
    resolve_noise_levels,
    resolve_probe_layer_ids,
    resolve_probe_layer_specs,
    resolve_relative_depth_layers,
)


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            z_dim=3,
            latents_mean=[0.25, 0.25, 0.25],
            latents_std=[0.5, 0.5, 0.5],
        )
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.last_input: torch.Tensor | None = None

    def encode(self, x: torch.Tensor):
        self.last_input = x.detach().clone()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x + 0.5))


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0)
        self.last_sample: torch.Tensor | None = None
        self.last_timestep: torch.Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device | str | None = None) -> None:
        _ = device
        self.sigmas = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1)
        self.timesteps = torch.linspace(1000.0, 0.0, int(num_inference_steps) + 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.last_sample = original_samples.detach().clone()
        self.last_timestep = timesteps.detach().clone()
        factor = timesteps.view(-1, 1, 1, 1, 1).to(dtype=original_samples.dtype) / 1000.0
        return original_samples + noise * factor


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: object | None = None,
    ) -> torch.Tensor:
        _ = encoder_hidden_states, temb, rotary_emb
        return hidden_states + self.delta


class _FakeTransformer(torch.nn.Module):
    def __init__(self, depth: int = 40) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_FakeTransformerBlock(delta=float(index + 1)) for index in range(depth)]
        )
        self.config = SimpleNamespace(in_channels=3, patch_size=(1, 2, 2))
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ):
        batch_size = hidden_states.shape[0]
        output = hidden_states.flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            output = block(output, encoder_hidden_states, timestep, None)
        if not return_dict:
            return (output,)
        return SimpleNamespace(sample=output)


class _FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        _ = prompts
        max_length = int(kwargs["max_length"])
        return SimpleNamespace(
            input_ids=torch.ones(1, max_length, dtype=torch.long),
            attention_mask=torch.ones(1, max_length, dtype=torch.long),
        )


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _ = attention_mask
        batch_size, sequence_length = input_ids.shape
        hidden = torch.ones(batch_size, sequence_length, 32, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_load_components(
    self: WanVideoAdapter,
) -> tuple[torch.nn.Module, torch.nn.Module, object, object, torch.nn.Module]:
    return (
        _FakeVAE(),
        _FakeTransformer(depth=40),
        _FakeScheduler(),
        _FakeTokenizer(),
        _FakeTextEncoder(),
    )


def _fake_load_components_30(
    self: WanVideoAdapter,
) -> tuple[torch.nn.Module, torch.nn.Module, object, object, torch.nn.Module]:
    return (
        _FakeVAE(),
        _FakeTransformer(depth=30),
        _FakeScheduler(),
        _FakeTokenizer(),
        _FakeTextEncoder(),
    )


def _make_minimal_config(tmp: Path) -> Path:
    config_path = tmp / "backbones.yaml"
    payload = {
        "wan_video": {
            "default_variant": "wan2_1_t2v_14b_diffusers",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "default_noise_levels": [1.0, 0.5, 0.1],
            "model_block_depths": {
                "wan2_1_t2v_1_3b": 30,
                "wan2_1_t2v_14b": 40,
            },
            "variants": {
                "wan2_1_t2v_1_3b_diffusers": {
                    "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "model_name": "wan2_1_t2v_1_3b",
                    "frames_per_clip": 16,
                    "crop_size": 224,
                    "patch_size": 2,
                    "patch_size_t": 1,
                    "torch_dtype": "float32",
                    "num_inference_steps": 10,
                    "max_sequence_length": 8,
                },
                "wan2_1_t2v_14b_diffusers": {
                    "hf_model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                    "model_name": "wan2_1_t2v_14b",
                    "frames_per_clip": 17,
                    "crop_size": 224,
                    "patch_size": 2,
                    "patch_size_t": 1,
                    "torch_dtype": "float32",
                    "num_inference_steps": 10,
                    "max_sequence_length": 8,
                }
            },
        }
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


class WanVideoLayerMappingTests(unittest.TestCase):
    def test_default_depth_mapping(self) -> None:
        result = resolve_relative_depth_layers(
            "wan2_1_t2v_14b",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"wan2_1_t2v_14b": 40},
        )
        self.assertEqual(result, (10, 20, 30, 40))

    def test_default_noise_levels(self) -> None:
        result = resolve_noise_levels([1.0, 0.5, 0.1])
        self.assertEqual(result, (1.0, 0.5, 0.1))

    def test_default_depth_mapping_for_1_3b(self) -> None:
        result = resolve_relative_depth_layers(
            "wan2_1_t2v_1_3b",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"wan2_1_t2v_1_3b": 30},
        )
        self.assertEqual(result, (8, 15, 22, 30))

    def test_probe_layer_ids_flatten_noise_and_depth(self) -> None:
        result = resolve_probe_layer_ids(
            "wan2_1_t2v_14b",
            relative_depths=[0.25, 0.5, 0.75, 1.0],
            noise_levels=[1.0, 0.5, 0.1],
            model_block_depths={"wan2_1_t2v_14b": 40},
        )
        self.assertEqual(result, tuple(range(1, 13)))

    def test_probe_layer_specs_expose_noise_depth_mapping(self) -> None:
        specs = resolve_probe_layer_specs(
            "wan2_1_t2v_14b",
            relative_depths=[0.25, 0.5],
            noise_levels=[1.0, 0.1],
            model_block_depths={"wan2_1_t2v_14b": 40},
        )
        self.assertEqual(specs[0].probe_layer_id, 1)
        self.assertEqual(specs[0].depth_layer_id, 10)
        self.assertEqual(specs[0].noise_label, "noise_1")
        self.assertEqual(specs[-1].probe_layer_id, 4)
        self.assertEqual(specs[-1].depth_layer_id, 20)


class WanVideoAdapterTests(unittest.TestCase):
    def test_missing_sentencepiece_raises_early(self) -> None:
        with mock.patch("models.wan_video_adapter.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "sentencepiece"):
                _ensure_wan_runtime_support()

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                WanVideoAdapter(config_path=Path("/tmp/missing_wan_backbones.yaml"))

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with self.assertRaises(ValueError):
                with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                    WanVideoAdapter(config_path=config_path, variant="nope")

    def test_extract_shape_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                adapter = WanVideoAdapter(config_path=config_path)
            features = adapter.extract(torch.randn(2, 3, 16, 8, 8))

        self.assertEqual(features.selected_layers, tuple(range(1, 13)))
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
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                adapter = WanVideoAdapter(config_path=config_path)
            features = adapter.extract(torch.randn(1, 3, 17, 8, 8), layer_ids=[2, 11])

        self.assertEqual(features.selected_layers, (2, 11))
        self.assertEqual(set(features.tokens_by_layer), {2, 11})
        self.assertNotIn(1, features.tokens_by_layer)

    def test_1_3b_variant_metadata_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components_30):
                adapter = WanVideoAdapter(
                    config_path=config_path,
                    variant="wan2_1_t2v_1_3b_diffusers",
                )
            features = adapter.extract(torch.randn(1, 3, 16, 8, 8), layer_ids=[1, 2, 3, 4])

        self.assertEqual(features.metadata["variant"], "wan2_1_t2v_1_3b_diffusers")
        self.assertEqual(features.metadata["model_name"], "wan2_1_t2v_1_3b")
        self.assertEqual(features.metadata["transformer_block_count"], 30)
        self.assertEqual(features.selected_layers, (1, 2, 3, 4))
        self.assertEqual(features.metadata["layer_spec_by_id"]["1"]["depth_layer_id"], 8)
        self.assertEqual(features.metadata["layer_spec_by_id"]["2"]["depth_layer_id"], 15)
        self.assertEqual(features.metadata["layer_spec_by_id"]["3"]["depth_layer_id"], 22)
        self.assertEqual(features.metadata["layer_spec_by_id"]["4"]["depth_layer_id"], 30)

    def test_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                adapter = WanVideoAdapter(config_path=config_path)
            features = adapter.extract(torch.randn(1, 3, 17, 8, 8), layer_ids=[1])

        required = {
            "model_name",
            "hf_model_id",
            "config_path",
            "variant",
            "extract_source",
            "frames_per_clip",
            "crop_size",
            "noise_levels",
            "noise_sigmas",
            "noise_timesteps",
            "layer_spec_by_id",
        }
        self.assertTrue(required.issubset(features.metadata.keys()))
        self.assertEqual(features.metadata["extract_source"], "diffusion_transformer_blocks")
        self.assertEqual(features.metadata["layer_spec_by_id"]["1"]["depth_layer_id"], 10)
        self.assertEqual([round(value, 2) for value in features.metadata["noise_sigmas"]], [1.0, 0.5, 0.1])
        self.assertEqual(features.metadata["noise_timesteps"], [1000.0, 500.0, 100.0])

    def test_extract_aligns_to_wan_temporal_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                adapter = WanVideoAdapter(config_path=config_path, normalize_input=True)
            adapter.extract(torch.zeros(1, 3, 16, 8, 8), layer_ids=[1])

        self.assertIsNotNone(adapter._vae.last_input)
        self.assertEqual(adapter._vae.last_input.shape[2], 17)
        self.assertTrue(torch.allclose(adapter._vae.last_input, torch.full_like(adapter._vae.last_input, -1.0)))
        self.assertIsNotNone(adapter._scheduler.last_sample)
        expected_latents = torch.full_like(adapter._scheduler.last_sample, -1.5)
        self.assertTrue(torch.allclose(adapter._scheduler.last_sample, expected_latents))

    def test_cache_metadata_and_layer_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            kwargs = {"config_path": config_path}
            metadata = resolve_backbone_cache_metadata("wan_video", kwargs)
            label = resolve_backbone_layer_label("wan_video", kwargs, 2)

        self.assertEqual(metadata["selected_layers"], list(range(1, 13)))
        self.assertEqual(label, "noise_1.0_block_20")

    def test_cache_metadata_and_layer_labels_for_1_3b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            kwargs = {"config_path": config_path, "variant": "wan2_1_t2v_1_3b_diffusers"}
            metadata = resolve_backbone_cache_metadata("wan_video", kwargs)
            label = resolve_backbone_layer_label("wan_video", kwargs, 2)

        self.assertEqual(metadata["variant"], "wan2_1_t2v_1_3b_diffusers")
        self.assertEqual(metadata["model_name"], "wan2_1_t2v_1_3b")
        self.assertEqual(metadata["selected_layers"], list(range(1, 13)))
        self.assertEqual(label, "noise_1.0_block_15")


class WanVideoRegistryTests(unittest.TestCase):
    def test_adapter_is_registered(self) -> None:
        self.assertIn("wan_video", get_registered_adapters())

    def test_registry_factory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(WanVideoAdapter, "_load_components", _fake_load_components):
                adapter = create_adapter("wan_video", config_path=config_path)

        self.assertIsInstance(adapter, WanVideoAdapter)


if __name__ == "__main__":
    unittest.main()
