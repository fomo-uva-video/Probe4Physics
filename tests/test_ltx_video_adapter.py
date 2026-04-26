from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from models.ltx_video_adapter import (
    LTXVideoAdapter,
    resolve_noise_levels,
    resolve_probe_layer_ids,
    resolve_probe_layer_specs,
    resolve_relative_depth_layers,
)
from models.registry import create_adapter, get_registered_adapters


class _FakeTemporalDownsampler(torch.nn.Module):
    def __init__(self, stride_t: int) -> None:
        super().__init__()
        self.stride = (int(stride_t), 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeTemporalStage(torch.nn.Module):
    def __init__(self, stride_t: int = 1) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        if int(stride_t) > 1:
            modules.append(_FakeTemporalDownsampler(stride_t))
        self.downsamplers = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeVAE(torch.nn.Module):
    def __init__(self, *, temporal_strides: tuple[int, ...] = (1, 2, 2, 2)) -> None:
        super().__init__()
        self.encoder = SimpleNamespace(
            down_blocks=torch.nn.ModuleList(
                [_FakeTemporalStage(stride_t=stride) for stride in temporal_strides]
            )
        )
        self.spatial_compression_ratio = 8
        self.temporal_compression_ratio = 8
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.last_input: torch.Tensor | None = None

    def encode(self, x: torch.Tensor):
        self.last_input = x.detach().clone()
        latents = x + 0.5
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        factor = timestep.view(-1, 1, 1, 1, 1).to(dtype=sample.dtype) / 1000.0
        return sample + noise * factor


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: object | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = encoder_hidden_states, temb, image_rotary_emb, encoder_attention_mask
        return hidden_states + self.delta


class _FakeTransformer(torch.nn.Module):
    def __init__(self, depth: int = 28) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [_FakeTransformerBlock(delta=float(index + 1)) for index in range(depth)]
        )
        self.config = SimpleNamespace(in_channels=3)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        return_dict: bool = True,
    ):
        _ = timestep, encoder_attention_mask, num_frames, height, width
        output = hidden_states
        for block in self.transformer_blocks:
            output = block(
                hidden_states=output,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep,
                image_rotary_emb=None,
                encoder_attention_mask=encoder_attention_mask,
            )
        if not return_dict:
            return (output,)
        return SimpleNamespace(sample=output)


class _FakeTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **kwargs):
        _ = prompts, kwargs
        return SimpleNamespace(
            input_ids=torch.ones(1, self.model_max_length, dtype=torch.long),
            attention_mask=torch.ones(1, self.model_max_length, dtype=torch.long),
        )


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _ = attention_mask
        batch_size, sequence_length = input_ids.shape
        hidden = torch.ones(batch_size, sequence_length, 16, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_load_components(
    self: LTXVideoAdapter,
) -> tuple[torch.nn.Module, torch.nn.Module, object, object, torch.nn.Module]:
    return (
        _FakeVAE(),
        _FakeTransformer(depth=28),
        _FakeScheduler(),
        _FakeTokenizer(),
        _FakeTextEncoder(),
    )


def _fake_load_components_temporal(
    self: LTXVideoAdapter,
) -> tuple[torch.nn.Module, torch.nn.Module, object, object, torch.nn.Module]:
    return (
        _FakeVAE(temporal_strides=(1, 2, 2, 2)),
        _FakeTransformer(depth=28),
        _FakeScheduler(),
        _FakeTokenizer(),
        _FakeTextEncoder(),
    )


def _make_minimal_config(tmp: Path) -> Path:
    config_path = tmp / "backbones.yaml"
    payload = {
        "ltx_video": {
            "default_variant": "ltxv_13b_0_9_8_distilled",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "default_noise_levels": [0.9, 0.5, 0.1],
            "model_block_depths": {
                "ltx_transformer_28": 28,
            },
            "variants": {
                "ltxv_13b_0_9_8_distilled": {
                    "hf_model_id": "Lightricks/LTX-Video-0.9.8-13B-distilled",
                    "model_name": "ltx_transformer_28",
                    "vae_subfolder": "vae",
                    "frames_per_clip": 16,
                    "crop_size": 224,
                    "patch_size": 1,
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
            "ltx_transformer_28",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"ltx_transformer_28": 28},
        )
        self.assertEqual(result, (7, 14, 21, 28))

    def test_default_noise_levels(self) -> None:
        result = resolve_noise_levels([0.9, 0.5, 0.1])
        self.assertEqual(result, (0.9, 0.5, 0.1))

    def test_probe_layer_ids_flatten_noise_and_depth(self) -> None:
        result = resolve_probe_layer_ids(
            "ltx_transformer_28",
            relative_depths=[0.25, 0.5, 0.75, 1.0],
            noise_levels=[0.9, 0.5, 0.1],
            model_block_depths={"ltx_transformer_28": 28},
        )
        self.assertEqual(result, tuple(range(1, 13)))

    def test_probe_layer_specs_expose_noise_depth_mapping(self) -> None:
        specs = resolve_probe_layer_specs(
            "ltx_transformer_28",
            relative_depths=[0.25, 0.5],
            noise_levels=[0.9, 0.1],
            model_block_depths={"ltx_transformer_28": 28},
        )
        self.assertEqual(specs[0].probe_layer_id, 1)
        self.assertEqual(specs[0].depth_layer_id, 7)
        self.assertEqual(specs[0].noise_label, "noise_1")
        self.assertEqual(specs[-1].probe_layer_id, 4)
        self.assertEqual(specs[-1].depth_layer_id, 14)


class LTXVideoAdapterTests(unittest.TestCase):
    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                LTXVideoAdapter(config_path=Path("/tmp/missing_ltx_backbones.yaml"))

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with self.assertRaises(ValueError):
                with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                    LTXVideoAdapter(config_path=config_path, variant="nope")

    def test_extract_shape_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = LTXVideoAdapter(config_path=config_path)

            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

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
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = LTXVideoAdapter(config_path=config_path)

            clips = torch.randn(1, 3, 16, 224, 224)
            features = adapter.extract(clips, layer_ids=[2, 11])

        self.assertEqual(features.selected_layers, (2, 11))
        self.assertEqual(set(features.tokens_by_layer), {2, 11})
        self.assertNotIn(1, features.tokens_by_layer)

    def test_extract_invalid_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = LTXVideoAdapter(config_path=config_path)

            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 224, 224), layer_ids=[99])

    def test_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
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
            "noise_levels",
            "noise_timesteps",
            "layer_spec_by_id",
        }
        self.assertTrue(required.issubset(features.metadata.keys()))
        self.assertEqual(features.metadata["extract_source"], "diffusion_transformer_blocks")
        self.assertEqual(features.metadata["layer_spec_by_id"]["1"]["depth_layer_id"], 7)

    def test_extract_aligns_temporal_length_for_ltx_downsamplers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components_temporal):
                adapter = LTXVideoAdapter(config_path=config_path)
            clips = torch.randn(1, 3, 16, 224, 224)
            features = adapter.extract(clips)

        self.assertEqual(adapter._temporal_downsample_strides, (2, 2, 2))
        self.assertEqual(features.metadata.get("temporal_downsample_strides"), [2, 2, 2])

    def test_extract_normalizes_to_minus_one_one_before_vae_encode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = LTXVideoAdapter(config_path=config_path, normalize_input=True)

            clips = torch.zeros(1, 3, 16, 224, 224)
            adapter.extract(clips, layer_ids=[1])

        self.assertIsNotNone(adapter._vae.last_input)
        self.assertTrue(torch.allclose(adapter._vae.last_input, torch.full_like(adapter._vae.last_input, -1.0)))

    def test_extract_can_disable_input_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = LTXVideoAdapter(config_path=config_path, normalize_input=False)

            clips = torch.zeros(1, 3, 16, 224, 224)
            adapter.extract(clips, layer_ids=[1])

        self.assertIsNotNone(adapter._vae.last_input)
        self.assertTrue(torch.allclose(adapter._vae.last_input, torch.zeros_like(adapter._vae.last_input)))
        self.assertEqual(
            features := adapter.preprocessing_metadata()["pixel_normalization"],
            "identity",
        )


class LTXVideoRegistryTests(unittest.TestCase):
    def test_adapter_is_registered(self) -> None:
        self.assertIn("ltx_video", get_registered_adapters())

    def test_registry_factory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _make_minimal_config(Path(tmp))
            with mock.patch.object(LTXVideoAdapter, "_load_components", _fake_load_components):
                adapter = create_adapter("ltx_video", config_path=config_path)

        self.assertIsInstance(adapter, LTXVideoAdapter)


if __name__ == "__main__":
    unittest.main()
