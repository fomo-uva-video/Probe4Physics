from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from models.jepa_v1_adapter import JEPAV1Adapter, resolve_relative_depth_layers
from models.registry import (
    create_adapter,
    enforce_single_jepa_namespace,
    get_registered_adapters,
    reset_runtime_guard_for_tests,
)


def _write_fake_repo_layout(root: Path) -> None:
    (root / "src" / "models").mkdir(parents=True, exist_ok=True)
    (root / "src" / "models" / "vision_transformer.py").write_text("# stub\n", encoding="utf-8")


class _FakeFeatureEncoder(torch.nn.Module):
    def __init__(self, out_layers=None, **_: object) -> None:
        super().__init__()
        self.out_layers = tuple(out_layers or [])
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        batch_size = clips.shape[0]
        outputs: list[torch.Tensor] = []
        for index, _ in enumerate(self.out_layers, start=1):
            outputs.append(torch.full((batch_size, 4, 3), float(index)))
        return outputs


class _FakeCheckpointEncoder(torch.nn.Module):
    def __init__(self, out_layers=None, **_: object) -> None:
        super().__init__()
        self.out_layers = tuple(out_layers or [])
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        batch_size = clips.shape[0]
        outputs: list[torch.Tensor] = []
        for _ in self.out_layers:
            outputs.append(torch.zeros((batch_size, 2, 2), dtype=clips.dtype))
        return outputs


def _fake_load_pretrained_noop(*, encoder, pretrained: str, checkpoint_key: str = "target_encoder"):
    _ = pretrained, checkpoint_key
    return encoder


def _fake_load_pretrained_with_fallback(
    *,
    encoder,
    pretrained: str,
    checkpoint_key: str = "target_encoder",
):
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = checkpoint.get(checkpoint_key, checkpoint.get("encoder", {}))
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(cleaned, strict=False)
    return encoder


def _build_fake_vit_module(encoder_ctor):
    return types.SimpleNamespace(vit_large=encoder_ctor, vit_huge=encoder_ctor)


class JEPAV1LayerMappingTests(unittest.TestCase):
    def test_relative_depth_mapping_vit_large(self) -> None:
        self.assertEqual(resolve_relative_depth_layers("vit_large"), (6, 12, 18, 24))

    def test_relative_depth_mapping_vit_huge(self) -> None:
        self.assertEqual(resolve_relative_depth_layers("vit_huge"), (8, 16, 24, 32))


class JEPAV1RuntimeGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_guard_blocks_different_namespace(self) -> None:
        enforce_single_jepa_namespace("jepa_v1")
        with self.assertRaises(RuntimeError):
            enforce_single_jepa_namespace("jepa_v2")


class JEPAV1AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_missing_submodule_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "fake_checkpoint.pt"
            torch.save({"target_encoder": {}}, checkpoint)

            with self.assertRaises(FileNotFoundError):
                JEPAV1Adapter(
                    repo_root=Path(tmp) / "missing",
                    checkpoint_path=checkpoint,
                )

    def test_checkpoint_fallback_and_prefix_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_repo_layout(root)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {"encoder": {"module.backbone.proj.weight": torch.ones((2, 2))}},
                checkpoint,
            )
            fake_vit = _build_fake_vit_module(_FakeCheckpointEncoder)

            with mock.patch(
                "models.jepa_v1_adapter._import_official_jepa_modules",
                return_value=fake_vit,
            ), mock.patch(
                "models.jepa_v1_adapter._load_pretrained_official",
                side_effect=_fake_load_pretrained_with_fallback,
            ):
                adapter = JEPAV1Adapter(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    model_name="vit_large",
                    checkpoint_key="target_encoder",
                )

            self.assertTrue(torch.allclose(adapter._encoder.proj.weight, torch.ones((2, 2))))

    def test_extract_structure_and_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_repo_layout(root)
            checkpoint = root / "checkpoint.pt"
            torch.save({"target_encoder": {}}, checkpoint)
            fake_vit = _build_fake_vit_module(_FakeFeatureEncoder)

            with mock.patch(
                "models.jepa_v1_adapter._import_official_jepa_modules",
                return_value=fake_vit,
            ), mock.patch(
                "models.jepa_v1_adapter._load_pretrained_official",
                side_effect=_fake_load_pretrained_noop,
            ):
                adapter = JEPAV1Adapter(repo_root=root, checkpoint_path=checkpoint)
                clips = torch.randn(2, 3, 16, 224, 224)
                features = adapter.extract(clips, layer_ids=[12, 24])

        self.assertEqual(features.selected_layers, (12, 24))
        self.assertEqual(set(features.tokens_by_layer), {12, 24})
        self.assertEqual(tuple(features.tokens_by_layer[12].shape), (2, 4, 3))
        self.assertEqual(tuple(features.pooled_by_layer[12].shape), (2, 3))
        self.assertEqual(features.metadata["model_name"], "vit_large")
        self.assertEqual(features.metadata["checkpoint_key"], "target_encoder")


class JEPAV1RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_jepa_v1_is_registered(self) -> None:
        self.assertIn("jepa_v1", get_registered_adapters())

    def test_registry_factory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_repo_layout(root)
            checkpoint = root / "checkpoint.pt"
            torch.save({"target_encoder": {}}, checkpoint)
            fake_vit = _build_fake_vit_module(_FakeFeatureEncoder)

            with mock.patch(
                "models.jepa_v1_adapter._import_official_jepa_modules",
                return_value=fake_vit,
            ), mock.patch(
                "models.jepa_v1_adapter._load_pretrained_official",
                side_effect=_fake_load_pretrained_noop,
            ):
                adapter = create_adapter(
                    "jepa_v1",
                    repo_root=root,
                    checkpoint_path=checkpoint,
                )
        self.assertIsInstance(adapter, JEPAV1Adapter)


if __name__ == "__main__":
    unittest.main()
