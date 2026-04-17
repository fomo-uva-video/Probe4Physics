from __future__ import annotations

import tempfile
import types
import unittest
import yaml
from pathlib import Path
from unittest import mock

import torch

from models.jepa_v2_adapter import JEPAV2Adapter, resolve_relative_depth_layers
from models.registry import (
    create_adapter,
    enforce_single_jepa_namespace,
    get_registered_adapters,
    reset_runtime_guard_for_tests,
)


# ---------------------------------------------------------------------------
# Fake repo / helpers
# ---------------------------------------------------------------------------

def _write_fake_repo_layout(root: Path) -> None:
    """Create the minimal directory tree expected by JEPAV2Adapter._validate_repo_layout."""
    (root / "src" / "models").mkdir(parents=True, exist_ok=True)
    (root / "src" / "models" / "vision_transformer.py").write_text("# stub\n", encoding="utf-8")


def _make_minimal_config(tmp: Path, *, default_variant: str = "vitl_256") -> Path:
    """Write a minimal jepa_v2 backbones.yaml to ``tmp``."""
    checkpoints_dir = tmp / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    payload = {
        "jepa_v2": {
            "default_variant": default_variant,
            "checkpoints_dir": str(checkpoints_dir),
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {
                "vit_large": 24,
                "vit_giant_xformers": 40,
            },
            "variants": {
                "vitl_256": {
                    "checkpoint_filename": "vitl.pt",
                    "model_name": "vit_large",
                    "checkpoint_key": "target_encoder",
                    "crop_size": 256,
                    "patch_size": 16,
                    "frames_per_clip": 16,
                    "tubelet_size": 2,
                },
                "vitg_256": {
                    "checkpoint_filename": "vitg.pt",
                    "model_name": "vit_giant_xformers",
                    "checkpoint_key": "target_encoder",
                    "crop_size": 256,
                    "patch_size": 16,
                    "frames_per_clip": 16,
                    "tubelet_size": 2,
                },
            },
        }
    }
    config_path = tmp / "backbones.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _write_fake_checkpoint(path: Path, key: str = "target_encoder") -> None:
    torch.save({key: {}}, path)


class _FakeV2Encoder(torch.nn.Module):
    """Minimal ViT stub: returns one [B, 4, 8] tensor per requested out_layer."""

    def __init__(self, out_layers=None, **_: object) -> None:
        super().__init__()
        self.out_layers = list(out_layers or [])
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        b = clips.shape[0]
        return [torch.full((b, 4, 8), float(i + 1)) for i in range(len(self.out_layers))]


class _FakeV2CheckpointEncoder(torch.nn.Module):
    """Stub that actually loads weights — used to verify prefix stripping."""

    def __init__(self, out_layers=None, **_: object) -> None:
        super().__init__()
        self.out_layers = list(out_layers or [])
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        b = clips.shape[0]
        return [torch.zeros(b, 2, 2) for _ in self.out_layers]


def _build_fake_vit_module(encoder_ctor) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        vit_large=encoder_ctor,
        vit_giant_xformers=encoder_ctor,
    )


# ---------------------------------------------------------------------------
# Layer mapping tests
# ---------------------------------------------------------------------------

class JEPAV2LayerMappingTests(unittest.TestCase):
    """Verify relative-depth → 1-based block-id mapping for V-JEPA 2 models."""

    def test_vit_large_default_depths(self) -> None:
        # 24 blocks × [0.25, 0.5, 0.75, 1.0] → [6, 12, 18, 24]
        result = resolve_relative_depth_layers(
            "vit_large",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"vit_large": 24},
        )
        self.assertEqual(result, (6, 12, 18, 24))

    def test_vit_giant_xformers_default_depths(self) -> None:
        # 40 blocks × [0.25, 0.5, 0.75, 1.0] → [10, 20, 30, 40]
        result = resolve_relative_depth_layers(
            "vit_giant_xformers",
            [0.25, 0.5, 0.75, 1.0],
            model_block_depths={"vit_giant_xformers": 40},
        )
        self.assertEqual(result, (10, 20, 30, 40))

    def test_custom_single_depth(self) -> None:
        result = resolve_relative_depth_layers(
            "vit_large",
            [1.0],
            model_block_depths={"vit_large": 24},
        )
        self.assertEqual(result, (24,))

    def test_unknown_model_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "vit_unknown",
                [0.5],
                model_block_depths={"vit_large": 24},
            )

    def test_depth_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "vit_large",
                [0.0],
                model_block_depths={"vit_large": 24},
            )

    def test_depth_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relative_depth_layers(
                "vit_large",
                [1.5],
                model_block_depths={"vit_large": 24},
            )

    def test_deduplication(self) -> None:
        # Two depths that round to the same block should produce only one id.
        result = resolve_relative_depth_layers(
            "vit_large",
            [0.25, 0.26],
            model_block_depths={"vit_large": 24},
        )
        self.assertEqual(len(set(result)), len(result))


# ---------------------------------------------------------------------------
# Runtime guard tests
# ---------------------------------------------------------------------------

class JEPAV2RuntimeGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_guard_blocks_jepa_v1_after_jepa_v2(self) -> None:
        enforce_single_jepa_namespace("jepa_v2")
        with self.assertRaises(RuntimeError):
            enforce_single_jepa_namespace("jepa_v1")

    def test_same_namespace_allowed_twice(self) -> None:
        enforce_single_jepa_namespace("jepa_v2")
        # Should not raise — same namespace is idempotent.
        enforce_single_jepa_namespace("jepa_v2")

    def test_v2_and_v2_1_share_namespace(self) -> None:
        """Both V-JEPA 2 and 2.1 use 'jepa_v2' so they can coexist."""
        enforce_single_jepa_namespace("jepa_v2")
        # Registering again with the same key must be fine.
        enforce_single_jepa_namespace("jepa_v2")


# ---------------------------------------------------------------------------
# Adapter construction / extraction tests
# ---------------------------------------------------------------------------

class JEPAV2AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    # --- error path tests ---

    def test_missing_repo_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitl.pt"
            _write_fake_checkpoint(checkpoint)

            with self.assertRaises(FileNotFoundError):
                JEPAV2Adapter(
                    repo_root=tmp_path / "missing_repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                )

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)

            with self.assertRaises(FileNotFoundError):
                JEPAV2Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=tmp_path / "nonexistent.pt",
                    config_path=config_path,
                )

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            JEPAV2Adapter(
                repo_root="third_party/vjepa2",
                config_path=Path("/tmp/does_not_exist.yaml"),
            )

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = _make_minimal_config(tmp_path)
            _write_fake_repo_layout(tmp_path / "repo")

            with self.assertRaises(ValueError):
                JEPAV2Adapter(
                    repo_root=tmp_path / "repo",
                    config_path=config_path,
                    variant="does_not_exist",
                )

    def test_unknown_model_name_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitl.pt"
            _write_fake_checkpoint(checkpoint)

            # Return a module that has no "vit_weird" constructor.
            empty_module = types.SimpleNamespace()

            with mock.patch(
                "models.jepa_v2_adapter._import_vjepa2_vit_module",
                return_value=empty_module,
            ):
                with self.assertRaises(ValueError):
                    JEPAV2Adapter(
                        repo_root=tmp_path / "repo",
                        checkpoint_path=checkpoint,
                        config_path=config_path,
                        model_name="vit_weird",
                    )

    # --- checkpoint loading tests ---

    def test_checkpoint_key_fallback_to_encoder(self) -> None:
        """If the primary checkpoint_key is missing the adapter falls back to 'encoder'."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitl.pt"
            # Save under "encoder" (not "target_encoder").
            torch.save({"encoder": {"proj.weight": torch.ones(2, 2)}}, checkpoint)

            fake_vit = _build_fake_vit_module(_FakeV2CheckpointEncoder)

            with mock.patch(
                "models.jepa_v2_adapter._import_vjepa2_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_large",
                )
            self.assertTrue(torch.allclose(adapter._encoder.proj.weight, torch.ones(2, 2)))

    def test_checkpoint_prefix_stripping(self) -> None:
        """'module.' and 'backbone.' prefixes must be stripped from checkpoint keys."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitl.pt"
            torch.save(
                {"target_encoder": {"module.backbone.proj.weight": torch.ones(2, 2)}},
                checkpoint,
            )

            fake_vit = _build_fake_vit_module(_FakeV2CheckpointEncoder)

            with mock.patch(
                "models.jepa_v2_adapter._import_vjepa2_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_large",
                )
            self.assertTrue(torch.allclose(adapter._encoder.proj.weight, torch.ones(2, 2)))

    # --- extract shape contract tests ---

    def _make_adapter(self, tmp_path: Path, model_name: str = "vit_large") -> JEPAV2Adapter:
        _write_fake_repo_layout(tmp_path / "repo")
        config_path = _make_minimal_config(tmp_path)
        checkpoint = tmp_path / "checkpoints" / "vitl.pt"
        _write_fake_checkpoint(checkpoint)
        fake_vit = _build_fake_vit_module(_FakeV2Encoder)

        with mock.patch(
            "models.jepa_v2_adapter._import_vjepa2_vit_module",
            return_value=fake_vit,
        ):
            return JEPAV2Adapter(
                repo_root=tmp_path / "repo",
                checkpoint_path=checkpoint,
                config_path=config_path,
                model_name=model_name,
            )

    def test_extract_tokens_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

        # Default layers for vit_large: (6, 12, 18, 24)
        self.assertEqual(features.selected_layers, (6, 12, 18, 24))
        for layer_id in features.selected_layers:
            tokens = features.tokens_by_layer[layer_id]
            pooled = features.pooled_by_layer[layer_id]
            self.assertEqual(tokens.ndim, 3, f"layer {layer_id} tokens must be [B, N, D]")
            self.assertEqual(pooled.ndim, 2, f"layer {layer_id} pooled must be [B, D]")
            self.assertEqual(tokens.shape[0], 2)
            self.assertEqual(pooled.shape[0], 2)
            # pooled = mean over N → same D
            self.assertEqual(tokens.shape[2], pooled.shape[1])

    def test_extract_layer_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips, layer_ids=[6, 24])

        self.assertEqual(features.selected_layers, (6, 24))
        self.assertEqual(set(features.tokens_by_layer), {6, 24})
        self.assertNotIn(12, features.tokens_by_layer)
        self.assertNotIn(18, features.tokens_by_layer)

    def test_extract_invalid_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 224, 224), layer_ids=[99])

    def test_extract_empty_layer_ids_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 224, 224), layer_ids=[])

    def test_extract_wrong_ndim_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(2, 3, 224, 224))  # 4-D, not 5-D

    def test_extract_non_tensor_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(TypeError):
                adapter.extract([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_encoder_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
        for param in adapter._encoder.parameters():
            self.assertFalse(param.requires_grad, "All encoder parameters must be frozen.")

    def test_metadata_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            features = adapter.extract(torch.randn(1, 3, 16, 224, 224))

        for key in (
            "model_name",
            "checkpoint_path",
            "config_path",
            "variant",
            "checkpoint_key",
            "patch_size",
            "tubelet_size",
            "frames_per_clip",
            "crop_size",
        ):
            self.assertIn(key, features.metadata, f"metadata missing key: {key!r}")

    def test_pooled_is_mean_of_tokens(self) -> None:
        """pooled_by_layer[L] must equal tokens_by_layer[L].mean(dim=1)."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            clips = torch.randn(2, 3, 16, 224, 224)
            features = adapter.extract(clips)

        for layer_id, tokens in features.tokens_by_layer.items():
            expected_pooled = tokens.mean(dim=1)
            self.assertTrue(
                torch.allclose(features.pooled_by_layer[layer_id], expected_pooled),
                f"pooled_by_layer[{layer_id}] must be the mean of tokens_by_layer[{layer_id}]",
            )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class JEPAV2RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_jepa_v2_is_registered(self) -> None:
        self.assertIn("jepa_v2", get_registered_adapters())

    def test_registry_factory_returns_correct_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitl.pt"
            _write_fake_checkpoint(checkpoint)
            fake_vit = _build_fake_vit_module(_FakeV2Encoder)

            with mock.patch(
                "models.jepa_v2_adapter._import_vjepa2_vit_module",
                return_value=fake_vit,
            ):
                adapter = create_adapter(
                    "jepa_v2",
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                )
        self.assertIsInstance(adapter, JEPAV2Adapter)


if __name__ == "__main__":
    unittest.main()
