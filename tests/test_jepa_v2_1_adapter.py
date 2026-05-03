from __future__ import annotations

import tempfile
import types
import unittest
import yaml
from pathlib import Path
from unittest import mock

import torch

from models.jepa_v2_1_adapter import JEPAV2_1Adapter
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
    """Create the minimal files expected by JEPAV2_1Adapter._validate_repo_layout."""
    (root / "src" / "models").mkdir(parents=True, exist_ok=True)
    (root / "src" / "models" / "vision_transformer.py").write_text("# stub\n", encoding="utf-8")
    (root / "app" / "vjepa_2_1" / "models").mkdir(parents=True, exist_ok=True)
    (root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py").write_text(
        "# stub\n", encoding="utf-8"
    )


def _make_minimal_config(
    tmp: Path,
    *,
    default_variant: str = "vitb_384",
) -> Path:
    """Write a minimal jepa_v2_1 backbones.yaml to ``tmp``."""
    checkpoints_dir = tmp / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    payload = {
        "jepa_v2_1": {
            "default_variant": default_variant,
            "checkpoints_dir": str(checkpoints_dir),
            "model_block_depths": {
                "vit_base": 12,
                "vit_large": 24,
                "vit_giant_xformers": 40,
                "vit_gigantic_xformers": 48,
            },
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "variants": {
                "vitb_384": {
                    "checkpoint_filename": "vitb.pt",
                    "model_name": "vit_base",
                    "checkpoint_key": "ema_encoder",
                    "crop_size": 384,
                    "patch_size": 16,
                    "frames_per_clip": 16,
                    "tubelet_size": 2,
                },
                "vitg_384": {
                    "checkpoint_filename": "vitg.pt",
                    "model_name": "vit_giant_xformers",
                    "checkpoint_key": "target_encoder",
                    "crop_size": 384,
                    "patch_size": 16,
                    "frames_per_clip": 16,
                    "tubelet_size": 2,
                },
                "vitG_384": {
                    "checkpoint_filename": "vitG.pt",
                    "model_name": "vit_gigantic_xformers",
                    "checkpoint_key": "target_encoder",
                    "crop_size": 384,
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


def _write_fake_checkpoint(path: Path, key: str = "ema_encoder") -> None:
    torch.save({key: {}}, path)


class _FakeV2_1Encoder(torch.nn.Module):
    """
    Minimal V-JEPA 2.1 ViT stub.

    Returns one ``[B, 4, 8]`` tensor per ``out_layers`` entry (same contract as
    the real encoder when ``out_layers`` is not None).  Accepts all extra
    kwargs that the real V-JEPA 2.1 ViT constructor requires.
    """

    def __init__(
        self,
        out_layers=None,
        *,
        img_size=384,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        **_: object,
    ) -> None:
        super().__init__()
        self.out_layers = list(out_layers or [])
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, clips: torch.Tensor) -> list[torch.Tensor]:
        b = clips.shape[0]
        return [torch.full((b, 4, 8), float(i + 1)) for i in range(len(self.out_layers))]


class _FakeV2_1CheckpointEncoder(torch.nn.Module):
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
        vit_base=encoder_ctor,
        vit_large=encoder_ctor,
        vit_giant_xformers=encoder_ctor,
        vit_gigantic_xformers=encoder_ctor,
    )


# ---------------------------------------------------------------------------
# Runtime guard tests
# ---------------------------------------------------------------------------

class JEPAV2_1RuntimeGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_v2_1_blocks_jepa_v1(self) -> None:
        """V-JEPA 2.1 uses 'jepa_v2' namespace, which conflicts with 'jepa_v1'."""
        enforce_single_jepa_namespace("jepa_v2")
        with self.assertRaises(RuntimeError):
            enforce_single_jepa_namespace("jepa_v1")

    def test_v2_and_v2_1_coexist(self) -> None:
        """V-JEPA 2 and 2.1 both use 'jepa_v2' — they must coexist in one process."""
        enforce_single_jepa_namespace("jepa_v2")  # simulates jepa_v2_adapter import
        enforce_single_jepa_namespace("jepa_v2")  # simulates jepa_v2_1_adapter import


# ---------------------------------------------------------------------------
# Adapter construction / extraction tests
# ---------------------------------------------------------------------------

class JEPAV2_1AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    # --- error path tests ---

    def test_missing_repo_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitb.pt"
            _write_fake_checkpoint(checkpoint, key="ema_encoder")

            with self.assertRaises(FileNotFoundError):
                JEPAV2_1Adapter(
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
                JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=tmp_path / "nonexistent.pt",
                    config_path=config_path,
                )

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            JEPAV2_1Adapter(
                repo_root="third_party/vjepa2",
                config_path=Path("/tmp/does_not_exist.yaml"),
            )

    def test_unknown_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)

            with self.assertRaises(ValueError):
                JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    config_path=config_path,
                    variant="does_not_exist",
                )

    def test_unknown_model_name_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitb.pt"
            _write_fake_checkpoint(checkpoint, key="ema_encoder")

            empty_module = types.SimpleNamespace()
            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=empty_module,
            ):
                with self.assertRaises(ValueError):
                    JEPAV2_1Adapter(
                        repo_root=tmp_path / "repo",
                        checkpoint_path=checkpoint,
                        config_path=config_path,
                        model_name="vit_weird",
                    )

    # --- checkpoint loading tests ---

    def test_checkpoint_key_fallback_to_encoder(self) -> None:
        """Adapter falls back to 'encoder' key when the primary key is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitb.pt"
            torch.save({"encoder": {"proj.weight": torch.ones(2, 2)}}, checkpoint)

            fake_vit = _build_fake_vit_module(_FakeV2_1CheckpointEncoder)

            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_base",
                )
            self.assertTrue(torch.allclose(adapter._encoder.proj.weight, torch.ones(2, 2)))

    def test_checkpoint_prefix_stripping(self) -> None:
        """'module.' and 'backbone.' prefixes must be stripped from checkpoint keys."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitb.pt"
            torch.save(
                {"ema_encoder": {"module.backbone.proj.weight": torch.ones(2, 2)}},
                checkpoint,
            )

            fake_vit = _build_fake_vit_module(_FakeV2_1CheckpointEncoder)

            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_base",
                )
            self.assertTrue(torch.allclose(adapter._encoder.proj.weight, torch.ones(2, 2)))

    # --- extract shape contract tests ---

    def _make_adapter(
        self,
        tmp_path: Path,
        model_name: str = "vit_base",
        checkpoint_key: str = "ema_encoder",
    ) -> JEPAV2_1Adapter:
        _write_fake_repo_layout(tmp_path / "repo")
        config_path = _make_minimal_config(tmp_path)
        checkpoint = tmp_path / "checkpoints" / "vitb.pt"
        _write_fake_checkpoint(checkpoint, key=checkpoint_key)
        fake_vit = _build_fake_vit_module(_FakeV2_1Encoder)

        with mock.patch(
            "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
            return_value=fake_vit,
        ):
            return JEPAV2_1Adapter(
                repo_root=tmp_path / "repo",
                checkpoint_path=checkpoint,
                config_path=config_path,
                model_name=model_name,
                checkpoint_key=checkpoint_key,
            )

    def test_selected_layers_are_hierarchical_for_vit_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
        # depth=12 hierarchical layers → 1-based [3, 6, 9, 12]
        self.assertEqual(adapter.selected_layers, (3, 6, 9, 12))

    def test_selected_layers_are_hierarchical_for_vit_giant_xformers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path, default_variant="vitg_384")
            checkpoint = tmp_path / "checkpoints" / "vitg.pt"
            _write_fake_checkpoint(checkpoint, key="target_encoder")
            fake_vit = _build_fake_vit_module(_FakeV2_1Encoder)

            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_giant_xformers",
                    checkpoint_key="target_encoder",
                )
        # depth=40 hierarchical layers → 1-based [10, 20, 30, 40]
        self.assertEqual(adapter.selected_layers, (10, 20, 30, 40))

    def test_selected_layers_are_hierarchical_for_vit_gigantic_xformers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path, default_variant="vitG_384")
            checkpoint = tmp_path / "checkpoints" / "vitG.pt"
            _write_fake_checkpoint(checkpoint, key="target_encoder")
            fake_vit = _build_fake_vit_module(_FakeV2_1Encoder)

            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=fake_vit,
            ):
                adapter = JEPAV2_1Adapter(
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                    model_name="vit_gigantic_xformers",
                    checkpoint_key="target_encoder",
                )
        # depth=48 hierarchical layers → 1-based [12, 24, 38, 48]
        self.assertEqual(adapter.selected_layers, (12, 24, 38, 48))

    def test_extract_tokens_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            clips = torch.randn(2, 3, 16, 384, 384)
            features = adapter.extract(clips)

        self.assertEqual(features.selected_layers, (3, 6, 9, 12))
        for layer_id in features.selected_layers:
            tokens = features.tokens_by_layer[layer_id]
            pooled = features.pooled_by_layer[layer_id]
            self.assertEqual(tokens.ndim, 3, f"layer {layer_id} tokens must be [B, N, D]")
            self.assertEqual(pooled.ndim, 2, f"layer {layer_id} pooled must be [B, D]")
            self.assertEqual(tokens.shape[0], 2)
            self.assertEqual(pooled.shape[0], 2)
            self.assertEqual(tokens.shape[2], pooled.shape[1])

    def test_extract_layer_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            clips = torch.randn(2, 3, 16, 384, 384)
            features = adapter.extract(clips, layer_ids=[6, 12])

        self.assertEqual(features.selected_layers, (6, 12))
        self.assertEqual(set(features.tokens_by_layer), {6, 12})
        self.assertNotIn(3, features.tokens_by_layer)
        self.assertNotIn(9, features.tokens_by_layer)

    def test_extract_non_hierarchical_layer_raises(self) -> None:
        """Requesting a layer not in the hierarchical set must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            # Layer 7 is not a hierarchical layer for depth=12 (valid: 3,6,9,12).
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 384, 384), layer_ids=[7])

    def test_extract_invalid_layer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 384, 384), layer_ids=[99])

    def test_extract_empty_layer_ids_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))
            with self.assertRaises(ValueError):
                adapter.extract(torch.randn(1, 3, 16, 384, 384), layer_ids=[])

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
            features = adapter.extract(torch.randn(1, 3, 16, 384, 384))

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
            clips = torch.randn(2, 3, 16, 384, 384)
            features = adapter.extract(clips)

        for layer_id, tokens in features.tokens_by_layer.items():
            expected = tokens.mean(dim=1)
            self.assertTrue(
                torch.allclose(features.pooled_by_layer[layer_id], expected),
                f"pooled_by_layer[{layer_id}] must be the mean of tokens_by_layer[{layer_id}]",
            )

    def test_out_layers_passed_to_encoder_are_zero_indexed(self) -> None:
        """Encoder must receive 0-based out_layers, not 1-based user-facing ids."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._make_adapter(Path(tmp))

        # For vit_base (depth=12), hierarchical 0-based: [2, 5, 8, 11]
        # user-facing 1-based selected_layers: [3, 6, 9, 12]
        expected_zero = tuple(s - 1 for s in adapter.selected_layers)
        self.assertEqual(adapter._out_layers_zero_indexed, expected_zero)
        self.assertEqual(
            sorted(adapter._encoder.out_layers),
            sorted(expected_zero),
        )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class JEPAV2_1RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_guard_for_tests()

    def tearDown(self) -> None:
        reset_runtime_guard_for_tests()

    def test_jepa_v2_1_is_registered(self) -> None:
        self.assertIn("jepa_v2_1", get_registered_adapters())

    def test_registry_factory_returns_correct_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_repo_layout(tmp_path / "repo")
            config_path = _make_minimal_config(tmp_path)
            checkpoint = tmp_path / "checkpoints" / "vitb.pt"
            _write_fake_checkpoint(checkpoint, key="ema_encoder")
            fake_vit = _build_fake_vit_module(_FakeV2_1Encoder)

            with mock.patch(
                "models.jepa_v2_1_adapter._import_vjepa2_1_vit_module",
                return_value=fake_vit,
            ):
                adapter = create_adapter(
                    "jepa_v2_1",
                    repo_root=tmp_path / "repo",
                    checkpoint_path=checkpoint,
                    config_path=config_path,
                )
        self.assertIsInstance(adapter, JEPAV2_1Adapter)


if __name__ == "__main__":
    unittest.main()
