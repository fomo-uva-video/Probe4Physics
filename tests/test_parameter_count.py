from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "models" / "parameter_count.py"
SPEC = importlib.util.spec_from_file_location("parameter_count_under_test", MODULE_PATH)
assert SPEC is not None
parameter_count = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = parameter_count
SPEC.loader.exec_module(parameter_count)


class _FakeTensor:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = int(count)
        self.requires_grad = bool(requires_grad)

    def numel(self) -> int:
        return self._count


class _FakeModule:
    def named_parameters(self, recurse: bool = True):
        _ = recurse
        return iter(
            [
                ("blocks.0.attn.qkv.weight", _FakeTensor(10, requires_grad=True)),
                ("blocks.0.norm.weight", _FakeTensor(2, requires_grad=False)),
                ("head.weight", _FakeTensor(5, requires_grad=True)),
            ]
        )

    def named_buffers(self, recurse: bool = True):
        _ = recurse
        return iter([("running", _FakeTensor(7, requires_grad=False))])


class ParameterCountTests(unittest.TestCase):
    def test_count_module_parameters_groups_by_prefix(self) -> None:
        result = parameter_count.count_module_parameters(
            _FakeModule(),
            group_depth=1,
            include_buffers=True,
        )

        self.assertEqual(result["total"], 17)
        self.assertEqual(result["trainable"], 15)
        self.assertEqual(result["frozen"], 2)
        self.assertEqual(result["buffers"], {"total": 7, "tensors": 1})
        groups = {item["group"]: item for item in result["groups"]}
        self.assertEqual(groups["blocks"]["total"], 12)
        self.assertEqual(groups["head"]["total"], 5)

    def test_build_vit_parameter_table_reads_configured_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "backbones.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "jepa_v2_1": {
                            "variants": {
                                "vitl_384": {
                                    "model_name": "vit_large",
                                    "crop_size": 384,
                                    "patch_size": 16,
                                    "frames_per_clip": 16,
                                    "tubelet_size": 2,
                                },
                                "vitG_384": {
                                    "model_name": "vit_gigantic_xformers",
                                    "crop_size": 384,
                                    "patch_size": 16,
                                    "frames_per_clip": 16,
                                    "tubelet_size": 2,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            rows = parameter_count.build_vit_parameter_table(
                config_path=config_path,
                backbones=("jepa_v2_1",),
            )

        self.assertEqual([row["variant"] for row in rows], ["vitG_384", "vitl_384"])
        rows_by_variant = {row["variant"]: row for row in rows}
        self.assertFalse(rows_by_variant["vitG_384"]["is_default_variant"])
        self.assertGreater(
            rows_by_variant["vitG_384"]["total_parameters"],
            rows_by_variant["vitl_384"]["total_parameters"],
        )
        self.assertGreater(rows_by_variant["vitG_384"]["patch_embed_img_params"], 0)
        self.assertEqual(rows_by_variant["vitl_384"]["size_label"], "ViT-L")

    def test_select_size_comparison_keeps_vitl_and_largest(self) -> None:
        rows = [
            {
                "backbone": "demo",
                "variant": "vitb",
                "model_name": "vit_base",
                "depth": 12,
                "embed_dim": 768,
                "total_parameters": 1,
            },
            {
                "backbone": "demo",
                "variant": "vitl",
                "model_name": "vit_large",
                "depth": 24,
                "embed_dim": 1024,
                "total_parameters": 2,
            },
            {
                "backbone": "demo",
                "variant": "vitg_256",
                "model_name": "vit_giant_xformers",
                "depth": 40,
                "embed_dim": 1408,
                "total_parameters": 3,
                "is_default_variant": False,
            },
            {
                "backbone": "demo",
                "variant": "vitg_384",
                "model_name": "vit_giant_xformers",
                "depth": 40,
                "embed_dim": 1408,
                "total_parameters": 3,
                "is_default_variant": True,
            },
        ]

        selected = parameter_count.select_vit_size_comparison_rows(rows)

        self.assertEqual([row["variant"] for row in selected], ["vitl", "vitg_384"])


if __name__ == "__main__":
    unittest.main()
