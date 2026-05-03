from __future__ import annotations

import unittest
from unittest import mock

from experiments.health.run import run_health_layers


def _backbone_cfg_fixture() -> dict[str, object]:
    return {
        "jepa_v1": {
            "default_variant": "vith16_384",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {"vit_huge": 32},
            "variants": {"vith16_384": {"model_name": "vit_huge"}},
        },
        "jepa_v2": {
            "default_variant": "vitg_384",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {"vit_giant_xformers": 40},
            "variants": {"vitg_384": {"model_name": "vit_giant_xformers"}},
        },
        "jepa_v2_1": {
            "default_variant": "vitG_384",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {"vit_gigantic_xformers": 48},
            "variants": {"vitG_384": {"model_name": "vit_gigantic_xformers"}},
        },
        "videomae": {
            "default_variant": "vit_huge_16_224",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {"vit_huge": 32},
            "variants": {"vit_huge_16_224": {"model_name": "vit_huge"}},
        },
        "videomae_v2": {
            "default_variant": "vit_giant_16_224",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "model_block_depths": {"vit_giant": 40},
            "variants": {"vit_giant_16_224": {"model_name": "vit_giant"}},
        },
        "ltx_video": {
            "default_variant": "ltxv_13b_0_9_8_distilled",
            "default_relative_depths": [0.25, 0.5, 0.75, 1.0],
            "default_noise_levels": [0.9, 0.5, 0.1],
            "model_block_depths": {"ltx_transformer_48": 48},
            "variants": {"ltxv_13b_0_9_8_distilled": {"model_name": "ltx_transformer_48"}},
        },
    }


def _benchmark_cfg(backbone_name: str, layer_ids: list[int] | None = None) -> dict[str, object]:
    return {
        "backbone": {"name": backbone_name, "kwargs": {}},
        "feature_cache": {"layer_ids": layer_ids if layer_ids is not None else []},
    }


class HealthLayersRunTests(unittest.TestCase):
    def test_layer_mapping_report_passes_for_valid_config(self) -> None:
        backbone_cfg = _backbone_cfg_fixture()
        mvp_cfg = _benchmark_cfg("jepa_v1", [])
        intphys2_cfg = _benchmark_cfg("jepa_v2_1", [12, 24, 38, 48])
        ssv2_cfg = _benchmark_cfg("videomae", [8, 16, 24, 32])

        with mock.patch(
            "experiments.health.run._load_yaml",
            side_effect=[backbone_cfg, mvp_cfg, intphys2_cfg, ssv2_cfg],
        ):
            report = run_health_layers({})

        self.assertTrue(report["ok"])
        self.assertEqual(report["exit_code"], 0)
        self.assertIn("Probe4Physics Layer Health Check", report["human_report"])

        jepa21 = next(
            item
            for item in report["variants"]
            if item.get("name") == "jepa_v2_1" and item.get("variant") == "vitG_384"
        )
        ltx = next(
            item
            for item in report["variants"]
            if item.get("name") == "ltx_video" and item.get("variant") == "ltxv_13b_0_9_8_distilled"
        )
        self.assertEqual(jepa21["selected_layers_1_based"], [12, 24, 38, 48])
        self.assertEqual(ltx["selected_layers_1_based"], list(range(1, 13)))
        self.assertEqual(jepa21["backbone_layer_ids_1_based"][0], 1)
        self.assertEqual(jepa21["backbone_layer_ids_1_based"][-1], 48)

    def test_invalid_requested_layers_fail_and_strict_exit_sets_nonzero(self) -> None:
        backbone_cfg = _backbone_cfg_fixture()
        mvp_cfg = _benchmark_cfg("jepa_v1", [7])  # jepa_v1 default selection is [8, 16, 24, 32]
        intphys2_cfg = _benchmark_cfg("jepa_v2_1", [])
        ssv2_cfg = _benchmark_cfg("videomae", [])

        with mock.patch(
            "experiments.health.run._load_yaml",
            side_effect=[backbone_cfg, mvp_cfg, intphys2_cfg, ssv2_cfg],
        ):
            report = run_health_layers({"strict_exit": True})

        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], 1)
        mvp_request = next(item for item in report["benchmark_layer_requests"] if item.get("benchmark") == "mvp")
        self.assertEqual(mvp_request["missing_requested_layers"], [7])
        self.assertEqual(mvp_request["check"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
