from __future__ import annotations

import unittest
from unittest import mock

from experiments.health.run import run_health


def _dataset_result(name: str) -> dict[str, object]:
    return {
        "kind": "dataset",
        "name": name,
        "status": "pass",
        "checks": [],
    }


def _failing_dataset_result(name: str) -> dict[str, object]:
    return {
        "kind": "dataset",
        "name": name,
        "status": "fail",
        "checks": [
            {
                "name": "fixture_missing",
                "status": "fail",
                "detail": "missing fixture",
            }
        ],
    }


class HealthRunTests(unittest.TestCase):
    def test_lightweight_mode_skips_synthetic_forward(self) -> None:
        backbone_cfg = {
            "jepa_v1": {
                "variants": {"vith16_384": {"checkpoint_filename": "vith16-384.pth.tar"}},
                "checkpoints_dir": "data/checkpoints/jepa_v1",
            },
            "jepa_v2": {
                "variants": {"vitg_384": {"checkpoint_filename": "vitg-384.pth"}},
                "checkpoints_dir": "data/checkpoints/jepa_v2",
            },
            "jepa_v2_1": {
                "variants": {"vitG_384": {"checkpoint_filename": "vjepa2_1_vitG_384.pt"}},
                "checkpoints_dir": "data/checkpoints/jepa_v2_1",
            },
            "videomae": {
                "variants": {"vit_huge_16_224": {"hf_model_id": "demo/videomae-huge"}},
            },
            "videomae_v2": {
                "variants": {"vit_giant_16_224": {"hf_model_id": "demo/videomaev2-giant"}},
            },
            "ltx_video": {
                "variants": {"ltxv_13b_0_9_8_distilled": {"hf_model_id": "demo/ltx-video"}},
            },
        }

        with mock.patch("experiments.health.run._load_yaml", side_effect=[backbone_cfg, {}, {}, {}]):
            with mock.patch("experiments.health.run.get_registered_adapters", return_value=tuple(backbone_cfg)):
                with mock.patch("experiments.health.run._check_mvp", return_value=_dataset_result("mvp")):
                    with mock.patch("experiments.health.run._check_intphys2", return_value=_dataset_result("intphys2")):
                        with mock.patch("experiments.health.run._check_ssv2", return_value=_dataset_result("ssv2")):
                            with mock.patch("experiments.health.run._run_backbone_probe_subprocess") as probe_mock:
                                report = run_health({})

        self.assertFalse(report["mode"]["synthetic_forward"])
        self.assertEqual(probe_mock.call_count, 0)
        self.assertIn("lightweight static checks", report["human_report"])

    def test_deep_mode_runs_synthetic_forward(self) -> None:
        backbone_cfg = {
            "jepa_v1": {
                "variants": {"vith16_384": {"checkpoint_filename": "vith16-384.pth.tar"}},
                "checkpoints_dir": "data/checkpoints/jepa_v1",
            },
            "jepa_v2": {
                "variants": {"vitg_384": {"checkpoint_filename": "vitg-384.pth"}},
                "checkpoints_dir": "data/checkpoints/jepa_v2",
            },
            "jepa_v2_1": {
                "variants": {"vitG_384": {"checkpoint_filename": "vjepa2_1_vitG_384.pt"}},
                "checkpoints_dir": "data/checkpoints/jepa_v2_1",
            },
            "videomae": {
                "variants": {"vit_huge_16_224": {"hf_model_id": "demo/videomae-huge"}},
            },
            "videomae_v2": {
                "variants": {"vit_giant_16_224": {"hf_model_id": "demo/videomaev2-giant"}},
            },
            "ltx_video": {
                "variants": {"ltxv_13b_0_9_8_distilled": {"hf_model_id": "demo/ltx-video"}},
            },
        }

        with mock.patch("experiments.health.run._load_yaml", side_effect=[backbone_cfg, {}, {}, {}]):
            with mock.patch("experiments.health.run.get_registered_adapters", return_value=tuple(backbone_cfg)):
                with mock.patch("experiments.health.run._check_mvp", return_value=_dataset_result("mvp")):
                    with mock.patch("experiments.health.run._check_intphys2", return_value=_dataset_result("intphys2")):
                        with mock.patch("experiments.health.run._check_ssv2", return_value=_dataset_result("ssv2")):
                            with mock.patch(
                                "experiments.health.run._run_backbone_probe_subprocess",
                                return_value=(True, "ok"),
                            ) as probe_mock:
                                report = run_health({"synthetic_forward": True, "device": "cuda"})

        self.assertTrue(report["mode"]["synthetic_forward"])
        self.assertEqual(report["mode"]["device"], "cuda")
        self.assertEqual(probe_mock.call_count, 6)
        first_call = probe_mock.call_args_list[0]
        self.assertEqual(first_call.args[1]["device"], "cuda")
        self.assertIn("deep synthetic-forward smoke (device=cuda)", report["human_report"])

    def test_failed_checks_are_report_only_by_default(self) -> None:
        backbone_cfg = {
            "jepa_v1": {"variants": {"vith16_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v1"},
            "jepa_v2": {"variants": {"vitg_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v2"},
            "jepa_v2_1": {"variants": {"vitG_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v2_1"},
            "videomae": {"variants": {"vit_huge_16_224": {}}},
            "videomae_v2": {"variants": {"vit_giant_16_224": {}}},
            "ltx_video": {"variants": {"ltxv_13b_0_9_8_distilled": {}}},
        }

        with mock.patch("experiments.health.run._load_yaml", side_effect=[backbone_cfg, {}, {}, {}]):
            with mock.patch("experiments.health.run.get_registered_adapters", return_value=tuple(backbone_cfg)):
                with mock.patch("experiments.health.run._check_mvp", return_value=_failing_dataset_result("mvp")):
                    with mock.patch("experiments.health.run._check_intphys2", return_value=_dataset_result("intphys2")):
                        with mock.patch("experiments.health.run._check_ssv2", return_value=_dataset_result("ssv2")):
                            report = run_health({})

        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], 0)

    def test_strict_exit_returns_nonzero_for_failed_checks(self) -> None:
        backbone_cfg = {
            "jepa_v1": {"variants": {"vith16_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v1"},
            "jepa_v2": {"variants": {"vitg_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v2"},
            "jepa_v2_1": {"variants": {"vitG_384": {}}, "checkpoints_dir": "data/checkpoints/jepa_v2_1"},
            "videomae": {"variants": {"vit_huge_16_224": {}}},
            "videomae_v2": {"variants": {"vit_giant_16_224": {}}},
            "ltx_video": {"variants": {"ltxv_13b_0_9_8_distilled": {}}},
        }

        with mock.patch("experiments.health.run._load_yaml", side_effect=[backbone_cfg, {}, {}, {}]):
            with mock.patch("experiments.health.run.get_registered_adapters", return_value=tuple(backbone_cfg)):
                with mock.patch("experiments.health.run._check_mvp", return_value=_failing_dataset_result("mvp")):
                    with mock.patch("experiments.health.run._check_intphys2", return_value=_dataset_result("intphys2")):
                        with mock.patch("experiments.health.run._check_ssv2", return_value=_dataset_result("ssv2")):
                            report = run_health({"strict_exit": True})

        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
