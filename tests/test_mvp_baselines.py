from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from benchmarks.mvp import features_displacement, features_single_frame
from benchmarks.mvp import eval as mvp_eval


class MVPBaselineTests(unittest.TestCase):
    def test_single_frame_clip_repeats_one_frame_from_original_clip(self) -> None:
        clip = torch.arange(1 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 3, 4, 2, 2)
        clip_fn = features_single_frame._make_single_frame_clip_fn()

        with mock.patch(
            "benchmarks.mvp.features_single_frame._decode_video_clip",
            return_value=clip,
        ):
            repeated = clip_fn({"sample_id": "mvp_sample_a", "video_path": "ignored.mp4"}, 4, 224)

        chosen_idx = features_single_frame._frame_index_for_sample("mvp_sample_a", 4)
        expected = clip[:, :, chosen_idx : chosen_idx + 1, :, :].expand(-1, -1, 4, -1, -1)
        self.assertTrue(torch.equal(repeated, expected))

    def test_single_frame_extraction_forces_test_split(self) -> None:
        config = {
            "split_name": "val",
            "feature_cache": {"split_names": ["train", "val", "test"]},
        }

        with mock.patch(
            "benchmarks.mvp.features_single_frame.run_mvp_feature_extraction",
            return_value={"ok": True},
        ) as extract:
            with mock.patch("benchmarks.mvp.features_single_frame._write_single_frame_metadata"):
                features_single_frame.run_mvp_single_frame_extraction(config)

        called_config = extract.call_args.args[0]
        self.assertEqual(called_config["feature_cache"]["split_names"], ["test"])
        self.assertEqual(called_config["split_name"], "test")
        self.assertEqual(called_config["baseline_tag"], "single_frame")

    def test_displacement_extraction_forces_test_split(self) -> None:
        config = {
            "split_name": "train",
            "feature_cache": {"split_names": ["train", "val", "test"]},
        }

        with mock.patch(
            "benchmarks.mvp.features_displacement.run_mvp_feature_extraction",
            return_value={"ok": True},
        ) as extract:
            with mock.patch("benchmarks.mvp.features_displacement.resolve_expected_feature_cache_paths") as paths:
                paths.return_value = SimpleNamespace(index_path=Path("/tmp/missing-index.parquet"))
                features_displacement.run_mvp_displacement_extraction(config)

        called_config = extract.call_args.args[0]
        self.assertEqual(called_config["feature_cache"]["split_names"], ["test"])
        self.assertEqual(called_config["split_name"], "test")
        self.assertEqual(called_config["baseline_tag"], "displacement")

    def test_mvp_label_override_uses_yes_no_choice_indices(self) -> None:
        sample = SimpleNamespace(
            sample_id="sample",
            pair_id="pair",
            question="Is it possible?",
            choices=("no", "yes"),
            answer_idx=0,
            plausibility_label=0,
            yes_choice_idx=1,
            no_choice_idx=0,
            video_a_ref="video.mp4",
            video_b_ref="",
            split="test",
        )

        overridden = mvp_eval._apply_label_override(sample, 1)

        self.assertEqual(overridden.answer_idx, 1)
        self.assertEqual(overridden.plausibility_label, 1)


if __name__ == "__main__":
    unittest.main()
