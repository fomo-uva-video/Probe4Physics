from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from benchmarks.intphys2 import features_single_frame
from training import run_probe
from training.run_probe import EvalContext


class IntPhys2SingleFrameTests(unittest.TestCase):
    def test_single_frame_clip_repeats_one_frame_from_original_clip(self) -> None:
        clip = torch.arange(1 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 3, 4, 2, 2)
        clip_fn = features_single_frame._make_single_frame_clip_fn()

        with mock.patch(
            "benchmarks.intphys2.features_single_frame._decode_video_clip",
            return_value=clip,
        ):
            repeated = clip_fn({"sample_id": "sample_a", "video_path": "ignored.mp4"}, 4, 224)

        chosen_idx = features_single_frame._frame_index_for_sample("sample_a", 4)
        expected = clip[:, :, chosen_idx : chosen_idx + 1, :, :].expand(-1, -1, 4, -1, -1)
        self.assertTrue(torch.equal(repeated, expected))

    def test_single_frame_eval_probe_scores_all_true_and_all_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_file = root / "probe_predictions.json"
            pred_file.write_text("{}", encoding="utf-8")

            spec = SimpleNamespace(
                eval_runner=lambda cfg: {
                    "metrics": {
                        "accuracy": 100.0 if int(cfg["label_override"]) == 1 else 0.0,
                        "roc_auc": None,
                        "voe_accuracy": 0.0,
                    },
                    "output_dir": str(root / f"single_frame_{'all_true' if int(cfg['label_override']) == 1 else 'all_false'}"),
                }
            )
            context = EvalContext(
                spec=spec,
                manifest={"signature": "single_frame_sig"},
                index=None,
                features=None,
                probe=object(),
                checkpoint_path=root / "probe_best.pt",
                current_signature="single_frame_sig",
            )

            config = {
                "split_name": "test",
                "probe": {
                    "name": "linear",
                    "feature_view": "pooled",
                    "layer": "last",
                    "eval_output_dir": str(root / "results"),
                    "eval_output_subdir": "single_frame_eval",
                    "checkpoint_path": str(root / "probe_best.pt"),
                },
            }

            with mock.patch("training.run_probe._load_eval_context", return_value=context):
                with mock.patch(
                    "training.run_probe._write_prediction_payload_from_context",
                    return_value=pred_file,
                ):
                    summary = run_probe.run_intphys2_single_frame_eval_probe(config)

            self.assertIn("metrics_by_label_mode", summary)
            self.assertIn("all_true", summary["metrics_by_label_mode"])
            self.assertIn("all_false", summary["metrics_by_label_mode"])
            self.assertEqual(summary["metrics_by_label_mode"]["all_true"]["accuracy"], 100.0)
            self.assertEqual(summary["metrics_by_label_mode"]["all_false"]["accuracy"], 0.0)
            summary_path = root / "results" / "single_frame_eval" / "probe_eval_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("single_frame_evals", payload)
