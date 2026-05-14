from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from benchmarks.intphys2 import features_displacement
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

    def test_single_frame_extraction_forces_test_split(self) -> None:
        config = {
            "split_name": "val",
            "feature_cache": {"split_names": ["train", "val", "test"]},
        }

        with mock.patch(
            "benchmarks.intphys2.features_single_frame.run_intphys2_feature_extraction",
            return_value={"ok": True},
        ) as extract:
            with mock.patch("benchmarks.intphys2.features_single_frame._write_single_frame_metadata"):
                features_single_frame.run_intphys2_single_frame_extraction(config)

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
            "benchmarks.intphys2.features_displacement.run_intphys2_feature_extraction",
            return_value={"ok": True},
        ) as extract:
            with mock.patch(
                "benchmarks.intphys2.features_displacement.resolve_expected_feature_cache_paths"
            ) as resolve_paths:
                resolve_paths.return_value = SimpleNamespace(index_path=Path("/tmp/missing-index.parquet"))
                features_displacement.run_intphys2_displacement_extraction(config)

        called_config = extract.call_args.args[0]
        self.assertEqual(called_config["feature_cache"]["split_names"], ["test"])
        self.assertEqual(called_config["split_name"], "test")
        self.assertEqual(called_config["baseline_tag"], "displacement")

    def test_single_frame_eval_probe_scores_original_all_true_and_all_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_file = root / "probe_predictions.json"
            pred_file.write_text("{}", encoding="utf-8")

            def fake_eval(cfg):
                label_override = cfg.get("label_override")
                if label_override is None:
                    metrics = {"accuracy": 42.0, "roc_auc": None, "voe_accuracy": 12.5}
                    label_name = "original"
                else:
                    label_value = int(label_override)
                    metrics = {
                        "accuracy": 100.0 if label_value == 1 else 0.0,
                        "roc_auc": None,
                        "voe_accuracy": 0.0,
                    }
                    label_name = "all_true" if label_value == 1 else "all_false"
                return {
                    "metrics": metrics,
                    "output_dir": str(root / f"single_frame_{label_name}"),
                }

            spec = SimpleNamespace(
                eval_runner=fake_eval,
                objective_metric="voe_accuracy",
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
                "split_name": "val",
                "feature_cache": {"split_names": ["train", "val", "test"]},
                "probe": {
                    "name": "linear",
                    "feature_view": "pooled",
                    "layer": "last",
                    "eval_output_dir": str(root / "results"),
                    "eval_output_subdir": "single_frame_eval",
                    "checkpoint_path": str(root / "probe_best.pt"),
                },
            }

            with mock.patch("training.run_probe._load_eval_context", return_value=context) as load_context:
                with mock.patch(
                    "training.run_probe._write_prediction_payload_from_context",
                    return_value=pred_file,
                ):
                    summary = run_probe.run_intphys2_single_frame_eval_probe(config)

            called_config = load_context.call_args.args[1]
            self.assertEqual(called_config["feature_cache"]["split_names"], ["test"])
            self.assertEqual(called_config["split_name"], "test")
            self.assertIn("metrics_by_label_mode", summary)
            self.assertEqual(summary["primary_label_mode"], "original")
            self.assertIn("original", summary["metrics_by_label_mode"])
            self.assertIn("all_true", summary["metrics_by_label_mode"])
            self.assertIn("all_false", summary["metrics_by_label_mode"])
            self.assertEqual(summary["metrics_by_label_mode"]["original"]["accuracy"], 42.0)
            self.assertEqual(summary["objective_metric"], 12.5)
            self.assertEqual(summary["metrics_by_label_mode"]["all_true"]["accuracy"], 100.0)
            self.assertEqual(summary["metrics_by_label_mode"]["all_false"]["accuracy"], 0.0)
            summary_path = root / "results" / "single_frame_eval" / "probe_eval_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_evals", payload)
            self.assertIn("single_frame_evals", payload)
            self.assertTrue((root / "results" / "baseline_label_metrics.csv").exists())

    def test_displacement_eval_probe_scores_original_all_true_and_all_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_file = root / "probe_predictions.json"
            pred_file.write_text("{}", encoding="utf-8")

            def fake_eval(cfg):
                label_override = cfg.get("label_override")
                if label_override is None:
                    metrics = {"accuracy": 55.0, "roc_auc": 0.5, "voe_accuracy": 25.0}
                    label_name = "original"
                else:
                    label_value = int(label_override)
                    metrics = {
                        "accuracy": 90.0 if label_value == 1 else 10.0,
                        "roc_auc": None,
                        "voe_accuracy": 0.0,
                    }
                    label_name = "all_true" if label_value == 1 else "all_false"
                return {
                    "metrics": metrics,
                    "output_dir": str(root / f"displacement_{label_name}"),
                }

            spec = SimpleNamespace(
                eval_runner=fake_eval,
                objective_metric="voe_accuracy",
            )
            context = EvalContext(
                spec=spec,
                manifest={"signature": "displacement_sig"},
                index=None,
                features=None,
                probe=object(),
                checkpoint_path=root / "probe_best.pt",
                current_signature="displacement_sig",
            )
            config = {
                "split_name": "val",
                "feature_cache": {"split_names": ["train", "val", "test"]},
                "probe": {
                    "name": "linear",
                    "feature_view": "pooled",
                    "layer": "last",
                    "eval_output_dir": str(root / "results"),
                    "eval_output_subdir": "displacement_eval",
                    "checkpoint_path": str(root / "probe_best.pt"),
                },
            }

            with mock.patch("training.run_probe._load_eval_context", return_value=context) as load_context:
                with mock.patch(
                    "training.run_probe._write_prediction_payload_from_context",
                    return_value=pred_file,
                ):
                    summary = run_probe.run_intphys2_displacement_eval_probe(config)

            called_config = load_context.call_args.args[1]
            self.assertEqual(called_config["feature_cache"]["split_names"], ["test"])
            self.assertEqual(called_config["split_name"], "test")
            self.assertEqual(called_config["baseline_tag"], "displacement")
            self.assertEqual(summary["primary_label_mode"], "original")
            self.assertEqual(summary["metrics_by_label_mode"]["original"]["accuracy"], 55.0)
            self.assertEqual(summary["metrics_by_label_mode"]["all_true"]["accuracy"], 90.0)
            self.assertEqual(summary["metrics_by_label_mode"]["all_false"]["accuracy"], 10.0)
            summary_path = root / "results" / "displacement_eval" / "probe_eval_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_evals", payload)
            self.assertIn("displacement_evals", payload)
