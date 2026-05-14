from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.baseline_postprocess import run_baseline_label_backfill


class BaselinePostprocessTests(unittest.TestCase):
    def test_backfill_adds_original_metrics_and_writes_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mvp_single_frame"
            run_dir = root / "mvp_single_frame_jepa_v1_vith16_384_linear_123"
            layer_dir = run_dir / "layer_8"
            all_true_dir = layer_dir / "single_frame_all_true"
            all_false_dir = layer_dir / "single_frame_all_false"
            all_true_dir.mkdir(parents=True)
            all_false_dir.mkdir(parents=True)

            prediction_file = layer_dir / "probe_predictions.json"
            prediction_file.write_text("{}", encoding="utf-8")
            (all_true_dir / "metrics.json").write_text(
                json.dumps({"accuracy": 100.0, "pair_consistency": 100.0}),
                encoding="utf-8",
            )
            (all_false_dir / "metrics.json").write_text(
                json.dumps({"accuracy": 0.0, "pair_consistency": 0.0}),
                encoding="utf-8",
            )
            (all_true_dir / "run_config.snapshot.yaml").write_text(
                "\n".join(
                    [
                        "annotation_file: annotations.jsonl",
                        "official_repo_root: third_party/minimal_video_pairs",
                        "videos_root: videos",
                        "cache_dir: cache",
                        "split_name: test",
                        "seed: 42",
                        "predictor:",
                        "  mode: from_file",
                        f"  prediction_file: {prediction_file}",
                        f"output_dir: {layer_dir}",
                        "output_subdir: single_frame_all_true",
                        "split:",
                        "  dir: splits",
                        "label_override: 1",
                    ]
                ),
                encoding="utf-8",
            )
            summary_path = layer_dir / "probe_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(layer_dir / "probe_best.pt"),
                        "dataset": "mvp_single_frame",
                        "feature_signature": "sig",
                        "metrics": {"accuracy": 100.0, "pair_consistency": 100.0},
                        "metrics_by_label_mode": {
                            "all_true": {"accuracy": 100.0, "pair_consistency": 100.0},
                            "all_false": {"accuracy": 0.0, "pair_consistency": 0.0},
                        },
                        "objective_metric": 100.0,
                        "objective_metric_name": "pair_consistency",
                        "prediction_file": str(prediction_file),
                        "probe_eval_dir": str(layer_dir),
                        "probe_name": "linear",
                        "single_frame_evals": {
                            "all_true": {
                                "label_mode": "all_true",
                                "label_value": 1,
                                "probe_eval_dir": str(all_true_dir),
                                "metrics": {"accuracy": 100.0, "pair_consistency": 100.0},
                                "base_eval": {"output_dir": str(all_true_dir)},
                            },
                            "all_false": {
                                "label_mode": "all_false",
                                "label_value": 0,
                                "probe_eval_dir": str(all_false_dir),
                                "metrics": {"accuracy": 0.0, "pair_consistency": 0.0},
                                "base_eval": {"output_dir": str(all_false_dir)},
                            },
                        },
                        "split_name": "test",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            def fake_mvp_eval(config):
                output_dir = Path(config["output_dir"]) / str(config["output_subdir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                metrics = {"accuracy": 77.0, "pair_consistency": 66.0}
                (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                return {"output_dir": str(output_dir), "metrics": metrics}

            with mock.patch("training.baseline_postprocess.run_mvp_eval", side_effect=fake_mvp_eval):
                result = run_baseline_label_backfill(
                    {"baseline_backfill": {"roots": [str(root)]}}
                )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_label_mode"], "original")
            self.assertEqual(payload["metrics_by_label_mode"]["original"]["accuracy"], 77.0)
            self.assertEqual(payload["objective_metric"], 66.0)
            self.assertTrue((run_dir / "baseline_label_metrics.csv").exists())
            self.assertTrue((root / "baseline_label_metrics_all.csv").exists())
            self.assertEqual(
                result["baseline_backfill"]["totals"]["summaries_updated"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
