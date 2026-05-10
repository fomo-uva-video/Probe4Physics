from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training.intphys2_postprocess import run_intphys2_backfill_roc_auc
from training.run_probe import _write_train_eval_summary_csv


class IntPhys2BackfillRocAucTests(unittest.TestCase):
    def test_backfill_updates_saved_intphys_artifacts_and_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            intphys_root = repo_root / "artifacts" / "probes" / "intphys2"
            run_dir = intphys_root / "toy_run"
            layer_dir = run_dir / "layer_8"
            train_dir = layer_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)

            split_eval_dirs: dict[str, str] = {}
            split_evals: dict[str, dict[str, object]] = {}
            metrics_by_split: dict[str, dict[str, object]] = {}
            for split_name in ("train", "val", "test"):
                probe_eval_dir = layer_dir / "eval" / split_name
                self._write_probe_eval_dir(probe_eval_dir, split_name)
                split_eval_dirs[split_name] = str(probe_eval_dir)
                split_evals[split_name] = {
                    "split_name": split_name,
                    "probe_eval_dir": str(probe_eval_dir),
                    "metrics": self._base_metrics(),
                    "base_eval": {"metrics": self._base_metrics()},
                }
                metrics_by_split[split_name] = self._base_metrics()

            summary = {
                "dataset": "intphys2",
                "probe_name": "linear",
                "feature_view": "pooled",
                "train_split": "train",
                "objective_metric_name": "voe_accuracy",
                "split_name": "test",
                "reported_splits": ["train", "val", "test"],
                "layers": [
                    {
                        "layer": 8,
                        "layer_label": "8",
                        "checkpoint": str(train_dir / "trial_0000" / "probe_best.pt"),
                        "objective_metric": 100.0,
                        "train": {"objective_metric": "voe_accuracy"},
                        "eval": {
                            "split_name": "test",
                            "reported_splits": ["train", "val", "test"],
                            "probe_eval_dir": str(layer_dir / "eval" / "test"),
                            "split_eval_dirs": split_eval_dirs,
                            "split_evals": split_evals,
                            "metrics_by_split": metrics_by_split,
                            "metrics": self._base_metrics(),
                            "base_eval": {"metrics": self._base_metrics()},
                        },
                    }
                ],
            }

            summary_json_path = run_dir / "train_eval_summary.json"
            summary_json_path.parent.mkdir(parents=True, exist_ok=True)
            summary_json_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_train_eval_summary_csv(run_dir / "train_eval_summary.csv", summary)

            docs_dir = repo_root / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(run_dir / "train_eval_summary.csv", docs_dir / "train_eval_summary.csv")

            best_summary_csv = intphys_root / "intphys2_optuna_best_summary.csv"
            self._write_best_summary_csv(best_summary_csv, train_dir)

            result = run_intphys2_backfill_roc_auc(
                {
                    "postprocess": {
                        "artifacts_root": str(intphys_root),
                        "best_summary_csvs": [str(best_summary_csv)],
                        "copied_summary_csvs": [str(docs_dir / "train_eval_summary.csv")],
                    }
                }
            )

            self.assertEqual(result["updated_train_eval_summaries"], 1)
            self.assertGreaterEqual(result["updated_eval_dirs"], 3)
            self.assertEqual(result["updated_best_summary_csvs"], 1)
            self.assertEqual(result["updated_copied_summary_csvs"], 1)

            metrics = json.loads(
                (layer_dir / "eval" / "test" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertAlmostEqual(metrics["roc_auc"], 1.0)

            probe_eval_summary = json.loads(
                (layer_dir / "eval" / "test" / "probe_eval_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertAlmostEqual(probe_eval_summary["metrics"]["roc_auc"], 1.0)
            self.assertAlmostEqual(probe_eval_summary["base_eval"]["metrics"]["roc_auc"], 1.0)

            train_eval_summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                train_eval_summary["layers"][0]["eval"]["metrics_by_split"]["test"]["roc_auc"],
                1.0,
            )
            self.assertAlmostEqual(
                train_eval_summary["layers"][0]["eval"]["metrics"]["roc_auc"],
                1.0,
            )

            fieldnames, rows = self._read_csv(run_dir / "train_eval_summary.csv")
            self.assertIn("train_roc_auc", fieldnames)
            self.assertIn("val_roc_auc", fieldnames)
            self.assertIn("test_roc_auc", fieldnames)
            self.assertEqual(rows[0]["test_roc_auc"], "1.0")

            docs_fieldnames, docs_rows = self._read_csv(docs_dir / "train_eval_summary.csv")
            self.assertEqual(fieldnames, docs_fieldnames)
            self.assertEqual(rows, docs_rows)

            best_fieldnames, best_rows = self._read_csv(best_summary_csv)
            self.assertIn("train_roc_auc", best_fieldnames)
            self.assertIn("val_roc_auc", best_fieldnames)
            self.assertIn("test_roc_auc", best_fieldnames)
            self.assertEqual(best_rows[0]["val_roc_auc"], "1.0")

    def _write_probe_eval_dir(self, probe_eval_dir: Path, split_name: str) -> None:
        probe_eval_dir.mkdir(parents=True, exist_ok=True)
        self._write_predictions_csv(probe_eval_dir / "predictions.csv")
        metrics = self._base_metrics()
        (probe_eval_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary = {
            "split_name": split_name,
            "probe_eval_dir": str(probe_eval_dir),
            "metrics": dict(metrics),
            "base_eval": {"metrics": dict(metrics)},
        }
        (probe_eval_dir / "probe_eval_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_predictions_csv(self, path: Path) -> None:
        rows = [
            {
                "sample_id": "a",
                "scene_id": "scene_1",
                "condition": "continuity",
                "plausibility": 1,
                "pred_idx": 1,
                "score": "0.95",
                "is_correct": 1,
                "video_ref": "videos/a.mp4",
            },
            {
                "sample_id": "b",
                "scene_id": "scene_1",
                "condition": "continuity",
                "plausibility": 1,
                "pred_idx": 1,
                "score": "0.80",
                "is_correct": 1,
                "video_ref": "videos/b.mp4",
            },
            {
                "sample_id": "c",
                "scene_id": "scene_1",
                "condition": "continuity",
                "plausibility": 0,
                "pred_idx": 0,
                "score": "0.20",
                "is_correct": 1,
                "video_ref": "videos/c.mp4",
            },
            {
                "sample_id": "d",
                "scene_id": "scene_1",
                "condition": "continuity",
                "plausibility": 0,
                "pred_idx": 0,
                "score": "0.05",
                "is_correct": 1,
                "video_ref": "videos/d.mp4",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_best_summary_csv(self, path: Path, train_dir: Path) -> None:
        rows = [
            {
                "probe": "linear",
                "backbone": "toy_backbone",
                "run_timestamp": "20260510T000000Z",
                "layer": "8",
                "objective_metric": "voe_accuracy",
                "best_value": "100.0",
                "best_trial_number": "0",
                "n_trials": "1",
                "input_dim": "4",
                "num_classes": "2",
                "batch_size": "4",
                "epochs": "1",
                "lr": "0.001",
                "weight_decay": "0.0",
                "hidden_dims": "",
                "dropout": "",
                "num_heads": "",
                "num_self_attn_blocks": "",
                "mlp_ratio": "",
                "train_dir": str(train_dir),
                "best_output_dir": str(train_dir / "trial_0000"),
                "best_checkpoint": str(train_dir / "trial_0000" / "probe_best.pt"),
            }
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _base_metrics(self) -> dict[str, object]:
        return {
            "accuracy": 100.0,
            "n_samples": 4,
            "n_scenes": 1,
            "voe_accuracy": 100.0,
        }

    def _read_csv(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
