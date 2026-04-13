from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.mvp import MVPBenchmark, MVPPrediction, OfficialIntegrationError
from evaluation.mvp_data import load_mvp_rows
from evaluation.mvp_eval import run_mvp_eval


class MVPBenchmarkStrictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.fixture = self.repo_root / "tests" / "fixtures" / "mvp_mini_fixture.jsonl"
        self.official_repo = self.repo_root / "third_party" / "minimal_video_pairs"

    def test_import_failure_is_hard_error(self) -> None:
        with self.assertRaises(OfficialIntegrationError):
            MVPBenchmark(self.repo_root / "third_party" / "does_not_exist")

    def test_official_scoring_path(self) -> None:
        benchmark = MVPBenchmark(self.official_repo)
        rows = load_mvp_rows(self.fixture)
        samples = benchmark.load_samples(rows, split="train")

        # Pair alpha all-correct, pair beta has one wrong -> accuracy=75, pair=50.
        predictions = [
            MVPPrediction(sample_id="pair_alpha_0", pred_idx=0),
            MVPPrediction(sample_id="pair_alpha_1", pred_idx=1),
            MVPPrediction(sample_id="pair_beta_0", pred_idx=0),
            MVPPrediction(sample_id="pair_beta_1", pred_idx=0),
        ]

        metrics = benchmark.evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.accuracy, 75.0)
        self.assertAlmostEqual(metrics.pair_consistency, 50.0)
        self.assertEqual(metrics.n_samples, 4)
        self.assertEqual(metrics.n_pairs, 2)


class MVPSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.selection_fixture = self.repo_root / "tests" / "fixtures" / "mvp_selection_fixture.jsonl"

    def _base_config(self, output_dir: Path, output_subdir: str) -> dict:
        return {
            "annotation_file": str(self.selection_fixture),
            "videos_root": str(output_dir / "videos"),
            "cache_dir": str(output_dir / "cache"),
            "official_repo_root": str(self.repo_root / "third_party" / "minimal_video_pairs"),
            "split": "train",
            "max_pairs": 0,
            "seed": 7,
            "decode": {"backend": "pyav", "num_frames": 16, "fps": 4, "sampling": "uniform"},
            "materialize_missing": False,
            "download_timeout_seconds": 10,
            "predictor": {"mode": "oracle", "prediction_file": ""},
            "selection": {
                "enabled": True,
                "subset": "intuitive_physics",
                "plausibility_only": True,
                "plausibility_keywords": [
                    "is this video physically plausible",
                    "is the outcome of the experiment plausible",
                ],
                "require_binary_yes_no": True,
                "yes_aliases": ["yes", "plausible", "possible"],
                "no_aliases": ["no", "implausible", "impossible"],
                "include_pair_ids": [],
                "exclude_pair_ids": [],
                "include_question_contains": [],
                "drop_incomplete_pairs": True,
                "artifacts": {"enabled": True},
            },
            "output_dir": str(output_dir),
            "output_subdir": output_subdir,
        }

    def test_selection_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._base_config(tmp_path / "out", "selection_run")
            Path(config["videos_root"]).mkdir(parents=True, exist_ok=True)

            result = run_mvp_eval(config)
            run_dir = Path(result["output_dir"])

            self.assertEqual(result["metrics"]["n_samples"], 4)
            self.assertEqual(result["metrics"]["n_pairs"], 2)
            self.assertAlmostEqual(result["metrics"]["accuracy"], 100.0)
            self.assertAlmostEqual(result["metrics"]["pair_consistency"], 100.0)

            kept_csv = run_dir / "selection_kept.csv"
            dropped_csv = run_dir / "selection_dropped.csv"
            report_json = run_dir / "selection_report.json"

            self.assertTrue(kept_csv.exists())
            self.assertTrue(dropped_csv.exists())
            self.assertTrue(report_json.exists())

            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["total_rows"], 11)
            self.assertEqual(report["kept_rows"], 4)
            self.assertEqual(report["dropped_rows"], 7)
            self.assertEqual(report["dropped_by_reason"]["subset_mismatch"], 2)
            self.assertEqual(report["dropped_by_reason"]["not_plausibility_question"], 2)
            self.assertEqual(report["dropped_by_reason"]["not_binary_yes_no"], 2)
            self.assertEqual(report["dropped_by_reason"]["incomplete_pair_1"], 1)

            header = kept_csv.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("question", header)
            self.assertIn("choices", header)

    def test_include_pair_ids_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._base_config(tmp_path / "out", "include_pair_run")
            Path(config["videos_root"]).mkdir(parents=True, exist_ok=True)
            config["selection"]["include_pair_ids"] = ["pair_included"]

            result = run_mvp_eval(config)
            self.assertEqual(result["metrics"]["n_samples"], 2)
            self.assertEqual(result["metrics"]["n_pairs"], 1)

    def test_missing_annotations_with_autodownload_disabled_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._base_config(tmp_path / "out", "missing_annotations")
            Path(config["videos_root"]).mkdir(parents=True, exist_ok=True)
            config["annotation_file"] = str(tmp_path / "does_not_exist.jsonl")
            config["annotations"] = {"auto_download": False}

            with self.assertRaises(FileNotFoundError):
                run_mvp_eval(config)


class MVPEndToEndReproTests(unittest.TestCase):
    def test_repeated_run_same_metrics_and_selection_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        fixture = repo_root / "tests" / "fixtures" / "mvp_selection_fixture.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "out"
            videos_root = tmp_path / "videos"
            videos_root.mkdir(parents=True, exist_ok=True)

            config = {
                "annotation_file": str(fixture),
                "videos_root": str(videos_root),
                "cache_dir": str(tmp_path / "cache"),
                "official_repo_root": str(repo_root / "third_party" / "minimal_video_pairs"),
                "split": "train",
                "max_pairs": 0,
                "seed": 7,
                "decode": {"backend": "pyav", "num_frames": 16, "fps": 4, "sampling": "uniform"},
                "materialize_missing": False,
                "download_timeout_seconds": 10,
                "predictor": {"mode": "oracle", "prediction_file": ""},
                "selection": {
                    "enabled": True,
                    "subset": "intuitive_physics",
                    "plausibility_only": True,
                    "plausibility_keywords": [
                        "is this video physically plausible",
                        "is the outcome of the experiment plausible",
                    ],
                    "require_binary_yes_no": True,
                    "yes_aliases": ["yes", "plausible", "possible"],
                    "no_aliases": ["no", "implausible", "impossible"],
                    "include_pair_ids": [],
                    "exclude_pair_ids": [],
                    "include_question_contains": [],
                    "drop_incomplete_pairs": True,
                    "artifacts": {"enabled": True},
                },
                "output_dir": str(output_root),
                "output_subdir": "run1",
            }

            result_1 = run_mvp_eval(config)
            run1 = Path(result_1["output_dir"])

            config["output_subdir"] = "run2"
            result_2 = run_mvp_eval(config)
            run2 = Path(result_2["output_dir"])

            metrics_1 = json.loads((run1 / "metrics.json").read_text(encoding="utf-8"))
            metrics_2 = json.loads((run2 / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_1, metrics_2)

            report_1 = json.loads((run1 / "selection_report.json").read_text(encoding="utf-8"))
            report_2 = json.loads((run2 / "selection_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_1, report_2)


if __name__ == "__main__":
    unittest.main()
