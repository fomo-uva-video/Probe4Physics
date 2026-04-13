from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.mvp import MVPBenchmark, MVPPrediction, OfficialIntegrationError
from evaluation.mvp.data import load_mvp_rows
from evaluation.mvp.eval import ConfigError, run_mvp_eval
from evaluation.mvp.init import run_mvp_init
from evaluation.splitting import build_strata, build_units, materialize_sample_assignments, split_units


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


class SplittingCoreTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_keeps_pairs(self) -> None:
        rows = []
        for idx in range(12):
            pair_id = f"pair_{idx:02d}"
            source = "intphys" if idx % 2 == 0 else "clevrer"
            question_template = "is this video physically plausible"
            rows.append(
                {
                    "sample_id": f"{pair_id}_0",
                    "pair_id": pair_id,
                    "source": source,
                    "question_template": question_template,
                }
            )
            rows.append(
                {
                    "sample_id": f"{pair_id}_1",
                    "pair_id": pair_id,
                    "source": source,
                    "question_template": question_template,
                }
            )

        units = build_units(rows, "pair_id")
        strata = build_strata(
            units,
            keys=["source", "question_template"],
            rare_strata_min_units=1,
            other_label="OTHER",
        )
        for unit in units:
            unit["stratum"] = strata[unit["unit_id"]]

        split_a = split_units(
            units,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42,
            stratified=True,
        )
        split_b = split_units(
            units,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42,
            stratified=True,
        )
        self.assertEqual(split_a, split_b)

        assignments = materialize_sample_assignments(rows, split_a, "pair_id")
        split_by_pair: dict[str, set[str]] = {}
        for row in assignments:
            split_by_pair.setdefault(str(row["pair_id"]), set()).add(str(row["split"]))

        for pair_id, split_names in split_by_pair.items():
            self.assertEqual(len(split_names), 1, f"Pair {pair_id} got split across sets: {split_names}")


class MVPInitEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.official_repo = self.repo_root / "third_party" / "minimal_video_pairs"

    def _write_large_annotation_fixture(self, path: Path, n_pairs: int = 30) -> None:
        rows: list[dict] = []
        questions = [
            "Is this video physically plausible after the collision?",
            "Is the outcome of the experiment plausible?",
        ]
        sources = ["intphys", "clevrer"]
        for i in range(n_pairs):
            pair_id = f"pair_{i:04d}"
            question = questions[i % len(questions)]
            source = sources[i % len(sources)]
            rows.append(
                {
                    "video_id": f"{pair_id}_0",
                    "subset": "intuitive_physics",
                    "source": source,
                    "video_path": f"subset/{pair_id}_0.mp4",
                    "question": question,
                    "candidates": ["Yes", "No"],
                    "answer": "Yes",
                    "label": 1,
                }
            )
            rows.append(
                {
                    "video_id": f"{pair_id}_1",
                    "subset": "intuitive_physics",
                    "source": source,
                    "video_path": f"subset/{pair_id}_1.mp4",
                    "question": question,
                    "candidates": ["Yes", "No"],
                    "answer": "No",
                    "label": 0,
                }
            )

        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _split_cfg(self, output_root: Path) -> dict:
        return {
            "dir": str(output_root / "mvp" / "fixture_60_20_20"),
            "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
            "group_key": "pair_id",
            "stratify_keys": ["source", "question_template"],
            "rare_strata_min_units": 1,
            "rare_bucket_label": "OTHER",
            "seed": 7,
            "template_coverage_check": False,
        }

    def _selection_cfg(self) -> dict:
        return {
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
        }

    def _init_config(self, output_root: Path, annotation_file: Path) -> dict:
        return {
            "annotation_file": str(annotation_file),
            "official_repo_root": str(self.official_repo),
            "annotations": {
                "auto_download": False,
                "dataset_id": "facebook/minimal_video_pairs",
                "hf_cache_dir": str(output_root / "hf"),
                "subsets": [
                    "human_object_interactions",
                    "intuitive_physics",
                    "robot_object_interactions",
                    "temporal_reasoning",
                ],
            },
            "selection": self._selection_cfg(),
            "split": self._split_cfg(output_root / "splits"),
        }

    def _eval_config(self, output_root: Path, annotation_file: Path) -> dict:
        return {
            "annotation_file": str(annotation_file),
            "videos_root": str(output_root / "videos"),
            "cache_dir": str(output_root / "cache"),
            "official_repo_root": str(self.official_repo),
            "annotations": {"auto_download": False},
            "split_name": "train",
            "max_pairs": 0,
            "seed": 7,
            "materialize_missing": False,
            "download_timeout_seconds": 10,
            "predictor": {"mode": "oracle", "prediction_file": ""},
            "split": self._split_cfg(output_root / "splits"),
            "output_dir": str(output_root / "results"),
            "output_subdir": "eval_run",
        }

    def test_init_writes_split_and_selection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_full_fixture.jsonl"
            self._write_large_annotation_fixture(annotation_file, n_pairs=30)
            result = run_mvp_init(self._init_config(tmp_path, annotation_file))

            split_dir = Path(result["split_dir"])
            self.assertTrue((split_dir / "split_pairs.parquet").exists())
            self.assertTrue((split_dir / "manifest.json").exists())
            self.assertTrue((split_dir / "selection_kept.csv").exists())
            self.assertTrue((split_dir / "selection_dropped.csv").exists())
            self.assertTrue((split_dir / "selection_report.json").exists())

            manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("annotation_sha256", manifest)
            self.assertIn("selection_sha256", manifest)
            self.assertIn("stats", manifest)
            self.assertEqual(manifest["selection"]["report"]["kept_rows"], 60)

    def test_eval_uses_precomputed_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_full_fixture.jsonl"
            self._write_large_annotation_fixture(annotation_file, n_pairs=30)
            run_mvp_init(self._init_config(tmp_path, annotation_file))

            eval_cfg = self._eval_config(tmp_path, annotation_file)
            Path(eval_cfg["videos_root"]).mkdir(parents=True, exist_ok=True)
            result = run_mvp_eval(eval_cfg)
            metrics = result["metrics"]

            self.assertGreater(metrics["n_samples"], 0)
            self.assertGreater(metrics["n_pairs"], 0)
            self.assertAlmostEqual(metrics["accuracy"], 100.0)
            self.assertAlmostEqual(metrics["pair_consistency"], 100.0)

    def test_eval_fails_on_annotation_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_full_fixture.jsonl"
            self._write_large_annotation_fixture(annotation_file, n_pairs=30)
            run_mvp_init(self._init_config(tmp_path, annotation_file))

            bad_annotation = tmp_path / "bad_annotation.jsonl"
            bad_annotation.write_text(
                annotation_file.read_text(encoding="utf-8") + '{"video_id":"extra_0","question":"x"}\n',
                encoding="utf-8",
            )

            eval_cfg = self._eval_config(tmp_path, bad_annotation)
            Path(eval_cfg["videos_root"]).mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ConfigError):
                run_mvp_eval(eval_cfg)


if __name__ == "__main__":
    unittest.main()
