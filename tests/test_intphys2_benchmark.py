from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.intphys2 import (
    IntPhys2Benchmark,
    IntPhys2Metrics,
    IntPhys2Prediction,
    IntPhys2Sample,
)
from benchmarks.intphys2.core import asdict_metrics
from benchmarks.intphys2.data import (
    derive_sample_id,
    load_intphys2_rows,
    normalize_intphys2_row,
)
from benchmarks.intphys2.eval import ConfigError, run_intphys2_eval
from benchmarks.intphys2.init import InitConfigError, run_intphys2_init


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata_rows(n_scenes: int = 4, split: str = "main") -> list[dict]:
    """Build a synthetic IntPhys2 metadata fixture with quadruplets."""
    rows = []
    conditions = ["permanence", "immutability", "continuity", "solidity"]
    for i in range(n_scenes):
        scene_id = f"scene_{i:04d}"
        condition = conditions[i % len(conditions)]
        # 2 possible, 2 impossible per scene
        for j, plausibility in enumerate([1, 1, 0, 0]):
            rows.append(
                {
                    "video_path": f"videos/{scene_id}/clip_{j}.mp4",
                    "scene_id": scene_id,
                    "condition": condition,
                    "plausibility": plausibility,
                    "split": split,
                }
            )
    return rows


def _write_metadata_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# data.py tests
# ---------------------------------------------------------------------------

class NormalizeRowTests(unittest.TestCase):
    def test_canonical_fields_pass_through(self) -> None:
        row = {
            "video_path": "v.mp4",
            "scene_id": "s1",
            "condition": "permanence",
            "plausibility": 1,
            "split": "main",
        }
        norm = normalize_intphys2_row(row)
        self.assertEqual(norm["video_path"], "v.mp4")
        self.assertEqual(norm["scene_id"], "s1")
        self.assertEqual(norm["plausibility"], 1)

    def test_alias_columns_are_resolved(self) -> None:
        row = {
            "filename": "clip.mp4",
            "scene": "sc42",
            "principle": "solidity",
            "label": 0,
            "subset": "debug",
        }
        norm = normalize_intphys2_row(row)
        self.assertEqual(norm["video_path"], "clip.mp4")
        self.assertEqual(norm["scene_id"], "sc42")
        self.assertEqual(norm["condition"], "solidity")
        self.assertEqual(norm["plausibility"], 0)
        self.assertEqual(norm["split"], "debug")

    def test_missing_fields_default_to_empty(self) -> None:
        norm = normalize_intphys2_row({})
        self.assertEqual(norm["video_path"], "")
        self.assertEqual(norm["scene_id"], "")
        self.assertEqual(norm["condition"], "")
        self.assertEqual(norm["plausibility"], -1)
        self.assertEqual(norm["split"], "")


class DerivesSampleIdTests(unittest.TestCase):
    def test_explicit_sample_id_is_used(self) -> None:
        row = {"sample_id": "s42", "video_path": "v.mp4"}
        self.assertEqual(derive_sample_id(row), "s42")

    def test_video_path_hash_is_stable(self) -> None:
        row = {"video_path": "scene_0001/clip_0.mp4"}
        sid1 = derive_sample_id(row)
        sid2 = derive_sample_id(row)
        self.assertEqual(sid1, sid2)
        self.assertEqual(len(sid1), 16)

    def test_different_paths_give_different_ids(self) -> None:
        a = derive_sample_id({"video_path": "scene_0001/clip_0.mp4"})
        b = derive_sample_id({"video_path": "scene_0001/clip_1.mp4"})
        self.assertNotEqual(a, b)


class LoadIntPhys2RowsTests(unittest.TestCase):
    def test_load_csv(self) -> None:
        rows = _make_metadata_rows(n_scenes=2)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = Path(tmp.name)
        _write_metadata_csv(path, rows)
        loaded = load_intphys2_rows(path)
        self.assertEqual(len(loaded), 8)
        path.unlink()

    def test_load_jsonl(self) -> None:
        rows = _make_metadata_rows(n_scenes=2)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = Path(tmp.name)
        _write_metadata_jsonl(path, rows)
        loaded = load_intphys2_rows(path)
        self.assertEqual(len(loaded), 8)
        path.unlink()

    def test_unsupported_extension_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_intphys2_rows(Path("metadata.xyz"))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_intphys2_rows(Path("/does/not/exist.csv"))


# ---------------------------------------------------------------------------
# core.py tests
# ---------------------------------------------------------------------------

class IntPhys2BenchmarkLoadSamplesTests(unittest.TestCase):
    def _make_rows(self) -> list[dict]:
        return _make_metadata_rows(n_scenes=2, split="main")

    def test_load_samples_returns_correct_count(self) -> None:
        benchmark = IntPhys2Benchmark()
        rows = self._make_rows()
        samples = benchmark.load_samples(rows, split="main")
        self.assertEqual(len(samples), 8)

    def test_samples_have_correct_split(self) -> None:
        benchmark = IntPhys2Benchmark()
        rows = self._make_rows()
        samples = benchmark.load_samples(rows, split="debug")
        self.assertTrue(all(s.split == "debug" for s in samples))

    def test_sample_ids_are_stable_across_calls(self) -> None:
        benchmark = IntPhys2Benchmark()
        rows = self._make_rows()
        s1 = benchmark.load_samples(rows, split="main")
        s2 = benchmark.load_samples(rows, split="main")
        ids1 = [s.sample_id for s in s1]
        ids2 = [s.sample_id for s in s2]
        self.assertEqual(ids1, ids2)

    def test_plausibility_values_are_0_or_1(self) -> None:
        benchmark = IntPhys2Benchmark()
        rows = self._make_rows()
        samples = benchmark.load_samples(rows, split="main")
        self.assertTrue(all(s.plausibility in (0, 1) for s in samples))


class IntPhys2BenchmarkEvaluateTests(unittest.TestCase):
    def _make_samples(self) -> list[IntPhys2Sample]:
        benchmark = IntPhys2Benchmark()
        return benchmark.load_samples(_make_metadata_rows(n_scenes=2), split="main")

    def test_oracle_predictions_give_100_accuracy(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(sample_id=s.sample_id, pred_idx=s.plausibility)
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.accuracy, 100.0)

    def test_inverted_predictions_give_0_accuracy(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(sample_id=s.sample_id, pred_idx=1 - s.plausibility)
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.accuracy, 0.0)

    def test_oracle_voe_is_100(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(
                sample_id=s.sample_id,
                pred_idx=s.plausibility,
                score=float(s.plausibility),
            )
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.voe_accuracy, 100.0)

    def test_inverted_scores_give_0_voe(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(
                sample_id=s.sample_id,
                pred_idx=1 - s.plausibility,
                score=float(1 - s.plausibility),
            )
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.voe_accuracy, 0.0)

    def test_n_scenes_is_correct(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(sample_id=s.sample_id, pred_idx=s.plausibility)
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        self.assertEqual(metrics.n_scenes, 2)
        self.assertEqual(metrics.n_samples, 8)

    def test_accuracy_by_condition_has_all_conditions(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(sample_id=s.sample_id, pred_idx=s.plausibility)
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        # 2 scenes = 2 different conditions (permanence and immutability for scenes 0 and 1)
        self.assertGreater(len(metrics.accuracy_by_condition), 0)

    def test_asdict_metrics_is_serializable(self) -> None:
        samples = self._make_samples()
        predictions = [
            IntPhys2Prediction(sample_id=s.sample_id, pred_idx=s.plausibility)
            for s in samples
        ]
        metrics = IntPhys2Benchmark().evaluate(samples, predictions)
        d = asdict_metrics(metrics)
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# init.py tests
# ---------------------------------------------------------------------------

class IntPhys2InitTests(unittest.TestCase):
    def _init_config(self, output_root: Path, metadata_file: Path) -> dict:
        return {
            "metadata_file": str(metadata_file),
            "split": {"dir": str(output_root / "splits" / "intphys2")},
        }

    def test_init_writes_split_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=8, split="main")
            _write_metadata_csv(metadata_file, rows)

            result = run_intphys2_init(self._init_config(tmp_path, metadata_file))

            split_dir = Path(result["split_dir"])
            self.assertTrue((split_dir / "split_scenes.parquet").exists())
            self.assertTrue((split_dir / "manifest.json").exists())

            manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("metadata_sha256", manifest)
            self.assertEqual(manifest["stats"]["n_scenes"], 8)
            self.assertEqual(manifest["stats"]["n_samples"], 32)

    def test_init_missing_metadata_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                run_intphys2_init(
                    {
                        "metadata_file": str(Path(tmp) / "does_not_exist.csv"),
                        "split": {"dir": str(Path(tmp) / "splits")},
                    }
                )

    def test_init_invalid_plausibility_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=2)
            rows[0]["plausibility"] = 5  # invalid
            _write_metadata_csv(metadata_file, rows)

            with self.assertRaises(InitConfigError):
                run_intphys2_init(self._init_config(tmp_path, metadata_file))

    def test_init_missing_config_keys_raises(self) -> None:
        with self.assertRaises(InitConfigError):
            run_intphys2_init({"metadata_file": "x.csv"})  # missing 'split'


# ---------------------------------------------------------------------------
# eval.py tests
# ---------------------------------------------------------------------------

class IntPhys2EvalTests(unittest.TestCase):
    def _init_config(self, output_root: Path, metadata_file: Path) -> dict:
        return {
            "metadata_file": str(metadata_file),
            "split": {"dir": str(output_root / "splits" / "intphys2")},
        }

    def _eval_config(self, output_root: Path, metadata_file: Path) -> dict:
        return {
            "metadata_file": str(metadata_file),
            "split_name": "main",
            "seed": 42,
            "predictor": {"mode": "oracle", "prediction_file": ""},
            "split": {"dir": str(output_root / "splits" / "intphys2")},
            "output_dir": str(output_root / "results"),
            "output_subdir": "eval_run",
        }

    def test_oracle_eval_gives_100_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=4, split="main")
            _write_metadata_csv(metadata_file, rows)

            run_intphys2_init(self._init_config(tmp_path, metadata_file))

            result = run_intphys2_eval(self._eval_config(tmp_path, metadata_file))
            metrics = result["metrics"]

            self.assertAlmostEqual(metrics["accuracy"], 100.0)
            self.assertGreater(metrics["n_samples"], 0)
            self.assertGreater(metrics["n_scenes"], 0)

    def test_eval_fails_on_metadata_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=4, split="main")
            _write_metadata_csv(metadata_file, rows)

            run_intphys2_init(self._init_config(tmp_path, metadata_file))

            # Overwrite the metadata with a different file
            bad_metadata = tmp_path / "bad_metadata.csv"
            rows_extra = rows + [
                {
                    "video_path": "extra/clip.mp4",
                    "scene_id": "extra_scene",
                    "condition": "permanence",
                    "plausibility": 1,
                    "split": "main",
                }
            ]
            _write_metadata_csv(bad_metadata, rows_extra)

            eval_cfg = self._eval_config(tmp_path, bad_metadata)

            with self.assertRaises(ConfigError):
                run_intphys2_eval(eval_cfg)

    def test_eval_fails_on_missing_split_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=2, split="main")
            _write_metadata_csv(metadata_file, rows)
            # Do NOT run init — split artifacts are missing

            eval_cfg = self._eval_config(tmp_path, metadata_file)

            with self.assertRaises(FileNotFoundError):
                run_intphys2_eval(eval_cfg)

    def test_eval_random_predictor_runs_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=4, split="main")
            _write_metadata_csv(metadata_file, rows)

            run_intphys2_init(self._init_config(tmp_path, metadata_file))

            eval_cfg = self._eval_config(tmp_path, metadata_file)
            eval_cfg["predictor"] = {"mode": "random", "prediction_file": ""}
            result = run_intphys2_eval(eval_cfg)

            self.assertIn("metrics", result)
            self.assertIn("output_dir", result)

    def test_eval_from_file_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_file = tmp_path / "metadata.csv"
            rows = _make_metadata_rows(n_scenes=2, split="main")
            _write_metadata_csv(metadata_file, rows)

            run_intphys2_init(self._init_config(tmp_path, metadata_file))

            # Build oracle predictions from a file
            benchmark = IntPhys2Benchmark()
            all_rows = load_intphys2_rows(metadata_file)
            norm_rows = [normalize_intphys2_row(r) for r in all_rows]
            samples = benchmark.load_samples(norm_rows, split="main")
            pred_dict = {s.sample_id: {"pred_idx": s.plausibility, "score": float(s.plausibility)} for s in samples}
            pred_file = tmp_path / "preds.json"
            pred_file.write_text(json.dumps(pred_dict), encoding="utf-8")

            eval_cfg = self._eval_config(tmp_path, metadata_file)
            eval_cfg["predictor"] = {"mode": "from_file", "prediction_file": str(pred_file)}
            eval_cfg["output_subdir"] = "from_file_run"
            result = run_intphys2_eval(eval_cfg)

            self.assertAlmostEqual(result["metrics"]["accuracy"], 100.0)
            self.assertAlmostEqual(result["metrics"]["voe_accuracy"], 100.0)


# ---------------------------------------------------------------------------
# run.py integration tests
# ---------------------------------------------------------------------------

class RunCommandIntPhys2Tests(unittest.TestCase):
    def test_intphys2_commands_are_registered(self) -> None:
        try:
            import run
        except ModuleNotFoundError:
            self.skipTest("hydra-core not installed; skipping run.py import test")

        expected = {
            "init.intphys2",
            "eval.intphys2",
            "extract.intphys2",
            "train.linear.intphys2",
            "eval.linear.intphys2",
        }
        self.assertTrue(expected.issubset(set(run.COMMANDS)))

    def test_intphys2_experiment_is_registered(self) -> None:
        from experiments.registry import get_experiment

        spec = get_experiment("intphys2.jepa_v1.linear")
        self.assertEqual(spec.name, "intphys2.jepa_v1.linear")
        self.assertIn("extract.intphys2", spec.pipeline)
        self.assertIn("train.linear.intphys2", spec.pipeline)
        self.assertIn("eval.linear.intphys2", spec.pipeline)


if __name__ == "__main__":
    unittest.main()
