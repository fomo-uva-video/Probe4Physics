from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from benchmarks.ssv2 import (
    SSv2Benchmark,
    SSv2Metrics,
    SSv2Prediction,
    SSv2Sample,
    resolve_expected_feature_cache_paths,
)
from benchmarks.ssv2.core import asdict_metrics
from benchmarks.ssv2.data import (
    SSV2_NUM_CLASSES,
    derive_sample_id,
    load_ssv2_annotations,
    load_ssv2_labels,
)
from benchmarks.ssv2.eval import ConfigError, run_ssv2_eval
from benchmarks.ssv2.init import InitConfigError, run_ssv2_init


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, int] = {f"action_{i}": i for i in range(5)}


def _make_labels_file(tmp: Path, label_map: dict[str, int] | None = None) -> Path:
    """Write a labels.json file and return its path."""
    data = label_map if label_map is not None else _LABEL_MAP
    path = tmp / "labels.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_annotation_rows(n: int = 10, split_labels: dict[str, int] | None = None) -> list[dict]:
    """Build synthetic SSv2 annotation rows."""
    lmap = split_labels if split_labels is not None else _LABEL_MAP
    label_names = list(lmap.keys())
    rows = []
    for i in range(n):
        label_name = label_names[i % len(label_names)]
        rows.append(
            {
                "id": str(10000 + i),
                "label": label_name,
                "template": label_name.replace("_", " [something]"),
            }
        )
    return rows


def _write_annotation_file(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _make_full_init_config(tmp: Path) -> dict:
    """Write annotation files and return a valid init config."""
    train_rows = _make_annotation_rows(n=10)
    val_rows = _make_annotation_rows(n=5)
    labels_file = _make_labels_file(tmp)
    train_file = tmp / "train.json"
    val_file = tmp / "val.json"
    _write_annotation_file(train_file, train_rows)
    _write_annotation_file(val_file, val_rows)
    return {
        "train_annotation_file": str(train_file),
        "val_annotation_file": str(val_file),
        "labels_file": str(labels_file),
        "seed": 42,
        "split": {
            "dir": str(tmp / "splits" / "ssv2"),
            "max_samples_per_class": 200,
        },
    }


# ---------------------------------------------------------------------------
# data.py tests
# ---------------------------------------------------------------------------

class DeriveSampleIdTests(unittest.TestCase):
    def test_returns_string_of_id_field(self) -> None:
        row = {"id": "123456"}
        self.assertEqual(derive_sample_id(row), "123456")

    def test_numeric_id_converted_to_string(self) -> None:
        row = {"id": 999}
        self.assertEqual(derive_sample_id(row), "999")

    def test_different_ids_give_different_sample_ids(self) -> None:
        a = derive_sample_id({"id": "1"})
        b = derive_sample_id({"id": "2"})
        self.assertNotEqual(a, b)


class LoadSSv2LabelsTests(unittest.TestCase):
    def test_loads_correct_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            data = {"action_push": 0, "action_pull": 1, "action_drop": 2}
            path.write_text(json.dumps(data), encoding="utf-8")
            label_map = load_ssv2_labels(path)
            self.assertEqual(label_map["action_push"], 0)
            self.assertEqual(label_map["action_pull"], 1)
            self.assertEqual(label_map["action_drop"], 2)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ssv2_labels(Path("/does/not/exist.json"))

    def test_non_dict_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ssv2_labels(path)


class LoadSSv2AnnotationsTests(unittest.TestCase):
    def test_loads_list_of_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            rows = _make_annotation_rows(n=5)
            _write_annotation_file(path, rows)
            loaded = load_ssv2_annotations(path)
            self.assertEqual(len(loaded), 5)
            self.assertIn("id", loaded[0])
            self.assertIn("label", loaded[0])

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ssv2_annotations(Path("/does/not/exist.json"))

    def test_non_list_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps({"key": "val"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ssv2_annotations(path)


# ---------------------------------------------------------------------------
# core.py tests
# ---------------------------------------------------------------------------

def _make_samples(n: int = 6) -> list[SSv2Sample]:
    label_names = list(_LABEL_MAP.keys())
    return [
        SSv2Sample(
            sample_id=str(i),
            label_idx=i % len(_LABEL_MAP),
            label_name=label_names[i % len(label_names)],
            template=label_names[i % len(label_names)],
            video_ref=f"{i}.webm",
            split="val",
        )
        for i in range(n)
    ]


class SSv2BenchmarkLoadSamplesTests(unittest.TestCase):
    def test_loads_correct_count(self) -> None:
        benchmark = SSv2Benchmark()
        rows = _make_annotation_rows(n=8)
        # Simulate pre-normalised rows (as stored in split_clips.parquet)
        parquet_rows = [
            {
                "sample_id": str(r["id"]),
                "label_idx": _LABEL_MAP.get(str(r["label"]), 0),
                "label_name": str(r["label"]),
                "template": str(r.get("template", r["label"])),
                "video_ref": f"{r['id']}.webm",
                "split": "val",
            }
            for r in rows
        ]
        samples = benchmark.load_samples(parquet_rows, split="val")
        self.assertEqual(len(samples), 8)

    def test_samples_are_sorted_by_sample_id(self) -> None:
        benchmark = SSv2Benchmark()
        rows = [
            {"sample_id": "20", "label_idx": 1, "label_name": "a", "template": "a", "video_ref": "20.webm", "split": "val"},
            {"sample_id": "10", "label_idx": 0, "label_name": "b", "template": "b", "video_ref": "10.webm", "split": "val"},
        ]
        samples = benchmark.load_samples(rows, split="val")
        self.assertEqual(samples[0].sample_id, "10")
        self.assertEqual(samples[1].sample_id, "20")

    def test_split_field_is_overridden_by_argument(self) -> None:
        benchmark = SSv2Benchmark()
        rows = [{"sample_id": "1", "label_idx": 0, "label_name": "a", "template": "a", "video_ref": "1.webm", "split": "train"}]
        samples = benchmark.load_samples(rows, split="val")
        self.assertEqual(samples[0].split, "val")


class SSv2BenchmarkEvaluateTests(unittest.TestCase):
    def test_oracle_top1_is_100(self) -> None:
        samples = _make_samples(n=5)
        predictions = [SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx) for s in samples]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.top1_accuracy, 100.0)

    def test_wrong_predictions_give_0_top1(self) -> None:
        samples = _make_samples(n=5)
        n_classes = len(_LABEL_MAP)
        predictions = [
            SSv2Prediction(sample_id=s.sample_id, pred_idx=(s.label_idx + 1) % n_classes)
            for s in samples
        ]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.top1_accuracy, 0.0)

    def test_top5_with_scores_uses_top5_indices(self) -> None:
        # 10 classes; correct answer is class 3; put class 3 in top-5 but not top-1
        n_classes = 10
        samples = [
            SSv2Sample(
                sample_id="s0",
                label_idx=3,
                label_name="c3",
                template="c3",
                video_ref="s0.webm",
                split="val",
            )
        ]
        # scores: class 0 is highest (top-1 wrong), class 3 is 4th highest (in top-5)
        scores = [0.5, 0.2, 0.15, 0.08, 0.03] + [0.004] * 5
        predictions = [SSv2Prediction(sample_id="s0", pred_idx=0, scores=tuple(scores))]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        self.assertAlmostEqual(metrics.top1_accuracy, 0.0)
        self.assertAlmostEqual(metrics.top5_accuracy, 100.0)

    def test_top5_without_scores_degrades_to_top1(self) -> None:
        samples = _make_samples(n=4)
        # No scores vector supplied
        predictions = [SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx) for s in samples]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        # top5 == top1 when no scores
        self.assertAlmostEqual(metrics.top5_accuracy, metrics.top1_accuracy)

    def test_n_samples_and_n_classes_are_correct(self) -> None:
        samples = _make_samples(n=5)
        predictions = [SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx) for s in samples]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        self.assertEqual(metrics.n_samples, 5)
        self.assertEqual(metrics.n_classes, 5)  # 5 distinct labels for n=5 with 5 classes in _LABEL_MAP

    def test_top1_by_template_is_populated(self) -> None:
        samples = _make_samples(n=5)
        predictions = [SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx) for s in samples]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        self.assertGreater(len(metrics.top1_by_template), 0)

    def test_asdict_metrics_is_serializable(self) -> None:
        samples = _make_samples(n=5)
        predictions = [SSv2Prediction(sample_id=s.sample_id, pred_idx=s.label_idx) for s in samples]
        metrics = SSv2Benchmark().evaluate(samples, predictions)
        d = asdict_metrics(metrics)
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# init.py tests
# ---------------------------------------------------------------------------

class SSv2InitTests(unittest.TestCase):
    def test_init_writes_split_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _make_full_init_config(tmp_path)
            result = run_ssv2_init(cfg)

            split_dir = Path(result["split_dir"])
            self.assertTrue((split_dir / "split_clips.parquet").exists())
            self.assertTrue((split_dir / "manifest.json").exists())

            manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("annotation_sha256", manifest)
            self.assertIn("stats", manifest)
            self.assertGreater(manifest["stats"]["n_samples"], 0)

    def test_init_manifest_contains_n_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _make_full_init_config(tmp_path)
            result = run_ssv2_init(cfg)
            manifest = json.loads(
                (Path(result["split_dir"]) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["stats"]["n_classes"], len(_LABEL_MAP))

    def test_init_missing_annotation_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_file = _make_labels_file(tmp_path)
            with self.assertRaises(FileNotFoundError):
                run_ssv2_init(
                    {
                        "train_annotation_file": str(tmp_path / "no_train.json"),
                        "val_annotation_file": str(tmp_path / "no_val.json"),
                        "labels_file": str(labels_file),
                        "seed": 42,
                        "split": {"dir": str(tmp_path / "splits")},
                    }
                )

    def test_init_non_contiguous_labels_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Labels with a gap: indices 0, 1, 3 (missing 2)
            bad_labels = {"a": 0, "b": 1, "c": 3}
            labels_file = tmp_path / "labels.json"
            labels_file.write_text(json.dumps(bad_labels), encoding="utf-8")

            train_rows = [{"id": "1", "label": "a", "template": "a"}]
            val_rows = [{"id": "2", "label": "b", "template": "b"}]
            train_file = tmp_path / "train.json"
            val_file = tmp_path / "val.json"
            _write_annotation_file(train_file, train_rows)
            _write_annotation_file(val_file, val_rows)

            with self.assertRaises(InitConfigError):
                run_ssv2_init(
                    {
                        "train_annotation_file": str(train_file),
                        "val_annotation_file": str(val_file),
                        "labels_file": str(labels_file),
                        "seed": 42,
                        "split": {"dir": str(tmp_path / "splits")},
                    }
                )

    def test_init_missing_config_keys_raises(self) -> None:
        with self.assertRaises(InitConfigError):
            run_ssv2_init({"train_annotation_file": "x.json"})  # missing required keys

    def test_subset_by_class_limits_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Use max_per_class=2 with many samples; only 2 per class should survive
            label_map = {"action_0": 0, "action_1": 1}
            labels_file = tmp_path / "labels.json"
            labels_file.write_text(json.dumps(label_map), encoding="utf-8")

            # 20 samples per class
            train_rows = []
            for i in range(40):
                label = "action_0" if i < 20 else "action_1"
                train_rows.append({"id": str(i), "label": label, "template": label})
            val_rows = [{"id": "100", "label": "action_0", "template": "action_0"}]

            train_file = tmp_path / "train.json"
            val_file = tmp_path / "val.json"
            _write_annotation_file(train_file, train_rows)
            _write_annotation_file(val_file, val_rows)

            result = run_ssv2_init(
                {
                    "train_annotation_file": str(train_file),
                    "val_annotation_file": str(val_file),
                    "labels_file": str(labels_file),
                    "seed": 42,
                    "split": {
                        "dir": str(tmp_path / "splits"),
                        "max_samples_per_class": 2,
                    },
                }
            )
            # train: 2 per class × 2 classes = 4 max; val: 1 sample
            manifest = json.loads(
                (Path(result["split_dir"]) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertLessEqual(
                manifest["stats"]["samples_by_split"].get("train", 0), 4
            )


# ---------------------------------------------------------------------------
# eval.py tests
# ---------------------------------------------------------------------------

class SSv2EvalTests(unittest.TestCase):
    def _setup(self, tmp_path: Path) -> tuple[dict, dict]:
        """Create split artifacts and return (init_config, eval_config)."""
        init_cfg = _make_full_init_config(tmp_path)
        run_ssv2_init(init_cfg)
        eval_cfg = {
            "train_annotation_file": init_cfg["train_annotation_file"],
            "val_annotation_file": init_cfg["val_annotation_file"],
            "labels_file": init_cfg["labels_file"],
            "split_name": "val",
            "seed": 42,
            "predictor": {"mode": "oracle", "prediction_file": ""},
            "split": init_cfg["split"],
            "output_dir": str(tmp_path / "results"),
            "output_subdir": "eval_run",
        }
        return init_cfg, eval_cfg

    def test_oracle_eval_gives_100_top1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, eval_cfg = self._setup(tmp_path)
            result = run_ssv2_eval(eval_cfg)
            metrics = result["metrics"]
            self.assertAlmostEqual(metrics["top1_accuracy"], 100.0)
            self.assertGreater(metrics["n_samples"], 0)

    def test_random_predictor_runs_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, eval_cfg = self._setup(tmp_path)
            eval_cfg["predictor"] = {"mode": "random", "prediction_file": ""}
            eval_cfg["output_subdir"] = "random_run"
            result = run_ssv2_eval(eval_cfg)
            self.assertIn("metrics", result)
            self.assertIn("output_dir", result)

    def test_eval_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, eval_cfg = self._setup(tmp_path)
            result = run_ssv2_eval(eval_cfg)
            out = Path(result["output_dir"])
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "predictions.csv").exists())
            self.assertTrue((out / "summary.md").exists())
            self.assertTrue((out / "provenance.json").exists())

    def test_eval_fails_on_missing_split_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            init_cfg = _make_full_init_config(tmp_path)
            # Do NOT run init — split artifacts are missing
            eval_cfg = {
                "train_annotation_file": init_cfg["train_annotation_file"],
                "val_annotation_file": init_cfg["val_annotation_file"],
                "labels_file": init_cfg["labels_file"],
                "split_name": "val",
                "seed": 42,
                "predictor": {"mode": "oracle", "prediction_file": ""},
                "split": init_cfg["split"],
                "output_dir": str(tmp_path / "results"),
                "output_subdir": "no_init_run",
            }
            with self.assertRaises(FileNotFoundError):
                run_ssv2_eval(eval_cfg)

    def test_eval_from_file_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            init_cfg, eval_cfg = self._setup(tmp_path)

            # Build oracle prediction file from parquet
            try:
                import pandas as pd
            except ImportError:
                self.skipTest("pandas not installed")

            split_dir = Path(init_cfg["split"]["dir"])
            parquet_rows = pd.read_parquet(split_dir / "split_clips.parquet").to_dict(orient="records")
            val_rows = [r for r in parquet_rows if str(r.get("split", "")) == "val"]

            pred_dict = {
                str(r["sample_id"]): {"pred_idx": int(r["label_idx"]), "scores": []}
                for r in val_rows
            }
            pred_file = tmp_path / "preds.json"
            pred_file.write_text(json.dumps(pred_dict), encoding="utf-8")

            eval_cfg["predictor"] = {"mode": "from_file", "prediction_file": str(pred_file)}
            eval_cfg["output_subdir"] = "from_file_run"
            result = run_ssv2_eval(eval_cfg)
            self.assertAlmostEqual(result["metrics"]["top1_accuracy"], 100.0)

    def test_eval_missing_config_key_raises(self) -> None:
        with self.assertRaises(ConfigError):
            run_ssv2_eval({"split_name": "val"})  # many keys missing

    def test_eval_unknown_predictor_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, eval_cfg = self._setup(tmp_path)
            eval_cfg["predictor"] = {"mode": "unsupported", "prediction_file": ""}
            eval_cfg["output_subdir"] = "bad_mode_run"
            with self.assertRaises(ConfigError):
                run_ssv2_eval(eval_cfg)


# ---------------------------------------------------------------------------
# run.py / experiments/registry integration tests
# ---------------------------------------------------------------------------

class RunCommandSSv2Tests(unittest.TestCase):
    def test_ssv2_commands_are_registered(self) -> None:
        try:
            import run
        except ModuleNotFoundError:
            self.skipTest("hydra-core not installed; skipping run.py import test")

        expected = {
            "init.ssv2",
            "eval.ssv2",
            "extract.ssv2",
            "train.linear.ssv2",
            "eval.linear.ssv2",
        }
        self.assertTrue(expected.issubset(set(run.COMMANDS)))

    def test_ssv2_alias_is_registered(self) -> None:
        try:
            import run
        except ModuleNotFoundError:
            self.skipTest("hydra-core not installed; skipping run.py import test")

        self.assertEqual(run.ALIASES.get("ssv2"), "eval.ssv2")

    def test_ssv2_experiment_is_registered(self) -> None:
        from experiments.registry import get_experiment

        spec = get_experiment("ssv2.jepa_v1.linear")
        self.assertEqual(spec.name, "ssv2.jepa_v1.linear")
        self.assertIn("extract.ssv2", spec.pipeline)
        self.assertIn("train.linear.ssv2", spec.pipeline)
        self.assertIn("eval.linear.ssv2", spec.pipeline)

    def test_all_three_experiments_are_listed(self) -> None:
        from experiments.registry import list_experiments

        names = {spec.name for spec in list_experiments()}
        self.assertIn("mvp.jepa_v1.linear", names)
        self.assertIn("intphys2.jepa_v1.linear", names)
        self.assertIn("ssv2.jepa_v1.linear", names)


class SSv2FeatureCacheConfigTests(unittest.TestCase):
    def _write_backbones_config(self, path: Path, *, default_variant: str) -> Path:
        payload = {
            "ltx_video": {
                "default_variant": default_variant,
                "default_relative_depths": [0.5, 1.0],
                "model_block_depths": {"ltx_vae_5": 5},
                "variants": {
                    "ltxv_13b_0_9_8_dev": {
                        "hf_model_id": "demo/ltx-dev",
                        "model_name": "ltx_vae_5",
                        "frames_per_clip": 16,
                        "crop_size": 224,
                        "patch_size": 4,
                        "patch_size_t": 1,
                    },
                    "ltx_2b_0_9_8_distilled": {
                        "hf_model_id": "demo/ltx-2b-distilled",
                        "model_name": "ltx_vae_5",
                        "frames_per_clip": 16,
                        "crop_size": 224,
                        "patch_size": 4,
                        "patch_size_t": 1,
                    },
                },
            }
        }
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def _base_config(self, root: Path, *, config_path: Path) -> dict:
        split_dir = root / "splits" / "ssv2"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.json").write_text(
            json.dumps({"annotation_sha256": {"train": "fixture_train_sha"}}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "train_annotation_file": str(root / "train.json"),
            "val_annotation_file": str(root / "val.json"),
            "labels_file": str(root / "labels.json"),
            "videos_root": str(root / "videos"),
            "cache_dir": str(root / "cache"),
            "split": {"dir": str(split_dir)},
            "backbone": {
                "name": "ltx_video",
                "kwargs": {
                    "config_path": str(config_path),
                },
            },
            "feature_cache": {
                "dir": str(root / "features"),
                "split_names": ["train"],
                "layer_ids": [],
                "include_pooled": True,
                "include_tokens": True,
            },
        }

    def test_decode_crop_size_changes_cache_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_backbones_config(
                root / "backbones.yaml",
                default_variant="ltxv_13b_0_9_8_dev",
            )
            cfg_a = self._base_config(root, config_path=config_path)
            cfg_a["decode"] = {"num_frames": 2, "sampling": "uniform", "crop_size": 4}
            cfg_b = self._base_config(root, config_path=config_path)
            cfg_b["decode"] = {"num_frames": 2, "sampling": "uniform", "crop_size": 8}

            paths_a = resolve_expected_feature_cache_paths(cfg_a)
            paths_b = resolve_expected_feature_cache_paths(cfg_b)

        self.assertNotEqual(paths_a.signature, paths_b.signature)

    def test_default_variant_changes_cache_signature_when_variant_not_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path_a = self._write_backbones_config(
                root / "backbones_a.yaml",
                default_variant="ltxv_13b_0_9_8_dev",
            )
            config_path_b = self._write_backbones_config(
                root / "backbones_b.yaml",
                default_variant="ltx_2b_0_9_8_distilled",
            )
            cfg_a = self._base_config(root, config_path=config_path_a)
            cfg_b = self._base_config(root, config_path=config_path_b)

            paths_a = resolve_expected_feature_cache_paths(cfg_a)
            paths_b = resolve_expected_feature_cache_paths(cfg_b)

        self.assertNotEqual(paths_a.signature, paths_b.signature)
        self.assertNotEqual(paths_a.cache_dir, paths_b.cache_dir)


if __name__ == "__main__":
    unittest.main()
