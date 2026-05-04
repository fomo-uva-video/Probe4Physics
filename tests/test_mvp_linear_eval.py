from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from probes.linear import LinearProbe
import training.run_probe as run_probe
from training.run_probe import ProbeConfigError, run_mvp_eval_probe


class _StaticProbe:
    def __init__(self, predictions: list[int]) -> None:
        self._predictions = predictions

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        del x, batch_size
        return torch.tensor(self._predictions, dtype=torch.long)


class MVPLInearEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.official_repo = self.repo_root / "third_party" / "minimal_video_pairs"

    def _write_split_artifacts(self, split_dir: Path, annotation_file: Path) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "pair_id": "pair_alpha",
                    "split": "train",
                    "stratum": "all",
                    "source": "intphys",
                    "question_template": "is this video physically plausible after the collision",
                    "n_samples": 2,
                    "sample_ids_json": json.dumps(["pair_alpha_0", "pair_alpha_1"]),
                },
                {
                    "pair_id": "pair_beta",
                    "split": "train",
                    "stratum": "all",
                    "source": "intphys",
                    "question_template": "is this video physically plausible after the collision",
                    "n_samples": 2,
                    "sample_ids_json": json.dumps(["pair_beta_0", "pair_beta_1"]),
                },
            ]
        ).to_parquet(split_dir / "split_pairs.parquet", index=False)

        digest = hashlib.sha256(annotation_file.read_bytes()).hexdigest()
        (split_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "annotation_sha256": digest,
                    "selection_sha256": "fixture_selection_sha",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_eval_hard_fails_on_feature_signature_mismatch(self) -> None:
        fake_bundle = {
            "manifest": {"signature": "cache_sig"},
            "index": pd.DataFrame(
                [
                    {
                        "feature_index": 0,
                        "sample_id": "s0",
                        "pair_id": "p0",
                        "split": "test",
                        "answer_idx": 0,
                    }
                ]
            ),
            "pooled": {
                "selected_layers": [24],
                "by_layer": {24: torch.randn(1, 8)},
            },
            "tokens": None,
            "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
        }

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "linear.pt"
            probe = LinearProbe(input_dim=8, num_classes=2, device="cpu")
            probe.save(ckpt, metadata={"feature_signature": "other_sig"})

            config = {
                "split_name": "test",
                "probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "name": "linear",
                    "checkpoint_path": str(ckpt),
                    "device": "cpu",
                },
            }

            with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                with mock.patch("training.run_probe.run_mvp_eval") as mocked_eval:
                    with self.assertRaises(ProbeConfigError):
                        run_mvp_eval_probe(config)

            mocked_eval.assert_not_called()

    def test_eval_hard_fails_when_semantic_columns_are_missing(self) -> None:
        fake_bundle = {
            "manifest": {"signature": "cache_sig"},
            "index": pd.DataFrame(
                [
                    {
                        "feature_index": 0,
                        "sample_id": "s0",
                        "pair_id": "p0",
                        "split": "test",
                        "answer_idx": 0,
                    }
                ]
            ),
            "pooled": {
                "selected_layers": [24],
                "by_layer": {24: torch.randn(1, 8)},
            },
            "tokens": None,
            "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
        }

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "linear.pt"
            torch.save(
                {
                    "metadata": {
                        "feature_signature": "cache_sig",
                        "target_type": "semantic_plausibility",
                    }
                },
                ckpt,
            )

            config = {
                "split_name": "test",
                "probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "name": "linear",
                    "checkpoint_path": str(ckpt),
                    "device": "cpu",
                },
            }

            with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                with self.assertRaisesRegex(ProbeConfigError, "Re-run `python run.py extract.mvp`"):
                    run_mvp_eval_probe(config)

    def test_eval_translates_semantic_predictions_back_to_sample_specific_choice_indices(self) -> None:
        fake_bundle = {
            "manifest": {"signature": "cache_sig"},
            "index": pd.DataFrame(
                [
                    {
                        "feature_index": 0,
                        "sample_id": "yes_first_plausible",
                        "pair_id": "pair_a",
                        "split": "test",
                        "answer_idx": 0,
                        "plausibility_label": 1,
                        "yes_choice_idx": 0,
                        "no_choice_idx": 1,
                    },
                    {
                        "feature_index": 1,
                        "sample_id": "yes_first_implausible",
                        "pair_id": "pair_a",
                        "split": "test",
                        "answer_idx": 1,
                        "plausibility_label": 0,
                        "yes_choice_idx": 0,
                        "no_choice_idx": 1,
                    },
                    {
                        "feature_index": 2,
                        "sample_id": "no_first_plausible",
                        "pair_id": "pair_b",
                        "split": "test",
                        "answer_idx": 1,
                        "plausibility_label": 1,
                        "yes_choice_idx": 1,
                        "no_choice_idx": 0,
                    },
                    {
                        "feature_index": 3,
                        "sample_id": "no_first_implausible",
                        "pair_id": "pair_b",
                        "split": "test",
                        "answer_idx": 0,
                        "plausibility_label": 0,
                        "yes_choice_idx": 1,
                        "no_choice_idx": 0,
                    },
                    {
                        "feature_index": 4,
                        "sample_id": "train_anchor",
                        "pair_id": "pair_train",
                        "split": "train",
                        "answer_idx": 0,
                        "plausibility_label": 1,
                        "yes_choice_idx": 0,
                        "no_choice_idx": 1,
                    },
                    {
                        "feature_index": 5,
                        "sample_id": "val_anchor",
                        "pair_id": "pair_val",
                        "split": "val",
                        "answer_idx": 1,
                        "plausibility_label": 0,
                        "yes_choice_idx": 0,
                        "no_choice_idx": 1,
                    },
                ]
            ),
            "pooled": {
                "selected_layers": [24],
                "by_layer": {24: torch.randn(6, 8)},
            },
            "tokens": None,
            "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
        }

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "linear.pt"
            torch.save(
                {
                    "metadata": {
                        "feature_signature": "cache_sig",
                        "target_type": "semantic_plausibility",
                    }
                },
                ckpt,
            )

            config = {
                "split_name": "test",
                "probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "name": "linear",
                    "checkpoint_path": str(ckpt),
                    "device": "cpu",
                    "eval_output_dir": tmp,
                    "eval_output_subdir": "eval_run",
                },
            }

            single_split_spec = replace(run_probe.DATASET_SPECS["mvp"], report_splits=("test",))
            with mock.patch.dict(run_probe.DATASET_SPECS, {"mvp": single_split_spec}, clear=False):
                with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                    with mock.patch(
                        "training.run_probe.load_probe_from_checkpoint",
                        return_value=(_StaticProbe([1, 0, 1, 0]), {"metadata": {"feature_signature": "cache_sig", "target_type": "semantic_plausibility"}}),
                    ):
                        with mock.patch(
                            "training.run_probe.run_mvp_eval",
                            return_value={"metrics": {"accuracy": 100.0, "pair_consistency": 100.0}},
                        ) as mocked_eval:
                            summary = run_mvp_eval_probe(config)

            pred_file = Path(summary["prediction_file"])
            pred_payload = json.loads(pred_file.read_text(encoding="utf-8"))
            self.assertEqual(
                pred_payload,
                {
                    "no_first_implausible": 0,
                    "no_first_plausible": 1,
                    "yes_first_implausible": 1,
                    "yes_first_plausible": 0,
                },
            )
            mocked_eval.assert_called_once()

    def test_eval_reports_perfect_accuracy_when_semantic_predictions_are_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotation_file = tmp_path / "mvp_fixture.jsonl"
            rows = [
                {
                    "video_id": "pair_alpha_0",
                    "subset": "intuitive_physics",
                    "source": "intphys",
                    "video_path": "subset/pair_alpha_0.mp4",
                    "question": "Is this video physically plausible after the collision?",
                    "candidates": ["Yes", "No"],
                    "answer": "Yes",
                },
                {
                    "video_id": "pair_alpha_1",
                    "subset": "intuitive_physics",
                    "source": "intphys",
                    "video_path": "subset/pair_alpha_1.mp4",
                    "question": "Is this video physically plausible after the collision?",
                    "candidates": ["Yes", "No"],
                    "answer": "No",
                },
                {
                    "video_id": "pair_beta_0",
                    "subset": "intuitive_physics",
                    "source": "intphys",
                    "video_path": "subset/pair_beta_0.mp4",
                    "question": "Is this video physically plausible after the collision?",
                    "candidates": ["No", "Yes"],
                    "answer": "Yes",
                },
                {
                    "video_id": "pair_beta_1",
                    "subset": "intuitive_physics",
                    "source": "intphys",
                    "video_path": "subset/pair_beta_1.mp4",
                    "question": "Is this video physically plausible after the collision?",
                    "candidates": ["No", "Yes"],
                    "answer": "No",
                },
            ]
            with annotation_file.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

            split_dir = tmp_path / "splits" / "mvp_train_only"
            self._write_split_artifacts(split_dir, annotation_file)

            fake_bundle = {
                "manifest": {"signature": "cache_sig"},
                "index": pd.DataFrame(
                    [
                        {
                            "feature_index": 0,
                            "sample_id": "pair_alpha_0",
                            "pair_id": "pair_alpha",
                            "split": "train",
                            "answer_idx": 0,
                            "plausibility_label": 1,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 1,
                            "sample_id": "pair_alpha_1",
                            "pair_id": "pair_alpha",
                            "split": "train",
                            "answer_idx": 1,
                            "plausibility_label": 0,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 2,
                            "sample_id": "pair_beta_0",
                            "pair_id": "pair_beta",
                            "split": "train",
                            "answer_idx": 1,
                            "plausibility_label": 1,
                            "yes_choice_idx": 1,
                            "no_choice_idx": 0,
                        },
                        {
                            "feature_index": 3,
                            "sample_id": "pair_beta_1",
                            "pair_id": "pair_beta",
                            "split": "train",
                            "answer_idx": 0,
                            "plausibility_label": 0,
                            "yes_choice_idx": 1,
                            "no_choice_idx": 0,
                        },
                        {
                            "feature_index": 4,
                            "sample_id": "pair_gamma_0",
                            "pair_id": "pair_gamma",
                            "split": "val",
                            "answer_idx": 0,
                            "plausibility_label": 1,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 5,
                            "sample_id": "pair_delta_0",
                            "pair_id": "pair_delta",
                            "split": "test",
                            "answer_idx": 1,
                            "plausibility_label": 0,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                    ]
                ),
                "pooled": {
                    "selected_layers": [24],
                    "by_layer": {24: torch.randn(6, 8)},
                },
                "tokens": None,
                "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
            }

            ckpt = tmp_path / "linear.pt"
            probe = LinearProbe(input_dim=8, num_classes=2, device="cpu")
            probe.save(
                ckpt,
                metadata={
                    "feature_signature": "cache_sig",
                    "target_type": "semantic_plausibility",
                    "probe_name": "linear",
                },
            )

            config = {
                "annotation_file": str(annotation_file),
                "official_repo_root": str(self.official_repo),
                "videos_root": str(tmp_path / "videos"),
                "cache_dir": str(tmp_path / "cache"),
                "annotations": {"auto_download": False},
                "split": {"dir": str(split_dir)},
                "split_name": "train",
                "max_pairs": 0,
                "seed": 7,
                "materialize_missing": False,
                "download_timeout_seconds": 10,
                "probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "name": "linear",
                    "checkpoint_path": str(ckpt),
                    "device": "cpu",
                    "eval_output_dir": str(tmp_path / "results"),
                    "eval_output_subdir": "probe_eval",
                },
            }

            single_split_spec = replace(run_probe.DATASET_SPECS["mvp"], report_splits=("train",))
            with mock.patch.dict(run_probe.DATASET_SPECS, {"mvp": single_split_spec}, clear=False):
                with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                    with mock.patch(
                        "training.run_probe.load_probe_from_checkpoint",
                        return_value=(
                            _StaticProbe([1, 0, 1, 0]),
                            {
                                "type": "linear",
                                "metadata": {
                                    "feature_signature": "cache_sig",
                                    "target_type": "semantic_plausibility",
                                    "probe_name": "linear",
                                },
                            },
                        ),
                    ):
                        summary = run_mvp_eval_probe(config)

            metrics = summary["base_eval"]["metrics"]
            self.assertAlmostEqual(metrics["accuracy"], 100.0)
            self.assertAlmostEqual(metrics["pair_consistency"], 100.0)
            pred_payload = json.loads(Path(summary["prediction_file"]).read_text(encoding="utf-8"))
            self.assertEqual(pred_payload["pair_alpha_0"], 0)
            self.assertEqual(pred_payload["pair_alpha_1"], 1)
            self.assertEqual(pred_payload["pair_beta_0"], 1)
            self.assertEqual(pred_payload["pair_beta_1"], 0)


if __name__ == "__main__":
    unittest.main()
