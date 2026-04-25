from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from probes.base import ProbeFitResult
from training.mvp_linear import LinearProbeConfigError, run_mvp_train_linear


class _CapturingProbe:
    def __init__(self) -> None:
        self.x_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.x_val: torch.Tensor | None = None
        self.y_val: torch.Tensor | None = None
        self.saved_metadata: dict[str, object] | None = None
        self.logged_epochs: list[dict[str, float]] = []

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        seed: int = 42,
        epoch_logger=None,
    ) -> ProbeFitResult:
        del epochs, lr, batch_size, weight_decay, seed
        self.x_train = x_train.clone()
        self.y_train = y_train.clone()
        self.x_val = None if x_val is None else x_val.clone()
        self.y_val = None if y_val is None else y_val.clone()
        if epoch_logger is not None:
            row = {
                "epoch": 1.0,
                "train_loss": 0.1,
                "train_accuracy": 100.0,
            }
            if y_val is not None:
                row["val_loss"] = 0.2
                row["val_accuracy"] = 100.0
            self.logged_epochs.append(dict(row))
            epoch_logger(row)
        return ProbeFitResult(
            train_loss=0.1,
            train_accuracy=100.0,
            val_loss=0.2 if y_val is not None else None,
            val_accuracy=100.0 if y_val is not None else None,
            n_epochs=1,
            history=[],
        )

    def save(self, path: str | Path, metadata: dict[str, object] | None = None) -> None:
        self.saved_metadata = metadata or {}
        torch.save({"metadata": self.saved_metadata}, path)


class MVPLinearSemanticTrainingTests(unittest.TestCase):
    def _fake_bundle(self) -> dict[str, object]:
        index = pd.DataFrame(
            [
                {
                    "feature_index": 0,
                    "sample_id": "train_yes_first_plausible",
                    "pair_id": "pair_a",
                    "split": "train",
                    "answer_idx": 0,
                    "plausibility_label": 1,
                    "yes_choice_idx": 0,
                    "no_choice_idx": 1,
                },
                {
                    "feature_index": 1,
                    "sample_id": "train_no_first_plausible",
                    "pair_id": "pair_b",
                    "split": "train",
                    "answer_idx": 1,
                    "plausibility_label": 1,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 2,
                    "sample_id": "val_yes_first_implausible",
                    "pair_id": "pair_c",
                    "split": "val",
                    "answer_idx": 1,
                    "plausibility_label": 0,
                    "yes_choice_idx": 0,
                    "no_choice_idx": 1,
                },
                {
                    "feature_index": 3,
                    "sample_id": "test_no_first_implausible",
                    "pair_id": "pair_d",
                    "split": "test",
                    "answer_idx": 0,
                    "plausibility_label": 0,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
            ]
        )
        pooled = {
            "selected_layers": [24],
            "by_layer": {24: torch.randn(4, 8)},
        }
        return {
            "manifest": {"signature": "semantic_sig"},
            "index": index,
            "pooled": pooled,
            "tokens": None,
            "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
        }

    def test_train_uses_semantic_plausibility_labels_not_positional_answer_idx(self) -> None:
        fake_probe = _CapturingProbe()
        fake_bundle = self._fake_bundle()

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 13,
                "linear_probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "device": "cpu",
                    "output_dir": tmp,
                    "output_subdir": "train_run",
                },
            }

            with mock.patch("training.mvp_linear.load_feature_cache_for_config", return_value=fake_bundle):
                with mock.patch("training.mvp_linear.create_probe", return_value=fake_probe):
                    result = run_mvp_train_linear(config)

            self.assertTrue(Path(result["checkpoint"]).exists())
            self.assertTrue(Path(result["checkpoint_last"]).exists())
            self.assertTrue(Path(result["checkpoint_best"]).exists())
            self.assertFalse((Path(tmp) / "train_run" / "linear_probe.pt").exists())

        self.assertEqual(result["n_train"], 2)
        self.assertEqual(result["n_val"], 1)
        self.assertEqual(fake_probe.y_train.tolist(), [1, 1])
        self.assertEqual(fake_probe.y_val.tolist(), [0])
        self.assertNotEqual(
            fake_probe.y_train.tolist(),
            fake_bundle["index"].loc[fake_bundle["index"]["split"] == "train", "answer_idx"].tolist(),
        )
        self.assertEqual(fake_probe.saved_metadata["target_type"], "semantic_plausibility")
        self.assertEqual(fake_probe.saved_metadata["positive_label"], 1)
        self.assertEqual(fake_probe.saved_metadata["negative_label"], 0)

    def test_train_hard_fails_when_semantic_columns_are_missing(self) -> None:
        fake_bundle = self._fake_bundle()
        fake_bundle["index"] = fake_bundle["index"].drop(columns=["plausibility_label"])

        config = {
            "linear_probe": {
                "feature_view": "pooled",
                "layer": "last",
                "device": "cpu",
            }
        }

        with mock.patch("training.mvp_linear.load_feature_cache_for_config", return_value=fake_bundle):
            with self.assertRaisesRegex(LinearProbeConfigError, "Re-run `python run.py extract.mvp`"):
                run_mvp_train_linear(config)

    def test_train_logs_to_wandb_when_enabled(self) -> None:
        fake_probe = _CapturingProbe()
        fake_bundle = self._fake_bundle()
        fake_logger = mock.Mock()

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 13,
                "linear_probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "device": "cpu",
                    "output_dir": tmp,
                    "output_subdir": "train_run",
                    "wandb": {
                        "enabled": True,
                        "project": "probe4physics-test",
                    },
                },
            }

            with mock.patch("training.mvp_linear.load_feature_cache_for_config", return_value=fake_bundle):
                with mock.patch("training.mvp_linear.create_probe", return_value=fake_probe):
                    with mock.patch("training.mvp_linear.init_wandb_train_logger", return_value=fake_logger):
                        result = run_mvp_train_linear(config)

        fake_logger.log_epoch.assert_called_once()
        fake_logger.log_summary.assert_called_once_with(result)
        fake_logger.finish.assert_called_once()
        self.assertEqual(result["fit"]["train_accuracy"], 100.0)
        self.assertEqual(result["fit"]["val_accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
