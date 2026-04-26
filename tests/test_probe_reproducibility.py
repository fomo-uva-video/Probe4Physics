from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from probes.linear import LinearProbe
from training import run_probe


class ProbeReproducibilityTests(unittest.TestCase):
    def _fake_mvp_bundle(self) -> dict[str, object]:
        index = pd.DataFrame(
            [
                {
                    "feature_index": 0,
                    "sample_id": "train_neg_a",
                    "pair_id": "pair_train_a",
                    "split": "train",
                    "answer_idx": 0,
                    "plausibility_label": 0,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 1,
                    "sample_id": "train_pos_a",
                    "pair_id": "pair_train_b",
                    "split": "train",
                    "answer_idx": 1,
                    "plausibility_label": 1,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 2,
                    "sample_id": "train_neg_b",
                    "pair_id": "pair_train_c",
                    "split": "train",
                    "answer_idx": 0,
                    "plausibility_label": 0,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 3,
                    "sample_id": "train_pos_b",
                    "pair_id": "pair_train_d",
                    "split": "train",
                    "answer_idx": 1,
                    "plausibility_label": 1,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 4,
                    "sample_id": "val_neg",
                    "pair_id": "pair_val_a",
                    "split": "val",
                    "answer_idx": 0,
                    "plausibility_label": 0,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 5,
                    "sample_id": "val_pos",
                    "pair_id": "pair_val_b",
                    "split": "val",
                    "answer_idx": 1,
                    "plausibility_label": 1,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 6,
                    "sample_id": "test_neg",
                    "pair_id": "pair_test_a",
                    "split": "test",
                    "answer_idx": 0,
                    "plausibility_label": 0,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
                {
                    "feature_index": 7,
                    "sample_id": "test_pos",
                    "pair_id": "pair_test_b",
                    "split": "test",
                    "answer_idx": 1,
                    "plausibility_label": 1,
                    "yes_choice_idx": 1,
                    "no_choice_idx": 0,
                },
            ]
        )
        pooled = {
            "selected_layers": [24],
            "by_layer": {
                24: torch.tensor(
                    [
                        [-2.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [-1.0, 0.5, 0.0, 0.0],
                        [1.0, 0.5, 0.0, 0.0],
                        [-1.5, 0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0, 0.0],
                        [-1.2, -0.2, 0.0, 0.0],
                        [1.2, -0.2, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            },
        }
        return {
            "manifest": {"signature": "repro_sig"},
            "index": index,
            "pooled": pooled,
            "tokens": None,
            "paths": SimpleNamespace(cache_dir=Path("/tmp/fake")),
        }

    def test_seed_training_runtime_repeats_linear_probe_initialization(self) -> None:
        run_probe._seed_training_runtime(42, deterministic=False)
        first_probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")
        first_state = {key: value.detach().clone() for key, value in first_probe.model.state_dict().items()}

        run_probe._seed_training_runtime(42, deterministic=False)
        second_probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")
        second_state = {key: value.detach().clone() for key, value in second_probe.model.state_dict().items()}

        self.assertEqual(set(first_state), set(second_state))
        for key in first_state:
            self.assertTrue(torch.equal(first_state[key], second_state[key]), key)

    def test_same_seed_train_run_produces_identical_histories_and_checkpoints(self) -> None:
        base_config = {
            "seed": 42,
            "probe": {
                "name": "linear",
                "feature_view": "pooled",
                "layer": "last",
                "device": "cpu",
                "epochs": 5,
                "lr": 1e-1,
                "batch_size": 2,
                "weight_decay": 0.0,
            },
        }
        fake_bundle = self._fake_mvp_bundle()

        with tempfile.TemporaryDirectory() as tmp:
            config_a = copy.deepcopy(base_config)
            config_a["probe"]["output_dir"] = tmp
            config_a["probe"]["output_subdir"] = "run_a"

            config_b = copy.deepcopy(base_config)
            config_b["probe"]["output_dir"] = tmp
            config_b["probe"]["output_subdir"] = "run_b"

            with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                with mock.patch("training.run_probe.init_wandb_train_logger", return_value=None):
                    summary_a = run_probe.run_mvp_train_probe(config_a)
            with mock.patch("training.run_probe.load_mvp_feature_cache", return_value=fake_bundle):
                with mock.patch("training.run_probe.init_wandb_train_logger", return_value=None):
                    summary_b = run_probe.run_mvp_train_probe(config_b)

            self.assertEqual(summary_a["fit"]["history"], summary_b["fit"]["history"])
            self.assertEqual(summary_a["fit"]["best_epoch"], summary_b["fit"]["best_epoch"])
            self.assertEqual(summary_a["fit"]["best_val_accuracy"], summary_b["fit"]["best_val_accuracy"])

            checkpoint_a = torch.load(summary_a["checkpoint"], map_location="cpu")
            checkpoint_b = torch.load(summary_b["checkpoint"], map_location="cpu")
            self.assertEqual(set(checkpoint_a["state_dict"]), set(checkpoint_b["state_dict"]))
            for key in checkpoint_a["state_dict"]:
                self.assertTrue(
                    torch.equal(checkpoint_a["state_dict"][key], checkpoint_b["state_dict"][key]),
                    key,
                )


if __name__ == "__main__":
    unittest.main()
