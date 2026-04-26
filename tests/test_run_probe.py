from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from training import run_probe
from probes.mlp import MLPProbe


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}
        self.value: float | None = None
        self.reported: list[tuple[int, float]] = []
        self.prune_on_step: int | None = None

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        del high, log
        value = float(low) * float(self.number + 1)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices):
        index = min(self.number, len(choices) - 1)
        value = choices[index]
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value: object) -> None:
        self.user_attrs[name] = value

    def report(self, value: float, step: int) -> None:
        self.reported.append((int(step), float(value)))

    def should_prune(self) -> bool:
        if self.prune_on_step is None or not self.reported:
            return False
        return self.reported[-1][0] >= self.prune_on_step


class _FakeStudy:
    def __init__(self, study_name: str) -> None:
        self.study_name = study_name
        self.trials: list[_FakeTrial] = []
        self.best_trial: _FakeTrial | None = None

    def optimize(self, objective, n_trials: int, n_jobs: int, timeout: int | None) -> None:
        del n_jobs, timeout
        for number in range(n_trials):
            trial = _FakeTrial(number)
            trial.value = float(objective(trial))
            self.trials.append(trial)
        self.best_trial = max(self.trials, key=lambda item: float(item.value))


class _FakeOptunaModule:
    class TrialPruned(Exception):
        pass

    class samplers:
        class TPESampler:
            def __init__(self, seed: int) -> None:
                self.seed = seed

    class pruners:
        class MedianPruner:
            def __init__(self, *, n_startup_trials: int, n_warmup_steps: int, interval_steps: int) -> None:
                self.n_startup_trials = n_startup_trials
                self.n_warmup_steps = n_warmup_steps
                self.interval_steps = interval_steps

    @staticmethod
    def create_study(
        *,
        study_name: str,
        direction: str,
        sampler,
        pruner,
        storage: str,
        load_if_exists: bool,
    ) -> _FakeStudy:
        del direction, sampler, pruner, storage, load_if_exists
        return _FakeStudy(study_name)


class RunProbeTests(unittest.TestCase):
    def test_probe_cfg_auto_selects_tokens_for_temporal_attn(self) -> None:
        cfg = run_probe._probe_cfg({"probe": {"name": "temporal_attn"}})
        self.assertEqual(cfg["feature_view"], "tokens")

    def test_load_probe_from_checkpoint_roundtrip_for_mlp(self) -> None:
        probe = MLPProbe(
            input_dim=4,
            num_classes=2,
            hidden_dims=[8],
            dropout=0.1,
            device="cpu",
        )
        x = torch.randn(8, 4)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "probe.pt"
            probe.save(ckpt, metadata={"probe_name": "mlp"})
            loaded, payload = run_probe.load_probe_from_checkpoint(
                ckpt,
                device="cpu",
                expected_probe_name="mlp",
            )

        self.assertEqual(payload["type"], "mlp")
        self.assertEqual(tuple(loaded.predict(x).shape), (8,))

    def test_apply_trial_parameters_keeps_fixed_epochs_when_epoch_search_disabled(self) -> None:
        probe_cfg = run_probe._probe_cfg(
            {
                "probe": {
                    "name": "linear",
                    "epochs": 123,
                    "optuna": {
                        "search_space": {
                            "epochs": {
                                "enabled": False,
                                "choices": [20, 50, 100],
                            }
                        }
                    },
                }
            }
        )
        trial_cfg = {"probe": {"name": "linear", "epochs": 123}}

        trial = _FakeTrial(0)
        run_probe._apply_trial_parameters(trial_cfg, "mvp", probe_cfg, trial)

        self.assertEqual(trial_cfg["probe"]["epochs"], 123)
        self.assertNotIn("epochs", trial.params)

    def test_apply_trial_parameters_samples_epochs_when_enabled(self) -> None:
        probe_cfg = run_probe._probe_cfg(
            {
                "probe": {
                    "name": "linear",
                    "epochs": 123,
                    "optuna": {
                        "search_space": {
                            "epochs": {
                                "enabled": True,
                                "choices": [33, 44],
                            }
                        }
                    },
                }
            }
        )
        trial_cfg = {"probe": {"name": "linear", "epochs": 123}}

        trial = _FakeTrial(1)
        run_probe._apply_trial_parameters(trial_cfg, "mvp", probe_cfg, trial)

        self.assertEqual(trial_cfg["probe"]["epochs"], 44)
        self.assertEqual(trial.params["epochs"], 44)

    def test_train_eval_runs_requested_layers_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "split_name": "val",
                "probe": {
                    "name": "linear",
                    "output_dir": tmp,
                    "output_subdir": "layer_sweep",
                    "layer": "last",
                    "layers": [8, "last"],
                    "wandb": {"group": "probe_group"},
                    "optuna": {
                        "enabled": False,
                        "study_name": "probe_study",
                    },
                },
            }
            train_calls: list[dict[str, object]] = []
            eval_calls: list[dict[str, object]] = []

            def _fake_train_workflow(dataset, cfg, probe_cfg, *, output_dir=None):
                del dataset
                train_calls.append(
                    {
                        "layer": probe_cfg["layer"],
                        "wandb_group": cfg["probe"].get("wandb", {}).get("group", ""),
                        "study_name": probe_cfg["optuna"]["study_name"],
                        "output_dir": str(output_dir),
                    }
                )
                checkpoint = Path(output_dir or tmp) / "probe_best.pt"
                return {
                    "checkpoint": str(checkpoint),
                    "output_dir": str(output_dir),
                }

            def _fake_eval(dataset, cfg, probe_cfg, *, output_dir=None, split_name=None):
                del dataset, cfg
                eval_calls.append(
                    {
                        "layer": probe_cfg["layer"],
                        "checkpoint": probe_cfg["checkpoint_path"],
                        "split_name": split_name,
                        "output_dir": str(output_dir),
                    }
                )
                metric = 1.0 if probe_cfg["layer"] == 8 else 2.0
                return {
                    "objective_metric": metric,
                    "probe_eval_dir": str(output_dir),
                }

            with mock.patch("training.run_probe._run_train_workflow", side_effect=_fake_train_workflow):
                with mock.patch("training.run_probe._run_single_eval", side_effect=_fake_eval):
                    summary = run_probe.run_probe_train_eval("mvp", config)

        self.assertEqual([item["layer"] for item in train_calls], [8, "last"])
        self.assertEqual([item["layer"] for item in eval_calls], [8, "last"])
        self.assertEqual([item["split_name"] for item in eval_calls], ["val", "val"])
        self.assertEqual(
            [item["wandb_group"] for item in train_calls],
            ["probe_group_layer_8", "probe_group_layer_last"],
        )
        self.assertEqual(
            [item["study_name"] for item in train_calls],
            ["probe_study_layer_8", "probe_study_layer_last"],
        )
        self.assertEqual(summary["best_layer"], "last")
        self.assertEqual(summary["best_layer_label"], "last")
        self.assertEqual(len(summary["layers"]), 2)

    def test_make_optuna_epoch_pruning_callback_reports_val_accuracy(self) -> None:
        trial = _FakeTrial(0)
        probe_cfg = run_probe._probe_cfg({"probe": {"name": "linear"}})

        callback = run_probe._make_optuna_epoch_pruning_callback(_FakeOptunaModule(), trial, probe_cfg)
        self.assertIsNotNone(callback)
        assert callback is not None

        callback({"epoch": 1.0, "val_accuracy": 77.5})

        self.assertEqual(trial.reported, [(1, 77.5)])

    def test_make_optuna_epoch_pruning_callback_raises_when_trial_should_prune(self) -> None:
        trial = _FakeTrial(0)
        trial.prune_on_step = 2
        probe_cfg = run_probe._probe_cfg({"probe": {"name": "linear"}})

        callback = run_probe._make_optuna_epoch_pruning_callback(_FakeOptunaModule(), trial, probe_cfg)
        self.assertIsNotNone(callback)
        assert callback is not None

        callback({"epoch": 1.0, "val_accuracy": 55.0})
        with self.assertRaises(_FakeOptunaModule.TrialPruned):
            callback({"epoch": 2.0, "val_accuracy": 54.0})

    def test_make_optuna_epoch_pruning_callback_requires_val_accuracy(self) -> None:
        trial = _FakeTrial(0)
        probe_cfg = run_probe._probe_cfg({"probe": {"name": "linear"}})

        callback = run_probe._make_optuna_epoch_pruning_callback(_FakeOptunaModule(), trial, probe_cfg)
        self.assertIsNotNone(callback)
        assert callback is not None

        with self.assertRaises(run_probe.ProbeConfigError):
            callback({"epoch": 1.0, "train_accuracy": 80.0})

    def test_optuna_train_returns_best_trial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 7,
                "probe": {
                    "name": "linear",
                    "output_dir": tmp,
                    "output_subdir": "study_run",
                    "optuna": {
                        "enabled": True,
                        "n_trials": 2,
                        "sampler_seed": 7,
                    },
                },
            }

            def _fake_train(dataset, cfg, probe_cfg, *, output_dir=None, finish_logger=True, epoch_callback=None):
                del dataset, cfg, probe_cfg, finish_logger, epoch_callback
                trial_name = Path(output_dir or tmp).name
                checkpoint = Path(output_dir or tmp) / "probe_best.pt"
                return (
                    {
                        "checkpoint": str(checkpoint),
                        "output_dir": str(output_dir),
                        "dataset": "mvp",
                        "probe_name": "linear",
                    },
                    None,
                )

            def _fake_eval(dataset, cfg, probe_cfg, *, output_dir=None, split_name=None):
                del dataset, cfg, probe_cfg, split_name
                trial_name = Path(output_dir or tmp).parent.name
                trial_number = int(trial_name.split("_")[-1])
                return {
                    "objective_metric": float(trial_number + 1),
                    "probe_eval_dir": str(output_dir),
                }

            with mock.patch("training.run_probe._import_optuna", return_value=_FakeOptunaModule()):
                with mock.patch("training.run_probe._run_single_train", side_effect=_fake_train):
                    with mock.patch("training.run_probe._run_single_eval", side_effect=_fake_eval):
                        summary = run_probe.run_probe_train("mvp", config)

        self.assertEqual(summary["study_name"], "study_run")
        self.assertEqual(summary["n_trials"], 2)
        self.assertEqual(summary["best_trial_number"], 1)
        self.assertEqual(summary["best_value"], 2.0)
        self.assertEqual(len(summary["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
