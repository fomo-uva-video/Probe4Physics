from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_probe
from probes.mlp import MLPProbe


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}
        self.value: float | None = None

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
    class samplers:
        class TPESampler:
            def __init__(self, seed: int) -> None:
                self.seed = seed

    @staticmethod
    def create_study(
        *,
        study_name: str,
        direction: str,
        sampler,
        storage: str,
        load_if_exists: bool,
    ) -> _FakeStudy:
        del direction, sampler, storage, load_if_exists
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

            def _fake_train(dataset, cfg, probe_cfg, *, output_dir=None, finish_logger=True):
                del dataset, cfg, probe_cfg, finish_logger
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

            with mock.patch("run_probe._import_optuna", return_value=_FakeOptunaModule()):
                with mock.patch("run_probe._run_single_train", side_effect=_fake_train):
                    with mock.patch("run_probe._run_single_eval", side_effect=_fake_eval):
                        summary = run_probe.run_probe_train("mvp", config)

        self.assertEqual(summary["study_name"], "study_run")
        self.assertEqual(summary["n_trials"], 2)
        self.assertEqual(summary["best_trial_number"], 1)
        self.assertEqual(summary["best_value"], 2.0)
        self.assertEqual(len(summary["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
