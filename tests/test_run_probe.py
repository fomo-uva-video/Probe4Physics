from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import pandas as pd
import torch
from omegaconf import OmegaConf
from training import run_probe
from probes.linear import LinearProbe
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


class _FakeChunkedTokenStore:
    def __init__(self, *, token_length: int = 2, feature_dim: int = 4) -> None:
        self.selected_layers = (12,)
        self._token_length = int(token_length)
        self._feature_dim = int(feature_dim)

    def input_dim(self, layer: int) -> int:
        self._assert_layer(layer)
        return self._feature_dim

    def chunk_id_for_feature_index(self, feature_index: int) -> int:
        return int(feature_index) // 2

    def read_feature(self, feature_index: int, *, layer: int, reduction: str = "tokens") -> torch.Tensor:
        self._assert_layer(layer)
        base = float(int(feature_index) + 1)
        tokens = torch.full(
            (self._token_length, self._feature_dim),
            base,
            dtype=torch.float32,
        )
        if reduction == "tokens":
            return tokens
        if reduction == "tokens_mean":
            return tokens.mean(dim=0)
        raise ValueError(f"Unsupported reduction: {reduction}")

    def _assert_layer(self, layer: int) -> None:
        if int(layer) not in self.selected_layers:
            raise ValueError(f"Unsupported layer: {layer}")


class RunProbeTests(unittest.TestCase):
    def test_compose_mvp_config_defaults_optuna_metric_to_pair_consistency(self) -> None:
        cfg = run_probe._compose_config("mvp", [])
        config = OmegaConf.to_container(cfg, resolve=True)
        self.assertIsInstance(config, dict)
        assert isinstance(config, dict)

        self.assertEqual(config["probe"]["optuna"]["metric"], "pair_consistency")

    def test_resolve_objective_metric_name_uses_mvp_pair_consistency_by_default(self) -> None:
        probe_cfg = run_probe._probe_cfg({"probe": {"name": "linear"}})

        self.assertEqual(run_probe._resolve_objective_metric_name("mvp", probe_cfg), "pair_consistency")

    def test_resolve_objective_metric_name_prefers_explicit_override(self) -> None:
        probe_cfg = run_probe._probe_cfg(
            {
                "probe": {
                    "name": "linear",
                    "optuna": {
                        "metric": "accuracy",
                    },
                }
            }
        )

        self.assertEqual(run_probe._resolve_objective_metric_name("mvp", probe_cfg), "accuracy")

    def test_extract_objective_metric_uses_pair_consistency_for_mvp_default(self) -> None:
        probe_cfg = run_probe._probe_cfg({"probe": {"name": "linear"}})
        metrics = {
            "accuracy": 72.0,
            "pair_consistency": 61.5,
        }

        self.assertEqual(run_probe._extract_objective_metric("mvp", metrics, probe_cfg), 61.5)

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

    def test_run_single_train_forwards_eval_batch_size_to_probe_fit(self) -> None:
        class _CapturingLinearProbe(LinearProbe):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.seen_eval_batch_size: int | None = None

            def fit(self, *args, eval_batch_size=None, **kwargs):
                self.seen_eval_batch_size = eval_batch_size
                return super().fit(*args, eval_batch_size=eval_batch_size, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 7,
                "probe": {
                    "name": "linear",
                    "batch_size": 8,
                    "eval_batch_size": 3,
                    "epochs": 2,
                    "wandb": {"enabled": False},
                    "optuna": {"enabled": False},
                },
            }
            probe_cfg = run_probe._probe_cfg(config)
            bundle = {
                "paths": SimpleNamespace(cache_dir=Path(tmp)),
                "manifest": {"signature": "sig", "stats": {"n_classes": 2}},
                "index": pd.DataFrame(
                    [
                        {"feature_index": 0, "split": "train", "label_idx": 0},
                        {"feature_index": 1, "split": "train", "label_idx": 1},
                        {"feature_index": 2, "split": "train", "label_idx": 0},
                        {"feature_index": 3, "split": "val", "label_idx": 1},
                        {"feature_index": 4, "split": "val", "label_idx": 0},
                    ]
                ),
                "pooled": {"selected_layers": [8], "by_layer": {8: torch.randn(5, 4)}},
                "tokens": None,
            }
            probe = _CapturingLinearProbe(input_dim=4, num_classes=2, device="cpu")
            fake_spec = SimpleNamespace(load_bundle=lambda cfg, view: bundle, report_splits=())

            with mock.patch.dict(run_probe.DATASET_SPECS, {"ssv2": fake_spec}, clear=False):
                with mock.patch("training.run_probe._create_probe_instance", return_value=probe):
                    with mock.patch("training.run_probe.init_wandb_train_logger", return_value=None):
                        summary, _logger = run_probe._run_single_train(
                            "ssv2",
                            config,
                            probe_cfg,
                            output_dir=Path(tmp) / "train",
                        )

        self.assertEqual(probe.seen_eval_batch_size, 3)
        self.assertEqual(summary["fit"]["n_epochs"], 2)
        self.assertIsNotNone(summary["fit"]["best_val_accuracy"])

    def test_run_single_train_streams_chunked_mvp_tokens_mean(self) -> None:
        class _CapturingLinearProbe(LinearProbe):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.fit_loader_called = False

            def fit_loader(self, train_loader, **kwargs):
                self.fit_loader_called = True
                return super().fit_loader(train_loader, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 7,
                "probe": {
                    "name": "linear",
                    "feature_view": "tokens_mean",
                    "layer": 12,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "epochs": 2,
                    "wandb": {"enabled": False},
                    "optuna": {"enabled": False},
                },
            }
            probe_cfg = run_probe._probe_cfg(config)
            bundle = {
                "paths": SimpleNamespace(cache_dir=Path(tmp)),
                "manifest": {"signature": "sig", "stats": {"n_classes": 2}},
                "index": pd.DataFrame(
                    [
                        {
                            "feature_index": 0,
                            "split": "train",
                            "plausibility_label": 0,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 1,
                            "split": "train",
                            "plausibility_label": 1,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 2,
                            "split": "train",
                            "plausibility_label": 0,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 3,
                            "split": "val",
                            "plausibility_label": 1,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                    ]
                ),
                "pooled": None,
                "tokens": None,
                "token_store": _FakeChunkedTokenStore(token_length=2, feature_dim=4),
            }
            probe = _CapturingLinearProbe(input_dim=4, num_classes=2, device="cpu")
            fake_spec = SimpleNamespace(load_bundle=lambda cfg, view: bundle, report_splits=("train", "val"))

            with mock.patch.dict(run_probe.DATASET_SPECS, {"mvp": fake_spec}, clear=False):
                with mock.patch("training.run_probe._create_probe_instance", return_value=probe):
                    with mock.patch("training.run_probe.init_wandb_train_logger", return_value=None):
                        summary, _logger = run_probe._run_single_train(
                            "mvp",
                            config,
                            probe_cfg,
                            output_dir=Path(tmp) / "train",
                        )

        self.assertTrue(probe.fit_loader_called)
        self.assertEqual(summary["input_dim"], 4)
        self.assertEqual(summary["fit"]["n_epochs"], 2)

    def test_run_single_eval_streams_chunked_mvp_tokens(self) -> None:
        class _StaticChunkedProbe:
            def predict_logits_loader(self, loader):
                rows = 0
                for batch in loader:
                    xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                    rows += int(xb.shape[0])
                return torch.tensor(
                    [[0.1, 0.9], [0.9, 0.1]][:rows],
                    dtype=torch.float32,
                )

        with tempfile.TemporaryDirectory() as tmp:
            context = run_probe.EvalContext(
                spec=run_probe.DATASET_SPECS["mvp"],
                manifest={"signature": "sig"},
                index=pd.DataFrame(
                    [
                        {
                            "feature_index": 0,
                            "sample_id": "s0",
                            "pair_id": "p0",
                            "split": "test",
                            "answer_idx": 0,
                            "plausibility_label": 1,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                        {
                            "feature_index": 1,
                            "sample_id": "s1",
                            "pair_id": "p1",
                            "split": "test",
                            "answer_idx": 1,
                            "plausibility_label": 0,
                            "yes_choice_idx": 0,
                            "no_choice_idx": 1,
                        },
                    ]
                ),
                features=run_probe.ChunkedTokenSelection(
                    store=_FakeChunkedTokenStore(token_length=2, feature_dim=4),
                    layer=12,
                    reduction="tokens",
                ),
                probe=_StaticChunkedProbe(),
                checkpoint_path=Path(tmp) / "probe_best.pt",
                current_signature="sig",
            )
            config = {
                "split_name": "test",
                "probe": {
                    "name": "temporal_attn",
                    "feature_view": "tokens",
                    "layer": 12,
                    "device": "cpu",
                    "eval_batch_size": 2,
                    "eval_output_dir": tmp,
                    "eval_output_subdir": "eval_run",
                },
            }
            probe_cfg = run_probe._probe_cfg(config)

            with mock.patch(
                "training.run_probe.run_mvp_eval",
                return_value={"metrics": {"pair_consistency": 100.0}},
            ) as mocked_eval:
                summary = run_probe._run_single_eval_from_context(
                    "mvp",
                    config,
                    probe_cfg,
                    context,
                    output_dir=Path(tmp) / "eval_run",
                    split_name="test",
                )
                pred_payload = json.loads(Path(summary["prediction_file"]).read_text(encoding="utf-8"))
                self.assertEqual(pred_payload, {"s0": 0, "s1": 1})
                mocked_eval.assert_called_once()

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
                "split_name": "test",
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
            rows: list[dict[str, str]] = []

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

            def _fake_eval(dataset, cfg, probe_cfg, *, output_dir=None):
                del dataset
                eval_calls.append(
                    {
                        "layer": probe_cfg["layer"],
                        "checkpoint": probe_cfg["checkpoint_path"],
                        "split_name": cfg["split_name"],
                        "output_dir": str(output_dir),
                    }
                )
                metric = 1.0 if probe_cfg["layer"] == 8 else 2.0
                return {
                    "objective_metric": metric,
                    "probe_eval_dir": str(output_dir),
                    "reported_splits": ["train", "val", "test"],
                    "metrics_by_split": {
                        "train": {
                            "accuracy": metric + 10.0,
                            "n_samples": 20,
                        },
                        "val": {
                            "accuracy": metric + 20.0,
                        },
                        "test": {
                            "accuracy": metric + 30.0,
                        },
                    },
                }

            with mock.patch("training.run_probe._run_train_workflow", side_effect=_fake_train_workflow):
                with mock.patch("training.run_probe._run_report_eval", side_effect=_fake_eval):
                    summary = run_probe.run_probe_train_eval("mvp", config)
            csv_path = Path(tmp) / "layer_sweep" / "train_eval_summary.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([item["layer"] for item in train_calls], [8, "last"])
        self.assertEqual([item["layer"] for item in eval_calls], [8, "last"])
        self.assertEqual([item["split_name"] for item in eval_calls], ["test", "test"])
        self.assertEqual(
            [item["wandb_group"] for item in train_calls],
            ["probe_group_layer_8", "probe_group_layer_last"],
        )
        self.assertEqual(
            [item["study_name"] for item in train_calls],
            ["probe_study_layer_8", "probe_study_layer_last"],
        )
        self.assertEqual(summary["reported_splits"], ["train", "val", "test"])
        self.assertIsNone(summary["best_layer"])
        self.assertIsNone(summary["best_layer_label"])
        self.assertEqual(len(summary["layers"]), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            list(rows[0].keys()),
            [
                "layer",
                "layer_label",
                "objective_metric_name",
                "objective_metric",
                "train_accuracy",
                "train_n_samples",
                "val_accuracy",
                "test_accuracy",
            ],
        )
        self.assertEqual(rows[0]["objective_metric_name"], "pair_consistency")
        self.assertEqual(rows[0]["layer"], "8")
        self.assertEqual(rows[0]["objective_metric"], "1.0")
        self.assertEqual(rows[0]["train_accuracy"], "11.0")
        self.assertEqual(rows[0]["val_accuracy"], "21.0")
        self.assertEqual(rows[0]["test_accuracy"], "31.0")
        self.assertEqual(rows[1]["layer"], "last")
        self.assertEqual(rows[1]["test_accuracy"], "32.0")

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
            eval_splits: list[str] = []

            def _fake_train(dataset, cfg, probe_cfg, *, output_dir=None, finish_logger=True, epoch_callback=None):
                del dataset, cfg, probe_cfg, finish_logger, epoch_callback
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
                del dataset, probe_cfg
                eval_splits.append(str(split_name or cfg["split_name"]))
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
        self.assertEqual(summary["objective_metric"], "pair_consistency")
        self.assertEqual(summary["best_trial_number"], 1)
        self.assertEqual(summary["best_value"], 2.0)
        self.assertEqual(len(summary["trials"]), 2)
        self.assertEqual(eval_splits, ["val", "val"])

    def test_run_multi_split_eval_writes_root_metrics_for_train_val_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "probe_best.pt"
            eval_root = Path(tmp) / "eval"
            config = {
                "split_name": "test",
                "probe": {
                    "name": "linear",
                },
            }
            probe_cfg = run_probe._probe_cfg(config)
            context = run_probe.EvalContext(
                spec=run_probe.DATASET_SPECS["mvp"],
                manifest={"signature": "sig"},
                index=None,
                features=torch.empty(0),
                probe=object(),
                checkpoint_path=checkpoint_path,
                current_signature="sig",
            )

            metric_by_split = {
                "train": 0.7,
                "val": 0.8,
                "test": 0.9,
            }

            def _fake_single_eval(dataset, cfg, resolved_probe_cfg, resolved_context, *, output_dir=None, split_name):
                del dataset, cfg, resolved_probe_cfg, resolved_context
                assert output_dir is not None
                split_dir = Path(output_dir)
                split_dir.mkdir(parents=True, exist_ok=True)
                metric = metric_by_split[str(split_name)]
                metrics = {"accuracy": metric}
                return {
                    "probe_eval_dir": str(split_dir),
                    "checkpoint": str(checkpoint_path),
                    "prediction_file": str(split_dir / "probe_predictions.json"),
                    "feature_signature": "sig",
                    "dataset": "mvp",
                    "probe_name": "linear",
                    "split_name": str(split_name),
                    "objective_metric": metric,
                    "metrics": metrics,
                    "base_eval": {
                        "output_dir": str(split_dir),
                        "metrics": metrics,
                    },
                }

            with mock.patch("training.run_probe._run_single_eval_from_context", side_effect=_fake_single_eval):
                summary = run_probe._run_multi_split_eval(
                    "mvp",
                    config,
                    probe_cfg,
                    context,
                    output_dir=eval_root,
                    primary_split="test",
                    report_splits=("train", "val", "test"),
                )

            metrics_payload = json.loads((eval_root / "metrics.json").read_text(encoding="utf-8"))
            summary_payload = json.loads((eval_root / "probe_eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["train"]["accuracy"], 0.7)
            self.assertEqual(metrics_payload["val"]["accuracy"], 0.8)
            self.assertEqual(metrics_payload["test"]["accuracy"], 0.9)
            self.assertEqual(summary["objective_metric"], 0.9)
            self.assertEqual(summary_payload["split_name"], "test")
            self.assertEqual(summary_payload["reported_splits"], ["train", "val", "test"])
            self.assertTrue((eval_root / "summary.md").exists())
            self.assertTrue((eval_root / "run_config.snapshot.yaml").exists())


if __name__ == "__main__":
    unittest.main()
