from __future__ import annotations

import unittest
from unittest import mock

import run


class RunCommandTests(unittest.TestCase):
    def test_expected_commands_are_registered(self) -> None:
        expected = {
            "init.mvp",
            "eval.mvp",
            "extract.mvp",
            "train.linear.mvp",
            "eval.linear.mvp",
            "exp.list",
            "exp.run",
        }
        self.assertTrue(expected.issubset(set(run.COMMANDS)))

    def test_exp_run_executes_pipeline_in_order(self) -> None:
        calls: list[str] = []

        def _make_handler(name: str):
            def _handler(_cfg):
                calls.append(name)
                return {"step": name}

            return _handler

        command_overrides = {
            "extract.mvp": run.CommandSpec("mvp", _make_handler("extract.mvp"), ""),
            "train.linear.mvp": run.CommandSpec("mvp", _make_handler("train.linear.mvp"), ""),
            "eval.linear.mvp": run.CommandSpec("mvp", _make_handler("eval.linear.mvp"), ""),
        }

        with mock.patch.dict(run.COMMANDS, command_overrides, clear=False):
            with mock.patch("run.has_valid_feature_cache", return_value=False):
                result = run._run_exp({"name": "mvp.jepa_v1.linear"})

        self.assertEqual(calls, ["extract.mvp", "train.linear.mvp", "eval.linear.mvp"])
        self.assertEqual(result["experiment"], "mvp.jepa_v1.linear")
        self.assertEqual(len(result["pipeline"]), 3)

    def test_exp_run_skips_extract_when_cache_exists(self) -> None:
        calls: list[str] = []

        def _make_handler(name: str):
            def _handler(_cfg):
                calls.append(name)
                return {"step": name}

            return _handler

        command_overrides = {
            "extract.mvp": run.CommandSpec("mvp", _make_handler("extract.mvp"), ""),
            "train.linear.mvp": run.CommandSpec("mvp", _make_handler("train.linear.mvp"), ""),
            "eval.linear.mvp": run.CommandSpec("mvp", _make_handler("eval.linear.mvp"), ""),
        }

        with mock.patch.dict(run.COMMANDS, command_overrides, clear=False):
            with mock.patch("run.has_valid_feature_cache", return_value=True):
                result = run._run_exp({"name": "mvp.jepa_v1.linear"})

        self.assertEqual(calls, ["train.linear.mvp", "eval.linear.mvp"])
        first_step = result["pipeline"][0]
        self.assertEqual(first_step["step"], "extract.mvp")
        self.assertTrue(first_step["skipped"])

    def test_exp_run_does_not_skip_extract_when_forced(self) -> None:
        calls: list[str] = []

        def _make_handler(name: str):
            def _handler(_cfg):
                calls.append(name)
                return {"step": name}

            return _handler

        command_overrides = {
            "extract.mvp": run.CommandSpec("mvp", _make_handler("extract.mvp"), ""),
            "train.linear.mvp": run.CommandSpec("mvp", _make_handler("train.linear.mvp"), ""),
            "eval.linear.mvp": run.CommandSpec("mvp", _make_handler("eval.linear.mvp"), ""),
        }

        with mock.patch.dict(run.COMMANDS, command_overrides, clear=False):
            with mock.patch("run.has_valid_feature_cache", return_value=True):
                run._run_exp(
                    {
                        "name": "mvp.jepa_v1.linear",
                        "feature_cache": {"force_reextract": True},
                    }
                )

        self.assertEqual(calls, ["extract.mvp", "train.linear.mvp", "eval.linear.mvp"])


if __name__ == "__main__":
    unittest.main()
