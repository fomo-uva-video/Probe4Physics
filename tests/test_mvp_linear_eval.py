from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from probes.linear import LinearProbe
from training.mvp_linear import LinearProbeConfigError, run_mvp_eval_linear


class MVPLInearEvalTests(unittest.TestCase):
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
                "linear_probe": {
                    "feature_view": "pooled",
                    "layer": "last",
                    "checkpoint_path": str(ckpt),
                    "device": "cpu",
                },
            }

            with mock.patch("training.mvp_linear.load_feature_cache_for_config", return_value=fake_bundle):
                with mock.patch("training.mvp_linear.run_mvp_eval") as mocked_eval:
                    with self.assertRaises(LinearProbeConfigError):
                        run_mvp_eval_linear(config)

            mocked_eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
