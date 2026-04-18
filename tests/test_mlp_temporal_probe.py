from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from probes.mlp import MLPProbe
from probes.temporal_attn import TemporalAttentiveProbe


class MLPProbeTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        probe = MLPProbe(
            input_dim=4,
            num_classes=2,
            hidden_dims=[16, 8],
            dropout=0.0,
            device="cpu",
        )

        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()
        fit = probe.fit(x, y, epochs=3, lr=1e-2, batch_size=16, seed=7)

        self.assertEqual(fit.n_epochs, 3)
        self.assertGreaterEqual(fit.train_loss, 0.0)
        self.assertGreaterEqual(fit.train_accuracy, 0.0)
        self.assertLessEqual(fit.train_accuracy, 100.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mlp.pt"
            probe.save(ckpt, metadata={"tag": "unit-test"})

            loaded = MLPProbe.load(ckpt, device="cpu")
            pred = loaded.predict(x)
            self.assertEqual(tuple(pred.shape), (64,))


class TemporalAttentiveProbeTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=8,
            num_classes=2,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        x = torch.randn(32, 6, 8)
        y = (x[:, :, 0].mean(dim=1) > 0).long()
        fit = probe.fit(x, y, epochs=2, lr=1e-2, batch_size=8, seed=7)

        self.assertEqual(fit.n_epochs, 2)
        self.assertGreaterEqual(fit.train_loss, 0.0)
        self.assertGreaterEqual(fit.train_accuracy, 0.0)
        self.assertLessEqual(fit.train_accuracy, 100.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "temporal_attn.pt"
            probe.save(ckpt, metadata={"tag": "unit-test"})

            loaded = TemporalAttentiveProbe.load(ckpt, device="cpu")
            pred = loaded.predict(x)
            self.assertEqual(tuple(pred.shape), (32,))

    def test_fit_rejects_non_token_input(self) -> None:
        probe = TemporalAttentiveProbe(
            input_dim=8,
            num_classes=2,
            embed_dim=8,
            num_heads=2,
            num_self_attn_blocks=1,
            mlp_ratio=2.0,
            dropout=0.0,
            device="cpu",
        )

        x_bad = torch.randn(32, 8)
        y = torch.randint(0, 2, (32,), dtype=torch.long)

        with self.assertRaises(ValueError):
            probe.fit(x_bad, y, epochs=1)


if __name__ == "__main__":
    unittest.main()
