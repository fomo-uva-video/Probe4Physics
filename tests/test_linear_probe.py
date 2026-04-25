from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from probes.linear import LinearProbe


class LinearProbeTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")

        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()
        fit = probe.fit(x, y, epochs=3, lr=1e-2, batch_size=16, seed=7)

        self.assertEqual(fit.n_epochs, 3)
        self.assertGreaterEqual(fit.train_loss, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "linear.pt"
            probe.save(ckpt, metadata={"tag": "unit-test"})

            loaded = LinearProbe.load(ckpt, device="cpu")
            pred = loaded.predict(x)
            self.assertEqual(tuple(pred.shape), (64,))

    def test_fit_calls_epoch_logger_once_per_epoch(self) -> None:
        probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")

        x = torch.randn(32, 4)
        y = (x[:, 0] > 0).long()
        seen: list[dict[str, float]] = []

        fit = probe.fit(
            x,
            y,
            epochs=3,
            lr=1e-2,
            batch_size=8,
            seed=7,
            epoch_logger=lambda row: seen.append(dict(row)),
        )

        self.assertEqual(fit.n_epochs, 3)
        self.assertEqual(len(seen), 3)
        self.assertEqual([int(row["epoch"]) for row in seen], [1, 2, 3])
        self.assertTrue(all("train_loss" in row for row in seen))
        self.assertTrue(all("train_accuracy" in row for row in seen))


if __name__ == "__main__":
    unittest.main()
