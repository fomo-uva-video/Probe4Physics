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

    def test_fit_tracks_best_validation_epoch(self) -> None:
        probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")

        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()

        fit = probe.fit(
            x,
            y,
            x_val=x,
            y_val=y,
            epochs=4,
            lr=1e-2,
            batch_size=16,
            seed=7,
        )

        self.assertEqual(fit.n_epochs, 4)
        self.assertIsNotNone(fit.best_epoch)
        self.assertGreaterEqual(int(fit.best_epoch), 1)
        self.assertLessEqual(int(fit.best_epoch), 4)
        self.assertIsNotNone(fit.best_val_accuracy)
        self.assertIsNotNone(probe.best_fit_state_dict())

    def test_validation_metrics_match_full_batch_and_batched_paths(self) -> None:
        probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")
        criterion = torch.nn.CrossEntropyLoss()

        x_val = torch.randn(31, 4)
        y_val = (x_val[:, 0] > 0).long()

        full_loss, full_acc = probe._compute_loss_accuracy(x_val, y_val, criterion, batch_size=len(x_val))
        batched_loss, batched_acc = probe._compute_loss_accuracy(x_val, y_val, criterion, batch_size=7)

        self.assertAlmostEqual(full_loss, batched_loss, places=6)
        self.assertAlmostEqual(full_acc, batched_acc, places=6)

    def test_fit_supports_batched_validation(self) -> None:
        probe = LinearProbe(input_dim=4, num_classes=2, device="cpu")

        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()

        fit = probe.fit(
            x,
            y,
            x_val=x,
            y_val=y,
            epochs=3,
            lr=1e-2,
            batch_size=16,
            eval_batch_size=5,
            seed=7,
        )

        self.assertEqual(fit.n_epochs, 3)
        self.assertEqual(len(fit.history), 3)
        self.assertTrue(all("val_loss" in row for row in fit.history))
        self.assertTrue(all("val_accuracy" in row for row in fit.history))
        self.assertIsNotNone(fit.best_epoch)
        self.assertIsNotNone(fit.best_val_accuracy)


if __name__ == "__main__":
    unittest.main()
