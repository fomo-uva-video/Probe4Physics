from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probes.base import ProbeFitResult
from probes.registry import register_probe


class LinearProbe:
    """Simple linear classifier probe over frozen features."""

    def __init__(self, input_dim: int, num_classes: int = 2, device: str | torch.device = "cpu") -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)

        self.model = nn.Linear(self.input_dim, self.num_classes)
        self.model.to(self.device)

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
    ) -> ProbeFitResult:
        torch.manual_seed(int(seed))

        x_train = x_train.to(dtype=torch.float32)
        y_train = y_train.to(dtype=torch.long)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        history: list[dict[str, float]] = []
        final_train_loss = 0.0

        for epoch_idx in range(int(epochs)):
            self.model.train()
            running = 0.0
            n_batches = 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running += float(loss.detach().cpu())
                n_batches += 1

            final_train_loss = running / max(n_batches, 1)
            val_loss = self._compute_loss(x_val, y_val, criterion) if x_val is not None and y_val is not None else None
            row = {
                "epoch": float(epoch_idx + 1),
                "train_loss": float(final_train_loss),
            }
            if val_loss is not None:
                row["val_loss"] = float(val_loss)
            history.append(row)

        return ProbeFitResult(
            train_loss=float(final_train_loss),
            val_loss=history[-1].get("val_loss") if history else None,
            n_epochs=int(epochs),
            history=history,
        )

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        logits = self.predict_logits(x, batch_size=batch_size)
        return logits.argmax(dim=1)

    def predict_logits(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        self.model.eval()
        x = x.to(dtype=torch.float32)

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], int(batch_size)):
                end = start + int(batch_size)
                xb = x[start:end].to(self.device)
                logits = self.model(xb)
                outputs.append(logits.cpu())

        if not outputs:
            return torch.empty((0, self.num_classes), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "type": "linear",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "state_dict": self.model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "LinearProbe":
        payload = torch.load(str(path), map_location="cpu")
        probe_type = str(payload.get("type", ""))
        if probe_type != "linear":
            raise ValueError(f"Checkpoint type mismatch: expected 'linear', got '{probe_type}'")

        probe = cls(
            input_dim=int(payload["input_dim"]),
            num_classes=int(payload["num_classes"]),
            device=device,
        )
        probe.model.load_state_dict(payload["state_dict"])
        probe.model.to(torch.device(device))
        return probe

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "probe_type": "linear",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
        }

    def _compute_loss(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_val.to(self.device, dtype=torch.float32))
            loss = criterion(logits, y_val.to(self.device, dtype=torch.long))
        return float(loss.detach().cpu())



def create_linear_probe(**kwargs: Any) -> LinearProbe:
    return LinearProbe(**kwargs)


register_probe("linear", create_linear_probe, replace=True)
