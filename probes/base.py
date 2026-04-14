from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ProbeFitResult:
    """Training summary returned by probe implementations."""

    train_loss: float
    val_loss: float | None
    n_epochs: int
    history: list[dict[str, float]] = field(default_factory=list)


class Probe(Protocol):
    """Contract for all probe implementations used by training code."""

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
        ...

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def predict_logits(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        ...
