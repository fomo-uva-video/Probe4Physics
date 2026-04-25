from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ProbeFitResult:
    """Training summary returned by probe implementations."""

    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    n_epochs: int
    best_epoch: int | None = None
    best_val_loss: float | None = None
    best_val_accuracy: float | None = None
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
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        ...

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def predict_logits(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        ...


class BaseClassifierProbe(ABC):
    """Shared training/inference/checkpoint scaffold for classifier probes."""

    def __init__(self, input_dim: int, num_classes: int = 2, device: str | torch.device = "cpu") -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)

        self.model = self._build_model()
        self.model.to(self.device)
        self._last_fit_best_state_dict: dict[str, Any] | None = None
        self._last_fit_best_epoch: int | None = None
        self._last_fit_best_val_loss: float | None = None
        self._last_fit_best_val_accuracy: float | None = None

    @abstractmethod
    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _checkpoint_type(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _init_kwargs_from_payload(
        cls,
        payload: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _validate_and_cast_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _checkpoint_extra_fields(self) -> dict[str, Any]:
        return {}

    def _compute_loss_accuracy(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._validate_and_cast_input(x_val).to(self.device))
            targets = y_val.to(self.device, dtype=torch.long)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            accuracy = ((preds == targets).float().mean() * 100.0).item()
        return float(loss.detach().cpu()), float(accuracy)

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
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        torch.manual_seed(int(seed))

        x_train = self._validate_and_cast_input(x_train)
        y_train = y_train.to(dtype=torch.long)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        history: list[dict[str, float]] = []
        final_train_loss = 0.0
        final_train_accuracy = 0.0
        best_epoch: int | None = None
        best_val_loss: float | None = None
        best_val_accuracy: float | None = None
        best_state_dict: dict[str, Any] | None = None

        for epoch_idx in range(int(epochs)):
            self.model.train()
            running = 0.0
            n_batches = 0
            n_correct = 0
            n_seen = 0

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
                preds = logits.argmax(dim=1)
                n_correct += int((preds == yb).sum().item())
                n_seen += int(yb.shape[0])

            final_train_loss = running / max(n_batches, 1)
            final_train_accuracy = (n_correct / max(n_seen, 1)) * 100.0
            row: dict[str, float] = {
                "epoch": float(epoch_idx + 1),
                "train_loss": float(final_train_loss),
                "train_accuracy": float(final_train_accuracy),
            }

            if x_val is not None and y_val is not None:
                val_loss, val_accuracy = self._compute_loss_accuracy(x_val, y_val, criterion)
                row["val_loss"] = float(val_loss)
                row["val_accuracy"] = float(val_accuracy)
                if (
                    best_val_accuracy is None
                    or float(val_accuracy) > float(best_val_accuracy)
                    or (
                        float(val_accuracy) == float(best_val_accuracy)
                        and (
                            best_val_loss is None
                            or float(val_loss) < float(best_val_loss)
                        )
                    )
                ):
                    best_epoch = int(epoch_idx + 1)
                    best_val_loss = float(val_loss)
                    best_val_accuracy = float(val_accuracy)
                    best_state_dict = copy.deepcopy(self.model.state_dict())

            history.append(row)
            if epoch_logger is not None:
                epoch_logger(dict(row))

        if best_state_dict is None:
            best_epoch = int(epochs)
            best_state_dict = copy.deepcopy(self.model.state_dict())

        self._last_fit_best_state_dict = best_state_dict
        self._last_fit_best_epoch = best_epoch
        self._last_fit_best_val_loss = best_val_loss
        self._last_fit_best_val_accuracy = best_val_accuracy

        return ProbeFitResult(
            train_loss=float(final_train_loss),
            train_accuracy=float(final_train_accuracy),
            val_loss=history[-1].get("val_loss") if history else None,
            val_accuracy=history[-1].get("val_accuracy") if history else None,
            n_epochs=int(epochs),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            history=history,
        )

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        logits = self.predict_logits(x, batch_size=batch_size)
        return logits.argmax(dim=1)

    def predict_logits(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        self.model.eval()
        x = self._validate_and_cast_input(x)

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
            "type": self._checkpoint_type(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "state_dict": self.model.state_dict(),
            "metadata": metadata or {},
            **self._checkpoint_extra_fields(),
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Any:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        kwargs = cls._init_kwargs_from_payload(payload, device)
        probe = cls(**kwargs)
        probe.model.load_state_dict(payload["state_dict"])
        probe.model.to(torch.device(device))
        return probe

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "probe_type": self._checkpoint_type(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            **self._checkpoint_extra_fields(),
        }

    def best_fit_state_dict(self) -> dict[str, Any] | None:
        if self._last_fit_best_state_dict is None:
            return None
        return copy.deepcopy(self._last_fit_best_state_dict)
