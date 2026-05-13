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
    early_stopped: bool = False
    early_stopping_patience: int | None = None
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
        eval_batch_size: int | None = None,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        seed: int = 42,
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        ...

    def predict(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def predict_logits(self, x: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
        ...

    def fit_loader(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        seed: int = 42,
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        ...

    def predict_logits_loader(self, loader: DataLoader) -> torch.Tensor:
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
        *,
        batch_size: int | None = None,
    ) -> tuple[float, float]:
        self.model.eval()
        x_val = self._validate_and_cast_input(x_val)
        targets = y_val.to(dtype=torch.long)
        resolved_batch_size = int(batch_size) if batch_size is not None else int(x_val.shape[0])
        resolved_batch_size = max(resolved_batch_size, 1)

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for start in range(0, int(x_val.shape[0]), resolved_batch_size):
                end = start + resolved_batch_size
                xb = x_val[start:end].to(self.device)
                yb = targets[start:end].to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                preds = logits.argmax(dim=1)

                batch_seen = int(yb.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_seen
                total_correct += int((preds == yb).sum().item())
                total_seen += batch_seen

        mean_loss = total_loss / max(total_seen, 1)
        accuracy = (total_correct / max(total_seen, 1)) * 100.0
        return float(mean_loss), float(accuracy)

    def _compute_loss_accuracy_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for batch in loader:
                xb, yb = self._unpack_training_batch(batch)
                xb = self._validate_and_cast_input(xb).to(self.device)
                yb = yb.to(dtype=torch.long, device=self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                preds = logits.argmax(dim=1)

                batch_seen = int(yb.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_seen
                total_correct += int((preds == yb).sum().item())
                total_seen += batch_seen

        mean_loss = total_loss / max(total_seen, 1)
        accuracy = (total_correct / max(total_seen, 1)) * 100.0
        return float(mean_loss), float(accuracy)

    def _unpack_training_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise ValueError(
                    "Training batches must contain exactly two items: (features, labels)."
                )
            xb, yb = batch
        else:
            raise TypeError(
                f"Training loader must yield (features, labels) tuples, got {type(batch)!r}."
            )
        if not isinstance(xb, torch.Tensor) or not isinstance(yb, torch.Tensor):
            raise TypeError("Training loader must yield tensor features and tensor labels.")
        return xb, yb

    def _unpack_inference_batch(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Inference loader yielded an empty batch.")
            xb = batch[0]
            if not isinstance(xb, torch.Tensor):
                raise TypeError(
                    f"Inference loader must yield tensor features, got {type(xb)!r}."
                )
            return xb
        raise TypeError(
            f"Inference loader must yield a tensor or tuple/list with tensor first element, got {type(batch)!r}."
        )

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
        eval_batch_size: int | None = None,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        seed: int = 42,
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        torch.manual_seed(int(seed))

        x_train = self._validate_and_cast_input(x_train)
        y_train = y_train.to(dtype=torch.long)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
        val_loader = None
        if x_val is not None and y_val is not None:
            x_val = self._validate_and_cast_input(x_val)
            y_val = y_val.to(dtype=torch.long)
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=int(eval_batch_size) if eval_batch_size is not None else int(batch_size),
                shuffle=False,
            )
        return self.fit_loader(
            loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            epoch_logger=epoch_logger,
        )

    def fit_loader(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        seed: int = 42,
        epoch_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> ProbeFitResult:
        torch.manual_seed(int(seed))

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
        patience = None if early_stopping_patience is None else max(1, int(early_stopping_patience))
        epochs_without_improvement = 0
        early_stopped = False

        for epoch_idx in range(int(epochs)):
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch_idx)

            self.model.train()
            running = 0.0
            n_batches = 0
            n_correct = 0
            n_seen = 0

            for batch in train_loader:
                xb, yb = self._unpack_training_batch(batch)
                xb = self._validate_and_cast_input(xb).to(self.device)
                yb = yb.to(dtype=torch.long, device=self.device)

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

            if val_loader is not None:
                val_loss, val_accuracy = self._compute_loss_accuracy_loader(
                    val_loader,
                    criterion,
                )
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
                    epochs_without_improvement = 0
                elif patience is not None:
                    epochs_without_improvement += 1

            history.append(row)
            if epoch_logger is not None:
                epoch_logger(dict(row))
            if (
                val_loader is not None
                and patience is not None
                and epochs_without_improvement >= patience
            ):
                early_stopped = True
                break

        if best_state_dict is None:
            best_epoch = len(history) if history else int(epochs)
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
            n_epochs=len(history),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            early_stopped=early_stopped,
            early_stopping_patience=patience,
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

    def predict_logits_loader(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                xb = self._unpack_inference_batch(batch)
                xb = self._validate_and_cast_input(xb).to(self.device)
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
