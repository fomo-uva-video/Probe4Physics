from __future__ import annotations

from typing import Any

import torch
from torch import nn

from probes.base import BaseClassifierProbe
from probes.registry import register_probe


class MLPProbe(BaseClassifierProbe):
    """Multi-layer perceptron probe over frozen features.

    Architecture: a configurable stack of Linear → LayerNorm → GELU blocks,
    followed by a final linear classifier.  Dropout can optionally be added
    between blocks for regularisation.

    Parameters
    ----------
    input_dim:
        Dimensionality of the frozen feature vectors (D).
    num_classes:
        Number of output classes (must be ≥ 2).
    hidden_dims:
        Sequence of hidden layer widths.  Defaults to ``[512]``.
    dropout:
        Dropout probability applied after each hidden block.  Set to 0.0 to
        disable (default).
    device:
        The torch device to place the model on.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hidden_dims: list[int] = list(hidden_dims) if hidden_dims else [512]
        self.dropout = float(dropout)
        super().__init__(input_dim=input_dim, num_classes=num_classes, device=device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        return nn.Sequential(*layers)

    def _checkpoint_type(self) -> str:
        return "mlp"

    def _checkpoint_extra_fields(self) -> dict[str, Any]:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
        }

    @classmethod
    def _init_kwargs_from_payload(
        cls,
        payload: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        probe_type = str(payload.get("type", ""))
        if probe_type != "mlp":
            raise ValueError(f"Checkpoint type mismatch: expected 'mlp', got '{probe_type}'")
        return {
            "input_dim": int(payload["input_dim"]),
            "num_classes": int(payload["num_classes"]),
            "hidden_dims": list(payload.get("hidden_dims", [512])),
            "dropout": float(payload.get("dropout", 0.0)),
            "device": device,
        }

    def _validate_and_cast_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MLPProbe expects feature tensors with shape [N, D], got {tuple(x.shape)}")
        return x.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public interface  (mirrors LinearProbe)
    # ------------------------------------------------------------------


def create_mlp_probe(**kwargs: Any) -> MLPProbe:
    return MLPProbe(**kwargs)


register_probe("mlp", create_mlp_probe, replace=True)
