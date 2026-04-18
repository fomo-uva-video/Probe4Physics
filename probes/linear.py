from __future__ import annotations

from typing import Any

import torch
from torch import nn

from probes.base import BaseClassifierProbe
from probes.registry import register_probe


class LinearProbe(BaseClassifierProbe):
    """Simple linear classifier probe over frozen features."""

    def _build_model(self) -> nn.Module:
        return nn.Linear(self.input_dim, self.num_classes)

    def _checkpoint_type(self) -> str:
        return "linear"

    @classmethod
    def _init_kwargs_from_payload(
        cls,
        payload: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        probe_type = str(payload.get("type", ""))
        if probe_type != "linear":
            raise ValueError(f"Checkpoint type mismatch: expected 'linear', got '{probe_type}'")
        return {
            "input_dim": int(payload["input_dim"]),
            "num_classes": int(payload["num_classes"]),
            "device": device,
        }

    def _validate_and_cast_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"LinearProbe expects feature tensors with shape [N, D], got {tuple(x.shape)}")
        return x.to(dtype=torch.float32)



def create_linear_probe(**kwargs: Any) -> LinearProbe:
    return LinearProbe(**kwargs)


register_probe("linear", create_linear_probe, replace=True)
