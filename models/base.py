from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class BackboneFeatures:
    """Canonical feature bundle returned by all backbone adapters."""

    tokens_by_layer: dict[int, torch.Tensor]
    pooled_by_layer: dict[int, torch.Tensor]
    selected_layers: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoBackboneAdapter(Protocol):
    """Protocol each backbone adapter must implement."""

    def extract(
        self,
        clips: torch.Tensor,
        layer_ids: Sequence[int] | None = None,
    ) -> BackboneFeatures:
        """Extract frozen features from preprocessed clips [B, C, T, H, W]."""
