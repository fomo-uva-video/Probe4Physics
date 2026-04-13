

"""Shared adapter contracts and registry utilities for video backbones.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Protocol, Sequence
import torch


@dataclass(frozen=True)
class BackboneFeatures:
    """Canonical feature bundle returned by all backbone adapters.

    Attributes:
        tokens_by_layer: Per-layer token features with shape [B, N, D].
        pooled_by_layer: Per-layer pooled clip features with shape [B, D].
        selected_layers: Ordered tuple of layer ids included in this output.
        metadata: Lightweight context for provenance/reproducibility.
    """

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


AdapterFactory = Callable[..., VideoBackboneAdapter]

_ADAPTER_FACTORIES: dict[str, AdapterFactory] = {}
_RUNTIME_GUARD_LOCK = Lock()
_ACTIVE_JEPA_NAMESPACE: str | None = None


def register_adapter(name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
    """Register an adapter factory under a stable string key."""

    key = str(name).strip()
    if not key:
        raise ValueError("Adapter name cannot be empty.")

    if key in _ADAPTER_FACTORIES and not replace:
        raise ValueError(f"Adapter '{key}' is already registered.")

    _ADAPTER_FACTORIES[key] = factory


def create_adapter(name: str, **kwargs: Any) -> VideoBackboneAdapter:
    """Instantiate a registered adapter by name."""

    key = str(name).strip()
    if key not in _ADAPTER_FACTORIES:
        known = ", ".join(sorted(_ADAPTER_FACTORIES)) or "<none>"
        raise KeyError(f"Unknown adapter '{key}'. Registered adapters: {known}")
    return _ADAPTER_FACTORIES[key](**kwargs)


def get_registered_adapters() -> tuple[str, ...]:
    """Return sorted adapter names currently available in this process."""

    return tuple(sorted(_ADAPTER_FACTORIES))


def enforce_single_jepa_namespace(namespace: str) -> None:
    """Ensure only one JEPA family namespace is active per Python process.

    JEPA repositories commonly expose modules via top-level `src.*`.
    Importing multiple JEPA families in one process can therefore shadow
    symbols unexpectedly. This guard fails fast on mixed namespaces.
    """

    global _ACTIVE_JEPA_NAMESPACE

    requested = str(namespace).strip()
    if not requested:
        raise ValueError("Namespace cannot be empty.")

    with _RUNTIME_GUARD_LOCK:
        if _ACTIVE_JEPA_NAMESPACE is None:
            _ACTIVE_JEPA_NAMESPACE = requested
            return

        if _ACTIVE_JEPA_NAMESPACE == requested:
            return

        raise RuntimeError(
            "A different JEPA namespace is already active in this process "
            f"('{_ACTIVE_JEPA_NAMESPACE}' vs '{requested}'). "
            "Run each JEPA family in a separate process to avoid `src.*` import collisions."
        )


def reset_runtime_guard_for_tests() -> None:
    """Test helper: clear runtime namespace state between test cases."""

    global _ACTIVE_JEPA_NAMESPACE
    with _RUNTIME_GUARD_LOCK:
        _ACTIVE_JEPA_NAMESPACE = None
