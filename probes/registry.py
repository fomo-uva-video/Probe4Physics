from __future__ import annotations

from typing import Any, Callable

from probes.base import Probe

ProbeFactory = Callable[..., Probe]

_PROBE_FACTORIES: dict[str, ProbeFactory] = {}


def register_probe(name: str, factory: ProbeFactory, *, replace: bool = False) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("Probe name cannot be empty.")

    if key in _PROBE_FACTORIES and not replace:
        raise ValueError(f"Probe '{key}' is already registered.")

    _PROBE_FACTORIES[key] = factory


def create_probe(name: str, **kwargs: Any) -> Probe:
    key = str(name).strip()
    if key not in _PROBE_FACTORIES:
        known = ", ".join(sorted(_PROBE_FACTORIES)) or "<none>"
        raise KeyError(f"Unknown probe '{key}'. Registered probes: {known}")
    return _PROBE_FACTORIES[key](**kwargs)


def get_registered_probes() -> tuple[str, ...]:
    return tuple(sorted(_PROBE_FACTORIES))
