"""Minimal settings container used by the stub backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Settings:
    """Lightweight substitute for the production settings object.

    The real service uses a richer configuration model. Here we accept any
    keyword arguments and expose them as attributes so tests and integrations
    can override behaviour without pulling the full backend in as a
    dependency.
    """

    environment: str = "test"
    values: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, environment: str | None = None, **kwargs: Any) -> None:
        object.__setattr__(self, "environment", environment or "test")
        storage: Dict[str, Any] = {}
        for key, value in kwargs.items():
            storage[key] = value
            # mirror uppercase keys as lowercase attributes for convenience
            object.__setattr__(self, key.lower(), value)
            object.__setattr__(self, key, value)
        object.__setattr__(self, "values", storage)

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - convenience
        payload = {"environment": self.environment}
        payload.update(self.values)
        return payload
