"""Compatibility wrapper around the canonical FastAPI application factory."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Optional

_core_app_factory = import_module("core.api.app_factory")
Settings = getattr(_core_app_factory, "Settings", None)
if Settings is None:  # pragma: no cover - defensive fallback
    from core.api.settings import Settings  # type: ignore


def create_app(
    *,
    settings: Optional[Settings] = None,
    retriever_provider: Optional[Callable[[], Any]] = None,
):
    """Proxy to ``core.api.app_factory.create_app`` while keeping defaults."""

    if settings is None:
        settings = Settings()
    if retriever_provider is None:
        retriever_provider = lambda: None  # noqa: E731 - simple default

    return _core_app_factory.create_app(settings=settings, retriever_provider=retriever_provider)


__all__ = ["create_app", "Settings"]
