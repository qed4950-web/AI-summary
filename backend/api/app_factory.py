"""Minimal FastAPI application factory for desktop-agent tests."""
from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import FastAPI

try:  # pragma: no cover - optional dependency guard
    from backend.api.settings import Settings  # type: ignore
except ImportError:  # pragma: no cover
    from .settings import Settings  # type: ignore


def create_app(
    *,
    settings: Optional[Settings] = None,
    retriever_provider: Optional[Callable[[], Any]] = None,
) -> FastAPI:
    """Return a stub FastAPI application used by test fixtures.

    The full backend service is not bundled with the desktop UI project, but
    the tests expect an application factory. This stub keeps the contract
    simple while allowing local integrations to attach additional routes when
    the real backend package is installed.
    """

    app = FastAPI(title="Stub Backend", version="0.0.0")
    app.state.settings = settings or Settings()
    app.state.retriever_provider = retriever_provider or (lambda: None)

    @app.on_event("startup")
    async def _init_retriever():  # pragma: no cover - simple stub
        provider = getattr(app.state, "retriever_provider", None)
        if callable(provider):
            try:
                retriever = provider()
            except Exception:  # noqa: BLE001 - diagnostics best effort
                retriever = None
            app.state.retriever = retriever

    @app.get("/health", tags=["internal"])
    def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": getattr(app.state, "settings", Settings()).environment,
        }

    return app
