"""HTTP client helpers for the FastAPI pipeline server."""

from __future__ import annotations

from typing import Any, Dict

import httpx

DEFAULT_API_BASE = "http://127.0.0.1:8080"


class APIClientError(RuntimeError):
    """Wrapper for user-friendly API errors."""


def _normalize_base(base_url: str | None) -> str:
    base = (base_url or DEFAULT_API_BASE).strip()
    if not base:
        base = DEFAULT_API_BASE
    return base[:-1] if base.endswith("/") else base


def trigger_pipeline_run(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_normalize_base(base_url)}/pipeline/run"
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        raise APIClientError(str(exc)) from exc
    if response.status_code >= 300:
        raise APIClientError(f"{response.status_code}: {response.text}")
    return response.json()


def fetch_pipeline_status(base_url: str) -> Dict[str, Any]:
    url = f"{_normalize_base(base_url)}/pipeline/status"
    try:
        response = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        raise APIClientError(str(exc)) from exc
    if response.status_code >= 300:
        raise APIClientError(f"{response.status_code}: {response.text}")
    return response.json()


def cancel_pipeline(base_url: str) -> Dict[str, Any]:
    url = f"{_normalize_base(base_url)}/pipeline/cancel"
    try:
        response = httpx.post(url, timeout=10.0)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        raise APIClientError(str(exc)) from exc
    if response.status_code >= 300:
        raise APIClientError(f"{response.status_code}: {response.text}")
    return response.json()
