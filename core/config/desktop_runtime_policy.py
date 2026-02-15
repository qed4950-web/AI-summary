"""Persistent desktop runtime policy shared by backend and desktop UI."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.config.paths import DATA_DIR

DESKTOP_RUNTIME_POLICY_ENV_PATH = "DESKTOP_RUNTIME_POLICY_PATH"
DESKTOP_RUNTIME_POLICY_PATH = DATA_DIR / "desktop_runtime_policy.json"
DESKTOP_RUNTIME_POLICY_HISTORY_ENV_PATH = "DESKTOP_RUNTIME_POLICY_HISTORY_PATH"
DESKTOP_RUNTIME_POLICY_HISTORY_PATH = DATA_DIR / "desktop_runtime_policy_history.jsonl"

DEFAULT_DESKTOP_RUNTIME_POLICY: Dict[str, Any] = {
    "privacy_mask_enabled": True,
    "max_file_links": 8,
    "max_reference_links": 5,
    "max_response_chars": 24000,
    "max_suggestion_chars": 120,
}


def _resolve_policy_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env_path = os.getenv(DESKTOP_RUNTIME_POLICY_ENV_PATH, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DESKTOP_RUNTIME_POLICY_PATH


def _resolve_policy_history_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env_path = os.getenv(DESKTOP_RUNTIME_POLICY_HISTORY_ENV_PATH, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DESKTOP_RUNTIME_POLICY_HISTORY_PATH


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: object, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    if max_value is not None and parsed > max_value:
        return default
    return parsed


def _normalize_runtime_policy(
    raw: Dict[str, Any],
    *,
    base: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    defaults = dict(DEFAULT_DESKTOP_RUNTIME_POLICY if base is None else base)
    normalized = dict(defaults)
    normalized["privacy_mask_enabled"] = _as_bool(raw.get("privacy_mask_enabled"), bool(defaults["privacy_mask_enabled"]))
    normalized["max_file_links"] = _as_int(raw.get("max_file_links"), int(defaults["max_file_links"]), min_value=1, max_value=64)
    normalized["max_reference_links"] = _as_int(
        raw.get("max_reference_links"),
        int(defaults["max_reference_links"]),
        min_value=1,
        max_value=64,
    )
    normalized["max_response_chars"] = _as_int(
        raw.get("max_response_chars"),
        int(defaults["max_response_chars"]),
        min_value=1200,
        max_value=120000,
    )
    normalized["max_suggestion_chars"] = _as_int(
        raw.get("max_suggestion_chars"),
        int(defaults["max_suggestion_chars"]),
        min_value=24,
        max_value=1024,
    )
    return normalized


def _load_env_fallback() -> Dict[str, Any]:
    return {
        "privacy_mask_enabled": os.getenv("DESKTOP_MASK_PII"),
        "max_file_links": os.getenv("DESKTOP_MAX_FILE_LINKS"),
        "max_reference_links": os.getenv("DESKTOP_MAX_REFERENCE_LINKS"),
        "max_response_chars": os.getenv("DESKTOP_MAX_RESPONSE_CHARS"),
        "max_suggestion_chars": os.getenv("DESKTOP_MAX_SUGGESTION_CHARS"),
    }


def load_desktop_runtime_policy(
    path: Path | None = None,
    *,
    use_env_fallback: bool = True,
) -> Dict[str, Any]:
    resolved_path = _resolve_policy_path(path)
    policy = dict(DEFAULT_DESKTOP_RUNTIME_POLICY)
    loaded_from_file = False
    if resolved_path.exists():
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            policy = _normalize_runtime_policy(payload, base=policy)
            loaded_from_file = True
    if use_env_fallback and not loaded_from_file:
        policy = _normalize_runtime_policy(_load_env_fallback(), base=policy)
    return policy


def _normalize_history_source(source: object) -> str:
    text = str(source or "").strip()
    return text or "save_desktop_runtime_policy"


def save_desktop_runtime_policy(
    policy: Dict[str, Any],
    path: Path | None = None,
    *,
    source: str = "save_desktop_runtime_policy",
) -> None:
    resolved_path = _resolve_policy_path(path)
    previous: Dict[str, Any] | None = None
    if resolved_path.exists():
        previous = load_desktop_runtime_policy(resolved_path, use_env_fallback=False)
    normalized = _normalize_runtime_policy(policy if isinstance(policy, dict) else {})
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if previous != normalized:
        append_desktop_runtime_policy_history(normalized, source=source)


def append_desktop_runtime_policy_history(
    policy: Dict[str, Any],
    path: Path | None = None,
    *,
    source: str = "save_desktop_runtime_policy",
) -> None:
    resolved_path = _resolve_policy_history_path(path)
    entry = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": _normalize_history_source(source),
        "policy": _normalize_runtime_policy(policy if isinstance(policy, dict) else {}),
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_desktop_runtime_policy_history(
    path: Path | None = None,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    resolved_path = _resolve_policy_history_path(path)
    if limit <= 0 or not resolved_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for raw in reversed(lines):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            entries.append(payload)
        if len(entries) >= limit:
            break
    return entries


__all__ = [
    "DEFAULT_DESKTOP_RUNTIME_POLICY",
    "DESKTOP_RUNTIME_POLICY_HISTORY_ENV_PATH",
    "DESKTOP_RUNTIME_POLICY_HISTORY_PATH",
    "DESKTOP_RUNTIME_POLICY_ENV_PATH",
    "DESKTOP_RUNTIME_POLICY_PATH",
    "append_desktop_runtime_policy_history",
    "load_desktop_runtime_policy_history",
    "load_desktop_runtime_policy",
    "save_desktop_runtime_policy",
]
