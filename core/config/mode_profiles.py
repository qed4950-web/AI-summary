"""Shared mode profile config for desktop runtime behavior."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from core.config.paths import DATA_DIR

MODE_ORDER: tuple[str, ...] = ("Auto", "Instant", "Thinking", "Pro")
MODE_PROFILES_PATH = DATA_DIR / "desktop_mode_profiles.json"

DEFAULT_MODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Auto": {
        "description": "난이도에 따라 자동 조절",
        "topk": None,
        "force_action": None,
        "thinking_status": "Thinking",
        "llm_max_new_tokens": 512,
        "llm_temperature": 0.10,
    },
    "Instant": {
        "description": "즉시 답변",
        "topk": 3,
        "force_action": "chat",
        "thinking_status": "Thinking fast",
        "llm_max_new_tokens": 256,
        "llm_temperature": 0.00,
    },
    "Thinking": {
        "description": "깊은 추론",
        "topk": 6,
        "force_action": "search",
        "thinking_status": "Thinking deep",
        "llm_max_new_tokens": 768,
        "llm_temperature": 0.15,
    },
    "Pro": {
        "description": "리서치 중심",
        "topk": 8,
        "force_action": "search",
        "thinking_status": "Thinking pro",
        "llm_max_new_tokens": 1024,
        "llm_temperature": 0.25,
    },
}


def _as_int(value: object, default: int, *, min_value: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _as_float(value: object, default: float, *, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _as_action(value: object, default: str | None) -> str | None:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", ""}:
        return None
    if normalized in {"chat", "search"}:
        return normalized
    return default


def _as_topk(value: object, default: int | None) -> int | None:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", ""}:
        return None
    try:
        parsed = int(normalized)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _merge_profile(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged["description"] = str(override.get("description", merged["description"])).strip() or merged["description"]
    merged["thinking_status"] = (
        str(override.get("thinking_status", merged["thinking_status"])).strip() or merged["thinking_status"]
    )
    merged["force_action"] = _as_action(override.get("force_action"), merged["force_action"])
    merged["topk"] = _as_topk(override.get("topk"), merged["topk"])
    merged["llm_max_new_tokens"] = _as_int(
        override.get("llm_max_new_tokens"),
        int(merged["llm_max_new_tokens"]),
    )
    merged["llm_temperature"] = _as_float(
        override.get("llm_temperature"),
        float(merged["llm_temperature"]),
    )
    return merged


def load_mode_profiles(path: Path = MODE_PROFILES_PATH) -> Dict[str, Dict[str, Any]]:
    profiles = {mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER}
    if not path.exists():
        return profiles
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return profiles
    if not isinstance(payload, dict):
        return profiles
    for mode in MODE_ORDER:
        raw = payload.get(mode)
        if isinstance(raw, dict):
            profiles[mode] = _merge_profile(profiles[mode], raw)
    return profiles


def save_mode_profiles(profiles: Dict[str, Dict[str, Any]], path: Path = MODE_PROFILES_PATH) -> None:
    sanitized: Dict[str, Dict[str, Any]] = {}
    base = {mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER}
    for mode in MODE_ORDER:
        raw = profiles.get(mode, {})
        sanitized[mode] = _merge_profile(base[mode], raw if isinstance(raw, dict) else {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MODE_PROFILES",
    "MODE_ORDER",
    "MODE_PROFILES_PATH",
    "load_mode_profiles",
    "save_mode_profiles",
]
