import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_SETTINGS: Dict[str, Any] = {
    "conversation": {
        "llm_backend": "",
        "llm_model": "",
        "llm_host": "",
        "llm_api_key": "",
        "llm_health_timeout": 20.0,
        "top_k": 8,
        "min_similarity": 0.35,
        "include_references": True,
    },
    "agents": {
        "meeting": {
            "recent_audio_files": [],
            "stt": {
                "backend": "auto",
                "model": "",
                "device": "",
                "compute": "",
                "download_dir": "",
            },
        },
        "photo": {
            "recent_roots": [],
        },
    }
}


class SettingsManager:
    """Lightweight JSON-backed settings helper for desktop UI."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self._cache = json.loads(json.dumps(DEFAULT_SETTINGS))
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("settings payload must be a dict")
            merged = json.loads(json.dumps(DEFAULT_SETTINGS))
            _deep_merge(merged, data)
            self._cache = merged
        except Exception:
            self._cache = json.loads(json.dumps(DEFAULT_SETTINGS))

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        cursor = self._cache
        for key in keys:
            if not isinstance(cursor, dict):
                return default
            cursor = cursor.get(key)
        return cursor if cursor is not None else default

    def set(self, value: Any, *keys: str) -> None:
        if not keys:
            raise ValueError("keys must not be empty")
        cursor = self._cache
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
        self._save()


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
