"""Shared loader for smart folder policy engine used across the desktop UI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.data_pipeline.policies.engine import PolicyEngine

REPO_ROOT = Path(__file__).resolve().parent.parent

_POLICY_CACHE: Optional[PolicyEngine] = None
_POLICY_PATH: Optional[Path] = None
_POLICY_MTIME: Optional[float] = None


def _candidate_policy_paths() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.getenv("INFOPILOT_POLICY_FILE")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    default_candidates = [
        REPO_ROOT / "core" / "config" / "smart_folders.json",
        REPO_ROOT / "core" / "config" / "smart_folder_policies.json",
        REPO_ROOT / "core" / "data_pipeline" / "policies" / "smart_folder_policy.json",
        REPO_ROOT / "core" / "data_pipeline" / "policies" / "examples" / "smart_folder_policy_sample.json",
    ]
    candidates.extend(default_candidates)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_policy_path() -> Optional[Path]:
    for candidate in _candidate_policy_paths():
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def get_policy_engine(force_reload: bool = False) -> PolicyEngine:
    """Return a cached PolicyEngine loaded from the best available policy file."""
    global _POLICY_CACHE, _POLICY_PATH, _POLICY_MTIME

    path = _resolve_policy_path()
    if path is None:
        _POLICY_CACHE = PolicyEngine.empty()
        _POLICY_PATH = None
        _POLICY_MTIME = None
        return _POLICY_CACHE

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if (
        force_reload
        or _POLICY_CACHE is None
        or _POLICY_PATH is None
        or _POLICY_PATH != path
        or (_POLICY_MTIME is not None and mtime is not None and _POLICY_MTIME != mtime)
    ):
        try:
            engine = PolicyEngine.from_file(path)
        except Exception:
            engine = PolicyEngine.empty()
        _POLICY_CACHE = engine
        _POLICY_PATH = path
        _POLICY_MTIME = mtime

    return _POLICY_CACHE


def get_policy_path() -> Optional[Path]:
    """Expose the resolved policy path for diagnostics."""
    return _POLICY_PATH or _resolve_policy_path()
