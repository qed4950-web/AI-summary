"""Compatibility layer that mirrors the legacy `infopilot_core` package layout.

This project internally migrated to `core.*`, but several components (tests,
UI, PyInstaller analyses) still import `infopilot_core.*`.  To stay backwards
compatible we forward every submodule import to the new location and expose the
same attributes so that `import infopilot_core` behaves like `import core`.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable

_ALIAS_TARGET = "core"
_SUBMODULES: Iterable[str] = (
    "agents",
    "conversation",
    "data_pipeline",
    "infra",
    "search",
    "utils",
)


def _install_aliases() -> ModuleType:
    target = importlib.import_module(_ALIAS_TARGET)
    sys.modules.setdefault("infopilot_core", target)

    for name in _SUBMODULES:
        module = importlib.import_module(f"{_ALIAS_TARGET}.{name}")
        sys.modules[f"infopilot_core.{name}"] = module
        setattr(target, name, module)

    # Convenience re-export for callers that expect attributes
    globals().update(target.__dict__)
    return target


_core_module = _install_aliases()


def __getattr__(item: str):  # pragma: no cover - passthrough
    return getattr(_core_module, item)


__all__ = getattr(_core_module, "__all__", [])
