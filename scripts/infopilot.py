"""Compatibility shim for the legacy `scripts.infopilot` module."""

from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path

# ---------------------------------------------------------------------------
# macOS PyTorch guardrails: avoid shared-memory errors before torch loads.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_SETTINGS", "1")


def _ensure_repo_root() -> None:
    """Insert repository root into sys.path so sibling imports succeed."""

    # Prefer the project root (two levels up) when possible.
    here = Path(__file__).resolve()
    for candidate in [here.parent, here.parents[1], here.parents[2]]:
        if (candidate / "scripts").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_ensure_repo_root()

# Now import the canonical implementation.
_module = importlib.import_module("scripts.pipeline.infopilot")
sys.modules[__name__] = _module

# Re-export symbols so legacy imports (e.g. ``import infopilot``) continue to work.
_exported: dict[str, object] = {}
for _name in dir(_module):
    if _name.startswith("__"):
        continue
    _exported[_name] = getattr(_module, _name)
globals().update(_exported)

__all__ = sorted(name for name in _exported if not name.startswith("_"))
__all__.append("main")

_EXCLUDED = {"__builtins__", "__all__", "main"}
__all__ = [name for name in globals() if not name.startswith("_") and name not in _EXCLUDED]
__all__.append("main")


def main() -> None:
    """Entry point retained for backwards compatibility."""

    _module.main()


if __name__ == "__main__":
    main()
