"""Compatibility layer providing legacy `src` module expected by UI components."""

import sys

from . import config  # re-export for convenience

# Alias legacy module names used by older joblib artefacts
try:
    import core.data_pipeline.pipeline as _pipeline_module

    sys.modules.setdefault("pipeline", _pipeline_module)
except Exception:
    pass

__all__ = ["config"]
