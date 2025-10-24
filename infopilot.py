"""Compatibility shim for the original ``infopilot`` module location."""

import os as _os
from importlib import import_module as _import_module
import sys as _sys

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("KMP_AFFINITY", "disabled")
_os.environ.setdefault("KMP_BLOCKTIME", "0")
_os.environ.setdefault("KMP_SETTINGS", "1")

_impl = _import_module("scripts.pipeline.infopilot")
_sys.modules[__name__] = _impl
