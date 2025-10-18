"""Compatibility shim for the original ``infopilot`` module location."""

from importlib import import_module as _import_module
import sys as _sys

_impl = _import_module("scripts.infopilot")
_sys.modules[__name__] = _impl
