"""Compatibility shims for legacy backend imports used by the test suite."""

from importlib import import_module

# Older integrations expect ``backend.api.session`` to expose the shared
# session registry. The canonical implementation lives under ``core.api``,
# so we lazily import and re-export the module here to keep the public
# surface compatible with existing tests.
session = import_module("core.api.session")

__all__ = ["session"]
