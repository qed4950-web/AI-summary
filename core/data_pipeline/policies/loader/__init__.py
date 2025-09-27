"""Helpers to load and validate smart folder policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import jsonschema

from core.utils import get_logger, resolve_repo_root

LOGGER = get_logger("policy.loader")


def load_policy(path: Path) -> Dict[str, Any]:
    """Load a single smart folder policy."""
    policies = load_policy_file(path)
    if not policies:
        raise ValueError(f"No policy found in {path}")
    return policies[0]


def load_policy_file(path: Path) -> List[Dict[str, Any]]:
    """Load one or more smart folder policies from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items: Iterable[Dict[str, Any]] = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Policy file must contain a JSON object or array of objects")

    schema = _load_schema()
    policies: List[Dict[str, Any]] = []
    for idx, data in enumerate(items):
        jsonschema.validate(data, schema)
        policies.append(data)
    LOGGER.debug("Loaded %d policies from %s", len(policies), path)
    return policies


def _load_schema() -> Dict[str, Any]:
    schema_path = resolve_repo_root() / "core" / "data_pipeline" / "policies" / "schema" / "smart_folder_policy.schema.json"
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        LOGGER.error("Policy schema missing at %s", schema_path)
        raise exc
