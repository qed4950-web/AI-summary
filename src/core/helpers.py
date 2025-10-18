"""Helper functions retained for UI compatibility."""
from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.data_pipeline.filefinder import FileFinder

from src.config import CORPUS_PARQUET, TOPIC_MODEL_PATH


def get_drives() -> List[Path]:
    """Return available drive roots similar to the legacy helper."""
    finder = FileFinder(scan_all_drives=True, start_from_current_drive_only=False)
    roots = finder.get_roots()
    # ensure unique resolved paths
    seen = set()
    unique: List[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    if not unique:
        # Fallback to home directory and root
        home = Path.home()
        unique = [home, Path("/")] if platform.system().lower() != "windows" else [home]
    return unique


def have_all_artifacts() -> bool:
    """Check whether the primary corpus and topic model artifacts exist."""
    return CORPUS_PARQUET.exists() and TOPIC_MODEL_PATH.exists()


_FILTER_TOKEN_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_+-]+):(?!//)(?P<value>.+)$")
_TRAILING_PUNCT_RE = re.compile(r"[.,;]+$")


def _strip_trailing_punct(value: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", value)


def parse_query_and_filters(query: str) -> Tuple[str, Optional[Dict[str, List[str]]]]:
    """Extract simple `key:value` filters from a query string."""
    if not query:
        return "", None

    filters: Dict[str, List[str]] = {}
    remaining_tokens: List[str] = []

    for raw_token in query.split():
        match = _FILTER_TOKEN_RE.match(raw_token)
        if match:
            key = match.group("key").lower()
            value = _strip_trailing_punct(match.group("value"))
            if key and value:
                filters.setdefault(key, []).append(value)
                continue
        remaining_tokens.append(raw_token)

    cleaned_query = " ".join(remaining_tokens).strip()
    return (cleaned_query, filters or None)
