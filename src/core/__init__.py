"""Legacy `src.core` namespace expected by UI code."""

from .helpers import get_drives, have_all_artifacts, parse_query_and_filters
from .corpus import CorpusBuilder
from .indexing import run_indexing

__all__ = [
    "get_drives",
    "have_all_artifacts",
    "parse_query_and_filters",
    "CorpusBuilder",
    "run_indexing",
]

