"""Legacy config facade expected by UI modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

from core.config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    CACHE_DIR,
    SUMMARIES_DIR,
    MODELS_DIR,
    TOPIC_MODEL_PATH,
    CORPUS_PATH,
)
from core.data_pipeline.filefinder import FileFinder

# Legacy constants
FOUND_FILES_CSV = DATA_DIR / "found_files.csv"
SUPPORTED_EXTS: Set[str] = set(FileFinder.DEFAULT_EXTS) | {
    ".txt",
    ".md",
}
EXCLUDE_DIRS: Set[str] = set(FileFinder.COMMON_SKIP_DIRS)
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35

MEETING_OUTPUT_DIR = DATA_DIR / "meetings"
PHOTO_OUTPUT_DIR = DATA_DIR / "photos"

# Backwards compatibility names
CORPUS_PARQUET = CORPUS_PATH

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_DIR",
    "SUMMARIES_DIR",
    "MODELS_DIR",
    "TOPIC_MODEL_PATH",
    "CORPUS_PARQUET",
    "CORPUS_PATH",
    "FOUND_FILES_CSV",
    "SUPPORTED_EXTS",
    "EXCLUDE_DIRS",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MEETING_OUTPUT_DIR",
    "PHOTO_OUTPUT_DIR",
]

