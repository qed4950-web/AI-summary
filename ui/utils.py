"""Utilities and constants shared across UI screens."""
from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from core.data_pipeline.pipeline import CorpusBuilder as _CorpusBuilder
from core.search.retriever import Retriever

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
FOUND_FILES_CSV = DATA_DIR / "found_files.csv"
MEETING_OUTPUT_DIR = DATA_DIR / "meetings"
PHOTO_OUTPUT_DIR = DATA_DIR / "photos"
CORPUS_PARQUET = CORPUS_PATH

# ---------------------------------------------------------------------------
# Scanning defaults
# ---------------------------------------------------------------------------
SUPPORTED_EXTS: Set[str] = set(FileFinder.DEFAULT_EXTS) | {".txt", ".md"}
EXCLUDE_DIRS: Set[str] = set(FileFinder.COMMON_SKIP_DIRS)
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Helpers for UI
# ---------------------------------------------------------------------------
def get_drives() -> List[Path]:
    """Return available drive roots similar to the legacy helper."""
    finder = FileFinder(scan_all_drives=True, start_from_current_drive_only=False)
    roots = finder.get_roots()

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
        home = Path.home()
        if platform.system().lower() == "windows":
            unique = [home]
        else:
            unique = [home, Path("/")]
    return unique


def have_all_artifacts() -> bool:
    """Check whether the primary corpus and topic model artifacts exist."""
    return CORPUS_PATH.exists() and TOPIC_MODEL_PATH.exists()


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


def rebuild_index(*, corpus_path: Path, cache_dir: Path) -> None:
    """Rebuild the ANN index using the modern retriever."""
    if not TOPIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"토픽 모델 파일을 찾을 수 없습니다: {TOPIC_MODEL_PATH}. "
            "먼저 전체 학습을 실행해 topic_model.joblib을 생성하세요."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(
        model_path=TOPIC_MODEL_PATH,
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        auto_refresh=False,
    )
    try:
        retriever.ready(rebuild=True, wait=True)
    finally:
        retriever.shutdown()


# 단순 별칭 (UI 코드에서 기존 이름 유지)
CorpusBuilder = _CorpusBuilder


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_DIR",
    "SUMMARIES_DIR",
    "MODELS_DIR",
    "TOPIC_MODEL_PATH",
    "CORPUS_PATH",
    "CORPUS_PARQUET",
    "FOUND_FILES_CSV",
    "SUPPORTED_EXTS",
    "EXCLUDE_DIRS",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MEETING_OUTPUT_DIR",
    "PHOTO_OUTPUT_DIR",
    "get_drives",
    "have_all_artifacts",
    "parse_query_and_filters",
    "rebuild_index",
    "CorpusBuilder",
]
