"""Utility helpers for robust text encoding detection and fallbacks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_FALLBACK_ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "cp949",
    "euc-kr",
    "utf-16",
    "latin-1",
]


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def detect_file_encodings(path: Path, extras: Optional[Iterable[str]] = None) -> List[str]:
    """Return a prioritized list of encodings to try for ``path``."""

    candidates: List[str] = []

    try:
        from charset_normalizer import from_path  # type: ignore

        result = from_path(str(path))
        if result:
            best = result.best()
            if best and best.encoding:
                candidates.append(best.encoding)
    except Exception:
        try:
            import chardet  # type: ignore

            with path.open("rb") as fb:
                raw = fb.read(256 * 1024)
            if raw:
                guess = chardet.detect(raw)
                encoding = guess.get("encoding")
                if encoding:
                    candidates.append(encoding)
        except Exception:
            pass

    if extras:
        candidates.extend(extras)

    candidates.extend(DEFAULT_FALLBACK_ENCODINGS)
    return _unique_preserve_order(candidates)


def open_text_with_fallback(path: Path, *, errors: str = "strict"):
    """Yield ``(encoding, file_object)`` pairs trying multiple encodings."""

    for encoding in detect_file_encodings(path):
        try:
            f = path.open("r", encoding=encoding, errors=errors)
        except UnicodeDecodeError:
            continue
        try:
            yield encoding, f
        except Exception:
            f.close()
            raise

