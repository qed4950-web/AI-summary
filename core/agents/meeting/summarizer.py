"""Summarisation helpers for the meeting agent."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, List

LOGGER = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%s; using %s", name, raw, default)
        return default


@dataclass
class SummariserConfig:
    model_name: str = os.getenv("MEETING_SUMMARY_MODEL", "gogamza/kobart-base-v2")
    max_length: int = _int_env("MEETING_SUMMARY_MAXLEN", 128)
    min_length: int = _int_env("MEETING_SUMMARY_MINLEN", 32)
    chunk_char_limit: int = _int_env("MEETING_SUMMARY_CHUNK_CHARS", 1800)


class KoBARTSummariser:
    """Chunked KoBART summarisation with lazy pipeline initialisation."""

    def __init__(self, config: SummariserConfig | None = None) -> None:
        self.config = config or SummariserConfig()
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for KoBART summarisation") from exc

        LOGGER.info("Loading KoBART summariser model: %s", self.config.model_name)
        self._pipeline = pipeline(
            "summarization",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
        )
        return self._pipeline

    def summarise(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        chunks = self._chunk_text(text, self.config.chunk_char_limit)
        summarise_chunk = self._make_chunk_summariser()
        partials = [summarise_chunk(chunk) for chunk in chunks if chunk.strip()]
        partials = [item for item in partials if item]

        if not partials:
            return ""

        if len(partials) == 1:
            return partials[0]

        combined = " ".join(partials)
        final = summarise_chunk(combined)
        return final or combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_chunk_summariser(self) -> Callable[[str], str]:
        pipeline = self._ensure_pipeline()
        max_length = self.config.max_length
        min_length = self.config.min_length

        def _summarise_chunk(chunk: str) -> str:
            try:
                result: List[dict] = pipeline(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )
                if not result:
                    return ""
                return (result[0].get("summary_text") or "").strip()
            except Exception as exc:  # pragma: no cover - inference guard
                LOGGER.exception("KoBART summarisation failed: %s", exc)
                return ""

        return _summarise_chunk

    def _chunk_text(self, text: str, limit: int) -> List[str]:
        if limit <= 0:
            return [text]

        sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]
        if not sentences:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            length = len(sentence)
            if current and current_len + length > limit:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = length
            else:
                current.append(sentence)
                current_len += length

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]
