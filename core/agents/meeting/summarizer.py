"""Summarisation helpers and backends for the meeting agent."""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

LOGGER = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
DEFAULT_OLLAMA_PROMPT = textwrap.dedent(
    """
    You are a helpful meeting assistant.
    Summarise the following meeting transcript in the user's language.
    Highlight key decisions and actionable follow-up items.
    Respond using concise bullet points.

    Transcript:
    {transcript}
    """
)


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
    """Common configuration shared by the supported summariser backends."""

    model_name: str = os.getenv("MEETING_SUMMARY_MODEL", "gogamza/kobart-base-v2")
    english_model_name: str = os.getenv("MEETING_SUMMARY_EN_MODEL", "facebook/bart-large-cnn")
    max_length: int = _int_env("MEETING_SUMMARY_MAXLEN", 128)
    min_length: int = _int_env("MEETING_SUMMARY_MINLEN", 32)
    max_new_tokens: int = _int_env("MEETING_SUMMARY_MAX_NEW_TOKENS", 128)
    num_beams: int = _int_env("MEETING_SUMMARY_NUM_BEAMS", 4)
    model_max_input_chars: int = _int_env("MEETING_SUMMARY_MODEL_MAX_CHARS", 1024)
    chunk_char_limit: int = _int_env("MEETING_SUMMARY_CHUNK_CHARS", 1800)

    ollama_model: str = os.getenv("MEETING_SUMMARY_OLLAMA_MODEL", "llama3")
    ollama_host: str = os.getenv("MEETING_SUMMARY_OLLAMA_HOST", "")
    ollama_prompt: str = os.getenv("MEETING_SUMMARY_OLLAMA_PROMPT", DEFAULT_OLLAMA_PROMPT)

    bitnet_model: str = os.getenv("MEETING_SUMMARY_BITNET_MODEL", "bitnet/b1.58-instruct")
    bitnet_max_length: int = _int_env("MEETING_SUMMARY_BITNET_MAXLEN", 220)
    bitnet_min_length: int = _int_env("MEETING_SUMMARY_BITNET_MINLEN", 60)


class BaseSummariser(Protocol):
    """Protocol describing the summariser interface."""

    def summarise(self, text: str) -> str:
        ...

    @staticmethod
    def is_available() -> bool:
        ...


class KoBARTSummariser:
    """Chunked KoBART summarisation with lazy pipeline initialisation."""

    def __init__(self, config: SummariserConfig | None = None) -> None:
        self.config = config or SummariserConfig()
        self._pipelines: Dict[str, Any] = {}
        self._device = self._resolve_device()

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("transformers") is not None

    def _resolve_device(self) -> int:
        try:
            override = os.getenv("MEETING_SUMMARY_DEVICE")
            if override:
                override = override.strip().lower()
                if override in {"cpu", "-1"}:
                    return -1
                if override in {"cuda", "gpu", "0"}:
                    return 0
                if override.isdigit():
                    return int(override)
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except ImportError:  # pragma: no cover - torch optional
                return -1
        except Exception:  # pragma: no cover - defensive fallback
            return -1

    def _ensure_pipeline(self, flavour: str = "ko"):
        if flavour in self._pipelines:
            return self._pipelines[flavour]

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for KoBART summarisation") from exc

        if flavour == "en":
            model_name = self.config.english_model_name
        else:
            model_name = self.config.model_name

        device_label = self._device if isinstance(self._device, int) and self._device >= 0 else "cpu"
        LOGGER.info(
            "Loading %s summariser model: %s on device=%s",  # noqa: G004
            flavour.upper(),
            model_name,
            device_label,
        )
        task = "text2text-generation"
        self._pipelines[flavour] = pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            device=self._device,
        )
        return self._pipelines[flavour]

    def summarise(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        flavour = "en" if self._is_likely_english(text) else "ko"
        chunks = self._chunk_text(text, self.config.chunk_char_limit)
        summarise_chunk = self._make_chunk_summariser(flavour)
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
    def _make_chunk_summariser(self, flavour: str) -> Callable[[str], str]:
        transformer_pipeline = self._ensure_pipeline(flavour)
        max_length = self.config.max_length
        min_length = self.config.min_length
        max_new_tokens = self.config.max_new_tokens
        num_beams = self.config.num_beams
        max_input_chars = self.config.model_max_input_chars

        def _summarise_chunk(chunk: str) -> str:
            try:
                trimmed = chunk[:max_input_chars] if max_input_chars > 0 else chunk
                result: List[dict] = transformer_pipeline(
                    trimmed,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=num_beams,
                )
                if not result:
                    return ""
                payload = result[0].get("generated_text") or result[0].get("summary_text") or ""
                return payload.strip()
            except Exception as exc:  # pragma: no cover - inference guard
                LOGGER.exception("KoBART summarisation failed: %s", exc)
                return ""

        return _summarise_chunk

    def _is_likely_english(self, text: str) -> bool:
        # Quick heuristic based on Hangul vs ASCII letter ratio.
        if not text:
            return False

        hangul = sum(1 for ch in text if "가" <= ch <= "힣")
        latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        # Treat as English when Hangul is rare and there is a reasonable amount of Latin chars.
        return hangul == 0 and latin > 0 or (latin > 5 and hangul * 5 < latin)

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


class OllamaSummariser:
    """Summariser that delegates to a locally running Ollama server/CLI."""

    def __init__(self, config: SummariserConfig | None = None) -> None:
        self.config = config or SummariserConfig()
        self._ollama_cmd = shutil.which("ollama")

    @staticmethod
    def is_available() -> bool:
        return shutil.which("ollama") is not None

    def summarise(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if not self._ollama_cmd:
            raise RuntimeError("ollama executable not detected")

        prompt_template = self.config.ollama_prompt or DEFAULT_OLLAMA_PROMPT
        prompt = prompt_template.format(transcript=text)
        env = os.environ.copy()
        if self.config.ollama_host:
            env["OLLAMA_HOST"] = self.config.ollama_host

        try:
            result = subprocess.run(
                ["ollama", "run", self.config.ollama_model],
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise RuntimeError("ollama executable not found in PATH") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            error_text = stderr or stdout or "unknown error"
            raise RuntimeError(f"ollama run failed ({result.returncode}): {error_text}")

        return (result.stdout or "").strip()


class BitNetSummariser:
    """Summariser for BitNet / low-bit quantised models via HuggingFace pipeline."""

    def __init__(self, config: SummariserConfig | None = None) -> None:
        self.config = config or SummariserConfig()
        self._pipeline = None

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("transformers") is not None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for BitNet summarisation") from exc

        LOGGER.info("Loading BitNet summariser model: %s", self.config.bitnet_model)
        self._pipeline = pipeline(
            "summarization",
            model=self.config.bitnet_model,
            tokenizer=self.config.bitnet_model,
        )
        return self._pipeline

    def summarise(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        pipeline = self._ensure_pipeline()
        try:
            result: List[dict] = pipeline(
                text,
                max_length=self.config.bitnet_max_length,
                min_length=self.config.bitnet_min_length,
                do_sample=False,
            )
        except Exception as exc:  # pragma: no cover - inference guard
            LOGGER.exception("BitNet summarisation failed: %s", exc)
            return ""

        if not result:
            return ""
        return (result[0].get("summary_text") or "").strip()


_SUMMARY_BACKEND_ALIASES = {
    "kobart": "kobart",
    "kobart_chunk": "kobart",
    "ollama": "ollama",
    "bitnet": "bitnet",
}
_SUMMARY_BACKENDS = {
    "kobart": KoBARTSummariser,
    "ollama": OllamaSummariser,
    "bitnet": BitNetSummariser,
}


def create_summary_backend(name: str | None, config: SummariserConfig | None = None) -> Optional[BaseSummariser]:
    """Instantiate the configured summary backend, returning ``None`` for heuristics."""

    if not name:
        return None

    key = _SUMMARY_BACKEND_ALIASES.get(name.lower().strip(), name.lower().strip())
    if key in {"heuristic", "none", "placeholder"}:
        return None
    backend_cls = _SUMMARY_BACKENDS.get(key)
    if backend_cls is None:
        LOGGER.warning("Unknown summary backend '%s'; using heuristic summary", name)
        return None

    try:
        return backend_cls(config)
    except RuntimeError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("%s summariser unavailable: %s", key, exc)
        return None


def available_summary_backends() -> Dict[str, bool]:
    """Return availability information for the known summary backends."""

    availability: Dict[str, bool] = {"heuristic": True}
    for alias, key in _SUMMARY_BACKEND_ALIASES.items():
        backend_cls = _SUMMARY_BACKENDS.get(key)
        if backend_cls is None:
            availability[alias] = False
            continue
        try:
            availability[alias] = bool(getattr(backend_cls, "is_available")())
        except Exception:  # pragma: no cover - defensive
            availability[alias] = False
    return availability
