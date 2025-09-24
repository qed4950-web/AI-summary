"""Wrapper around faster-whisper to provide meeting STT."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from . import STTBackend, TranscriptionPayload

LOGGER = logging.getLogger(__name__)


class WhisperSTTBackend:
    """Lazy-initialised faster-whisper transcription backend."""

    def __init__(
        self,
        *,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.name = "whisper"
        self.model_size = model_size or os.getenv("MEETING_STT_MODEL", "small")
        self.device = device or os.getenv("MEETING_STT_DEVICE")
        self.compute_type = compute_type or os.getenv("MEETING_STT_COMPUTE", "int8")
        self.download_root = download_root or os.getenv("MEETING_STT_MODEL_DIR")
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel  # type: ignore

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        if self.compute_type:
            kwargs["compute_type"] = self.compute_type
        if self.download_root:
            kwargs["download_root"] = self.download_root

        LOGGER.info(
            "Loading faster-whisper model: size=%s device=%s compute=%s",  # noqa: G004
            self.model_size,
            kwargs.get("device", "auto"),
            kwargs.get("compute_type", "auto"),
        )
        self._model = WhisperModel(self.model_size, **kwargs)
        return self._model

    def _default_speaker(self, index: int) -> str:
        # Diarisation may be disabled or unsupported; fall back to speaker index.
        return f"speaker_{index + 1}"

    # ------------------------------------------------------------------
    # STTBackend API
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        diarize: bool = False,
        speaker_count: Optional[int] = None,
    ) -> TranscriptionPayload:
        model = self._ensure_model()

        kwargs = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        if language:
            kwargs["language"] = language
        if diarize:
            kwargs["diarize"] = True
            if speaker_count:
                kwargs["speaker_count"] = speaker_count

        segments_iter, info = model.transcribe(str(audio_path), **kwargs)

        segments = []
        text_chunks = []
        speaker_cycle = max(speaker_count or 0, 1)
        for idx, segment in enumerate(segments_iter):
            speaker = getattr(segment, "speaker", None)
            if not speaker:
                speaker = self._default_speaker(idx % speaker_cycle)
            chunk_text = (segment.text or "").strip()
            if chunk_text:
                text_chunks.append(chunk_text)
            segments.append(
                {
                    "start": round(float(getattr(segment, "start", 0.0)), 2),
                    "end": round(float(getattr(segment, "end", 0.0)), 2),
                    "speaker": speaker,
                    "text": chunk_text,
                }
            )

        joined_text = " ".join(text_chunks).strip()
        duration = getattr(info, "duration", None)
        detected_language = getattr(info, "language", None)

        return TranscriptionPayload(
            text=joined_text,
            segments=segments,
            duration_seconds=duration,
            language=detected_language or language,
        )
