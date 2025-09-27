"""Pipeline orchestrator for meeting transcription and summarisation."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from infopilot_core.utils import get_logger

from .models import (
    MeetingJobConfig,
    MeetingSummary,
    MeetingTranscriptionResult,
    StreamingSummarySnapshot,
)
from .stt import TranscriptionPayload, create_stt_backend
from .summarizer import SummariserConfig, available_summary_backends, create_summary_backend

LOGGER = get_logger("meeting.pipeline")

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

LANGUAGE_ALIASES = {
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    "ko-kr": "ko",
    "kr": "ko",
    "en": "en",
    "eng": "en",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "cmn": "zh",
    "chi": "zh",
    "mandarin": "zh",
}

DEFAULT_LANGUAGE = "ko"

ACTION_KEYWORDS = {
    "default": ["action", "todo", "follow", "follow-up"],
    "ko": ["action", "todo", "follow", "해야", "요청", "담당", "액션", "아이템", "후속"],
    "en": ["action", "todo", "follow", "follow-up", "owner", "next step"],
    "ja": ["対応", "タスク", "宿題", "確認", "引き続き"],
    "zh": ["行动", "待办", "跟进", "负责人", "任务"],
}

DECISION_KEYWORDS = {
    "default": ["decision", "decide", "approved", "agreed"],
    "ko": ["결정", "승인", "확정", "정리", "합의"],
    "en": ["decision", "approved", "agreed", "final"],
    "ja": ["決定", "合意", "承認", "確定"],
    "zh": ["决定", "批准", "确认", "定案"],
}

HIGHLIGHT_FALLBACK = {
    "ko": "회의 주요 내용을 식별하지 못했습니다.",
    "en": "No key highlights detected.",
    "ja": "主要なハイライトを検出できませんでした。",
    "zh": "未检测到关键要点。",
}

GENERIC_FALLBACK = {
    "ko": "관련 항목이 발견되지 않았습니다.",
    "en": "No related items found.",
    "ja": "該当する項目が見つかりませんでした。",
    "zh": "未找到相关条目。",
}

PII_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")

AVERAGE_SPEECH_WPM = 130

QUESTION_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "will",
    "can",
    "should",
    "could",
    "would",
    "please",
    "누가",
    "무엇",
    "언제",
    "어디",
    "왜",
    "어떻게",
    "무슨",
    "어느",
    "가능",
}

try:  # Optional dependency for spacing correction
    from pykospacing import Spacing  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Spacing = None  # type: ignore

try:  # Optional dependency for spell checking
    from hanspell import spell_checker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spell_checker = None  # type: ignore


class MeetingPipeline:
    """Meeting agent MVP pipeline.

    The implementation follows the assistant roadmap guidelines:
    - Load or transcribe audio into text (fallback to sidecar transcripts for MVP)
    - Split the transcript into diarisation-friendly segments
    - Generate highlights, action items, and decisions using lightweight heuristics
    - Persist artefacts so downstream smart folders and the 작업 센터 can ingest them
    """

    def __init__(
        self,
        *,
        stt_backend: Optional[str] = None,
        summary_backend: Optional[str] = None,
        stt_options: Optional[dict] = None,
    ) -> None:
        backend_env = os.getenv("MEETING_STT_BACKEND")
        requested_backend = stt_backend if stt_backend not in {None, ""} else backend_env
        self.stt_backend = self._resolve_stt_backend(requested_backend)

        summary_env = os.getenv("MEETING_SUMMARY_BACKEND")
        summary_backend_name = summary_backend if summary_backend not in {None, ""} else summary_env
        summary_backend_name = (summary_backend_name or "kobart").lower()

        self.summary_backend = summary_backend_name
        stt_opts = dict(stt_options or {})
        self._resource_info = _resource_diagnostics()
        if self.stt_backend == "whisper" and "device" not in stt_opts:
            if not self._resource_info.get("gpu_available"):
                stt_opts["device"] = "cpu"
        self._stt = create_stt_backend(self.stt_backend, **stt_opts)
        if self._stt is None and self.stt_backend not in {"placeholder", "none", "noop"}:
            LOGGER.warning("requested STT backend '%s' unavailable; proceeding without STT", self.stt_backend)

        # Lazy initialisation of post-processing helpers
        self._spacing_model = None
        save_transcript_env = os.getenv("MEETING_SAVE_TRANSCRIPT", "0").strip().lower()
        self._save_transcript = save_transcript_env not in {"", "0", "false", "no"}

        self._summary_config = SummariserConfig()
        self._summariser = create_summary_backend(self.summary_backend, self._summary_config)
        if self._summariser is None and self.summary_backend not in {"heuristic", "none", "placeholder"}:
            LOGGER.warning("summary backend '%s' unavailable; using heuristic summary", self.summary_backend)
            self.summary_backend = "heuristic"

        cache_env = os.getenv("MEETING_CACHE", "1").strip().lower()
        self._cache_enabled = cache_env not in {"", "0", "false", "no"}

        pii_env = os.getenv("MEETING_MASK_PII", "0").strip().lower()
        self._mask_pii_enabled = pii_env not in {"", "0", "false", "no"}

        chunk_env = os.getenv("MEETING_STT_CHUNK_SECONDS", "0").strip()
        self._chunk_seconds = self._coerce_positive_float(chunk_env, default=0.0)

    def start_streaming(
        self,
        job: MeetingJobConfig,
        *,
        update_interval: float = 60.0,
    ) -> "StreamingMeetingSession":
        return StreamingMeetingSession(self, job, update_interval=update_interval)

    def run(self, job: MeetingJobConfig) -> MeetingSummary:
        LOGGER.info(
            "meeting pipeline start: audio=%s backend=%s policy=%s",
            job.audio_path,
            self.stt_backend,
            job.policy_tag,
        )
        cached_summary = self._load_cache(job)
        if cached_summary is not None:
            LOGGER.info(
                "meeting pipeline cache hit: audio=%s summary_backend=%s",
                job.audio_path,
                self.summary_backend,
            )
            return cached_summary
        transcript = self._transcribe(job)
        summary = self._summarise(job, transcript)
        if self._mask_pii_enabled:
            self._mask_sensitive_content(transcription=transcript, summary=summary)
        self._persist(job, transcript, summary)
        LOGGER.info("meeting pipeline finished: saved=%s", summary.transcript_path.parent)
        return summary

    # ---------------------------------------------------------------------
    # Stage 1: Speech-to-text or transcript loading
    # ---------------------------------------------------------------------
    def _transcribe(self, job: MeetingJobConfig) -> MeetingTranscriptionResult:
        text = self._load_transcript_text(job.audio_path)
        if text is not None:
            duration = self._estimate_duration(job.audio_path, text)
            segments = self._segment_transcript(text, job, duration)
            language = self._detect_language(text, job.language)
        else:
            payload = self._invoke_stt_backend(job)
            text = payload.text
            duration = payload.duration_seconds or self._estimate_duration(job.audio_path, text)
            segments = payload.segments or self._segment_transcript(text, job, duration)
            language = self._detect_language(text, payload.language, job.language)

        normalised_segments = self._normalise_segments(segments, job)

        return MeetingTranscriptionResult(
            text=text,
            segments=normalised_segments,
            duration_seconds=duration,
            language=language,
        )

    def _detect_language(self, text: str, *hints: Optional[str]) -> str:
        for hint in hints:
            language = self._map_language_code(hint)
            if language:
                return language

        sample = (text or "").strip()[:500]
        if any("\uac00" <= char <= "\ud7a3" for char in sample):
            return "ko"
        if re.search("[ぁ-んァ-ン]", sample):
            return "ja"
        if re.search("[\u4e00-\u9fff]", sample):
            return "zh"
        return "en"

    def _map_language_code(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        code = value.lower().strip()
        mapped = LANGUAGE_ALIASES.get(code)
        if mapped:
            return mapped
        if code and code.split("-")[0] in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[code.split("-")[0]]
        return None

    @staticmethod
    def _coerce_positive_float(value: str, *, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _load_transcript_text(self, audio_path: Path) -> Optional[str]:
        # Sidecar transcript: <audio>.<ext>.txt or <audio>.txt
        for candidate in self._candidate_transcript_paths(audio_path):
            if candidate.exists():
                LOGGER.debug("loading sidecar transcript: %s", candidate)
                return candidate.read_text(encoding="utf-8").strip()

        if audio_path.suffix.lower() in {".txt", ".md"}:
            LOGGER.debug("treating %s as text transcript", audio_path)
            return audio_path.read_text(encoding="utf-8").strip()

        LOGGER.debug("no sidecar transcript detected for %s", audio_path)
        return None

    def _candidate_transcript_paths(self, audio_path: Path) -> Iterable[Path]:
        yield audio_path.with_suffix(audio_path.suffix + ".txt")
        yield audio_path.with_suffix(".txt")

    def _estimate_duration(self, audio_path: Path, transcript: str) -> float:
        try:
            import soundfile as sf  # type: ignore

            with sf.SoundFile(audio_path) as audio:
                return len(audio) / audio.samplerate
        except Exception:
            LOGGER.debug("soundfile not available for %s; estimating duration", audio_path)

        return self._estimate_text_duration(transcript)

    def _estimate_text_duration(self, transcript: str) -> float:
        words = max(len((transcript or "").split()), 1)
        minutes = words / AVERAGE_SPEECH_WPM
        return round(minutes * 60, 2)

    def _segment_transcript(
        self,
        transcript: str,
        job: MeetingJobConfig,
        duration_seconds: float,
    ) -> List[dict]:
        sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(transcript) if s.strip()]
        if not sentences:
            sentences = [transcript.strip() or "(empty transcript)"]

        segment_count = len(sentences)
        slice_duration = duration_seconds / segment_count if segment_count else 0.0
        segments: List[dict] = []
        cursor = 0.0
        for index, sentence in enumerate(sentences):
            start = round(cursor, 2)
            if index == segment_count - 1:
                end = duration_seconds
            else:
                end = round(cursor + slice_duration, 2)
            cursor = end
            segments.append(
                {
                    "start": start,
                    "end": max(end, start),
                    "speaker": f"speaker_{(index % (job.speaker_count or 1)) + 1}",
                    "text": sentence,
                }
            )
        return segments

    def _normalise_segments(
        self,
        segments: Optional[Sequence[dict]],
        job: MeetingJobConfig,
    ) -> List[dict]:
        if not segments:
            return []

        speaker_alias: Dict[str, str] = {}
        next_alias = 1
        normalised: List[dict] = []

        sorted_segments = sorted(
            segments,
            key=lambda item: (
                self._safe_time(item.get("start"), 0.0),
                self._safe_time(item.get("end"), 0.0),
            ),
        )

        fallback_cycle = job.speaker_count or 1

        for segment in sorted_segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue

            start = round(self._safe_time(segment.get("start"), 0.0), 2)
            end = round(self._safe_time(segment.get("end"), start), 2)
            if end < start:
                end = start

            raw_speaker = str(segment.get("speaker") or "").strip()
            if raw_speaker:
                speaker_label = speaker_alias.get(raw_speaker)
                if speaker_label is None:
                    speaker_label = f"speaker_{next_alias}"
                    speaker_alias[raw_speaker] = speaker_label
                    next_alias += 1
            else:
                cycle = fallback_cycle if fallback_cycle > 0 else max(len(speaker_alias), 1)
                index = (len(normalised) % cycle) + 1 if cycle else 1
                speaker_label = f"speaker_{index}"

            if normalised and normalised[-1]["speaker"] == speaker_label:
                normalised[-1]["text"] = f"{normalised[-1]['text']} {text}".strip()
                normalised[-1]["end"] = round(max(normalised[-1]["end"], end), 2)
            else:
                normalised.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker_label,
                        "text": text,
                    }
                )

        return normalised

    @staticmethod
    def _safe_time(value: Optional[float], default: float) -> float:
        try:
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _invoke_stt_backend(self, job: MeetingJobConfig) -> TranscriptionPayload:
        if self._stt is None:
            LOGGER.warning(
                "STT backend '%s' is not configured; using placeholder transcript",
                self.stt_backend,
            )
            return TranscriptionPayload(
                text="(transcription not available – STT backend disabled)",
                language=job.language,
            )

        try:
            payload = self._stt.transcribe(
                job.audio_path,
                language=job.language,
                diarize=job.diarize,
                speaker_count=job.speaker_count,
            )
            if not payload.text:
                raise ValueError("STT backend returned empty transcript")
            return self._postprocess_transcript(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("STT backend '%s' failed: %s", self.stt_backend, exc)
            if self._chunk_seconds > 0 and self._stt is not None:
                try:
                    chunk_payload = self._transcribe_in_chunks(job, language=job.language)
                    if chunk_payload.text:
                        LOGGER.info("chunked STT fallback succeeded for %s", job.audio_path)
                        return self._postprocess_transcript(chunk_payload)
                except Exception as chunk_exc:  # pragma: no cover - diagnostics
                    LOGGER.warning("chunked STT fallback failed: %s", chunk_exc)
            return TranscriptionPayload(
                text=f"(transcription failed: {exc})",
                language=job.language,
            )

    def _transcribe_in_chunks(
        self,
        job: MeetingJobConfig,
        *,
        language: Optional[str] = None,
    ) -> TranscriptionPayload:
        if self._chunk_seconds <= 0 or self._stt is None:
            raise RuntimeError("chunked transcription is disabled or STT backend missing")

        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise RuntimeError("soundfile is required for chunked STT") from exc

        segments: List[dict] = []
        texts: List[str] = []
        total_duration = 0.0
        detected_language = None

        with sf.SoundFile(job.audio_path) as audio:
            samplerate = audio.samplerate
            frames_per_chunk = int(self._chunk_seconds * samplerate)
            if frames_per_chunk <= 0:
                frames_per_chunk = int(600 * samplerate)

            chunk_index = 0
            while True:
                data = audio.read(frames_per_chunk)
                if data.size == 0:
                    break
                chunk_path = Path(tempfile.mkstemp(suffix=job.audio_path.suffix)[1])
                try:
                    sf.write(str(chunk_path), data, samplerate)
                    chunk_payload = self._stt.transcribe(
                        chunk_path,
                        language=language,
                        diarize=job.diarize,
                        speaker_count=job.speaker_count,
                    )
                finally:
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        LOGGER.debug("failed to remove temp chunk %s", chunk_path)

                chunk_duration = chunk_payload.duration_seconds
                if chunk_duration is None:
                    chunk_duration = len(data) / float(samplerate)

                offset = total_duration
                total_duration += chunk_duration

                if chunk_payload.language and not detected_language:
                    detected_language = chunk_payload.language

                if chunk_payload.text:
                    texts.append(chunk_payload.text.strip())

                chunk_segments = chunk_payload.segments or []
                if chunk_segments:
                    for segment in chunk_segments:
                        segment_text = (segment.get("text") or "").strip()
                        if not segment_text:
                            continue
                        start = self._safe_time(segment.get("start"), 0.0) + offset
                        end = self._safe_time(segment.get("end"), 0.0) + offset
                        segments.append(
                            {
                                "start": round(start, 2),
                                "end": round(max(end, start), 2),
                                "speaker": segment.get("speaker") or f"speaker_{(len(segments) % (job.speaker_count or 1)) + 1}",
                                "text": segment_text,
                            }
                        )
                elif chunk_payload.text:
                    segments.append(
                        {
                            "start": round(offset, 2),
                            "end": round(offset + chunk_duration, 2),
                            "speaker": f"speaker_{(chunk_index % (job.speaker_count or 1)) + 1}",
                            "text": chunk_payload.text.strip(),
                        }
                    )
                chunk_index += 1

        combined_text = " ".join(texts).strip()
        return TranscriptionPayload(
            text=combined_text,
            segments=segments,
            duration_seconds=total_duration,
            language=detected_language or language,
        )

    # ---------------------------------------------------------------------
    # Stage 2: Summary/action extraction
    # ---------------------------------------------------------------------
    def _summarise(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
    ) -> MeetingSummary:
        language = self._map_language_code(transcription.language) or self._map_language_code(job.language) or DEFAULT_LANGUAGE
        highlight_entries = self._extract_highlights(transcription.segments, language)
        action_entries = self._extract_action_items(transcription.segments, language)
        decision_entries = self._extract_decisions(transcription.segments, language)

        model_summary = ""
        if self._summariser is not None:
            try:
                model_summary = self._summariser.summarise(transcription.text)
            except Exception as exc:  # pragma: no cover - inference guard
                LOGGER.warning(
                    "%s summariser failed; falling back to heuristic summary: %s",
                    self.summary_backend,
                    exc,
                )
                self._summariser = None
                self.summary_backend = "heuristic"

        if model_summary:
            raw_summary = model_summary
        else:
            raw_summary = self._build_summary_text(highlight_entries, action_entries, decision_entries)

        structured_summary = {
            "highlights": [entry for entry in highlight_entries],
            "action_items": [entry for entry in action_entries],
            "decisions": [entry for entry in decision_entries],
        }

        highlights = [entry.get("text", "") for entry in highlight_entries]
        action_items = [entry.get("text", "") for entry in action_entries]
        decisions = [entry.get("text", "") for entry in decision_entries]

        transcript_path = job.output_dir / "transcript.txt"
        return MeetingSummary(
            highlights=highlights,
            action_items=action_items,
            decisions=decisions,
            raw_summary=raw_summary,
            transcript_path=transcript_path,
            structured_summary=structured_summary,
        )

    def _resolve_stt_backend(self, requested: Optional[str]) -> str:
        value = (requested or "").strip()
        if not value or value.lower() == "auto":
            return self._auto_select_stt_backend()
        return value.lower()

    def _auto_select_stt_backend(self) -> str:
        if self._whisper_available():
            LOGGER.info("Whisper backend detected; defaulting to 'whisper'")
            return "whisper"
        LOGGER.warning(
            "No STT backend configured or available; falling back to placeholder transcripts",
        )
        return "placeholder"

    @staticmethod
    def _whisper_available() -> bool:
        try:
            return importlib.util.find_spec("faster_whisper") is not None
        except Exception:  # pragma: no cover - defensive fallback
            return False

    def _extract_highlights(self, segments: Sequence[dict], language: str) -> List[dict]:
        entries: List[dict] = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            entries.append(
                {
                    "text": text,
                    "ref": self._format_timestamp(segment.get("start")),
                }
            )
            if len(entries) >= 3:
                break

        if not entries:
            return [{"text": self._fallback_message(language, HIGHLIGHT_FALLBACK)}]
        return entries

    def _extract_action_items(self, segments: Sequence[dict], language: str) -> List[dict]:
        keywords = self._keywords_for(language, ACTION_KEYWORDS)
        return self._collect_by_keywords(segments, keywords, language)

    def _extract_decisions(self, segments: Sequence[dict], language: str) -> List[dict]:
        keywords = self._keywords_for(language, DECISION_KEYWORDS)
        return self._collect_by_keywords(segments, keywords, language)

    def _collect_by_keywords(
        self,
        segments: Sequence[dict],
        keywords: Sequence[str],
        language: str,
    ) -> List[dict]:
        collected: List[dict] = []
        lowered_keywords = [kw.lower() for kw in keywords]
        for segment in segments:
            raw_text = segment.get("text")
            text = (raw_text or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if any(keyword in lowered for keyword in lowered_keywords):
                collected.append(
                    {
                        "text": text,
                        "ref": self._format_timestamp(segment.get("start")),
                    }
                )
        if not collected:
            fallback_segment = segments[0] if segments else {}
            fallback_text = (fallback_segment.get("text") or self._fallback_message(language, GENERIC_FALLBACK)).strip()
            entry = {"text": fallback_text}
            ref = self._format_timestamp(fallback_segment.get("start")) if fallback_segment else None
            if ref:
                entry["ref"] = ref
            collected.append(entry)
        return collected

    def _build_summary_text(
        self,
        highlights: Sequence[dict],
        action_items: Sequence[dict],
        decisions: Sequence[dict],
    ) -> str:
        def _join(entries: Sequence[dict]) -> str:
            if not entries:
                return "- (내용 없음)"
            return "- " + "\n- ".join(entry.get("text", "") for entry in entries)

        sections = [
            "요약:",
            _join(highlights),
            "",
            "액션 아이템:",
            _join(action_items),
            "",
            "결정 사항:",
            _join(decisions),
        ]
        return "\n".join(sections)

    # ---------------------------------------------------------------------
    # Stage 3: Persistence
    # ---------------------------------------------------------------------
    def _persist(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        summary.transcript_path.write_text(transcription.text, encoding="utf-8")

        participants = self._extract_participants(transcription.segments)
        cache_info = {
            "version": 1,
            "audio_fingerprint": self._audio_fingerprint(job.audio_path),
            "stt_backend": self.stt_backend,
            "summary_backend": self.summary_backend,
            "options": {
                "diarize": job.diarize,
                "speaker_count": job.speaker_count,
            },
        }
        summary_payload = {
            "meeting_meta": {
                "title": job.audio_path.stem or "meeting",
                "date": job.created_at.date().isoformat(),
                "participants": participants,
            },
            "summary": {
                "highlights": summary.structured_summary.get("highlights", []),
                "action_items": summary.structured_summary.get("action_items", []),
                "decisions": summary.structured_summary.get("decisions", []),
                "raw_summary": summary.raw_summary,
            },
            "duration_seconds": transcription.duration_seconds,
            "language": transcription.language,
            "policy_tag": job.policy_tag,
            "generated_by": {
                "stt_backend": self.stt_backend,
                "summary_backend": self.summary_backend,
            },
            "raw_summary": summary.raw_summary,
            "cache": cache_info,
            "pii_masked": self._mask_pii_enabled,
            "quality_metrics": self._compute_quality_metrics(transcription, summary),
        }
        feedback_info = self._queue_feedback_request(job, summary)
        if feedback_info:
            summary_payload["feedback"] = feedback_info
            summary_payload.setdefault("attachments", {})["feedback_queue"] = feedback_info.get("queue")
        summary_path = job.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        segments_path = job.output_dir / "segments.json"
        segments_path.write_text(
            json.dumps(transcription.segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata = {
            "audio_path": str(job.audio_path),
            "output_dir": str(job.output_dir),
            "language": transcription.language,
            "duration_seconds": transcription.duration_seconds,
            "policy_tag": job.policy_tag,
            "created_at": job.created_at.isoformat(),
        }
        metadata["cache"] = cache_info
        metadata["quality_metrics"] = summary_payload["quality_metrics"]
        metadata["pii_masked"] = self._mask_pii_enabled
        if feedback_info:
            metadata["feedback"] = feedback_info
        metadata_path = job.output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self._save_transcript:
            transcript_file = job.output_dir / "transcript.json"
            transcript_entries = [
                {
                    "speaker": segment.get("speaker"),
                    "start": self._format_timestamp(segment.get("start")),
                    "end": self._format_timestamp(segment.get("end")),
                    "text": segment.get("text"),
                }
                for segment in transcription.segments
            ]
            transcript_file.write_text(
                json.dumps(transcript_entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary_payload.setdefault("attachments", {})["transcript"] = transcript_file.name
            summary_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        LOGGER.info(
            "meeting artefacts saved: transcript=%s summary=%s segments=%s metadata=%s",
            summary.transcript_path,
            summary_path,
            segments_path,
            metadata_path,
        )

        self._record_for_search(job, transcription, summary, summary_payload["quality_metrics"])
        self._export_integrations(job, transcription, summary)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    def _postprocess_transcript(self, payload: TranscriptionPayload) -> TranscriptionPayload:
        text = payload.text
        if not text:
            return payload

        original = text
        text = self._apply_spacing(text)
        text = self._apply_spell_check(text)

        if text != original:
            payload.text = text
            if payload.segments:
                payload.segments = [
                    {**segment, "text": self._apply_spell_check(self._apply_spacing(segment.get("text", "")))}
                    for segment in payload.segments
                ]

        return payload

    def _apply_spacing(self, text: str) -> str:
        if not text or Spacing is None:
            return text

        try:
            if self._spacing_model is None:
                self._spacing_model = Spacing()
            return self._spacing_model(text)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Spacing correction failed: %s", exc)
            return text

    def _apply_spell_check(self, text: str) -> str:
        if not text or spell_checker is None:
            return text

        try:
            result = spell_checker.check(text)
            corrected = getattr(result, "checked", None)
            return corrected if corrected else text
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Spell check failed: %s", exc)
            return text

    def _format_timestamp(self, seconds: Optional[float]) -> Optional[str]:
        if seconds is None:
            return None
        try:
            total_seconds = max(int(round(seconds)), 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _extract_participants(self, segments: Sequence[dict]) -> List[str]:
        speakers: List[str] = []
        seen = set()
        for segment in segments:
            speaker = segment.get("speaker")
            if not speaker or speaker in seen:
                continue
            seen.add(speaker)
            speakers.append(speaker)
        return speakers

    def _keywords_for(self, language: str, mapping: Dict[str, Sequence[str]]) -> List[str]:
        lang = language if language in mapping else DEFAULT_LANGUAGE
        combined = list(mapping.get("default", []))
        combined.extend(mapping.get(lang, []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: List[str] = []
        for keyword in combined:
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(keyword)
        return result

    def _fallback_message(self, language: str, mapping: Dict[str, str]) -> str:
        return mapping.get(language) or mapping.get(DEFAULT_LANGUAGE) or ""

    def _compute_quality_metrics(
        self,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> Dict[str, float | int | str]:
        transcript_chars = len(transcription.text or "")
        summary_chars = len(summary.raw_summary or "")
        compression = summary_chars / transcript_chars if transcript_chars else 0.0
        rouge_scores = self._compute_rouge_metrics(transcription.text, summary.raw_summary)
        lfqa_scores = self._estimate_lfqa_metrics(transcription.text, summary.raw_summary)

        metrics: Dict[str, float | int | str] = {
            "transcript_chars": transcript_chars,
            "summary_chars": summary_chars,
            "compression_ratio": round(compression, 4) if compression else 0.0,
            "highlight_count": len(summary.highlights),
            "action_count": len(summary.action_items),
            "decision_count": len(summary.decisions),
        }
        metrics.update(rouge_scores)
        metrics.update(lfqa_scores)
        return metrics

    def _record_for_search(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
        quality_metrics: Dict[str, float | int | str],
    ) -> None:
        index_env = os.getenv("MEETING_VECTOR_INDEX")
        if index_env:
            index_path = Path(index_env)
        else:
            index_path = job.output_dir.parent / "meeting_vector_index.jsonl"

        entry = {
            "meeting_id": job.audio_path.stem,
            "created_at": job.created_at.isoformat(),
            "language": transcription.language,
            "summary": summary.raw_summary,
            "highlights": summary.structured_summary.get("highlights", []),
            "action_items": summary.structured_summary.get("action_items", []),
            "decisions": summary.structured_summary.get("decisions", []),
            "quality_metrics": quality_metrics,
            "source": str(job.audio_path),
            "output_dir": str(job.output_dir),
        }

        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - diagnostic only
            LOGGER.debug("failed to record meeting entry for search: %s", exc)

    def _mask_sensitive_content(
        self,
        *,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        transcription.text = self._mask_text(transcription.text)
        for segment in transcription.segments:
            segment["text"] = self._mask_text(segment.get("text"))
        summary.raw_summary = self._mask_text(summary.raw_summary)
        summary.highlights = [self._mask_text(text) for text in summary.highlights]
        summary.action_items = [self._mask_text(text) for text in summary.action_items]
        summary.decisions = [self._mask_text(text) for text in summary.decisions]
        for section in summary.structured_summary.values():
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict) and "text" in item:
                        item["text"] = self._mask_text(item.get("text"))

    def _mask_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        masked = PII_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        masked = PII_PHONE_RE.sub("[REDACTED_PHONE]", masked)
        return masked

    def _export_integrations(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        attachments: Dict[str, str] = {}

        tasks_path = job.output_dir / "tasks.json"
        tasks_payload = [
            {
                "title": item.get("text", ""),
                "reference": item.get("ref"),
                "status": "pending",
                "source_meeting": job.audio_path.stem,
            }
            for item in summary.structured_summary.get("action_items", [])
            if isinstance(item, dict)
        ]
        tasks_path.write_text(
            json.dumps(tasks_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        attachments["tasks"] = tasks_path.name

        calendar_path = job.output_dir / "meeting.ics"
        calendar_path.write_text(
            self._build_calendar_event(job, transcription, summary),
            encoding="utf-8",
        )
        attachments["calendar"] = calendar_path.name

        integrations_path = job.output_dir / "integrations.json"
        integrations_payload = {
            "meeting_id": job.audio_path.stem,
            "generated_at": job.created_at.isoformat(),
            "tasks_file": tasks_path.name,
            "calendar_file": calendar_path.name,
            "action_items": summary.structured_summary.get("action_items", []),
            "decisions": summary.structured_summary.get("decisions", []),
        }
        integrations_path.write_text(
            json.dumps(integrations_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        attachments["integrations"] = integrations_path.name

        summary_json = job.output_dir / "summary.json"
        if summary_json.exists():
            try:
                payload = json.loads(summary_json.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            payload.setdefault("attachments", {}).update(attachments)
            summary_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _build_calendar_event(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> str:
        start = job.created_at
        duration = transcription.duration_seconds or 0.0
        if duration <= 0:
            duration = 3600.0
        end = start + timedelta(seconds=duration)
        dtstamp = start.strftime("%Y%m%dT%H%M%SZ")
        dtstart = dtstamp
        dtend = end.strftime("%Y%m%dT%H%M%SZ")
        description = summary.raw_summary.replace("\n", "\\n")
        uid = f"{job.audio_path.stem}-{int(start.timestamp())}@infopilot.local"

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//InfoPilot//Meeting Agent//EN",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART:{dtstart}",
            f"DTEND:{dtend}",
            f"SUMMARY:{job.audio_path.stem or 'Meeting'}",
            f"DESCRIPTION:{description}",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
        return "\r\n".join(lines) + "\r\n"

    def _queue_feedback_request(
        self,
        job: MeetingJobConfig,
        summary: MeetingSummary,
    ) -> Optional[Dict[str, object]]:
        feedback_entry = {
            "meeting_id": job.audio_path.stem,
            "created_at": job.created_at.isoformat(),
            "summary_backend": self.summary_backend,
            "status": "pending",
            "highlights": summary.highlights,
            "action_items": summary.structured_summary.get("action_items", []),
            "decisions": summary.structured_summary.get("decisions", []),
        }

        local_queue = job.output_dir / "feedback_queue.jsonl"
        self._append_jsonl(local_queue, feedback_entry)

        global_queue_env = os.getenv("MEETING_FEEDBACK_INBOX")
        global_path: Optional[Path] = None
        if global_queue_env:
            global_path = Path(global_queue_env)
            self._append_jsonl(global_path, feedback_entry)

        return {
            "queue": local_queue.name,
            "status": "pending",
            "global_queue": str(global_path) if global_path else None,
        }


    # ------------------------------------------------------------------
    # Quality & feedback helpers
    # ------------------------------------------------------------------

    def _compute_rouge_metrics(self, reference: Optional[str], summary: Optional[str]) -> Dict[str, float]:
        reference_tokens = self._tokenize_for_metrics(reference)
        summary_tokens = self._tokenize_for_metrics(summary)
        if not reference_tokens or not summary_tokens:
            return {
                "rouge1_precision": 0.0,
                "rouge1_recall": 0.0,
                "rouge1_f": 0.0,
                "rougeL_precision": 0.0,
                "rougeL_recall": 0.0,
                "rougeL_f": 0.0,
            }

        rouge1 = self._rouge_n(reference_tokens, summary_tokens, n=1)
        rouge_l = self._rouge_l(reference_tokens, summary_tokens)

        return {
            "rouge1_precision": round(rouge1[0], 4),
            "rouge1_recall": round(rouge1[1], 4),
            "rouge1_f": round(rouge1[2], 4),
            "rougeL_precision": round(rouge_l[0], 4),
            "rougeL_recall": round(rouge_l[1], 4),
            "rougeL_f": round(rouge_l[2], 4),
        }

    def _estimate_lfqa_metrics(self, transcript: Optional[str], summary: Optional[str]) -> Dict[str, float | int]:
        questions = self._extract_question_keywords(transcript)
        if not questions:
            return {
                "lfqa_question_count": 0,
                "lfqa_coverage": 1.0,
            }

        summary_tokens = set(self._tokenize_for_metrics(summary))
        covered = 0
        for keywords in questions:
            if not keywords:
                covered += 1
                continue
            if any(token in summary_tokens for token in keywords):
                covered += 1

        coverage = covered / len(questions) if questions else 0.0
        return {
            "lfqa_question_count": len(questions),
            "lfqa_coverage": round(coverage, 4),
        }

    def _extract_question_keywords(self, text: Optional[str]) -> List[List[str]]:
        if not text:
            return []
        raw_questions = re.findall(r"[^?\n]+\?", text)
        questions: List[List[str]] = []
        for question in raw_questions:
            tokens = [token for token in self._tokenize_for_metrics(question) if token not in QUESTION_STOP_WORDS]
            questions.append(tokens)
        return questions

    def _tokenize_for_metrics(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r"[\w']+", text.lower())

    def _rouge_n(
        self,
        reference: Sequence[str],
        summary: Sequence[str],
        *,
        n: int,
    ) -> Tuple[float, float, float]:
        if n <= 0:
            return (0.0, 0.0, 0.0)

        def ngrams(tokens: Sequence[str]) -> Counter:
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        ref_counts = ngrams(reference)
        sum_counts = ngrams(summary)
        if not ref_counts or not sum_counts:
            return (0.0, 0.0, 0.0)

        overlap = sum((ref_counts & sum_counts).values())
        precision = overlap / max(sum(sum_counts.values()), 1)
        recall = overlap / max(sum(ref_counts.values()), 1)
        f_score = self._safe_f1(precision, recall)
        return (precision, recall, f_score)

    def _rouge_l(self, reference: Sequence[str], summary: Sequence[str]) -> Tuple[float, float, float]:
        lcs = self._lcs_length(reference, summary)
        if lcs == 0:
            return (0.0, 0.0, 0.0)
        precision = lcs / len(summary) if summary else 0.0
        recall = lcs / len(reference) if reference else 0.0
        f_score = self._safe_f1(precision, recall)
        return (precision, recall, f_score)

    def _lcs_length(self, reference: Sequence[str], summary: Sequence[str]) -> int:
        if not reference or not summary:
            return 0
        prev_row = [0] * (len(summary) + 1)
        for ref_token in reference:
            current = [0]
            for idx, sum_token in enumerate(summary, start=1):
                if ref_token == sum_token:
                    current.append(prev_row[idx - 1] + 1)
                else:
                    current.append(max(prev_row[idx], current[-1]))
            prev_row = current
        return prev_row[-1]

    @staticmethod
    def _safe_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("failed to append feedback entry to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _audio_fingerprint(self, audio_path: Path) -> Dict[str, int]:
        try:
            stat = audio_path.stat()
        except FileNotFoundError:
            return {}
        return {
            "size": stat.st_size,
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
        }

    def _matches_fingerprint(self, audio_path: Path, fingerprint: Dict[str, int]) -> bool:
        if not fingerprint:
            return False
        current = self._audio_fingerprint(audio_path)
        if not current:
            return False
        return (
            current.get("size") == fingerprint.get("size")
            and current.get("mtime_ns") == fingerprint.get("mtime_ns")
        )

    def _load_cache(self, job: MeetingJobConfig) -> Optional[MeetingSummary]:
        if not self._cache_enabled:
            return None

        summary_path = job.output_dir / "summary.json"
        segments_path = job.output_dir / "segments.json"
        metadata_path = job.output_dir / "metadata.json"
        transcript_path = job.output_dir / "transcript.txt"

        required = [summary_path, segments_path, metadata_path, transcript_path]
        if not all(path.exists() for path in required):
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.debug("failed to read meeting cache metadata: %s", exc)
            return None

        cache_info = metadata.get("cache") or {}
        if cache_info.get("version") != 1:
            return None
        if cache_info.get("stt_backend") != self.stt_backend:
            return None
        if cache_info.get("summary_backend") != self.summary_backend:
            return None

        options = cache_info.get("options", {})
        if bool(options.get("diarize")) != bool(job.diarize):
            return None
        if options.get("speaker_count") != job.speaker_count:
            return None

        if not self._matches_fingerprint(job.audio_path, cache_info.get("audio_fingerprint", {})):
            return None

        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.debug("failed to read cached summary payload: %s", exc)
            return None

        summary_section = summary_payload.get("summary", {})
        highlights = [entry.get("text", "") for entry in summary_section.get("highlights", [])]
        action_items = [entry.get("text", "") for entry in summary_section.get("action_items", [])]
        decisions = [entry.get("text", "") for entry in summary_section.get("decisions", [])]
        raw_summary = summary_section.get("raw_summary") or summary_payload.get("raw_summary", "")

        structured_summary = {
            "highlights": summary_section.get("highlights", []),
            "action_items": summary_section.get("action_items", []),
            "decisions": summary_section.get("decisions", []),
        }

        return MeetingSummary(
            highlights=highlights,
            action_items=action_items,
            decisions=decisions,
            raw_summary=raw_summary,
            transcript_path=transcript_path,
            structured_summary=structured_summary,
        )


def get_backend_diagnostics() -> Dict[str, Dict[str, bool]]:
    """Return availability information for STT and summary backends."""

    return {
        "stt": {
            "whisper": MeetingPipeline._whisper_available(),
        },
        "summary": available_summary_backends(),
        "resources": _resource_diagnostics(),
    }


def _resource_diagnostics() -> Dict[str, object]:
    info: Dict[str, object] = {
        "gpu_available": False,
    }
    try:
        import torch  # type: ignore

        info["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            try:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover - optional
                pass
    except Exception:
        info["gpu_available"] = False
    return info

class StreamingMeetingSession:
    """Stateful helper that supports streaming meeting transcription snapshots."""

    def __init__(
        self,
        pipeline: "MeetingPipeline",
        job: MeetingJobConfig,
        *,
        update_interval: float,
    ) -> None:
        self._pipeline = pipeline
        self._job = job
        self._update_interval = max(update_interval, 0.0)
        self._segments: List[dict] = []
        self._text_chunks: List[str] = []
        self._elapsed = 0.0
        self._since_snapshot = 0.0
        self._speaker_alias: Dict[str, str] = {}
        self._next_alias = 1
        self._final_summary: Optional[MeetingSummary] = None
        self._finalised = False

    def ingest(
        self,
        text: str,
        *,
        speaker: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Optional[StreamingSummarySnapshot]:
        if self._finalised:
            raise RuntimeError("streaming session already finalised")

        cleaned = (text or "").strip()
        if not cleaned:
            return None

        start_time, end_time = self._resolve_window(cleaned, start, end)
        speaker_label = self._normalise_speaker(speaker)

        segment = {
            "start": start_time,
            "end": end_time,
            "speaker": speaker_label,
            "text": cleaned,
        }
        self._segments.append(segment)
        self._text_chunks.append(cleaned)

        self._elapsed = max(self._elapsed, end_time)
        segment_duration = max(end_time - start_time, 0.0)
        self._since_snapshot += segment_duration

        if self._update_interval == 0 or self._since_snapshot >= self._update_interval:
            self._since_snapshot = 0.0
            return self.snapshot()
        return None

    def snapshot(self) -> StreamingSummarySnapshot:
        language = self._detect_language()
        highlights = self._pipeline._extract_highlights(self._segments, language)
        action_entries = self._pipeline._extract_action_items(self._segments, language)
        decision_entries = self._pipeline._extract_decisions(self._segments, language)
        summary_text = self._pipeline._build_summary_text(highlights, action_entries, decision_entries)

        return StreamingSummarySnapshot(
            summary_text=summary_text,
            highlights=[entry.get("text", "") for entry in highlights],
            action_items=[entry.get("text", "") for entry in action_entries],
            decisions=[entry.get("text", "") for entry in decision_entries],
            elapsed_seconds=self._elapsed,
            language=language,
        )

    def finalize(self) -> MeetingSummary:
        if self._final_summary is not None:
            return self._final_summary

        transcription = self._build_transcription_result()
        summary = self._pipeline._summarise(self._job, transcription)
        if self._pipeline._mask_pii_enabled:
            self._pipeline._mask_sensitive_content(transcription=transcription, summary=summary)
        self._pipeline._persist(self._job, transcription, summary)

        self._final_summary = summary
        self._finalised = True
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_window(
        self,
        text: str,
        start: Optional[float],
        end: Optional[float],
    ) -> Tuple[float, float]:
        if start is None:
            start_time = self._elapsed
        else:
            start_time = max(float(start), 0.0)

        if end is not None:
            end_time = max(float(end), start_time)
        else:
            duration = self._pipeline._estimate_text_duration(text)
            end_time = max(start_time + duration, start_time)

        return (round(start_time, 2), round(end_time, 2))

    def _normalise_speaker(self, speaker: Optional[str]) -> str:
        if not speaker:
            alias = self._speaker_alias.get("__default__")
            if alias:
                return alias
            alias = "speaker_1"
            self._speaker_alias["__default__"] = alias
            self._next_alias = max(self._next_alias, 2)
            return alias

        key = speaker.strip().lower()
        if key in self._speaker_alias:
            return self._speaker_alias[key]

        alias = f"speaker_{self._next_alias}"
        self._next_alias += 1
        self._speaker_alias[key] = alias
        return alias

    def _detect_language(self) -> str:
        text = " ".join(self._text_chunks)
        return self._pipeline._detect_language(text, self._job.language)

    def _build_transcription_result(self) -> MeetingTranscriptionResult:
        text = " ".join(self._text_chunks).strip()
        language = self._pipeline._detect_language(text, self._job.language)
        normalised_segments = self._pipeline._normalise_segments(self._segments, self._job)
        duration = self._elapsed if self._elapsed > 0 else self._pipeline._estimate_text_duration(text)
        return MeetingTranscriptionResult(
            text=text,
            segments=normalised_segments,
            duration_seconds=duration,
            language=language,
        )

class StreamingMeetingSession:
    """Stateful helper that supports streaming meeting transcription snapshots."""

    def __init__(
        self,
        pipeline: "MeetingPipeline",
        job: MeetingJobConfig,
        *,
        update_interval: float,
    ) -> None:
        self._pipeline = pipeline
        self._job = job
        self._update_interval = max(update_interval, 0.0)
        self._segments: List[dict] = []
        self._text_chunks: List[str] = []
        self._elapsed = 0.0
        self._since_snapshot = 0.0
        self._speaker_alias: Dict[str, str] = {}
        self._next_alias = 1
        self._final_summary: Optional[MeetingSummary] = None
        self._finalised = False

    def ingest(
        self,
        text: str,
        *,
        speaker: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Optional[StreamingSummarySnapshot]:
        if self._finalised:
            raise RuntimeError("streaming session already finalised")

        cleaned = (text or "").strip()
        if not cleaned:
            return None

        start_time, end_time = self._resolve_window(cleaned, start, end)
        speaker_label = self._normalise_speaker(speaker)

        segment = {
            "start": start_time,
            "end": end_time,
            "speaker": speaker_label,
            "text": cleaned,
        }
        self._segments.append(segment)
        self._text_chunks.append(cleaned)

        self._elapsed = max(self._elapsed, end_time)
        segment_duration = max(end_time - start_time, 0.0)
        self._since_snapshot += segment_duration

        if self._update_interval == 0 or self._since_snapshot >= self._update_interval:
            self._since_snapshot = 0.0
            return self.snapshot()
        return None

    def snapshot(self) -> StreamingSummarySnapshot:
        language = self._detect_language()
        highlights = self._pipeline._extract_highlights(self._segments, language)
        action_entries = self._pipeline._extract_action_items(self._segments, language)
        decision_entries = self._pipeline._extract_decisions(self._segments, language)
        summary_text = self._pipeline._build_summary_text(highlights, action_entries, decision_entries)

        return StreamingSummarySnapshot(
            summary_text=summary_text,
            highlights=[entry.get("text", "") for entry in highlights],
            action_items=[entry.get("text", "") for entry in action_entries],
            decisions=[entry.get("text", "") for entry in decision_entries],
            elapsed_seconds=self._elapsed,
            language=language,
        )

    def finalize(self) -> MeetingSummary:
        if self._final_summary is not None:
            return self._final_summary

        transcription = self._build_transcription_result()
        summary = self._pipeline._summarise(self._job, transcription)
        if self._pipeline._mask_pii_enabled:
            self._pipeline._mask_sensitive_content(transcription=transcription, summary=summary)
        self._pipeline._persist(self._job, transcription, summary)

        self._final_summary = summary
        self._finalised = True
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_window(
        self,
        text: str,
        start: Optional[float],
        end: Optional[float],
    ) -> Tuple[float, float]:
        if start is None:
            start_time = self._elapsed
        else:
            start_time = max(float(start), 0.0)

        if end is not None:
            end_time = max(float(end), start_time)
        else:
            duration = self._pipeline._estimate_text_duration(text)
            end_time = max(start_time + duration, start_time)

        return (round(start_time, 2), round(end_time, 2))

    def _normalise_speaker(self, speaker: Optional[str]) -> str:
        if not speaker:
            alias = self._speaker_alias.get("__default__")
            if alias:
                return alias
            alias = "speaker_1"
            self._speaker_alias["__default__"] = alias
            self._next_alias = max(self._next_alias, 2)
            return alias

        key = speaker.strip().lower()
        if key in self._speaker_alias:
            return self._speaker_alias[key]

        alias = f"speaker_{self._next_alias}"
        self._next_alias += 1
        self._speaker_alias[key] = alias
        return alias

    def _detect_language(self) -> str:
        text = " ".join(self._text_chunks)
        return self._pipeline._detect_language(text, self._job.language)

    def _build_transcription_result(self) -> MeetingTranscriptionResult:
        text = " ".join(self._text_chunks).strip()
        language = self._pipeline._detect_language(text, self._job.language)
        normalised_segments = self._pipeline._normalise_segments(self._segments, self._job)
        duration = self._elapsed if self._elapsed > 0 else self._pipeline._estimate_text_duration(text)
        return MeetingTranscriptionResult(
            text=text,
            segments=normalised_segments,
            duration_seconds=duration,
            language=language,
        )
