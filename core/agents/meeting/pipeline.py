"""Pipeline orchestrator for meeting transcription and summarisation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from infopilot_core.utils import get_logger

from .models import MeetingJobConfig, MeetingSummary, MeetingTranscriptionResult
from .stt import TranscriptionPayload, create_stt_backend
from .summarizer import KoBARTSummariser, SummariserConfig

LOGGER = get_logger("meeting.pipeline")

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

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
        backend_name = stt_backend if stt_backend not in {None, ""} else backend_env
        if not backend_name:
            backend_name = "placeholder"

        self.stt_backend = backend_name
        summary_env = os.getenv("MEETING_SUMMARY_BACKEND")
        summary_backend_name = summary_backend if summary_backend not in {None, ""} else summary_env
        if not summary_backend_name:
            summary_backend_name = "kobart"

        self.summary_backend = summary_backend_name
        self._stt = create_stt_backend(self.stt_backend, **(stt_options or {}))
        if self._stt is None and self.stt_backend not in {"placeholder", "none", "noop"}:
            LOGGER.warning("requested STT backend '%s' unavailable; proceeding without STT", self.stt_backend)

        # Lazy initialisation of post-processing helpers
        self._spacing_model = None
        save_transcript_env = os.getenv("MEETING_SAVE_TRANSCRIPT", "0").strip().lower()
        self._save_transcript = save_transcript_env not in {"", "0", "false", "no"}

        self._kobart: Optional[KoBARTSummariser] = None
        if self.summary_backend.lower() in {"kobart", "kobart_chunk"}:
            try:
                self._kobart = KoBARTSummariser(SummariserConfig())
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("KoBART summariser unavailable: %s", exc)
                self.summary_backend = "heuristic"

    def run(self, job: MeetingJobConfig) -> MeetingSummary:
        LOGGER.info(
            "meeting pipeline start: audio=%s backend=%s policy=%s",
            job.audio_path,
            self.stt_backend,
            job.policy_tag,
        )
        transcript = self._transcribe(job)
        summary = self._summarise(job, transcript)
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
            language = job.language
        else:
            payload = self._invoke_stt_backend(job)
            text = payload.text
            duration = payload.duration_seconds or self._estimate_duration(job.audio_path, text)
            segments = payload.segments or self._segment_transcript(text, job, duration)
            language = payload.language or job.language

        return MeetingTranscriptionResult(
            text=text,
            segments=segments,
            duration_seconds=duration,
            language=language,
        )

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

        average_wpm = 130
        words = max(len(transcript.split()), 1)
        minutes = words / average_wpm
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
            return TranscriptionPayload(
                text=f"(transcription failed: {exc})",
                language=job.language,
            )

    # ---------------------------------------------------------------------
    # Stage 2: Summary/action extraction
    # ---------------------------------------------------------------------
    def _summarise(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
    ) -> MeetingSummary:
        highlight_entries = self._extract_highlights(transcription.segments)
        action_entries = self._extract_action_items(transcription.segments)
        decision_entries = self._extract_decisions(transcription.segments)

        raw_summary = self._build_summary_text(highlight_entries, action_entries, decision_entries)
        if self._kobart is not None:
            try:
                kobart_summary = self._kobart.summarise(transcription.text)
            except Exception as exc:  # pragma: no cover - inference guard
                LOGGER.warning("KoBART summariser failed; falling back to heuristic summary: %s", exc)
                self._kobart = None
                self.summary_backend = "heuristic"
                kobart_summary = ""
            if kobart_summary:
                raw_summary = kobart_summary

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

    def _extract_highlights(self, segments: Sequence[dict]) -> List[dict]:
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
            return [{"text": "회의 주요 내용을 식별하지 못했습니다."}]
        return entries

    def _extract_action_items(self, segments: Sequence[dict]) -> List[dict]:
        keywords = ["action", "todo", "follow", "해야", "요청", "담당", "액션", "아이템"]
        return self._collect_by_keywords(segments, keywords)

    def _extract_decisions(self, segments: Sequence[dict]) -> List[dict]:
        keywords = ["결정", "승인", "확정", "정리", "합의"]
        return self._collect_by_keywords(segments, keywords)

    def _collect_by_keywords(self, segments: Sequence[dict], keywords: Sequence[str]) -> List[dict]:
        collected: List[dict] = []
        lowered_keywords = [kw.lower() for kw in keywords]
        for segment in segments:
            text = segment.get("text", "")
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
            fallback = segments[0]["text"] if segments else "관련 항목이 발견되지 않았습니다."
            collected.append({"text": fallback})
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
        }
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
