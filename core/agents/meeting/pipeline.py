"""Pipeline orchestrator for meeting transcription and summarisation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from infopilot_core.utils import get_logger

from .models import MeetingJobConfig, MeetingSummary, MeetingTranscriptionResult

LOGGER = get_logger("meeting.pipeline")


class MeetingPipeline:
    """Placeholder implementation for the meeting agent workflow."""

    def __init__(self, *, stt_backend: str = "placeholder", summary_backend: str = "llm") -> None:
        self.stt_backend = stt_backend
        self.summary_backend = summary_backend

    def run(self, job: MeetingJobConfig) -> MeetingSummary:
        LOGGER.info("meeting pipeline start: audio=%s backend=%s", job.audio_path, self.stt_backend)
        transcript = self._transcribe(job)
        summary = self._summarise(job, transcript)
        self._persist(job, transcript, summary)
        LOGGER.info("meeting pipeline finished: saved=%s", summary.transcript_path.parent)
        return summary

    def _transcribe(self, job: MeetingJobConfig) -> MeetingTranscriptionResult:
        duration = 0.0
        try:
            import soundfile as sf  # type: ignore

            with sf.SoundFile(job.audio_path) as audio:
                duration = len(audio) / audio.samplerate
        except Exception:
            LOGGER.debug("soundfile not available; duration fallback to 0")

        text = "(transcription placeholder – integrate Whisper/STT here)"
        return MeetingTranscriptionResult(
            text=text,
            segments=[
                {
                    "start": 0.0,
                    "end": duration,
                    "speaker": "unknown",
                    "text": text,
                }
            ],
            duration_seconds=duration,
            language=job.language,
        )

    def _summarise(self, job: MeetingJobConfig, transcription: MeetingTranscriptionResult) -> MeetingSummary:
        raw_summary = "(summary placeholder – integrate LLM/text rank summariser here)"
        highlights = ["회의 핵심 요약이 여기 표시됩니다."]
        actions = ["담당자 배정 후 추적 필요"]
        decisions = ["다음 회의 일정은 정책 엔진에 따라 공유"]
        transcript_path = job.output_dir / "transcript.txt"
        return MeetingSummary(
            highlights=highlights,
            action_items=actions,
            decisions=decisions,
            raw_summary=raw_summary,
            transcript_path=transcript_path,
        )

    def _persist(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = summary.transcript_path
        transcript_file.write_text(transcription.text, encoding="utf-8")
        summary_file = job.output_dir / "summary.json"
        import json

        summary_payload = {
            "highlights": summary.highlights,
            "action_items": summary.action_items,
            "decisions": summary.decisions,
            "raw_summary": summary.raw_summary,
            "duration_seconds": transcription.duration_seconds,
            "language": transcription.language,
            "policy_tag": job.policy_tag,
        }
        summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        segments_file = job.output_dir / "segments.json"
        segments_file.write_text(
            json.dumps(transcription.segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        LOGGER.info(
            "meeting artefacts saved: transcript=%s summary=%s segments=%s",
            transcript_file,
            summary_file,
            segments_file,
        )
