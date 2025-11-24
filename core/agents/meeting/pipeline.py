"""Pipeline orchestrator for meeting transcription and summarisation."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.utils import get_logger

try:  # Optional dependency handled gracefully.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from core.agents.taskgraph import TaskGraph, TaskContext, TaskCancelled

from .analytics import MeetingAnalyticsRecorder
from .audit import MeetingAuditLogger
from .context_adapter import ContextBundle, MeetingContextAdapter
from .context_store import MeetingContextStore
from .constants import (
    ACTION_KEYWORDS,
    AVERAGE_SPEECH_WPM,
    DEFAULT_LANGUAGE,
    DECISION_KEYWORDS,
    GENERIC_FALLBACK,
    HIGHLIGHT_FALLBACK,
    HIGHLIGHT_KEYWORDS,
    LANGUAGE_ALIASES,
    QUESTION_STOP_WORDS,
)
from .cache import audio_fingerprint, load_cached_summary
from .integrations import IntegrationConfig, load_provider_config, sync_action_items
from .llm.loader import OnDeviceModelLoader
from .models import (
    MeetingJobConfig,
    MeetingSummary,
    MeetingTranscriptionResult,
    StreamingSummarySnapshot,
)
from .persistence import (
    append_jsonl,
    export_integrations,
    record_analytics,
    record_audit,
    record_for_search,
    record_quality_alerts,
    sync_action_items_if_configured,
)
from .pii import mask_segments, mask_text
from .quality import compute_quality_metrics
from .reviewer import SummaryReviewer
from .speaker_id import SpeakerIdentifier, load_speaker_identifier
from .streaming import StreamingMeetingSession
from .stt import TranscriptionPayload, create_stt_backend
from .summarizer import SummariserConfig, available_summary_backends, create_summary_backend
from core.agents.supervisor import SummarySupervisor, SupervisorDecision
from .workflow import MeetingWorkflowEngine

LOGGER = get_logger("meeting.pipeline")

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")

try:  # Optional dependency for spacing correction
    from pykospacing import Spacing  # type: ignore
except ImportError:  # pragma: no cover - optional dependency (PyPI: python <3.12)
    try:
        from kosspacing import Spacing  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency (PyPI: python >=3.12)
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

        self._speaker_identifier: Optional[SpeakerIdentifier] = load_speaker_identifier()
        self._context_adapter = MeetingContextAdapter()
        self._analytics_recorder = MeetingAnalyticsRecorder()
        self._context_store = MeetingContextStore.from_env()
        self._integration_config: Optional[IntegrationConfig] = load_provider_config()
        self._on_device_loader = OnDeviceModelLoader.from_env()
        self._reviewer = SummaryReviewer.from_env()
        self._supervisor = SummarySupervisor.from_env("MEETING")
        self._audit_logger = MeetingAuditLogger.from_env()
        review_mode_env = (os.getenv("MEETING_SUMMARY_REVIEW_MODE") or "auto").strip().lower()
        self._review_mode = review_mode_env if review_mode_env in {"auto", "always", "manual", "off"} else "auto"
        supervisor_mode_env = (os.getenv("MEETING_SUPERVISOR_MODE") or "auto").strip().lower()
        self._supervisor_mode = supervisor_mode_env if supervisor_mode_env in {"auto", "always", "manual", "off"} else "auto"
        self._cancel_event: Optional[Any] = None
        self._last_events: List[Dict[str, Any]] = []
        self._last_review_backend: Optional[str] = None

    def start_streaming(
        self,
        job: MeetingJobConfig,
        *,
        update_interval: float = 60.0,
    ) -> "StreamingMeetingSession":
        return StreamingMeetingSession(self, job, update_interval=update_interval)

    # ------------------------------------------------------------------
    # TaskGraph stages
    # ------------------------------------------------------------------
    def _stage_transcription(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        job: MeetingJobConfig = context.job
        workflow: MeetingWorkflowEngine = context.extras["workflow"]

        transcript: Optional[MeetingTranscriptionResult] = None
        if not workflow.should_run("transcription"):
            transcript = workflow.load_transcription()
        if transcript is None:
            transcript = self._transcribe(job)
            workflow.store_transcription(transcript)
            workflow.mark_completed("transcription")

        context.set("transcript", transcript)

    def _stage_summary(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        job: MeetingJobConfig = context.job
        workflow: MeetingWorkflowEngine = context.extras["workflow"]
        transcript: MeetingTranscriptionResult = context.get("transcript")

        if transcript is None:
            raise RuntimeError("meeting pipeline summary stage requires transcript")

        context_bundle: Optional[ContextBundle] = None
        summary: Optional[MeetingSummary] = None
        self._last_review_backend = None
        context.extras["review_backend_used"] = None

        if not workflow.should_run("summary"):
            summary = workflow.load_summary()
            if summary is not None:
                summary.transcript_path = job.output_dir / "transcript.txt"
                if not isinstance(summary.attachments, dict):
                    summary.attachments = {}

        context.extras["quality_metrics"] = None
        context.extras["alerts"] = None
        context.extras["supervisor_decision"] = None
        review_performed = False

        if summary is None:
            context_bundle = self._collect_context_bundle(job)
            summary = self._summarise(job, transcript, context_bundle)
            issues, focus_keywords = self._evaluate_summary_quality(job, transcript, summary)
            if issues:
                summary.structured_summary["review_issues"] = issues
            if focus_keywords:
                summary.structured_summary["review_focus"] = focus_keywords
            if self._review_mode:
                summary.structured_summary["review_mode"] = self._review_mode
            review_enabled = self._reviewer.is_enabled() and self._review_mode not in {"off", "manual"}
            should_review = review_enabled and (self._review_mode == "always" or bool(issues))
            if should_review:
                reviewed = self._reviewer.review(
                    job,
                    summary,
                    transcript,
                    issues=issues,
                    focus_keywords=focus_keywords if focus_keywords else None,
                )
                if reviewed is not None:
                    summary = reviewed
                    review_performed = True
                    self._last_review_backend = self._reviewer.backend
                    context.extras["review_backend_used"] = self._reviewer.backend
                    issues, focus_keywords = self._evaluate_summary_quality(job, transcript, summary)
                    summary.structured_summary["review_issues"] = issues
                    if focus_keywords:
                        summary.structured_summary["review_focus"] = focus_keywords
            workflow.store_summary(summary)
            workflow.mark_completed("summary")

        if context_bundle is None:
            context_bundle = self._collect_context_bundle(job)
            if context_bundle.documents and not summary.attachments.get("context"):
                summary.attachments.setdefault("context", [])
                summary.attachments["context"] = [
                    {
                        "name": doc.target_name,
                        "kind": doc.kind,
                        "path": f"attachments/{doc.target_name}",
                        "preview": doc.preview,
                    }
                    for doc in context_bundle.documents
                ]
                if not summary.context:
                    summary.context = context_bundle.summary_prompt

        metrics = compute_quality_metrics(transcript, summary)
        alerts = self._detect_low_quality_summary(summary, metrics)
        supervisor_info: Optional[Dict[str, Any]] = None
        supervisor_enabled = self._supervisor.is_enabled() and self._supervisor_mode not in {"off", "manual"}

        if supervisor_enabled:
            decision = self._supervisor.decide(
                agent="meeting",
                summary=summary,
                metrics=metrics,
                issues=summary.structured_summary.get("review_issues"),
                alerts=alerts,
            )
            supervisor_info = decision.as_dict()
            summary.structured_summary["supervisor_decision"] = supervisor_info

            if decision.action == "review":
                can_review = self._reviewer.is_enabled() and self._review_mode not in {"off", "manual"}
                if can_review and not review_performed:
                    focus_override = decision.focus_keywords or summary.structured_summary.get("review_focus")
                    reviewed = self._reviewer.review(
                        job,
                        summary,
                        transcript,
                        issues=summary.structured_summary.get("review_issues") or [],
                        focus_keywords=focus_override,
                    )
                    if reviewed is not None:
                        summary = reviewed
                        review_performed = True
                        self._last_review_backend = self._reviewer.backend
                        context.extras["review_backend_used"] = self._reviewer.backend
                        issues, focus_keywords = self._evaluate_summary_quality(job, transcript, summary)
                        summary.structured_summary["review_issues"] = issues
                        if focus_keywords:
                            summary.structured_summary["review_focus"] = focus_keywords
                        metrics = compute_quality_metrics(transcript, summary)
                        alerts = self._detect_low_quality_summary(summary, metrics)
                        supervisor_info["follow_up"] = "reviewer_rerun"
            elif decision.action == "escalate":
                summary.structured_summary["requires_manual_review"] = True
                note = decision.notes or decision.reason
                if note:
                    summary.structured_summary["supervisor_notes"] = note

        if alerts:
            summary.structured_summary["alerts"] = alerts

        context.extras["quality_metrics"] = metrics
        context.extras["alerts"] = alerts
        context.extras["supervisor_decision"] = supervisor_info
        context.set("context_bundle", context_bundle)
        context.set("summary", summary)

    def _stage_finalise(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        job: MeetingJobConfig = context.job
        workflow: MeetingWorkflowEngine = context.extras["workflow"]
        transcript: MeetingTranscriptionResult = context.get("transcript")
        summary: MeetingSummary = context.get("summary")
        context_bundle: Optional[ContextBundle] = context.get("context_bundle")

        if summary is None or transcript is None:
            raise RuntimeError("meeting pipeline stages produced no summary or transcript")

        if context_bundle and context_bundle.documents and not summary.attachments.get("context"):
            summary.attachments["context"] = [
                {
                    "name": doc.target_name,
                    "kind": doc.kind,
                    "path": f"attachments/{doc.target_name}",
                    "preview": doc.preview,
                }
                for doc in context_bundle.documents
            ]

        if self._mask_pii_enabled:
            self._mask_sensitive_content(transcription=transcript, summary=summary)
        sync_action_items_if_configured(job, summary, self._integration_config)
        review_backend = context.extras.get("review_backend_used") or self._last_review_backend
        review_info: Optional[Dict[str, str]] = None
        if review_backend:
            review_info = {"backend": review_backend}
            review_model = getattr(self._reviewer, "model", None)
            if review_model:
                review_info["model"] = str(review_model)
        metrics = context.extras.get("quality_metrics")
        alerts = context.extras.get("alerts")
        supervisor_decision = context.extras.get("supervisor_decision")
        self._persist(
            job,
            transcript,
            summary,
            review_info=review_info,
            metrics=metrics,
            alerts=alerts,
            supervisor_info=supervisor_decision,
        )
        workflow.mark_completed("persistence")
        context.set("result", summary)

    def run(
        self,
        job: MeetingJobConfig,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> MeetingSummary:
        self._maybe_prepare_on_device_model()
        workflow = MeetingWorkflowEngine(job.output_dir, enable_resume=job.enable_resume)
        LOGGER.info(
            "meeting pipeline start: audio=%s backend=%s policy=%s",
            job.audio_path,
            self.stt_backend,
            job.policy_tag,
        )
        cached_summary = self._load_cache(job)
        if cached_summary is not None and not job.enable_resume:
            LOGGER.info(
                "meeting pipeline cache hit: audio=%s summary_backend=%s",
                job.audio_path,
                self.summary_backend,
            )
            return cached_summary
        context = TaskContext(pipeline=self, job=job)
        context.extras["workflow"] = workflow
        if progress_callback:
            context.extras["progress_callback"] = progress_callback
        if cancel_event:
            context.extras["cancel_event"] = cancel_event
        self._cancel_event = cancel_event

        graph = TaskGraph("meeting_pipeline")
        graph.add_stage("transcription", self._stage_transcription)
        graph.add_stage("summary", self._stage_summary, dependencies=("transcription",))
        graph.add_stage("finalise", self._stage_finalise, dependencies=("summary",))

        try:
            graph.run(context)
        finally:
            self._cancel_event = None

        events = context.stage_status()
        self._last_events = events
        for event in events:
            started = event.get("started_at")
            finished = event.get("finished_at")
            status = event.get("status")
            message = f"stage={event['stage']} status={status}"
            if started and finished:
                message += f" started={started} finished={finished}"
            if status == "failed" and event.get("error"):
                message += f" error={event['error']}"
            LOGGER.info("meeting pipeline stage: %s", message)

        summary: MeetingSummary = context.get("summary")
        if summary is None:
            raise RuntimeError("meeting pipeline did not produce a summary")
        LOGGER.info("meeting pipeline finished: saved=%s", summary.transcript_path.parent)
        return summary

    def last_events(self) -> List[Dict[str, Any]]:
        """Return the most recent TaskGraph stage events."""
        return list(self._last_events)

    def _ensure_not_cancelled(self) -> None:
        if (
            self._cancel_event
            and hasattr(self._cancel_event, "is_set")
            and callable(getattr(self._cancel_event, "is_set"))
            and self._cancel_event.is_set()
        ):
            LOGGER.info("meeting pipeline cancelled by user request")
            raise TaskCancelled("meeting pipeline cancelled")

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

    def _collect_context_bundle(self, job: MeetingJobConfig) -> ContextBundle:
        allowed_roots = {
            job.audio_path.parent.resolve(strict=False),
        }
        for path in job.context_dirs:
            try:
                allowed_roots.add(path.resolve(strict=False))
            except FileNotFoundError:
                allowed_roots.add(path.expanduser().resolve(strict=False))

        try:
            bundle = self._context_adapter.collect(
                job_audio=job.audio_path,
                output_dir=job.output_dir,
                extra_dirs=job.context_dirs,
                allowed_roots=allowed_roots,
            )
            self._record_context(job.audio_path.stem, bundle)
            return bundle
        except PermissionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("context collection failed: %s", exc)
            return ContextBundle(summary_prompt=None, documents=[])

    def _record_context(self, meeting_id: str, bundle: ContextBundle) -> None:
        if not self._context_store.is_enabled() or not bundle or not bundle.documents:
            return
        self._context_store.record_documents(meeting_id, bundle.documents)

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

        return self._apply_speaker_labels(job.audio_path, normalised)

    def _apply_speaker_labels(self, audio_path: Path, segments: List[dict]) -> List[dict]:
        if not segments:
            return segments
        if self._speaker_identifier is None:
            self._speaker_identifier = load_speaker_identifier()
        if self._speaker_identifier is None:
            return segments
        try:
            return self._speaker_identifier.label_segments(audio_path, segments)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("speaker identification failed: %s", exc)
            return segments

    @staticmethod
    def _safe_time(value: Optional[float], default: float) -> float:
        try:
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _invoke_stt_backend(self, job: MeetingJobConfig) -> TranscriptionPayload:
        if self._stt is None:
            raise RuntimeError(
                f"STT backend '{self.stt_backend}' is not configured or unavailable",
            )

        chunk_exception: Optional[Exception] = None
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
                    raise RuntimeError("chunked STT fallback returned empty transcript")
                except Exception as chunk_exc:  # pragma: no cover - diagnostics
                    chunk_exception = chunk_exc
                    LOGGER.warning("chunked STT fallback failed: %s", chunk_exc)
            failure = chunk_exception or exc
            raise RuntimeError(
                f"STT backend '{self.stt_backend}' failed to produce a transcript",
            ) from failure

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
                fd, tmp_name = tempfile.mkstemp(suffix=job.audio_path.suffix)
                os.close(fd)
                chunk_path = Path(tmp_name)
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
        context_bundle: Optional[ContextBundle] = None,
    ) -> MeetingSummary:
        language = self._map_language_code(transcription.language) or self._map_language_code(job.language) or DEFAULT_LANGUAGE
        highlight_entries = self._extract_highlights(transcription.segments, language)
        action_entries = self._extract_action_items(transcription.segments, language)
        decision_entries = self._extract_decisions(transcription.segments, language)

        context_prompt = context_bundle.summary_prompt if context_bundle else None
        summary_input = transcription.text
        if context_prompt:
            summary_input = (
                "Context:\n"
                f"{context_prompt}\n\n"
                "Transcript:\n"
                f"{transcription.text}"
            )

        model_summary = ""
        if self._summariser is not None:
            try:
                model_summary = self._summariser.summarise(summary_input)
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
        attachments: Dict[str, List[dict]] = {}
        if context_bundle and context_bundle.documents:
            attachments["context"] = [
                {
                    "name": doc.target_name,
                    "kind": doc.kind,
                    "path": f"attachments/{doc.target_name}",
                    "preview": doc.preview,
                }
                for doc in context_bundle.documents
            ]

        return MeetingSummary(
            highlights=highlights,
            action_items=action_items,
            decisions=decisions,
            raw_summary=raw_summary,
            transcript_path=transcript_path,
            structured_summary=structured_summary,
            context=context_prompt,
            attachments=attachments,
        )

    def _maybe_prepare_on_device_model(self) -> None:
        if not self._on_device_loader.is_configured():
            return
        try:
            self._on_device_loader.load()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("failed to prepare on-device model: %s", exc)

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
        scored: List[Tuple[float, dict]] = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            score = self._score_highlight(text, language)
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    {
                        "text": text,
                        "ref": self._format_timestamp(segment.get("start")),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        top_entries = [entry for _score, entry in scored[:3]]
        return top_entries

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
        lowered_keywords = [kw.lower() for kw in keywords]
        scored: List[Tuple[float, dict]] = []
        for segment in segments:
            raw_text = segment.get("text")
            text = (raw_text or "").strip()
            if not text:
                continue
            lowered = text.lower()
            match_count = sum(lowered.count(keyword) for keyword in lowered_keywords)
            if match_count == 0:
                continue
            score = self._score_segment(text, match_count)
            scored.append(
                (
                    score,
                    {
                        "text": text,
                        "ref": self._format_timestamp(segment.get("start")),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _score, entry in scored[:5]]

    def _score_highlight(self, text: str, language: str) -> float:
        words = text.split()
        if len(words) < 5:
            return 0.0
        score = min(len(words) / 6.0, 2.0)

        lowered = text.lower()
        highlight_keywords = self._keywords_for(language, HIGHLIGHT_KEYWORDS)
        if any(keyword in lowered for keyword in highlight_keywords):
            score += 1.2

        if any(char.isdigit() for char in text):
            score += 0.3

        if any(punct in text for punct in ("?", "!")):
            score += 0.2

        return score

    @staticmethod
    def _score_segment(text: str, match_count: int) -> float:
        words = text.split()
        base = min(len(words) / 5.0, 2.0)
        score = base + match_count
        if any(char.isdigit() for char in text):
            score += 0.2
        return score

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
        *,
        review_info: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float | int | str]] = None,
        alerts: Optional[List[str]] = None,
        supervisor_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        summary.transcript_path.write_text(transcription.text, encoding="utf-8")

        participants = self._extract_participants(transcription.segments)
        cache_info = {
            "version": 1,
            "audio_fingerprint": audio_fingerprint(job.audio_path),
            "stt_backend": self.stt_backend,
            "summary_backend": self.summary_backend,
            "options": {
                "diarize": job.diarize,
                "speaker_count": job.speaker_count,
            },
        }
        if metrics is None:
            metrics = compute_quality_metrics(transcription, summary)
        structured_summary = dict(summary.structured_summary) if isinstance(summary.structured_summary, dict) else {}
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
            "quality_metrics": metrics,
            "structured_summary": structured_summary,
        }
        if review_info:
            summary_payload["generated_by"]["review"] = dict(review_info)
        if summary.context:
            summary_payload["context_prompt"] = summary.context
        if summary.attachments:
            attachments = summary_payload.setdefault("attachments", {})
            for key, value in summary.attachments.items():
                attachments[key] = value
        if supervisor_info:
            summary_payload["supervisor"] = dict(supervisor_info)
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
        if structured_summary.get("requires_manual_review"):
            metadata["requires_manual_review"] = True
        if review_info:
            metadata.setdefault("generated_by", {})["review"] = dict(review_info)
        if supervisor_info:
            metadata.setdefault("supervisor", {})["decision"] = supervisor_info
        if alerts is None:
            alerts = self._detect_low_quality_summary(summary, summary_payload["quality_metrics"])
        if alerts:
            summary_payload["alerts"] = alerts
            metadata["alerts"] = alerts
            record_quality_alerts(job, alerts)
        if feedback_info:
            metadata["feedback"] = feedback_info
        metadata_path = job.output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._context_store.record_meeting_artifacts(
            job.audio_path.stem,
            transcription.text,
            summary.raw_summary,
            summary_payload.get("quality_metrics"),
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

        try:
            record_for_search(job, transcription, summary, summary_payload["quality_metrics"])
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("failed to record meeting entry for search: %s", exc)
        try:
            export_integrations(job, transcription, summary)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("integration export failed: %s", exc)
        try:
            record_analytics(self._analytics_recorder, job, transcription, summary, summary_payload["quality_metrics"])
        except Exception as exc:  # pragma: no cover - analytics optional
            LOGGER.warning("analytics recording failed: %s", exc)
        try:
            record_audit(
                self._audit_logger,
                job,
                transcription,
                summary,
                summary_payload,
                summary_path,
                metadata_path,
                segments_path,
                summary_backend=self.summary_backend,
                stt_backend=self.stt_backend,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("audit logging failed: %s", exc)

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
            speaker = segment.get("speaker_name") or segment.get("speaker")
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

    @staticmethod
    def _transcript_contains_keywords(
        transcription: MeetingTranscriptionResult,
        keywords: Sequence[str],
    ) -> bool:
        lowered_keywords = [kw.lower() for kw in keywords if kw]
        if not lowered_keywords:
            return False

        segments = transcription.segments or []
        for segment in segments:
            text = str(segment.get("text") or "").lower()
            if any(keyword in text for keyword in lowered_keywords):
                return True

        text = (transcription.text or "").lower()
        return any(keyword in text for keyword in lowered_keywords)

    def _evaluate_summary_quality(
        self,
        job: MeetingJobConfig,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> Tuple[List[str], List[str]]:
        issues: List[str] = []
        focus_keywords: set[str] = set()
        language = (
            self._map_language_code(transcription.language)
            or self._map_language_code(job.language)
            or DEFAULT_LANGUAGE
        )

        metrics = compute_quality_metrics(transcription, summary)
        compression = float(metrics.get("compression_ratio") or 0.0)
        transcript_chars = int(metrics.get("transcript_chars") or 0)
        summary_chars = int(metrics.get("summary_chars") or 0)

        highlight_keywords = self._keywords_for(language, HIGHLIGHT_KEYWORDS)
        action_keywords = self._keywords_for(language, ACTION_KEYWORDS)
        decision_keywords = self._keywords_for(language, DECISION_KEYWORDS)

        if not (summary.highlights and any(item.strip() for item in summary.highlights)):
            if self._transcript_contains_keywords(transcription, highlight_keywords):
                issues.append("핵심 요약이 비어 있어 주요 논점을 bullet 형태로 보완해 주세요.")
                focus_keywords.update(highlight_keywords)

        if not (summary.action_items and any(item.strip() for item in summary.action_items)):
            if self._transcript_contains_keywords(transcription, action_keywords):
                issues.append("액션 아이템이 누락되었습니다. 담당자와 후속 조치를 식별해 주세요.")
                focus_keywords.update(action_keywords)

        if not (summary.decisions and any(item.strip() for item in summary.decisions)):
            if self._transcript_contains_keywords(transcription, decision_keywords):
                issues.append("결정 사항이 비어 있습니다. 합의된 사항을 bullet 형태로 정리해 주세요.")
                focus_keywords.update(decision_keywords)

        if transcript_chars and compression < 0.01:
            issues.append("요약이 지나치게 짧습니다. 중요한 내용을 더 포함해 주세요.")
            focus_keywords.update(highlight_keywords)
            focus_keywords.update(action_keywords)
            focus_keywords.update(decision_keywords)

        if transcript_chars and compression > 0.5:
            issues.append("요약이 너무 길어 간결하지 않습니다. 핵심만 유지해 주세요.")

        if summary_chars == 0:
            issues.append("요약 텍스트가 비어 있습니다. 회의 내용을 간단히라도 요약해 주세요.")

        ordered_keywords = list(dict.fromkeys(kw for kw in focus_keywords if kw))
        return issues, ordered_keywords

    def _detect_low_quality_summary(
        self,
        summary: MeetingSummary,
        metrics: Dict[str, float | int | str],
    ) -> List[str]:
        alerts: List[str] = []
        if not (summary.highlights and any(item.strip() for item in summary.highlights)):
            alerts.append("highlight_missing")
        if not (summary.action_items and any(item.strip() for item in summary.action_items)):
            alerts.append("action_items_missing")
        if not (summary.decisions and any(item.strip() for item in summary.decisions)):
            alerts.append("decisions_missing")
        compression = float(metrics.get("compression_ratio") or 0.0)
        transcript_chars = int(metrics.get("transcript_chars") or 0)
        if transcript_chars and compression < 0.01:
            alerts.append("summary_too_short")
        if transcript_chars and compression > 0.5:
            alerts.append("summary_too_long")
        return alerts

    def _mask_sensitive_content(
        self,
        *,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        transcription.text = mask_text(transcription.text)
        transcription.segments = mask_segments(transcription.segments)
        summary.raw_summary = mask_text(summary.raw_summary)
        summary.highlights = [mask_text(text) for text in summary.highlights]
        summary.action_items = [mask_text(text) for text in summary.action_items]
        summary.decisions = [mask_text(text) for text in summary.decisions]
        for section in summary.structured_summary.values():
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict) and "text" in item:
                        item["text"] = mask_text(item.get("text"))

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
        append_jsonl(local_queue, feedback_entry)

        global_queue_env = os.getenv("MEETING_FEEDBACK_INBOX")
        global_path: Optional[Path] = None
        if global_queue_env:
            global_path = Path(global_queue_env)
            append_jsonl(global_path, feedback_entry)

        return {
            "queue": local_queue.name,
            "status": "pending",
            "global_queue": str(global_path) if global_path else None,
        }

    def _load_cache(self, job: MeetingJobConfig) -> Optional[MeetingSummary]:
        return load_cached_summary(
            job,
            stt_backend=self.stt_backend,
            summary_backend=self.summary_backend,
            cache_enabled=self._cache_enabled,
        )

    def _audio_fingerprint(self, audio_path: Path) -> Dict[str, int]:
        return audio_fingerprint(audio_path)


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

    def _sync_action_items(self, job: MeetingJobConfig, summary: MeetingSummary) -> None:
        if not self._integration_config:
            return
        entries = summary.structured_summary.get("action_items") or []
        if not entries:
            return
        try:
            sync_action_items(entries, self._integration_config, output_dir=job.output_dir)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("action item sync failed: %s", exc)

    def _mask_sensitive_content(
        self,
        *,
        transcription: MeetingTranscriptionResult,
        summary: MeetingSummary,
    ) -> None:
        transcription.text = mask_text(transcription.text)
        transcription.segments = mask_segments(transcription.segments)
        summary.raw_summary = mask_text(summary.raw_summary)
        summary.highlights = [mask_text(text) for text in summary.highlights]
        summary.action_items = [mask_text(text) for text in summary.action_items]
        summary.decisions = [mask_text(text) for text in summary.decisions]
        for section in summary.structured_summary.values():
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict) and "text" in item:
                        item["text"] = mask_text(item.get("text"))

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
        append_jsonl(local_queue, feedback_entry)

        global_queue_env = os.getenv("MEETING_FEEDBACK_INBOX")
        global_path: Optional[Path] = None
        if global_queue_env:
            global_path = Path(global_queue_env)
            append_jsonl(global_path, feedback_entry)

        return {
            "queue": local_queue.name,
            "status": "pending",
            "global_queue": str(global_path) if global_path else None,
        }


    # ------------------------------------------------------------------
    # Quality & feedback helpers
    # ------------------------------------------------------------------

    def _load_cache(self, job: MeetingJobConfig) -> Optional[MeetingSummary]:
        return load_cached_summary(
            job,
            stt_backend=self.stt_backend,
            summary_backend=self.summary_backend,
            cache_enabled=self._cache_enabled,
        )
