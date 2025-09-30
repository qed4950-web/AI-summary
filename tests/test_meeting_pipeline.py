from __future__ import annotations

import json
import os
from pathlib import Path

import argparse

import pytest

from core.agents.meeting import cli as meeting_cli
from core.agents.meeting import pipeline as meeting_pipeline_module
from core.agents.meeting.analytics import format_dashboard, load_dashboard
from core.agents.meeting.retraining import RetrainingQueueProcessor, process_next
from core.agents.meeting.retraining_runner import run_once
from core.agents.meeting.stt import TranscriptionPayload
from core.agents.meeting import (
    MeetingJobConfig,
    MeetingPipeline,
    MeetingTranscriptionResult,
    StreamingSummarySnapshot,
)


@pytest.mark.smoke
def test_meeting_pipeline_runs(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text(
        "프로젝트 일정 조율과 위험 검토를 진행했습니다. 액션 아이템은 김대리 확인 입니다. 출시 일정은 6월 3일로 결정되었습니다.",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    config = MeetingJobConfig(audio_path=audio, output_dir=output_dir)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    summary = pipeline.run(config)

    assert summary.transcript_path.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "segments.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert "액션 아이템" in summary.raw_summary
    assert any("액션" in item or "action" in item.lower() for item in summary.action_items)

    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "meeting_meta" in summary_payload
    assert summary_payload["summary"]["action_items"][0]["text"]
    assert "quality_metrics" in summary_payload
    metrics = summary_payload["quality_metrics"]
    assert metrics["rouge1_f"] >= 0.0
    assert metrics["rougeL_f"] >= 0.0
    assert "lfqa_coverage" in metrics
    assert summary_payload["feedback"]["status"] == "pending"
    assert summary_payload["attachments"]["feedback_queue"].endswith("feedback_queue.jsonl")
    assert (output_dir / summary_payload["attachments"]["feedback_queue"]).exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("quality_metrics", {}).get("summary_chars") > 0
    assert metadata.get("feedback", {}).get("status") == "pending"


def test_context_documents_are_packaged(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text("회의 전후 공유 자료를 확인했습니다.", encoding="utf-8")

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "meeting_agenda.txt").write_text("프로젝트 범위 정리", encoding="utf-8")

    output_dir = tmp_path / "out"
    job = MeetingJobConfig(audio_path=audio, output_dir=output_dir, context_dirs=[context_dir])
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    summary = pipeline.run(job)

    attachments = summary.attachments.get("context")
    assert attachments and attachments[0]["name"] == "meeting_agenda.txt"
    assert (output_dir / "attachments" / "meeting_agenda.txt").exists()
    assert summary.context and "프로젝트" in summary.context

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "context" in payload.get("attachments", {})
    assert payload.get("context_prompt")


def test_meeting_analytics_outputs(tmp_path: Path, capfd) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text("분석 대시보드를 위한 회의입니다.", encoding="utf-8")

    output_dir = tmp_path / "out"
    job = MeetingJobConfig(audio_path=audio, output_dir=output_dir)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    pipeline.run(job)

    analytics_dir = output_dir / "analytics"
    meeting_path = analytics_dir / f"{audio.stem}.json"
    assert meeting_path.exists()
    entry = json.loads(meeting_path.read_text(encoding="utf-8"))
    assert entry["meeting_id"] == audio.stem
    assert entry["speaker_stats"]
    assert entry["counts"]["action_items"] >= 0

    dashboard_path = analytics_dir / "dashboard.json"
    assert dashboard_path.exists()
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert dashboard["total_meetings"] == 1
    loaded_dashboard = load_dashboard(analytics_dir)
    assert loaded_dashboard == dashboard
    rendered = format_dashboard(loaded_dashboard)
    assert "총 회의 수" in rendered

    queue_path = analytics_dir / "training_queue.jsonl"
    assert queue_path.exists()
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert lines
    queue_entry = json.loads(lines[0])
    assert queue_entry["meeting_id"] == audio.stem

    processor = RetrainingQueueProcessor(analytics_dir)
    pending = processor.pending()
    assert pending and pending[0].meeting_id == audio.stem
    claimed = processor.claim_next()
    assert claimed and claimed.meeting_id == audio.stem
    processor.mark_processed(claimed, status="tested")

    claimed_flag = {
        "called": False,
    }

    queue_entry = processor.make_entry(meeting_id="manual")
    processor.mark_processed(queue_entry, status="seed")

    def fake_handler(entry):
        claimed_flag["called"] = True
        return "done"

    # process_next should return False when queue empty
    assert process_next(fake_handler, base_dir=analytics_dir) is False

    # Reinsert an entry and verify handler path
    entry = processor.make_entry(
        meeting_id="reinforced",
        summary_path="summary.json",
        transcript_path="transcript.txt",
        language="ko",
    )
    processor.enqueue(entry)
    assert process_next(fake_handler, base_dir=analytics_dir) is True
    assert claimed_flag["called"]
    assert run_once(base_dir=analytics_dir, handler=fake_handler) is False

    # CLI smoke checks
    dashboard_ns = argparse.Namespace(analytics_dir=str(analytics_dir), json=False)
    meeting_cli.dashboard_command(dashboard_ns)
    queue_list_ns = argparse.Namespace(analytics_dir=str(analytics_dir), json=False)
    meeting_cli.queue_list_command(queue_list_ns)
    queue_run_ns = argparse.Namespace(analytics_dir=str(analytics_dir), status="done", echo=True)
    meeting_cli.queue_run_command(queue_run_ns)
    output = capfd.readouterr().out
    assert "회의" in output or "대기" in output


def test_context_store_and_integrations_scaffolding(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"stub")
    rag_store = tmp_path / "rag"
    integration_out = tmp_path / "integrations"
    audit_log = tmp_path / "audit.jsonl"
    context_src = tmp_path / "context_src"
    context_src.mkdir()
    (context_src / "notes.txt").write_text("Prior agreement on budget.", encoding="utf-8")

    monkeypatch.setenv("MEETING_ONDEVICE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MEETING_RAG_ENABLED", "1")
    monkeypatch.setenv("MEETING_RAG_STORE", str(rag_store))
    monkeypatch.setenv("MEETING_INTEGRATIONS_PROVIDER", "local")
    monkeypatch.setenv("MEETING_INTEGRATIONS_OUT", str(integration_out))
    monkeypatch.setenv("MEETING_AUDIT_LOG", str(audit_log))

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text(
        "회의에서 액션 아이템으로 김대리가 문서를 정리하기로 했습니다. 출시 일정은 확정되었습니다.",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    job = MeetingJobConfig(
        audio_path=audio,
        output_dir=output_dir,
        context_dirs=[context_src],
    )

    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    assert pipeline._on_device_loader.is_configured()
    summary = pipeline.run(job)
    assert summary.highlights
    assert pipeline._on_device_loader.load() is not None

    store_file = rag_store / f"{audio.stem}.jsonl"
    assert store_file.exists()
    lines = store_file.read_text(encoding="utf-8").splitlines()
    assert any('"type": "summary"' in line for line in lines)
    assert any('"type": "transcript"' in line for line in lines)
    integration_file = integration_out / "action_items.json"
    assert integration_file.exists()

    data = json.loads(integration_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data
    assert audit_log.exists()
    audit_lines = audit_log.read_text(encoding="utf-8").splitlines()
    assert audit_lines


def test_cli_ingest_single_file(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    (tmp_path / "sample.wav.txt").write_text("간단한 회의 메모입니다.", encoding="utf-8")

    output_dir = tmp_path / "output"
    ns = argparse.Namespace(
        file=str(audio),
        input_dir=None,
        output_dir=str(output_dir),
        pattern="*.wav",
        recursive=False,
        echo=False,
    )

    meeting_cli.ingest_command(ns)
    assert (output_dir / "sample" / "summary.json").exists()


def test_streaming_session_finalize(tmp_path: Path) -> None:
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    job = MeetingJobConfig(audio_path=tmp_path / "live.wav", output_dir=tmp_path / "out")

    session = pipeline.start_streaming(job, update_interval=0.0)
    snapshot = session.ingest("첫 번째 의제는 프로젝트 일정 조율입니다.", speaker="호스트")
    assert isinstance(snapshot, StreamingSummarySnapshot)
    assert snapshot.highlights

    session.ingest("액션 아이템으로 김대리가 위험 요소를 정리합니다.", speaker="호스트")
    final_summary = session.finalize()

    assert final_summary.raw_summary
    assert final_summary.transcript_path.exists()

    summary_json = job.output_dir / "summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["feedback"]["status"] == "pending"
    queue_name = payload["attachments"]["feedback_queue"]
    assert (job.output_dir / queue_name).exists()


def test_workflow_resume_skips_summary_recompute(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text("테스트 회의를 진행했습니다.", encoding="utf-8")

    output_dir = tmp_path / "out"
    job = MeetingJobConfig(audio_path=audio, output_dir=output_dir, enable_resume=True)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    pipeline.run(job)
    state_file = output_dir / "workflow_state.json"
    assert state_file.exists()

    original_summarise = MeetingPipeline._summarise

    def fail_summarise(self, job_config, transcription, context_bundle=None):  # type: ignore[override]
        raise AssertionError("summary recomputation should be skipped when resuming")

    monkeypatch.setattr(MeetingPipeline, "_summarise", fail_summarise)

    try:
        summary = pipeline.run(job)
    finally:
        monkeypatch.setattr(MeetingPipeline, "_summarise", original_summarise)

    assert summary.raw_summary
    assert (output_dir / "checkpoints" / "summary.json").exists()


@pytest.mark.full
def test_pipeline_auto_selects_whisper_when_available(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder audio")
    output_dir = tmp_path / "out"

    class DummyWhisperBackend:
        name = "whisper"

        def transcribe(self, *_args, **_kwargs) -> TranscriptionPayload:
            return TranscriptionPayload(
                text="follow-up needed",
                segments=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "speaker_1",
                        "text": "Follow up work is required.",
                    }
                ],
                duration_seconds=1.0,
                language="ko",
            )

    def fake_create_backend(name: str, **_kwargs):
        if name == "whisper":
            return DummyWhisperBackend()
        return None

    monkeypatch.setattr(meeting_pipeline_module, "create_stt_backend", fake_create_backend)
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: True),
    )

    pipeline = MeetingPipeline(summary_backend="heuristic")
    assert pipeline.stt_backend == "whisper"

    job = MeetingJobConfig(audio_path=audio, output_dir=output_dir)
    summary = pipeline.run(job)

    assert summary.highlights  # STT 경로를 통해 결과 생성
    assert summary.transcript_path.exists()


@pytest.mark.full
def test_pipeline_auto_falls_back_to_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: False),
    )
    pipeline = MeetingPipeline(summary_backend="heuristic")
    assert pipeline.stt_backend == "placeholder"


@pytest.mark.full
def test_collect_by_keywords_handles_missing_text(monkeypatch) -> None:
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: False),
    )
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    segments = [
        {"speaker": "speaker_1"},
        {"text": None, "start": 1.0},
        {"text": "Follow up 작업 필요", "start": 2.0},
    ]

    collected = pipeline._collect_by_keywords(segments, ["follow"], "ko")
    assert collected[0]["text"].startswith("Follow up")

    fallback = pipeline._collect_by_keywords([{"speaker": "speaker_1"}], ["todo"], "ko")
    assert fallback == []


@pytest.mark.full
def test_chunked_stt_fallback(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")

    class FailingBackend:
        name = "whisper"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("primary transcription failed")

    backend = FailingBackend()

    monkeypatch.setenv("MEETING_STT_CHUNK_SECONDS", "60")
    monkeypatch.setenv("MEETING_MASK_PII", "0")
    def fake_create_backend(name: str, **_kwargs):  # type: ignore[override]
        return backend

    monkeypatch.setattr(meeting_pipeline_module, "create_stt_backend", fake_create_backend)

    chunk_payload = TranscriptionPayload(
        text="chunk transcription",
        segments=[{"start": 0.0, "end": 1.0, "speaker": "speaker_1", "text": "chunk transcription"}],
        duration_seconds=1.0,
        language="en",
    )

    chunk_calls = {"count": 0}

    def fake_transcribe_in_chunks(self, job, language=None):  # type: ignore[override]
        chunk_calls["count"] += 1
        return chunk_payload

    monkeypatch.setattr(MeetingPipeline, "_transcribe_in_chunks", fake_transcribe_in_chunks)

    job = MeetingJobConfig(audio_path=audio, output_dir=tmp_path)
    pipeline = MeetingPipeline(stt_backend="whisper", summary_backend="heuristic")
    summary = pipeline.run(job)

    assert summary.raw_summary
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["cache"]["stt_backend"] == "whisper"
    assert chunk_calls["count"] == 1


@pytest.mark.full
def test_context_collection_rejects_out_of_scope(monkeypatch, tmp_path: Path) -> None:
    scoped_dir = tmp_path / "scoped"
    scoped_dir.mkdir()
    audio = scoped_dir / "meeting.wav"
    audio.write_bytes(b"audio")
    transcript = scoped_dir / "meeting.wav.txt"
    transcript.write_text("회의 내용", encoding="utf-8")

    outside_dir = tmp_path.parent / "outside_context"
    outside_dir.mkdir(exist_ok=True)

    monkeypatch.setenv("MEETING_CONTEXT_PRE_DIR", str(outside_dir))

    job = MeetingJobConfig(audio_path=audio, output_dir=tmp_path / "out", context_dirs=[scoped_dir])
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    with pytest.raises(PermissionError):
        pipeline.run(job)

@pytest.mark.full
def test_highlight_scoring_prefers_keyword_segments() -> None:
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    segments = [
        {"text": "안녕하세요", "start": 0.0},
        {"text": "핵심 결론은 출시 일정을 6월로 조정한다는 것입니다.", "start": 1.0},
        {"text": "회의가 종료되었습니다", "start": 2.0},
    ]

    highlights = pipeline._extract_highlights(segments, "ko")
    assert highlights
    assert highlights[0]["text"].startswith("핵심 결론")


@pytest.mark.full
def test_model_summary_skips_heuristic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: False),
    )
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="ollama")

    class DummySummariser:
        def summarise(self, text: str) -> str:
            return f"요약된 내용: {text}"

    pipeline._summariser = DummySummariser()

    build_called = {"count": 0}

    def fake_build(self, *_args, **_kwargs):  # type: ignore[override]
        build_called["count"] += 1
        return "heuristic"

    monkeypatch.setattr(MeetingPipeline, "_build_summary_text", fake_build)

    job = MeetingJobConfig(audio_path=tmp_path / "dummy.wav", output_dir=tmp_path)
    transcription = MeetingTranscriptionResult(
        text="회의 결과 공유",
        segments=[{"text": "결정 사항을 공유합니다", "start": 0.0, "end": 1.0, "speaker": "speaker_1"}],
        duration_seconds=1.0,
        language="ko",
    )

    summary = pipeline._summarise(job, transcription)

    assert summary.raw_summary.startswith("요약된 내용")
    assert build_called["count"] == 0


@pytest.mark.full
def test_model_summary_fallback_uses_heuristic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: False),
    )
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="bitnet")

    class FailingSummariser:
        def summarise(self, _text: str) -> str:
            return ""

    pipeline._summariser = FailingSummariser()

    job = MeetingJobConfig(audio_path=tmp_path / "dummy.wav", output_dir=tmp_path)
    transcription = MeetingTranscriptionResult(
        text="회의 결과 공유",
        segments=[{"text": "결정 사항을 공유합니다", "start": 0.0, "end": 1.0, "speaker": "speaker_1"}],
        duration_seconds=1.0,
        language="ko",
    )

    summary = pipeline._summarise(job, transcription)

    assert summary.raw_summary.startswith("요약:")


@pytest.mark.full
def test_normalise_segments_aliases_and_merges(tmp_path: Path) -> None:
    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"audio")
    job = MeetingJobConfig(audio_path=audio, output_dir=tmp_path, speaker_count=2)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")

    segments = [
        {"speaker": "SPEAKER_00", "start": 1.5, "end": 2.0, "text": "첫 문장"},
        {"speaker": "SPEAKER_00", "start": 2.0, "end": 3.0, "text": "이어지는 문장"},
        {"speaker": "SPEAKER_01", "start": 4.0, "end": 5.0, "text": "다른 화자"},
        {"start": 6.0, "end": 7.0, "text": "미지정 화자"},
    ]

    normalised = pipeline._normalise_segments(segments, job)

    assert len(normalised) == 3
    assert normalised[0]["speaker"] == "speaker_1"
    assert normalised[0]["text"] == "첫 문장 이어지는 문장"
    assert normalised[1]["speaker"] == "speaker_2"
    assert normalised[2]["speaker"].startswith("speaker_")


@pytest.mark.full
def test_pipeline_reuses_cache(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text("회의 내용을 정리했습니다.", encoding="utf-8")

    output_dir = tmp_path / "out"
    job = MeetingJobConfig(audio_path=audio, output_dir=output_dir)

    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    first_summary = pipeline.run(job)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("cache", {}).get("version") == 1
    assert (tmp_path / "meeting_vector_index.jsonl").exists()
    assert (output_dir / "tasks.json").exists()
    action_items_path = output_dir / "action_items.json"
    assert action_items_path.exists()
    action_payload = json.loads(action_items_path.read_text(encoding="utf-8"))
    assert action_payload.get("meeting_id") == "meeting"
    assert "items" in action_payload
    assert (output_dir / "meeting.ics").exists()

    def fail_transcribe(self, *_args, **_kwargs):  # type: ignore[override]
        raise AssertionError("transcribe should not run when cache is valid")

    monkeypatch.setattr(MeetingPipeline, "_transcribe", fail_transcribe)
    pipeline_cached = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    second_summary = pipeline_cached.run(job)

    assert second_summary.raw_summary == first_summary.raw_summary
    assert second_summary.transcript_path == first_summary.transcript_path


def test_backend_diagnostics_structure(monkeypatch) -> None:
    monkeypatch.setattr(
        MeetingPipeline,
        "_whisper_available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        meeting_pipeline_module,
        "available_summary_backends",
        lambda: {"heuristic": True, "kobart": False},
    )

    diagnostics = meeting_pipeline_module.get_backend_diagnostics()

    assert diagnostics["stt"]["whisper"] is True
    assert diagnostics["summary"]["heuristic"] is True
    assert diagnostics["summary"]["kobart"] is False
    assert "resources" in diagnostics


@pytest.mark.full
def test_pii_masking_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEETING_MASK_PII", "1")
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    transcript = tmp_path / "meeting.wav.txt"
    transcript.write_text(
        "연락처는 contact@example.com 이고 전화번호는 +82 10-1234-5678 입니다.",
        encoding="utf-8",
    )

    job = MeetingJobConfig(audio_path=audio, output_dir=tmp_path)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="heuristic")
    summary = pipeline.run(job)

    assert "[REDACTED_EMAIL]" in summary.raw_summary
    assert "[REDACTED_PHONE]" in summary.raw_summary
    masked_transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert "contact@example.com" not in masked_transcript
    assert "[REDACTED_PHONE]" in masked_transcript

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("pii_masked") is True

    monkeypatch.delenv("MEETING_MASK_PII", raising=False)
