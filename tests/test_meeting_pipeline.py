from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.agents.meeting import pipeline as meeting_pipeline_module
from core.agents.meeting.stt import TranscriptionPayload
from infopilot_core.agents.meeting import (
    MeetingJobConfig,
    MeetingPipeline,
    MeetingTranscriptionResult,
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

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("quality_metrics", {}).get("summary_chars") > 0


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
    assert fallback[0]["text"] == "관련 항목이 발견되지 않았습니다."


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
