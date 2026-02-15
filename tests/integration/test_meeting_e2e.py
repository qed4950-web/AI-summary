from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.agents.meeting.pipeline import MeetingJobConfig, MeetingPipeline


@pytest.fixture
def mock_stt_backend() -> MagicMock:
    backend = MagicMock()
    payload = MagicMock()
    payload.text = "이것은 테스트 회의입니다. 중요한 결정이 있었습니다. 데이비드는 문서를 완료해야 합니다."
    payload.language = "ko"
    payload.duration_seconds = 6.0
    payload.segments = [
        {"start": 0.0, "end": 2.0, "speaker": "A", "text": "이것은 테스트 회의입니다."},
        {"start": 2.0, "end": 4.0, "speaker": "B", "text": "중요한 결정이 있었습니다."},
        {"start": 4.0, "end": 6.0, "speaker": "A", "text": "데이비드는 문서를 완료해야 합니다."},
    ]
    backend.transcribe.return_value = payload
    return backend


@pytest.fixture
def mock_summariser() -> MagicMock:
    summariser = MagicMock()
    summariser.summarise.return_value = """
## Highlights
- 테스트 회의 진행

## Decisions
- 결정 사항 없음

## Action Items
- [David] 문서 완료 (Due: TBD)

## Summary
이것은 테스트 회의 요약입니다.
"""
    return summariser


@pytest.fixture
def mock_policy_engine_permissive() -> MagicMock:
    engine = MagicMock()
    engine.allows.return_value = True
    engine.check.return_value = (True, "")
    return engine


@pytest.mark.smoke
@pytest.mark.integration
def test_meeting_pipeline_e2e(tmp_path, mock_stt_backend: MagicMock, mock_summariser: MagicMock) -> None:
    audio_path = tmp_path / "test_audio.mp3"
    output_dir = tmp_path / "output"
    audio_path.touch()
    output_dir.mkdir()

    job = MeetingJobConfig(
        audio_path=audio_path,
        output_dir=output_dir,
        language="ko",
        policy_tag="internal",
    )

    with (
        patch("core.agents.meeting.pipeline.create_stt_backend", return_value=mock_stt_backend),
        patch("core.agents.meeting.pipeline.create_summary_backend", return_value=mock_summariser),
        patch("core.agents.meeting.pipeline.load_provider_config", return_value=None),
        patch("core.agents.meeting.pipeline.MeetingContextStore") as mock_store,
        patch("core.agents.meeting.pipeline.MeetingAnalyticsRecorder") as mock_recorder,
        patch("core.agents.meeting.pipeline.MeetingAuditLogger") as mock_logger,
    ):
        mock_store.from_env.return_value.is_enabled.return_value = False
        mock_recorder.return_value = MagicMock()
        mock_logger.from_env.return_value.is_enabled.return_value = False

        pipeline = MeetingPipeline(stt_backend="mock", summary_backend="mock")
        pipeline._stt = mock_stt_backend
        pipeline._summariser = mock_summariser
        summary = pipeline.run(job)

    mock_stt_backend.transcribe.assert_called()
    mock_summariser.summarise.assert_called()
    assert (output_dir / "transcript.txt").exists()
    assert (output_dir / "summary.md").exists()
    assert summary.structured_summary["action_items"]
    assert "David" in str(summary.structured_summary["action_items"])


@pytest.mark.smoke
@pytest.mark.integration
def test_agent_saves_tasks(
    tmp_path,
    mock_stt_backend: MagicMock,
    mock_summariser: MagicMock,
    mock_policy_engine_permissive: MagicMock,
) -> None:
    from core.agents.base import AgentRequest
    from core.agents.meeting.agent import MeetingAgent

    audio_path = tmp_path / "agent_test.mp3"
    audio_path.touch()

    with (
        patch("core.agents.meeting.agent.TaskStore") as mock_task_store_class,
        patch("core.agents.meeting.pipeline.create_stt_backend", return_value=mock_stt_backend),
        patch("core.agents.meeting.pipeline.create_summary_backend", return_value=mock_summariser),
    ):
        mock_db = mock_task_store_class.return_value
        agent = MeetingAgent(policy_engine=mock_policy_engine_permissive)
        req = AgentRequest(query="요약해줘", context={"audio_path": str(audio_path), "output_dir": str(tmp_path)})
        agent.run(req)

    assert mock_db.add_task.call_count >= 1
