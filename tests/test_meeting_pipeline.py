from __future__ import annotations

import json
from pathlib import Path

import pytest

from infopilot_core.agents.meeting import MeetingJobConfig, MeetingPipeline


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
