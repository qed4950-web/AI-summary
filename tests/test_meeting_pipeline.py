from __future__ import annotations

from pathlib import Path

from infopilot_core.agents.meeting import MeetingJobConfig, MeetingPipeline


def test_meeting_pipeline_runs(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")

    output_dir = tmp_path / "out"
    config = MeetingJobConfig(audio_path=audio, output_dir=output_dir)
    pipeline = MeetingPipeline(stt_backend="placeholder", summary_backend="llm")
    summary = pipeline.run(config)

    assert summary.transcript_path.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "segments.json").exists()
    assert summary.highlights
