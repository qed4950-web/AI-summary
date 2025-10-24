"""Conversational wrapper for the meeting pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.agents import AgentRequest, AgentResult, ConversationalAgent
from core.config.paths import DATA_DIR
from core.agents.taskgraph import TaskCancelled
from core.utils import get_logger

from .models import MeetingJobConfig, MeetingSummary
from .pipeline import MeetingPipeline

LOGGER = get_logger("meeting.agent")


@dataclass
class MeetingAgentConfig:
    """Runtime configuration for the meeting assistant."""

    output_root: Path = DATA_DIR / "ami_outputs"
    language: str = "ko"
    policy_tag: Optional[str] = None


class MeetingAgent(ConversationalAgent):
    """Wraps ``MeetingPipeline`` to integrate with the orchestrator."""

    name = "meeting_summary"
    description = "회의 오디오를 전사하고 요약합니다. audio_path가 필요합니다."

    def __init__(self, config: Optional[MeetingAgentConfig] = None) -> None:
        self.config = config or MeetingAgentConfig()
        self.pipeline = MeetingPipeline()

    def prepare(self) -> None:
        # MeetingPipeline performs lazy loading; nothing extra required here.
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, request: AgentRequest) -> AgentResult:
        context = dict(request.context or {})
        progress_callback = context.pop("__progress_callback", None)
        cancel_event = context.pop("__cancel_event", None)
        audio_path_raw = context.get("audio_path")
        if not audio_path_raw:
            raise ValueError("회의 요약을 실행하려면 audio_path(오디오 파일 경로)가 필요합니다.")
        audio_path = Path(audio_path_raw).expanduser()
        if not audio_path.exists():
            raise ValueError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

        output_root = Path(context.get("output_dir") or self._default_output_root())
        output_dir = output_root / audio_path.stem
        language = context.get("language") or self.config.language
        context_dirs = [Path(p).expanduser() for p in context.get("context_dirs", [])]
        enable_resume = bool(context.get("enable_resume", False))
        job = MeetingJobConfig(
            audio_path=audio_path,
            output_dir=output_dir,
            language=language,
            policy_tag=context.get("policy_tag") or self.config.policy_tag,
            context_dirs=context_dirs,
            enable_resume=enable_resume,
        )
        LOGGER.info("meeting agent running job: %s", job)
        try:
            summary = self.pipeline.run(
                job,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except TaskCancelled as exc:
            LOGGER.info("meeting agent cancelled: %s", exc)
            raise ValueError("회의 요약이 취소되었습니다.") from exc

        events = self.pipeline.last_events()
        return AgentResult(
            content=self._format_summary(summary),
            metadata={
                "agent": self.name,
                "output_dir": str(output_dir),
                "transcript_path": str(summary.transcript_path),
                "language": summary.structured_summary.get("language"),
                "stages": events,
            },
        )

    def _default_output_root(self) -> Path:
        env_value = os.getenv("MEETING_OUTPUT_DIR")
        if env_value:
            return Path(env_value).expanduser()
        return self.config.output_root

    @staticmethod
    def _format_summary(summary: MeetingSummary) -> str:
        lines = ["🗂️ 회의 요약"]
        if summary.highlights:
            lines.append("\n핵심 요약:")
            for item in summary.highlights:
                lines.append(f"- {item}")
        if summary.decisions:
            lines.append("\n결정 사항:")
            for item in summary.decisions:
                lines.append(f"- {item}")
        if summary.action_items:
            lines.append("\n액션 아이템:")
            for item in summary.action_items:
                lines.append(f"- {item}")
        lines.append(f"\n원문 전체 요약:\n{summary.raw_summary.strip()}")
        lines.append(f"\n전사 파일: {summary.transcript_path}")
        return "\n".join(lines)
