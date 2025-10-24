"""Conversational wrapper for the photo pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from core.agents import AgentRequest, AgentResult, ConversationalAgent
from core.agents.taskgraph import TaskCancelled
from core.config.paths import DATA_DIR
from core.utils import get_logger

from .models import PhotoJobConfig, PhotoRecommendation
from .pipeline import PhotoPipeline

LOGGER = get_logger("photo.agent")


@dataclass
class PhotoAgentConfig:
    output_root: Path = DATA_DIR / "photo_outputs"
    policy_tag: Optional[str] = None


class PhotoAgent(ConversationalAgent):
    name = "photo_manager"
    description = "사진 폴더를 분석해 베스트샷과 중복을 정리합니다. roots가 필요합니다."

    def __init__(self, config: Optional[PhotoAgentConfig] = None) -> None:
        self.config = config or PhotoAgentConfig()
        self.pipeline = PhotoPipeline()

    def prepare(self) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, request: AgentRequest) -> AgentResult:
        context = dict(request.context or {})
        progress_callback = context.pop("__progress_callback", None)
        cancel_event = context.pop("__cancel_event", None)
        roots_raw = context.get("roots")
        if not roots_raw:
            raise ValueError("사진 비서를 사용하려면 roots(폴더 경로 목록)가 필요합니다.")
        roots = self._normalise_roots(roots_raw)
        if not roots:
            raise ValueError("유효한 사진 폴더 경로를 찾을 수 없습니다.")
        output_dir = Path(context.get("output_dir") or self._default_output_root())
        output_dir.mkdir(parents=True, exist_ok=True)
        job = PhotoJobConfig(
            roots=roots,
            output_dir=output_dir,
            policy_tag=context.get("policy_tag") or self.config.policy_tag,
            prefer_gpu=bool(context.get("prefer_gpu", False)),
        )
        LOGGER.info("photo agent running job: %s", job)
        try:
            recommendation = self.pipeline.run(
                job,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except TaskCancelled as exc:
            LOGGER.info("photo agent cancelled: %s", exc)
            raise ValueError("사진 정리가 취소되었습니다.") from exc

        events = self.pipeline.last_events()
        return AgentResult(
            content=self._format_recommendation(recommendation),
            metadata={
                "agent": self.name,
                "report_path": str(recommendation.report_path),
                "best_shots": [str(asset.path) for asset in recommendation.best_shots],
                "stages": events,
            },
        )

    def _default_output_root(self) -> Path:
        env_value = os.getenv("PHOTO_OUTPUT_DIR")
        if env_value:
            return Path(env_value).expanduser()
        return self.config.output_root

    @staticmethod
    def _normalise_roots(raw: Iterable[str | Path]) -> List[Path]:
        roots: List[Path] = []
        for item in raw:
            path = Path(item).expanduser()
            if path.exists():
                roots.append(path)
        return roots

    @staticmethod
    def _format_recommendation(recommendation: PhotoRecommendation) -> str:
        lines = ["📷 사진 정리 결과"]
        if recommendation.best_shots:
            lines.append("\n베스트 샷:")
            for asset in recommendation.best_shots[:10]:
                lines.append(f"- {asset.path}")
            if len(recommendation.best_shots) > 10:
                lines.append(f"... 총 {len(recommendation.best_shots)}장")
        if recommendation.duplicates:
            lines.append("\n중복 그룹:")
            for group in recommendation.duplicates[:5]:
                joined = " / ".join(str(asset.path) for asset in group[:3])
                lines.append(f"- {joined}")
        lines.append(f"\n리포트 파일: {recommendation.report_path}")
        return "\n".join(lines)
