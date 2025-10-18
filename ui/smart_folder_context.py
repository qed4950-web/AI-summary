"""Shared smart folder context model for UI components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class SmartFolderContext:
    """Lightweight representation of the currently selected smart folder."""

    folder_id: str
    label: str
    path: Optional[Path]
    scope: str
    policy: Optional[str]
    agent_type: str
    allowed_agents: FrozenSet[str]

    @property
    def is_global(self) -> bool:
        return (self.scope or "").lower() == "global" or (self.agent_type or "").lower() == "global"

    @property
    def path_display(self) -> str:
        if self.path is None:
            return "(경로 미지정)"
        return str(self.path)

    def allows_agent(self, agent: str) -> bool:
        normalized = (agent or "").strip().lower()
        if not normalized:
            return False
        if not self.allowed_agents:
            return True
        return normalized in self.allowed_agents or "global" in self.allowed_agents
