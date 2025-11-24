from __future__ import annotations

import traceback
from typing import Any, Dict

from PySide6 import QtCore

from core.agents import AgentRequest
from core.agents.document import DocumentAgent


class ChatWorker(QtCore.QThread):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, agent: DocumentAgent, query: str, context: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.agent = agent
        self.query = query
        self.context = context or {}

    def run(self) -> None:
        try:
            result = self.agent.run(AgentRequest(query=self.query, context=self.context))
            payload: Dict[str, Any] = {
                "answer": result.content,
                "hits": result.metadata.get("hits", []) if result.metadata else [],
                "suggestions": result.suggestions or [],
            }
            self.finished.emit(payload)
        except Exception:
            self.failed.emit(traceback.format_exc())
