
from __future__ import annotations

import webbrowser
import urllib.parse
from datetime import datetime
from typing import Any, Dict, Optional

from core.agents import AgentRequest, AgentResult, ConversationalAgent
from core.utils import get_logger

LOGGER = get_logger("agents.web")

class WebLauncherAgent(ConversationalAgent):
    """
    A lightweight agent that opens the user's default web browser
    for the given query. Does NOT scrape or summarize.
    """

    def __init__(self, name: str = "web_search"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Opens a web browser to search for information."

    def prepare(self) -> None:
        # No heavy model loading needed
        pass

    def run(self, request: AgentRequest) -> AgentResult:
        query = request.query.strip()
        if not query:
            return AgentResult(content="검색어를 입력해주세요.")

        # Construct search URL (Google by default)
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        LOGGER.info("Opening browser for query: %s", query)
        
        try:
            webbrowser.open(url)
            message = f"Google 검색 결과창을 열었습니다: '{query}'"
        except Exception as e:
            LOGGER.error("Failed to open browser: %s", e)
            message = f"브라우저를 여는 데 실패했습니다: {e}"

        return AgentResult(
            content=message,
            metadata={
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
        )
