"""High-level conversational orchestrator that routes queries to domain agents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from core.agents import AgentRequest, AgentResult, ConversationalAgent
from core.conversation.llm_client import LLMClient, LLMClientError
from core.utils import get_logger

LOGGER = get_logger("conversation.orchestrator")

MEETING_KEYWORDS = {"meeting", "회의", "녹음", "회의록", "transcribe", "transcription"}
PHOTO_KEYWORDS = {"사진", "photo", "이미지", "앨범", "gallery"}

COMMAND_PREFIXES = {
    "/search": "document_search",
    "/doc": "document_search",
    "/meeting": "meeting_summary",
    "/photo": "photo_manager",
}


DEFAULT_SYSTEM_PROMPT = """당신은 InfoPilot 오케스트레이터입니다. 아래 에이전트 중 하나를 선택해 사용자의 요청을 처리하세요.

에이전트:
1. document_search
   - 역할: 일반 대화, 문서 검색, 요약, 후속 질문 지원.
   - 기본 선택입니다. 확신이 서지 않을 때는 항상 document_search를 유지하세요.

2. meeting_summary
   - 역할: 회의·녹음 파일을 전사하고 요약합니다.
   - 사용자가 회의/녹음/오디오 파일을 명확히 언급하거나 "/meeting" 명령을 사용할 때만 선택하세요.
   - context.audio_path가 없으면 agent="follow_up", reason="needs_audio", context.message에 요청 문장을 넣으세요.

3. photo_manager
   - 역할: 사진 폴더를 분석하고 정리합니다.
   - 사용자가 사진/이미지/앨범을 명확히 언급하거나 "/photo" 명령을 사용할 때만 선택하세요.
   - context.roots가 없으면 agent="follow_up", reason="needs_roots", context.message를 설정하세요.

규칙:
- 추가 정보가 정말 필요할 때만 follow_up을 사용하세요.
- 명령("/meeting", "/photo", "/search", "/doc")이 있을 때는 해당 에이전트를 우선 고려하세요.
- 그렇지 않으면 agent="document_search"와 빈 context를 반환하세요.

응답 형식은 반드시 한 줄 JSON 객체입니다.
{"agent": "<agent-name 또는 follow_up>", "reason": "<선택 사항>", "context": {...}}
agent가 "follow_up"인 경우 reason과 context.message를 포함해야 합니다."""


@dataclass
class OrchestratorResponse:
    message: str
    agent: str
    metadata: Dict[str, object] = field(default_factory=dict)
    suggestions: Optional[List[str]] = None
    reason: Optional[str] = None


class AssistantOrchestrator:
    """Routes conversation turns to specialised agents using an LLM."""

    def __init__(
        self,
        agents: Iterable[ConversationalAgent],
        *,
        llm_client: Optional[LLMClient] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._agents: Dict[str, ConversationalAgent] = {agent.name: agent for agent in agents}
        if "document_search" not in self._agents:
            raise ValueError("orchestrator requires a document_search agent as fallback")
        self._history: List[Dict[str, str]] = []
        self._system_prompt = system_prompt
        self._llm_client = llm_client
        self._last_reason: Optional[str] = None
        self._last_agent: Optional[str] = None
        self._initialise_agents()

    def _initialise_agents(self) -> None:
        for agent in self._agents.values():
            try:
                agent.prepare()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("agent %s failed to prepare: %s", agent.name, exc)

    def attach_llm(self, client: Optional[LLMClient]) -> None:
        self._llm_client = client

    def handle(self, query: str, extra_context: Optional[Dict[str, object]] = None) -> OrchestratorResponse:
        self._history.append({"role": "user", "content": query})

        command_agent, normalized_query, command_context = self._detect_command(query)
        base_context: Dict[str, object] = dict(extra_context or {})
        if command_context:
            base_context.update(command_context)

        query_for_agent = normalized_query

        if command_agent:
            agent_name = command_agent
            context = base_context
            missing_reason = self._missing_context_reason(agent_name, context)
            if missing_reason:
                self._last_reason = missing_reason
                self._last_agent = "follow_up"
                message = self._default_follow_up_message(missing_reason)
                self._history.append({"role": "assistant", "content": message})
                LOGGER.info(
                    "command follow-up requested: agent=%s reason=%s",
                    agent_name,
                    missing_reason,
                )
                return OrchestratorResponse(message=message, agent="follow_up", reason=missing_reason)
            self._last_reason = "command"
            self._last_agent = agent_name
        else:
            agent_name, context = self._select_agent(query, base_context)

            query_for_agent = query
            self._last_agent = agent_name

        if agent_name == "follow_up":
            message = context.get("message")
            if not message:
                message = self._default_follow_up_message(self._last_reason)
            context["message"] = message
            self._history.append({"role": "assistant", "content": message})
            LOGGER.info(
                "orchestrator follow-up requested: reason=%s message=%s",
                self._last_reason,
                message,
            )
            self._last_agent = "follow_up"
            return OrchestratorResponse(message=message, agent="follow_up", reason=self._last_reason)

        agent = self._agents.get(agent_name, self._agents["document_search"])
        context_keys = [key for key in context.keys() if isinstance(key, str)]
        LOGGER.info(
            "orchestrator executing agent=%s reason=%s context_keys=%s",
            agent.name,
            self._last_reason,
            context_keys,
        )
        try:
            result = agent.run(AgentRequest(query=query_for_agent, context=context))
            response_text = result.content.strip() or "결과가 없습니다."
            self._history.append({"role": "assistant", "content": response_text})
            metadata_keys = list(result.metadata.keys()) if isinstance(result.metadata, dict) else []
            LOGGER.info(
                "agent %s completed; metadata_keys=%s reason=%s",
                agent.name,
                metadata_keys,
                self._last_reason,
            )
            self._last_agent = agent.name
            return OrchestratorResponse(
                message=response_text,
                agent=agent.name,
                metadata=result.metadata,
                suggestions=result.suggestions,
                reason=self._last_reason,
            )
        except ValueError as exc:
            message = str(exc)
            self._history.append({"role": "assistant", "content": message})
            self._last_agent = agent.name
            LOGGER.info(
                "agent %s returned validation message: %s (reason=%s)",
                agent.name,
                message,
                self._last_reason,
            )
            return OrchestratorResponse(message=message, agent=agent.name, reason=self._last_reason)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("agent %s execution failed", agent.name)
            fallback = "요청을 처리하는 중 오류가 발생했습니다. 다시 시도해 주세요."
            self._history.append({"role": "assistant", "content": fallback})
            self._last_agent = agent.name
            return OrchestratorResponse(
                message=fallback,
                agent=agent.name,
                metadata={"error": str(exc)},
                reason=self._last_reason,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_follow_up_message(reason: Optional[str]) -> str:
        if reason == "needs_audio":
            return "회의 오디오 파일 경로를 알려주세요."
        if reason == "needs_roots":
            return "사진이 들어 있는 폴더 경로를 알려주세요."
        return "추가 정보가 필요합니다."

    def _detect_command(self, query: str) -> tuple[Optional[str], str, Dict[str, object]]:
        stripped = query.lstrip()
        lower = stripped.lower()
        for prefix, agent in COMMAND_PREFIXES.items():
            if lower.startswith(prefix):
                remainder = stripped[len(prefix) :].strip()
                context: Dict[str, object] = {}
                if agent == "photo_manager" and remainder:
                    roots = [part.strip() for part in remainder.split(",") if part.strip()]
                    if roots:
                        context["roots"] = roots
                if agent == "meeting_summary" and remainder:
                    # Allow quick audio path specification
                    context["audio_path"] = remainder
                if agent == "document_search":
                    context["force_action"] = "search"
                return agent, remainder or query, context
        return None, query, {}

    def _select_agent(self, query: str, extra_context: Dict[str, object]) -> tuple[str, Dict[str, object]]:
        if self._llm_client is None:
            decision = self._heuristic_route(query, extra_context)
            self._last_reason = decision[2]
            if decision[0] == "follow_up":
                return decision[0], decision[1]
            agent, context, reason = decision
            self._last_reason = reason
            return agent, context

        try:
            payload = self._llm_client.generate(
                self._build_prompt(query),
                system=self._system_prompt,
                timeout=60.0,
            )
            parsed = self._parse_agent_response(payload)
            if parsed is None:
                self._last_reason = "parse_error"
                return "document_search", {}
            agent, context, reason = parsed
            context = {**context, **extra_context}
            agent, override_reason = self._enforce_guardrails(agent, query, context)
            if override_reason:
                reason = override_reason
            missing_reason = self._missing_context_reason(agent, context)
            if missing_reason:
                self._last_reason = missing_reason
                return "follow_up", {"reason": missing_reason}
            self._last_reason = reason
            return agent, context
        except (LLMClientError, TimeoutError) as exc:
            LOGGER.warning("llm agent selection failed: %s", exc)
            self._last_reason = "llm_error"
            return "document_search", {}

    def _build_prompt(self, query: str) -> str:
        history_segments = []
        for turn in self._history[-6:]:
            role = turn["role"]
            content = turn["content"]
            history_segments.append(f"{role.upper()}: {content}")
        history_text = "\n".join(history_segments)
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" for agent in self._agents.values()
        )
        last_agent = self._last_agent or "document_search"
        return (
            "대화 히스토리:\n"
            f"{history_text}\n\n"
            f"최근 선택된 에이전트: {last_agent}\n"
            "사용 가능한 에이전트 설명:\n"
            f"{agent_descriptions}\n\n"
            f"사용자 요청: {query}\n"
            "JSON으로만 응답하세요."
        )

    @staticmethod
    def _parse_agent_response(raw: str) -> Optional[tuple[str, Dict[str, object], Optional[str]]]:
        raw = raw.strip()
        candidate = raw
        if "```" in raw:
            # handle fenced code blocks
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and part.endswith("}"):
                    candidate = part
                    break
        try:
            data = json.loads(candidate)
            agent = str(data.get("agent") or "").strip()
            if not agent:
                return None
            context = data.get("context") or {}
            if not isinstance(context, dict):
                context = {}
            reason = data.get("reason")
            return agent, context, reason
        except json.JSONDecodeError:
            LOGGER.debug("failed to parse llm response: %s", raw)
            return None

    @staticmethod
    def _text_contains_keywords(text: str, keywords: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    def _enforce_guardrails(
        self,
        agent: str,
        query: str,
        context: Dict[str, object],
    ) -> tuple[str, Optional[str]]:
        if agent == "meeting_summary":
            if context.get("audio_path"):
                return agent, None
            if not self._text_contains_keywords(query, MEETING_KEYWORDS):
                LOGGER.info(
                    "guardrail: falling back to document_search for query without meeting keywords",
                )
                return "document_search", "guardrail_document"

        if agent == "photo_manager":
            if context.get("roots"):
                return agent, None
            if not self._text_contains_keywords(query, PHOTO_KEYWORDS):
                LOGGER.info(
                    "guardrail: falling back to document_search for query without photo keywords",
                )
                return "document_search", "guardrail_document"

        return agent, None

    # ------------------------------------------------------------------
    # Heuristic routing when LLM is unavailable
    # ------------------------------------------------------------------
    def _heuristic_route(self, query: str, extra_context: Dict[str, object]) -> tuple[str, Dict[str, object], Optional[str]]:
        return "document_search", dict(extra_context), "heuristic_document"

    def _missing_context_reason(self, agent: str, context: Dict[str, object]) -> Optional[str]:
        if agent == "meeting_summary":
            if not context.get("audio_path"):
                return "needs_audio"
        if agent == "photo_manager":
            roots = context.get("roots")
            if not roots or not isinstance(roots, (list, tuple)) or not any(str(r).strip() for r in roots):
                return "needs_roots"
        return None
