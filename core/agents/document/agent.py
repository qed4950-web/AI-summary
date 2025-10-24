"""Document assistant backed by LNPChat."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.agents import AgentRequest, AgentResult, ConversationalAgent
from core.conversation.lnp_chat import LNPChat
from core.data_pipeline.policies.engine import PolicyEngine


@dataclass
class DocumentAgentConfig:
    """Configuration for document assistant."""

    model_path: Path
    corpus_path: Path
    cache_dir: Path
    topk: int = 5
    min_similarity: float = 0.35
    translate: bool = False
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = 80
    rerank_batch_size: int = 16
    rerank_device: Optional[str] = None
    rerank_min_score: Optional[float] = 0.35
    lexical_weight: float = 0.0
    show_translation: bool = False
    translation_lang: str = "en"
    llm_backend: Optional[str] = None
    llm_model: Optional[str] = None
    llm_host: Optional[str] = None
    llm_options: Optional[Dict[str, str]] = None
    policy_engine: Optional[PolicyEngine] = None
    policy_scope: str = "auto"
    policy_agent: str = "knowledge_search"
    rebuild_index: bool = False


class DocumentAgent(ConversationalAgent):
    """Wraps LNPChat so it can be orchestrated alongside 다른 에이전트."""

    name = "document_search"
    description = "질문에 맞는 문서를 검색하고 필요한 경우 요약합니다."

    def __init__(self, config: DocumentAgentConfig) -> None:
        self.config = config
        self._chat = LNPChat(
            model_path=config.model_path,
            corpus_path=config.corpus_path,
            cache_dir=config.cache_dir,
            topk=config.topk,
            min_similarity=config.min_similarity,
            translate=config.translate,
            rerank=config.rerank,
            rerank_model=config.rerank_model,
            rerank_depth=config.rerank_depth,
            rerank_batch_size=config.rerank_batch_size,
            rerank_device=config.rerank_device,
            rerank_min_score=config.rerank_min_score,
            lexical_weight=config.lexical_weight,
            show_translation=config.show_translation,
            translation_lang=config.translation_lang,
            llm_backend=config.llm_backend,
            llm_model=config.llm_model or "llama3",
            llm_host=config.llm_host,
            llm_options=config.llm_options or {},
            policy_engine=config.policy_engine,
            policy_scope=config.policy_scope,
            policy_agent=config.policy_agent,
        )

    @property
    def chat(self) -> LNPChat:
        return self._chat

    @property
    def llm_client(self):
        return getattr(self._chat, "llm_client", None)

    def prepare(self) -> None:
        self._chat.ready(rebuild=self.config.rebuild_index)

    def run(self, request: AgentRequest) -> AgentResult:
        result = self._chat.ask(request.query)
        answer: str = result.get("answer") or ""
        suggestions = result.get("suggestions") or None
        metadata = {
            "hits": result.get("hits", []),
            "agent": self.name,
            "llm_summary": result.get("llm_summary"),
        }
        return AgentResult(content=answer, suggestions=suggestions, metadata=metadata)
