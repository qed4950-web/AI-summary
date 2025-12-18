# -*- coding: utf-8 -*-
"""
LNPChat: 자연어 대화로 문서 검색/추천
- Retriever(모델/코퍼스/인덱스)를 사용해 사용자 질의 → 유사 문서 Top-K
- 간단한 대화 히스토리, 진행 스피너, 후속질문 제안 포함
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.utils import get_logger

from core.conversation.prompting import ChatTurn, MemoryStore, PromptManager, ToolRouter
from core.conversation.translation_cache import TranslationCache
from core.policy.engine import PolicyEngine

# New Modules
from core.conversation.chat_ui import ChatUI
from core.conversation.retrieval_strategy import (
    init_retriever, 
    init_llm_client, 
    ensure_offline_transformers
)
from core.conversation.llm_client import LLMClient

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

LOGGER = get_logger("lnp.chat")

# ──────────────────────────
# Constants
# ──────────────────────────
CONFIRM_POSITIVES = {"응", "네", "예", "맞아", "좋아", "yes", "y", "sure", "ok", "그래", "ㅇㅋ", "넵"}
CONFIRM_NEGATIVES = {"아니", "아니오", "no", "n", "싫어", "괜찮아", "아냐", "안돼", "안해", "노"}

PROMPT_INSTRUCTION_SMALL_TALK = (
    "당신은 'InfoPilot'이라는 AI 비서입니다. 한국어로 친근하게 답하세요.\n"
    "규칙:\n"
    "1. 사용자가 '너의 이름'을 물으면 'InfoPilot'이라고 답하세요.\n"
    "2. 사용자가 '오늘 할 일'을 물으면 '파일 검색이나 문서 요약을 도와드릴까요?'라고 되물으세요.\n"
    "3. 그 외 일상 대화에는 짧고 친절하게 답하세요.\n"
    "4. 만약 문서나 파일 관련 질문이라면 '[SEARCH_INTENT]'라고만 답하세요.\n"
    "5. 정체성 방어:\n"
    "- '당신은 누구인가요?' -> '저는 InfoPilot입니다. 문서 검색과 요약을 돕는 AI 비서예요.'\n"
    "- 사용자가 당신(AI)의 이름을 지정하면, 그 이름으로 호칭을 변경하세요.\n"
    "- 주체(User vs AI)를 명확히 구분하세요."
)

def _split_tokens(text: str) -> List[str]:
    return text.split()

@dataclass
class LNPChat:
    """
    Local Neural Pilot Chat.
    Orchestrates search, translation, policy checks, and LLM summarization.
    """
    model_path: Path
    corpus_path: Path
    
    # Configuration
    cache_dir: Path = Path("data/cache")
    topk: int = 5
    min_similarity: float = 0.75
    translate: bool = False
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-large"
    rerank_depth: int = 80
    rerank_batch_size: int = 16
    rerank_device: Optional[str] = None
    rerank_min_score: Optional[float] = 0.35
    lexical_weight: float = 0.2
    show_translation: bool = False
    translation_lang: str = "en"
    auto_search: bool = False
    strict_search: bool = False
    
    # LLM Config
    llm_backend: Optional[str] = None
    llm_model: Optional[str] = None
    llm_host: Optional[str] = None
    llm_options: Optional[Dict[str, str]] = None
    llm_health_timeout: Optional[float] = None
    llm_timeout: Optional[float] = None
    
    # Policy Config
    policy_engine: Optional[PolicyEngine] = None
    policy_scope: str = "auto"
    policy_agent: str = "knowledge_search"
    
    # State
    memory: MemoryStore = field(default_factory=MemoryStore)
    llm_client: Optional[LLMClient] = None
    _translator: Optional[GoogleTranslator] = None
    _translation_cache: Optional[TranslationCache] = None
    
    # Components
    ui: ChatUI = field(default_factory=ChatUI)
    retr: Optional[Any] = None  # HybridRetriever
    tool_router: ToolRouter = field(default_factory=ToolRouter)
    prompt_manager: Optional[PromptManager] = None
    
    # Internal State
    _pending_confirmation: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ensure_offline_transformers()
        self._translation_cache = TranslationCache(self.cache_dir / "translation_cache.mb")
        self.prompt_manager = PromptManager(self.memory, tokenizer=_split_tokens)
        
        # Initialize LLM Client
        self._reset_llm_client()

    def _reset_llm_client(self):
        """Re-initialize the LLM client using the Strategy module."""
        self.llm_client = init_llm_client(
            backend=self.llm_backend,
            model=self.llm_model,
            host=self.llm_host,
            options=self.llm_options,
            health_timeout=self.llm_health_timeout
        )

    def ready(self, rebuild: bool = False, *, wait_timeout: float | None = 0.1) -> bool:
        """Initialize retrieval backend."""
        if self.retr is None:
            self._build_retriever(rebuild=rebuild)
            
        if self.retr and wait_timeout is not None:
            return self.retr.wait_until_ready(timeout=wait_timeout)
        return False

    def _build_retriever(self, rebuild: bool = False) -> None:
        """Delegate to retrieval strategy."""
        
        self.retr = init_retriever(
            model_path=self.model_path,
            corpus_path=self.corpus_path,
            cache_dir=self.cache_dir,
            topk=self.topk,
            min_similarity=self.min_similarity,
            use_rerank=self.rerank,
            rerank_model_name=self.rerank_model,
            rerank_depth=self.rerank_depth,
            rerank_batch_size=self.rerank_batch_size,
            rerank_device=self.rerank_device,
            rerank_min_score=self.rerank_min_score,
            lexical_weight=self.lexical_weight,
            rebuild=rebuild
        )
        if self.retr is None:
             LOGGER.warning("Retriever initialization failed or deferred.")

    def _ensure_preview_translator(self):
        if not self.translate or not GoogleTranslator:
            return
        if self._translator is None:
            try:
                self._translator = GoogleTranslator(source="auto", target="ko")
            except Exception as e:
                LOGGER.warning("Translator init failed: %s", e)

    def ask(self, query: str, topk: Optional[int] = None, *, force_action: Optional[str] = None) -> Dict[str, Any]:
        """Main interaction point."""
        k = topk if topk is not None else self.topk
        
        # 1. Command Check
        cmd_res = self._handle_commands(query)
        if cmd_res:
            return cmd_res

        # 2. Pending Confirmation
        confirm_res = self._handle_pending_confirmation(query, k)
        if confirm_res:
             if "answer" in confirm_res:
                 return confirm_res
             # If it returned query/action, we continue with that
             query = confirm_res.get("query", query)
             force_action = confirm_res.get("action", force_action)

        # 3. Action Determination
        action = self._determine_action(query, force_action)
        
        # 4. Execute
        return self._execute_action(action, query, k)

    def _handle_commands(self, query: str) -> Optional[Dict[str, Any]]:
        query_s = query.strip()
        if query_s in ("!reset", "!clear"):
            self.memory.clear()
            return {"answer": "대화 기록을 초기화했습니다."}
        return None

    def _handle_pending_confirmation(self, query: str, k: int) -> Optional[Dict[str, Any]]:
        if not self._pending_confirmation:
            return None
        
        pending = self._pending_confirmation
        self._pending_confirmation = None # Clear it
        
        q_norm = query.strip().lower()
        if q_norm in CONFIRM_POSITIVES:
            # User confirmed
            return {"query": pending["query"], "action": pending["action"]}
        elif q_norm in CONFIRM_NEGATIVES:
            # User denied
            return {"answer": "알겠습니다. 검색하지 않겠습니다."}
        
        # Ambiguous response -> treat as new query? 
        # For now, let's assume if it's not yes/no, it's a new turn
        pass 
        return None

    def _determine_action(self, query: str, force_action: Optional[str]) -> str:
        if force_action:
            return force_action
        
        # Heuristics
        if self.auto_search or self.strict_search:
            return "search"
            
        # Tool Router (LLM or Regex)
        # For MVP, explicit keywords or ToolRouter
        # ... (Simplified logic)
        return "search" # Default to search for now as this is a RAG agent

    def _execute_action(self, action: str, query: str, k: int) -> Dict[str, Any]:
        if action == "search":
            return self._search_and_answer(query, k)
        return self._search_and_answer(query, k) # Fallback

    def _search_and_answer(self, query: str, k: int) -> Dict[str, Any]:
        self.ready(wait_timeout=5.0)
        
        if not self.retr:
             return {"answer": "검색 엔진이 준비되지 않았습니다."}
             
        with self.ui.spinner("문서 검색 중..."):
             hits = self.retr.search(query, top_k=k)
        
        if not hits:
            return {"answer": "검색 결과가 없습니다.", "hits": []}
            
        # Summarize with LLM using the helper method (for test compatibility)
        summary = self._summarize_hits(query, hits)
        
        if summary:
            ans = summary
        else:
            context_str = "\n\n".join([h["content"] for h in hits])
            ans = "LLM을 사용할 수 없습니다. 검색 결과:\n" + context_str[:500]
            
        return {"answer": ans, "hits": hits, "suggestions": []}

    def _summarize_hits(self, query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
        """Summarize search hits using the attached LLM."""
        if not self.llm_client:
            return None
            
        context_str = "\n\n".join([
            f"[{h.get('path', 'Unknown')}]\n{h.get('content') or h.get('preview') or ''}"
            for h in hits
        ])
        
        # Improved RAG prompt with structure and citation requirements
        full_prompt = (
            f"## 작업\n"
            f"아래 검색 결과를 참고하여 사용자의 질문에 정확하고 유용하게 답변하세요.\n\n"
            f"## 규칙\n"
            f"1. 답변에는 반드시 출처(파일명)를 인용하세요.\n"
            f"2. 검색 결과에 없는 내용은 추측하지 마세요.\n"
            f"3. 핵심 내용을 먼저 말하고, 필요시 세부사항을 추가하세요.\n"
            f"4. 한국어로 자연스럽게 답변하세요.\n\n"
            f"## 사용자 질문\n{query}\n\n"
            f"## 검색 결과\n{context_str}"
        )
        
        system_prompt = (
            "당신은 InfoPilot입니다. 사용자의 로컬 문서를 검색하여 정확한 정보를 제공하는 AI 비서입니다. "
            "항상 출처를 밝히고, 검색 결과에 기반한 사실만 답변합니다."
        )
        
        try:
            response = self.llm_client.generate(full_prompt, system=system_prompt)
            return response
        except Exception as e:
            LOGGER.warning("LLM summarization failed: %s", e)
            return None
