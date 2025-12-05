# -*- coding: utf-8 -*-
"""
LNP Chat: 자연어 대화로 문서 검색/추천
- Retriever(모델/코퍼스/인덱스)를 사용해 사용자 질의 → 유사 문서 Top-K
- 간단한 대화 히스토리, 진행 스피너, 후속질문 제안 포함
"""
from __future__ import annotations
import re
import time
import threading
import os
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List
import textwrap

from core.data_pipeline.policies.engine import PolicyEngine
from core.config.paths import MODELS_DIR
from core.search.retriever import (
    Retriever,
    SessionState,
    _similarity_to_percent,
    _split_tokens,
)  # Step3 검색기 재사용

_PREVIEW_TOKEN_PATTERN = re.compile(r"(?u)(?:[가-힣]{1,}|[A-Za-z0-9]{2,})")
_GREETINGS = {
    "안녕",
    "안녕하세요",
    "안뇽",
    "하이",
    "ㅎㅇ",
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null"}:
        return default
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def _env_int(name: str, *, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if maximum is not None:
        value = min(value, maximum)
    return max(minimum, value)


def _ensure_offline_transformers() -> None:
    base_dir = MODELS_DIR / "sentence_transformers"
    if not base_dir.exists():
        return
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(base_dir))
    os.environ.setdefault("HF_HOME", str(base_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_ensure_offline_transformers()
from .translation_cache import TranslationCache
from .prompting import ChatTurn, MemoryStore, PromptManager, ToolRouter
from .llm_client import create_llm_client, LLMClient, LLMClientError

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# ──────────────────────────
# 콘솔 스피너 (즉시 피드백)
# ──────────────────────────
class Spinner:
    FRAMES = ["|", "/", "-", "\\"]
    def __init__(self, prefix="검색 준비", interval=0.12):
        self.prefix = prefix
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
        self._i = 0
    def start(self) -> None:
        if self._t:
            return

        def _run() -> None:
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                print(f"\r{self.prefix} {frame} ", end="", flush=True)

        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()

    def stop(self, clear=True) -> None:
        if not self._t:
            return
        self._stop.set()
        self._t.join()
        if clear:
            print("\r" + " " * 80 + "\r", end="", flush=True)

# ──────────────────────────
# 대화 상태
# ──────────────────────────
@dataclass
class LNPChat:
    model_path: Path
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 5
    translate: bool = False  # 기본은 다국어 Sentence-BERT로 번역 없이 처리
    rerank: bool = field(default_factory=lambda: _env_flag("LNPCHAT_RERANK", default=True))
    rerank_model: str = field(default_factory=lambda: os.getenv("LNPCHAT_RERANK_MODEL", "BAAI/bge-reranker-large"))
    rerank_depth: int = field(default_factory=lambda: _env_int("LNPCHAT_RERANK_DEPTH", default=80, minimum=10, maximum=256))
    rerank_batch_size: int = 16
    rerank_device: Optional[str] = None
    rerank_min_score: Optional[float] = 0.35
    lexical_weight: float = 0.2
    show_translation: bool = False
    translation_lang: str = "en"
    auto_search: bool = field(default_factory=lambda: _env_flag("LNPCHAT_AUTO_SEARCH", default=False))
    min_similarity: float = 0.75
    policy_engine: Optional[PolicyEngine] = None
    policy_scope: str = "auto"  # auto|policy|global
    policy_agent: str = "knowledge_search"
    llm_backend: Optional[str] = field(default_factory=lambda: os.getenv("LNPCHAT_LLM_BACKEND"))
    llm_model: str = field(default_factory=lambda: os.getenv("LNPCHAT_LLM_MODEL", "llama3"))
    llm_host: str = field(default_factory=lambda: os.getenv("LNPCHAT_LLM_HOST", ""))
    llm_options: Dict[str, str] = field(default_factory=dict)
    llm_health_timeout: Optional[float] = field(default=None)
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LNPCHAT_LLM_TIMEOUT", "30") or 30.0))

    retr: Optional[Retriever] = field(init=False, default=None)
    translator: Optional[Any] = field(init=False, default=None)
    ready_done: bool = field(init=False, default=False)
    translation_cache: Optional[TranslationCache] = field(init=False, default=None)
    preview_translator: Optional[Any] = field(init=False, default=None)
    index_loaded: bool = field(init=False, default=False)
    index_reasons: List[str] = field(init=False, default_factory=list)
    session_state: SessionState = field(init=False, default_factory=SessionState)
    last_query_text: str = field(init=False, default="")
    last_hits: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _policy_effective: bool = field(init=False, default=False)
    memory: MemoryStore = field(init=False)
    prompt_manager: PromptManager = field(init=False)
    tool_router: ToolRouter = field(init=False)
    llm_client: Optional[LLMClient] = field(init=False, default=None)
    pending_search: Optional[Dict[str, Any]] = field(init=False, default=None)
    last_selected_hit_index: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        memory_cap = _env_int("LNPCHAT_MEMORY_TURNS", default=40, minimum=5, maximum=200)
        self.memory = MemoryStore(capacity=memory_cap)
        self.prompt_manager = PromptManager(self.memory, tokenizer=_split_tokens)
        self.tool_router = ToolRouter()
        self.llm_client = self._init_llm_client()

    def _init_llm_client(self, *, require_health: bool = True) -> Optional[LLMClient]:
        backend = (self.llm_backend or "").strip()
        if not backend:
            return None
        def _env_flag(name: str, *, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            raw = raw.strip().lower()
            if raw in {"0", "false", "no", "off"}:
                return False
            if raw in {"1", "true", "yes", "on"}:
                return True
            return default
        require_llm = _env_flag("LNPCHAT_REQUIRE_LLM", default=False) if require_health else False
        health_timeout_env = os.getenv("LNPCHAT_LLM_HEALTH_TIMEOUT", "").strip()
        health_timeout_s: float
        if self.llm_health_timeout is not None:
            try:
                health_timeout_s = float(self.llm_health_timeout)
            except (TypeError, ValueError):
                health_timeout_s = 5.0
        elif health_timeout_env:
            try:
                health_timeout_s = float(health_timeout_env)
            except ValueError:
                health_timeout_s = 5.0
        else:
            health_timeout_s = 5.0
        if not math.isfinite(health_timeout_s) or health_timeout_s <= 0:
            health_timeout_s = 5.0
        health_timeout_s = max(1.0, min(30.0, health_timeout_s))
        try:
            client = create_llm_client(
                backend,
                model=self.llm_model or "llama3",
                host=self.llm_host or "",
                options=self.llm_options or {},
            )
        except LLMClientError as exc:
            if require_llm:
                raise SystemExit(f"LLM 백엔드 '{backend}' 초기화에 실패했습니다: {exc}") from exc
            print(f"⚠️ 로컬 LLM 초기화 실패: {exc}")
            return None
        if require_health and require_llm:
            health_prompt = os.getenv("LNPCHAT_LLM_HEALTH_PROMPT", "ping")
            system_prompt = "You are a health check responder. Reply briefly."
            try:
                client.generate(
                    health_prompt,
                    system=system_prompt,
                    timeout=health_timeout_s,
                )
            except LLMClientError as exc:
                raise SystemExit(f"LLM 백엔드 '{backend}'가 {health_timeout_s:.0f}s 내에 응답하지 않습니다: {exc}") from exc
            except Exception as exc:
                raise SystemExit(f"LLM 백엔드 '{backend}'에 연결하는 중 오류가 발생했습니다: {exc}") from exc
        return client

    def _reset_llm_client(self) -> bool:
        backend = (self.llm_backend or "").strip()
        if not backend:
            return False
        try:
            client = self._init_llm_client(require_health=False)
        except SystemExit as exc:
            print(f"⚠️ 로컬 LLM 재연결 실패: {exc}")
            return False
        if client is None:
            return False
        self.llm_client = client
        print("ℹ️ 로컬 LLM 세션을 다시 연결했습니다.")
        return True

    # 초기화: Retriever 및 번역기 준비
    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="인덱스 준비")
        spin.start()
        try:
            self.retr = self._build_retriever(use_rerank=self.rerank, rebuild=rebuild)
            if self.translate:
                if GoogleTranslator is None:
                    print("\n⚠️ 경고: 'deep-translator' 라이브러리를 찾을 수 없어 번역 기능이 비활성화됩니다.")
                    print("   해결: pip install deep-translator")
                else:
                    try:
                        self.translator = GoogleTranslator(source="auto", target="en")
                    except Exception as exc:
                        print("\n⚠️ 경고: 번역기 초기화에 실패해 번역 기능이 비활성화됩니다.")
                        print(f"   상세: {exc}")
            if self.show_translation:
                self.translation_lang = (self.translation_lang or "en").strip() or "en"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.translation_cache = TranslationCache(self.cache_dir / "translations.sqlite3")
            self.ready_done = True
            self._policy_effective = bool(self._resolve_policy_engine())
        finally:
            spin.stop()
        index_ready = self.retr.wait_until_ready(timeout=0.1)
        self._report_index_status(index_ready)

    def _build_retriever(self, *, use_rerank: bool, rebuild: bool) -> Retriever:
        attempt_args = dict(
            model_path=self.model_path,
            corpus_path=self.corpus_path,
            cache_dir=self.cache_dir,
            use_rerank=use_rerank,
            rerank_model=self.rerank_model,
            rerank_depth=self.rerank_depth,
            rerank_batch_size=self.rerank_batch_size,
            rerank_device=self.rerank_device,
            rerank_min_score=self.rerank_min_score,
            lexical_weight=self.lexical_weight,
            min_similarity=self.min_similarity,
        )
        try:
            retr = Retriever(**attempt_args)
            retr.ready(rebuild=rebuild, wait=False)
            return retr
        except RuntimeError as exc:
            message = str(exc).lower()
            meta_issue = "meta tensor" in message or "to_empty" in message
            if not meta_issue or not use_rerank:
                raise
            print("⚠️ CrossEncoder 로드 실패로 재정렬 기능을 비활성화하고 다시 시도합니다.")
            self.rerank = False
            attempt_args["use_rerank"] = False
            retr = Retriever(**attempt_args)
            retr.ready(rebuild=rebuild, wait=False)
            return retr

    def _ensure_preview_translator(self):
        if not self.show_translation:
            return None
        if self.preview_translator is not None:
            return self.preview_translator
        if GoogleTranslator is None:
            print("⚠️ 미리보기 번역을 위해 'deep-translator'가 필요하지만 설치되어 있지 않습니다.")
            self.show_translation = False
            return None
        try:
            self.preview_translator = GoogleTranslator(source="auto", target=self.translation_lang)
        except Exception as exc:
            print(f"⚠️ 미리보기 번역기 초기화 실패 → 번역 미표시: {exc}")
            self.preview_translator = None
            self.show_translation = False
        return self.preview_translator

    def _report_index_status(self, ready_flag: bool) -> None:
        translator_state = "활성" if self.translator else "비활성"
        index = None
        last_error: Optional[BaseException] = None
        if self.retr is not None:
            try:
                index = self.retr.index_manager.get_index(wait=False)
                last_error = self.retr.index_manager.last_error
            except Exception as exc:  # defensive guard
                last_error = exc

        self.index_loaded = index is not None
        self.index_reasons = []

        if self.index_loaded:
            doc_count = len(getattr(index, "doc_ids", []) or [])
            print(f"✅ LNP Chat 준비 완료 (문서 {doc_count:,}건 · 번역: {translator_state})")
            return

        if not ready_flag:
            self.index_reasons.append("인덱스를 아직 구축 중입니다. 잠시만 기다려주세요.")

        if last_error:
            msg = str(last_error).strip() or last_error.__class__.__name__
            if "유효 텍스트 문서가 없습니다" in msg:
                msg = "학습된 문서가 없어 인덱스를 만들 수 없습니다. scan/train 결과를 확인해주세요."
            self.index_reasons.append(msg)

        if not self.corpus_path.exists():
            self.index_reasons.append(f"코퍼스가 없습니다 → {self.corpus_path}")
        else:
            try:
                if self.corpus_path.stat().st_size == 0:
                    self.index_reasons.append(f"코퍼스가 비어 있습니다 → {self.corpus_path}")
            except OSError as exc:
                self.index_reasons.append(f"코퍼스 확인 실패: {exc}")

        cache_hint_added = False
        if not self.cache_dir.exists():
            self.index_reasons.append(f"인덱스 캐시 디렉터리가 없습니다 → {self.cache_dir}")
            cache_hint_added = True
        else:
            try:
                next(self.cache_dir.iterdir())
            except StopIteration:
                self.index_reasons.append("index_cache 디렉터리가 비어 있습니다.")
                cache_hint_added = True
            except OSError as exc:
                self.index_reasons.append(f"index_cache 확인 실패: {exc}")
                cache_hint_added = True

        self.index_reasons.append("python infopilot.py pipeline all --out data/found_files.csv 로 scan/train을 다시 실행해보세요.")
        if cache_hint_added:
            self.index_reasons.append("파이프라인 완료 후 --cache 옵션을 chat 명령과 동일하게 지정했는지 확인해주세요.")

        print("⚠️ 인덱스를 준비하지 못했습니다. (번역: " + translator_state + ")")
        for reason in self.index_reasons:
            print("   - " + reason)

    def _extract_context_terms(self) -> List[str]:
        terms: List[str] = []
        for hit in self.last_hits[:3]:
            raw_path = str(hit.get("path") or "").strip()
            if not raw_path:
                continue
            try:
                stem = Path(raw_path).stem
            except Exception:
                stem = raw_path
            if stem:
                terms.append(stem)
        return terms

    def _rewrite_query(self, query: str, tokens: Set[str]) -> Tuple[str, bool]:
        context_terms = self._extract_context_terms()
        return self.prompt_manager.rewrite_query(
            query,
            tokens,
            last_query=self.last_query_text or self.memory.last_user_text(),
            context_terms=context_terms,
        )

    def _augment_translations(self, hits: List[Dict[str, Any]]) -> None:
        if not (self.show_translation and hits):
            return
        cache = self.translation_cache
        if cache is None:
            return
        translator = self._ensure_preview_translator()
        for hit in hits:
            preview = str(hit.get("preview") or "").strip()
            if not preview:
                continue
            path = str(hit.get("path") or "")
            cached = cache.get(path, preview, self.translation_lang)
            if cached:
                hit["translation"] = cached
                continue
            if translator is None:
                continue
            try:
                translated = translator.translate(preview)
                if not isinstance(translated, str):
                    translated = getattr(translated, "text", "")
                translated = str(translated or "").strip()
                if translated:
                    cache.set(path, preview, self.translation_lang, translated)
                    hit["translation"] = translated
            except Exception as exc:
                print(f"⚠️ 문장 번역 실패(미리보기 유지): {exc}")

    @staticmethod
    def _highlight_preview(preview: str, query_tokens: Set[str]) -> str:
        if not preview or not query_tokens:
            return preview

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            return f"<<{token}>>" if token.lower() in query_tokens else token

        return _PREVIEW_TOKEN_PATTERN.sub(_replace, preview)

    @staticmethod
    def _wrap_preview(text: str, *, width: int = 140, limit: int = 2) -> List[str]:
        if not text:
            return []
        words = text.split()
        if not words:
            return [text.strip()][:limit]

        lines: List[str] = []
        current: List[str] = []

        for word in words:
            candidate = " ".join(current + [word]).strip()
            if len(candidate) <= width:
                current.append(word)
                continue
            if current:
                lines.append(" ".join(current))
            current = [word]
            if len(lines) >= limit:
                break

        if current and len(lines) < limit:
            lines.append(" ".join(current))

        if len(lines) > limit:
            lines = lines[:limit]

        if len(lines) == limit and len(words) > sum(len(line.split()) for line in lines):
            lines[-1] = lines[-1].rstrip() + " …"

        return lines

    def _ensure_ready(self) -> None:
        if not self.ready_done:
            self.ready(rebuild=False)
        if self.retr is None:
            raise RuntimeError("검색기 초기화에 실패했습니다. LNPChat.ready() 호출 결과를 확인하세요.")

    # 한 턴 처리
    def ask(self, query: str, topk: Optional[int] = None, *, force_action: Optional[str] = None) -> Dict[str, Any]:
        k = topk or self.topk
        query_tokens = {tok.lower() for tok in _split_tokens(query) if tok}
        consent_ack: Optional[str] = None
        stored_action: Optional[str] = None
        log_user_query = True

        normalized_query = query.strip().lower()
        normalized_head = normalized_query.split(maxsplit=1)[0] if normalized_query else ""
        if normalized_head in {"/quit", "/exit", "/dialogue"}:
            self.pending_search = None
            self._reset_llm_client()
            response = "검색 모드를 종료했어요. 이제 자유롭게 대화를 이어가세요."
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=response)
            return {
                "answer": response,
                "hits": [],
                "suggestions": ["무엇을 도와드릴까요?"],
            }

        if self.pending_search:
            if self._is_affirmative(query_tokens):
                confirmation = query.strip()
                if confirmation:
                    self.memory.add_turn(role="user", text=confirmation)
                stored = self.pending_search
                self.pending_search = None
                query = stored.get("query", query)
                k = stored.get("topk", k)
                query_tokens = {tok.lower() for tok in _split_tokens(query) if tok}
                stored_action = stored.get("action", "search")
                consent_ack = "네, 관련 문서를 찾아볼게요."
                log_user_query = False  # 원본 질문은 이미 기록됨
            elif self._is_negative(query_tokens):
                self.pending_search = None
                response = "알겠습니다. 문서 검색은 생략할게요."
                self.memory.add_turn(role="user", text=query)
                self.memory.add_turn(role="assistant", text=response)
                return {
                    "answer": response,
                    "hits": [],
                    "suggestions": ["다른 질문을 해보세요."],
                }
            else:
                self.pending_search = None

        hit_reference = self._respond_to_hit_reference(query)
        if hit_reference is not None:
            return hit_reference

        followup_reference = self._respond_to_selected_hit_followup(query)
        if followup_reference is not None:
            return followup_reference

        if self.llm_client is None and self._is_small_talk(query, query_tokens):
            friendly_answer = self._small_talk_reply(query)
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=friendly_answer)
            return {
                "answer": friendly_answer,
                "hits": [],
                "suggestions": ["문서를 검색하려면 궁금한 내용을 조금 더 구체적으로 적어 주세요."],
            }
        if self.llm_client is not None and self._is_small_talk(query, query_tokens):
            llm_reply = self._llm_chat_reply(query, mode="small_talk")
            if not llm_reply:
                llm_reply = self._small_talk_reply(query)
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=llm_reply)
            return {
                "answer": llm_reply,
                "hits": [],
                "suggestions": [],
                "llm_summary": llm_reply,
            }
        effective_policy = self._resolve_policy_engine()
        self._policy_effective = bool(effective_policy)

        if stored_action is not None:
            action = stored_action
        else:
            action = self.tool_router.select_action(
                query,
                use_translation=bool(self.translator),
                policy_active=bool(effective_policy),
                llm_available=self.llm_client is not None,
            )

        if force_action:
            forced = force_action.strip().lower()
            if forced in {"dialogue", "search", "search_and_summarize"}:
                action = forced
        elif (
            not self.auto_search
            and stored_action is None
            and action not in {"search", "search_and_summarize"}
        ):
            action = "dialogue"

        # 검색/요약이 실제로 필요할 때만 준비 비용을 지불한다.
        search_mode = action in {"search", "search_and_summarize"}
        if search_mode:
            self._ensure_ready()

        # [번역 기능] 사용자 질문을 영어로 번역
        contextual_query, used_context = self._rewrite_query(query, query_tokens)
        if used_context and search_mode:
            print(f"  (이전 질문 맥락을 반영해 '{contextual_query}'로 검색합니다.)")

        debug_query = query.strip().replace("\n", " ")
        if len(debug_query) > 60:
            debug_query = debug_query[:57] + "..."
        print(
            f"[LNPChat] action={action} search_mode={search_mode} "
            f"pending={bool(self.pending_search)} force_action={force_action} query='{debug_query}'"
        )

        if self.llm_client is not None and action == "dialogue":
            llm_reply = self._llm_chat_reply(query, mode="dialogue")
            if not llm_reply:
                llm_reply = self._fallback_dialogue_reply(query)
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=llm_reply)
            return {
                "answer": llm_reply,
                "hits": [],
                "suggestions": [],
                "llm_summary": llm_reply,
            }

        if (
            stored_action is None
            and self.llm_client is not None
            and self.pending_search is None
            and self._should_request_search_consent(action, query=query, tokens=query_tokens)
        ):
            dialogue = self._llm_chat_reply(query, mode="dialogue")
            if not dialogue:
                dialogue = "요청을 이해했어요."
            consent_prompt = (
                f"{dialogue.strip()}\n\n📂 관련 문서를 더 찾아볼까요? (네/응/Yes 또는 아니오/No)"
            )
            self.pending_search = {"query": query, "action": action, "topk": k}
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=consent_prompt)
            return {
                "answer": consent_prompt,
                "hits": [],
                "suggestions": ["네", "아니오"],
            }

        query_for_search = contextual_query
        if search_mode and self.translator:
            try:
                translated = self.translator.translate(query_for_search)
                query_for_search = translated if isinstance(translated, str) else getattr(translated, "text", query_for_search)
                print(f"  (질문 번역: '{contextual_query}' → '{query_for_search}')")
            except Exception as e:
                print(f"\n[경고] 질문 번역 실패. 원본 질문으로 검색합니다. 오류: {e}")

        if search_mode:
            self.session_state.add_query(contextual_query)

        # 스피너로 즉시 “살아있음” 표시
        index_ready = False
        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            index_ready = self.retr.wait_until_ready(timeout=0.4)
            hits = self.retr.search(query_for_search, top_k=k, session=self.session_state)
        finally:
            spin.stop()
        dt = time.time() - t0

        if index_ready:
            index_obj = self.retr.index_manager.get_index(wait=False)
            if index_obj is not None:
                self.index_loaded = True
                self.index_reasons.clear()

        # 히스토리 적재 (원본 query 기준)
        if log_user_query:
            self.memory.add_turn(role="user", text=query)
        filtered_hits, filtered_count = self._apply_policy_scope(hits)
        self.last_hits = filtered_hits
        self.last_query_text = contextual_query
        self.last_selected_hit_index = None

        self._augment_translations(filtered_hits)
        hits = filtered_hits

        llm_summary = None
        if hits and action == "search_and_summarize" and self.llm_client is not None:
            llm_summary = self._summarize_hits(query, hits)

        # 답변 생성(원본 query 기준)
        policy_note = ""
        if self._policy_effective and filtered_count:
            policy_note = f" (정책으로 {filtered_count}건 제외)"
        answer_lines: List[str] = []
        if consent_ack:
            answer_lines.append(consent_ack)
        answer_lines.append(f"‘{query}’에 대한 추천 문서 Top {len(hits)} (검색 {dt:.2f}s){policy_note}:")
        for i, h in enumerate(hits, 1):
            semantic_pct = _similarity_to_percent(h.get("vector_similarity"))
            overall_pct = _similarity_to_percent(h.get("similarity", h.get("vector_similarity")))
            lexical_component = h.get("lexical_score")
            lexical_pct = _similarity_to_percent(lexical_component) if lexical_component is not None else None
            score_breakdown = h.get("score_breakdown") or {}
            rerank_component = score_breakdown.get("rerank", h.get("rerank_score"))
            chunk_id = h.get("chunk_id")
            chunk_count = h.get("chunk_count")
            chunk_tokens = h.get("chunk_tokens")

            path_label = str(h.get("path") or "")
            ext_label = str(h.get("ext") or "")
            answer_lines.append(f"{i}. {path_label} [{ext_label}]")

            detail_bits: List[str] = [f"overall={overall_pct}"]
            if semantic_pct:
                detail_bits.append(f"semantic={semantic_pct}")
            if isinstance(lexical_component, (int, float)) and lexical_component > 0:
                lexical_pct = lexical_pct or _similarity_to_percent(lexical_component)
                detail_bits.append(f"lexical={lexical_pct}")
            ext_bonus = score_breakdown.get("extension_bonus")
            if isinstance(ext_bonus, (int, float)) and ext_bonus > 0:
                detail_bits.append(f"ext+{ext_bonus:.2f}")
            if isinstance(rerank_component, (int, float)):
                detail_bits.append(f"rerank={rerank_component:.2f}")
            try:
                chunk_idx_val = int(chunk_id) if chunk_id is not None else None
            except (TypeError, ValueError):
                chunk_idx_val = None
            try:
                chunk_count_val = int(chunk_count) if chunk_count is not None else None
            except (TypeError, ValueError):
                chunk_count_val = None
            try:
                chunk_token_val = int(chunk_tokens) if chunk_tokens is not None else None
            except (TypeError, ValueError):
                chunk_token_val = None

            if chunk_idx_val is not None:
                chunk_info = f"chunk {chunk_idx_val}"
                if chunk_count_val:
                    chunk_info += f"/{chunk_count_val}"
                if chunk_token_val:
                    chunk_info += f" ≈{chunk_token_val} tokens"
                detail_bits.append(chunk_info)
            answer_lines.append("   ▸ " + " | ".join(detail_bits))

            reasons = h.get("match_reasons") or []
            if reasons:
                answer_lines.append("   근거: " + " · ".join(reasons[:4]))

            preview_raw = str(h.get("preview") or "").strip()
            if preview_raw:
                highlighted = self._highlight_preview(preview_raw, query_tokens)
                preview_lines = self._wrap_preview(highlighted)
                if preview_lines:
                    answer_lines.append("   미리보기:")
                    for line_text in preview_lines:
                        answer_lines.append(f"     {line_text}")
            translation_text = h.get("translation") if self.show_translation else None
            if translation_text:
                answer_lines.append(f"   번역({self.translation_lang}): {translation_text}")
        if not hits:
            llm_fallback = self._llm_chat_reply(query, mode="no_hits") if self.llm_client else None
            if llm_fallback:
                fallback_lines = [llm_fallback, ""]
                if consent_ack:
                    fallback_lines.insert(0, consent_ack)
                answer_lines = fallback_lines
            elif consent_ack:
                answer_lines = [consent_ack, ""]
            answer_lines.append("관련 문서를 찾지 못했습니다. 표현을 바꿔보거나 더 구체적으로 적어주세요.")
            if not self.index_loaded:
                answer_lines.append("현재 인덱스를 사용할 수 없어 검색이 제한됩니다:")
                for reason in self.index_reasons:
                    answer_lines.append(f"   - {reason}")
            else:
                answer_lines.append("데이터셋에 해당 문서가 없다면 검색 결과 0건이 정상입니다.")
            if not index_ready:
                answer_lines.append("(인덱스를 준비 중입니다. 잠시 후 다시 시도해주세요.)")
            elif self.rerank and self.rerank_min_score is not None:
                answer_lines.append(
                    f"(Cross-Encoder 점수 {self.rerank_min_score:.2f} 미만 결과는 버렸습니다.)"
                )
            elif self.min_similarity > 0.0:
                answer_lines.append(
                    f"(유사도 {self.min_similarity:.2f} 미만 결과는 자동으로 제외됩니다.)"
                )

        answer = "\n".join(answer_lines)
        if llm_summary:
            composed = [llm_summary.strip(), ""]
            composed.extend(answer_lines)
            answer = "\n".join(composed)
        self.memory.add_turn(role="assistant", text=answer, hits=hits)

        result = {
            "answer": answer,
            "hits": hits,
            "suggestions": self._suggest_followups(query, hits),
        }
        if llm_summary:
            result["llm_summary"] = llm_summary
        return result

    @staticmethod
    def _is_small_talk(query: str, tokens: Set[str]) -> bool:
        if not tokens:
            return False
        if len(tokens) > 5:
            return False
        normalized = {token.strip("!?.") for token in tokens if token}
        if any(token in _GREETINGS for token in normalized):
            return True
        for token in normalized:
            for greeting in _GREETINGS:
                if greeting and greeting in token:
                    return True
        compact_query = "".join(query.split()).strip("!?.")
        if compact_query and compact_query in _GREETINGS:
            return True
        if normalized and normalized.issubset({"안녕", "하세요"}):
            return True
        return False

    def _small_talk_reply(self, query: str) -> str:
        stripped = query.strip()
        if not stripped:
            return "안녕하세요! 무슨 이야기를 나눠볼까요?"
        if len(stripped) <= 6:
            return "그 말이 재미있네요! 이어서 어떤 이야기를 나누면 좋을까요?"
        return f"{stripped}라는 이야기, 더 들려주시면 함께 생각해 볼게요."

    def _fallback_dialogue_reply(self, query: str) -> str:
        stripped = (query or "").strip()
        last_assistant = self.memory.last_assistant_text() or ""
        if not stripped:
            return "무엇이 궁금하신지 이야기해 주시면 함께 생각해 볼게요."
        if len(stripped) <= 8:
            return "방금 이야기가 인상적이었어요. 어떤 마음으로 말한 건지 조금 더 들려줄래?"
        if "재밌" in stripped:
            return "나도 웃음이 나네! 어떤 부분이 그렇게 재미있었는지 궁금해."
        if "말하고" in stripped or "대화" in stripped:
            return "나도 지금 계속 대화 중이야. 이어서 궁금한 걸 이야기해 줄래?"
        if stripped.endswith("?") and len(stripped) <= 18:
            return f"{stripped.rstrip('?')}에 대해서는 네 생각이 어때? 조금 더 얘기해 줘."
        if last_assistant:
            sample = " ".join(last_assistant.split())
            if len(sample) > 28:
                sample = sample[:28].rstrip() + "..."
            return f"방금 내가 \"{sample}\"이라고 했는데, 이어서 어떤 얘기를 나누면 좋을까?"
        return f"{stripped}라고 말해줘서 고마워. 조금만 더 자세히 이야기해 줄래?"

    def _compose_prompt(self, query: str, *, mode: str, include_history: bool, limit: int = 6) -> str:
        current = (query or "").strip()
        instruction_map = {
            "small_talk": (
                "친근하고 가볍게 대화를 이어가세요. 간결하면 좋지만 필요하면 최대 1000자까지 자세히 설명해도 됩니다. "
                "사실이 아닌 내용을 지어내지 말고, 모르는 내용은 솔직히 모른다고 답하세요."
            ),
            "no_hits": (
                "사용자에게 도움이 될 일반 지식을 한국어로 알려주세요. 핵심 위주로 정리하되 필요하면 1000자 이내에서 충분히 풀어 설명하세요. "
                "근거가 없는 내용은 절대 만들지 말고, 확실하지 않다면 모른다고 말하세요."
            ),
            "dialogue": (
                "이전 대화 흐름을 이어 자연스럽게 한국어로 답하세요. 가능한 간결하게, 하지만 필요한 경우 최대 1000자까지 자세히 쓸 수 있습니다. "
                "문서 검색이나 시스템 설명은 언급하지 말고, 사실이 확인되지 않은 내용은 지어내지 마세요."
            ),
        }
        instruction = instruction_map.get(
            mode,
            "친근하고 명확하게 한국어로 답하세요. 필요하면 최대 1000자까지 자세히 설명하되, 확인되지 않은 정보를 지어내지 말고 모르면 모른다고 답하세요.",
        )
        # 히스토리 주입을 제거하여 모델이 스스로 대화를 생성하는 환각(Hallucination) 방지
        # Llama 모델은 프롬프트에 'User:' 패턴이 포함되면 스스로 다음 턴을 예측하려는 경향이 있음
        return f"<|system|>\n{instruction}\n</s>\n<|user|>\n{current}\n</s>\n<|assistant|>\n"

    @staticmethod
    def _is_affirmative(tokens: Set[str]) -> bool:
        positives = {"응", "네", "예", "맞아", "좋아", "yes", "y", "sure", "ok", "그래", "ㅇㅋ", "넵"}
        return any(token in positives for token in tokens)

    @staticmethod
    def _is_negative(tokens: Set[str]) -> bool:
        negatives = {"아니", "아니오", "no", "n", "싫어", "괜찮아", "아냐", "안돼", "안해", "노"}
        return any(token in negatives for token in tokens)

    @staticmethod
    def _has_document_intent(query: str, tokens: Set[str]) -> bool:
        lowered_query = query.lower()
        normalized_tokens = {token.lower() for token in tokens if token}
        if lowered_query.startswith(("/search", "/doc")):
            return True

        doc_terms = {
            "문서", "자료", "파일", "보고서", "리포트", "레포트", "policy", "document", "documents",
            "documentation", "pdf", "ppt", "pptx", "자료집", "dataset", "데이터셋", "정책", "규정",
        }
        action_terms = {
            "찾아", "찾아줘", "검색", "search", "scan", "살펴", "추려", "추천", "list", "show",
            "요약", "요약해", "정리", "정리해", "정리해줘", "요약본", "알려줘",
        }

        has_doc = any(term in lowered_query for term in doc_terms) or bool(normalized_tokens & doc_terms)
        has_action = any(term in lowered_query for term in action_terms) or bool(normalized_tokens & action_terms)
        return has_doc and has_action

    def _should_use_llm_chat(self, query: str, tokens: Set[str]) -> bool:
        stripped = (query or "").strip()
        if not stripped:
            return False
        if self._has_document_intent(query, tokens):
            return False
        if self._is_small_talk(query, tokens):
            return True
        if len(stripped) <= 48:
            return True
        return "?" in stripped

    def _should_request_search_consent(self, action: str, *, query: str, tokens: Set[str]) -> bool:
        if action not in {"search", "search_and_summarize"}:
            return False
        normalized = (query or "").strip()
        if not normalized:
            return False
        if normalized.lower().startswith(("/search", "/doc")):
            return False
        if self._is_small_talk(query, tokens):
            return False
        if not self._has_document_intent(query, tokens):
            return False
        return True

    def _llm_chat_reply(self, query: str, *, mode: str) -> Optional[str]:
        client = self.llm_client
        if client is None:
            return None
        valid_modes = {"small_talk", "no_hits", "dialogue"}
        if mode not in valid_modes:
            mode = "dialogue"

        def _generate_with_retry(include_history: bool) -> Optional[str]:
            attempts = 0
            while attempts < 2:
                attempts += 1
                active = self.llm_client
                if active is None:
                    return None
                prompt = self._compose_prompt(query, mode=mode, include_history=include_history)
                try:
                    return active.generate(prompt, timeout=self._llm_timeout()).strip()
                except (LLMClientError, Exception) as exc:
                    label = "응답 실패" if isinstance(exc, LLMClientError) else "예외"
                    print(f"⚠️ 로컬 LLM {label}({mode}): {exc}")
                    if attempts >= 2 or not self._reset_llm_client():
                        return None
                    print("⟳ 로컬 LLM을 재연결한 뒤 다시 시도합니다...")
            return None

        reply = _generate_with_retry(include_history=True)
        if reply:
            return reply
        return _generate_with_retry(include_history=False)

    def _summarize_hits(self, query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
        client = self.llm_client
        if client is None:
            return None
        context_blocks: List[str] = []
        for idx, hit in enumerate(hits[:3], start=1):
            path_label = str(hit.get("path") or "")
            ext_label = str(hit.get("ext") or "")
            preview = str(hit.get("preview") or "").strip()
            snippet = preview[:400]
            block = textwrap.dedent(
                f"""
                {idx}. 경로: {path_label} [{ext_label}]
                   요약: {snippet}
                """
            ).strip()
            context_blocks.append(block)
        if not context_blocks:
            return None
        prompt = textwrap.dedent(
            f"""
            사용자 질문: {query}

            검색 결과 요약:
            {os.linesep.join(context_blocks)}

            위 정보를 근거로 질문에 명확하고 간결하게 답변해주세요.
            핵심 근거를 bullet 형식으로 제시하고, 부족한 정보가 있으면 추가 조사 필요성을 언급하세요.
            """
        ).strip()
        system_prompt = "You are a helpful assistant that summarises enterprise documents in Korean."
        try:
            summary = client.generate(prompt, system=system_prompt, timeout=self._llm_timeout()).strip()
        except LLMClientError as exc:
            print(f"⚠️ 로컬 LLM 응답 실패: {exc}")
            return None
        except Exception as exc:
            print(f"⚠️ 로컬 LLM 예외: {exc}")
            return None
        return summary or None

    def _respond_to_hit_reference(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.last_hits:
            return None
        ref_index = self._parse_hit_reference(query)
        if ref_index is None:
            return None
        total_hits = len(self.last_hits)
        if ref_index < 0 or ref_index >= total_hits:
            response = f"{ref_index + 1}번 문서는 찾을 수 없어요. 최근 검색 결과는 {total_hits}건입니다."
            self.memory.add_turn(role="user", text=query)
            self.memory.add_turn(role="assistant", text=response)
            return {
                "answer": response,
                "hits": self.last_hits,
                "suggestions": ["다른 번호를 지정해줘", "새로 검색해줘"],
            }
        hit = self.last_hits[ref_index]
        summary = self._summarize_hit_reference(query, hit, ref_index)
        self.memory.add_turn(role="user", text=query)
        self.memory.add_turn(role="assistant", text=summary, hits=[hit])
        self.last_selected_hit_index = ref_index
        suggestions = ["다른 문서도 설명해줘", "새 검색 시작"] if total_hits > 1 else ["새 검색 시작"]
        return {
            "answer": summary,
            "hits": [hit],
            "suggestions": suggestions,
        }

    def _respond_to_selected_hit_followup(self, query: str) -> Optional[Dict[str, Any]]:
        if (
            self.last_selected_hit_index is None
            or not self.last_hits
            or self.last_selected_hit_index >= len(self.last_hits)
        ):
            return None
        if self._parse_hit_reference(query) is not None:
            return None
        lowered = query.strip().lower()
        if not lowered:
            return None
        if not any(token in lowered for token in {"파일", "문서", "자료"}):
            return None
        verb_markers = {"열어", "확인", "보여", "설명", "읽어", "자세히", "다시", "봐봐", "검토"}
        if not any(marker in lowered for marker in verb_markers):
            return None
        idx = self.last_selected_hit_index
        hit = self.last_hits[idx]
        summary = self._summarize_hit_reference(query, hit, idx)
        self.memory.add_turn(role="user", text=query)
        self.memory.add_turn(role="assistant", text=summary, hits=[hit])
        return {
            "answer": summary,
            "hits": [hit],
            "suggestions": ["다른 문서도 설명해줘", "새 검색 시작"],
        }

    def _parse_hit_reference(self, query: str) -> Optional[int]:
        if not (query and query.strip()):
            return None
        trimmed = query.strip()
        lowered = trimmed.lower()
        if lowered.startswith("/doc"):
            parts = lowered.split()
            for part in parts[1:]:
                if part.isdigit():
                    return int(part) - 1
        match = re.search(r"(\d{1,3})\s*(?:번|번째)\s*(?:문서|파일|자료)?", trimmed)
        if match:
            return int(match.group(1)) - 1
        match = re.search(r"(?:file|doc)\s*(\d{1,3})", lowered)
        if match:
            return int(match.group(1)) - 1
        return None

    def _summarize_hit_reference(self, query: str, hit: Dict[str, Any], index: int) -> str:
        doc_path = str(hit.get("path") or "")
        doc_name = Path(doc_path).name or doc_path or f"{index + 1}번 문서"
        llm_summary = None
        if self._should_use_llm_for_hit(hit):
            llm_summary = self._summarize_hits(query, [hit])
        preview = (
            hit.get("preview")
            or hit.get("summary")
            or hit.get("chunk_text")
            or hit.get("text")
            or ""
        ).strip()
        if not preview:
            preview = self._load_hit_preview_from_file(hit) or ""
        snippet = ""
        if preview:
            snippet = preview[:800].strip()
        detail_lines: List[str] = [f"{index + 1}번 문서 '{doc_name}' 요약:"]
        if llm_summary:
            detail_lines.append(llm_summary.strip())
        elif snippet:
            detail_lines.append(snippet)
        else:
            detail_lines.append("문서 내용을 미리보기에서 찾지 못했습니다. 원본 파일을 직접 열어보세요.")
        extra_bits: List[str] = []
        similarity = hit.get("similarity", hit.get("vector_similarity"))
        similarity_pct = _similarity_to_percent(similarity) if similarity is not None else None
        if similarity_pct:
            extra_bits.append(f"유사도 {similarity_pct}")
        chunk_id = hit.get("chunk_id")
        chunk_count = hit.get("chunk_count")
        if chunk_id is not None and chunk_count:
            try:
                idx_val = int(chunk_id)
                extra_bits.append(f"chunk {idx_val}/{chunk_count}")
            except (TypeError, ValueError):
                pass
        if extra_bits:
            detail_lines.append("")
            detail_lines.append("세부 정보: " + " · ".join(extra_bits))
        if doc_path:
            detail_lines.append(f"파일 경로: {doc_path}")
        return "\n".join(detail_lines)

    def _load_hit_preview_from_file(self, hit: Dict[str, Any], *, max_chars: int = 1600) -> Optional[str]:
        path = hit.get("path")
        if not path:
            return None
        try:
            file_path = Path(path)
        except Exception:
            return None
        if not file_path.exists() or not file_path.is_file():
            return None
        text_extensions = {
            ".txt",
            ".md",
            ".rst",
            ".log",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".tsv",
            ".py",
            ".ipynb",
            ".sql",
        }
        if file_path.suffix.lower() not in text_extensions:
            return None
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                chunk = fh.read(max_chars)
                return chunk.strip()
        except Exception as exc:
            print(f"⚠️ 파일 미리보기 로드 실패: {file_path}: {exc}")
            return None

    def _should_use_llm_for_hit(self, hit: Dict[str, Any]) -> bool:
        if self.llm_client is None:
            return False
        ext = str(hit.get("ext") or "").lower()
        text_extensions = {
            ".txt",
            ".md",
            ".rst",
            ".log",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".tsv",
            ".py",
            ".ipynb",
            ".sql",
        }
        return ext not in text_extensions

    def _llm_timeout(self) -> float:
        try:
            return max(1.0, min(30.0, float(self.llm_timeout)))
        except Exception:
            return 8.0

    def _resolve_policy_engine(self) -> Optional[PolicyEngine]:
        engine = self.policy_engine
        if engine is None:
            return None
        try:
            if not engine.has_policies:
                return None
        except AttributeError:
            return None
        scope = (self.policy_scope or "auto").strip().lower()
        if scope == "global":
            return None
        if scope == "policy":
            return engine
        # auto
        return engine

    def _apply_policy_scope(self, hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        engine = self._resolve_policy_engine()
        if not engine:
            return hits, 0
        filtered: List[Dict[str, Any]] = []
        for hit in hits:
            path_val = hit.get("path")
            if not path_val:
                continue
            try:
                path_obj = Path(str(path_val))
            except Exception:
                continue
            if engine.allows(path_obj, agent=self.policy_agent, include_manual=False):
                filtered.append(hit)
        return filtered, max(0, len(hits) - len(filtered))

    # 후속 질문 제안
    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = []
        if hits:
            exts = {h["ext"].lower() for h in hits}
            if any(x in exts for x in [".xlsx", ".xls", ".xlsm", ".csv"]):
                base.append("표/컬럼 이름을 기준으로 다시 좁혀줘")
            if any(x in exts for x in [".pdf", ".ppt", ".pptx", ".doc", ".hwp"]):
                base.append("요약/키워드 중심으로 비슷한 문서 더 보여줘")
            base.append("기간(년도/월) 조건을 추가해서 다시 찾아줘")
            base.append("파일명에 포함된 키워드로 재검색")
        else:
            base.append("다른 표현으로 같은 의미의 질의를 시도")
            base.append("문서 유형(엑셀/한글/PDF 등)을 지정해서 검색")
        seen, out = set(), []
        for suggestion in base:
            if suggestion not in seen:
                out.append(suggestion)
                seen.add(suggestion)
        return out[:3]
