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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from retriever import Retriever, _similarity_to_percent, _split_tokens  # Step3 검색기 재사용

_PREVIEW_TOKEN_PATTERN = re.compile(r"(?u)(?:[가-힣]{1,}|[A-Za-z0-9]{2,})")
from translation_cache import TranslationCache

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
    def start(self):
        if self._t: return
        def _run():
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                print(f"\r{self.prefix} {frame} ", end="", flush=True)
        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()
    def stop(self, clear=True):
        if not self._t: return
        self._stop.set()
        self._t.join()
        if clear:
            print("\r" + " " * 80 + "\r", end="", flush=True)

# ──────────────────────────
# 대화 상태
# ──────────────────────────
@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    model_path: Path
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 5
    translate: bool = False # 기본은 다국어 Sentence-BERT로 번역 없이 처리
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = 80
    rerank_batch_size: int = 16
    rerank_device: Optional[str] = None
    rerank_min_score: Optional[float] = 0.35
    lexical_weight: float = 0.0
    show_translation: bool = False
    translation_lang: str = "en"
    min_similarity: float = 0.35

    retr: Optional[Retriever] = field(init=False, default=None)
    translator: Optional[Any] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)
    translation_cache: Optional[TranslationCache] = field(init=False, default=None)
    preview_translator: Optional[Any] = field(init=False, default=None)

    # 초기화: Retriever 및 번역기 준비
    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="인덱스 준비")
        spin.start()
        try:
            self.retr = Retriever(
                model_path=self.model_path,
                corpus_path=self.corpus_path,
                cache_dir=self.cache_dir,
                use_rerank=self.rerank,
                rerank_model=self.rerank_model,
                rerank_depth=self.rerank_depth,
                rerank_batch_size=self.rerank_batch_size,
                rerank_device=self.rerank_device,
                rerank_min_score=self.rerank_min_score,
                lexical_weight=self.lexical_weight,
                min_similarity=self.min_similarity,
            )
            self.retr.ready(rebuild=rebuild, wait=False)
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
        finally:
            spin.stop()
        if self.retr.wait_until_ready(timeout=0.1):
            print("✅ LNP Chat 준비 완료 (번역: " + ("활성" if self.translator else "비활성") + ")")
        else:
            print("⏳ 인덱스를 백그라운드에서 준비 중입니다. (번역: " + ("활성" if self.translator else "비활성") + ")")

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

    # 한 턴 처리
    def ask(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        if not self.ready_done:
            self.ready(rebuild=False)
        k = topk or self.topk
        query_tokens = {tok.lower() for tok in _split_tokens(query) if tok}

        # [번역 기능] 사용자 질문을 영어로 번역
        query_for_search = query
        if self.translator:
            try:
                translated = self.translator.translate(query)
                query_for_search = translated if isinstance(translated, str) else getattr(translated, "text", query)
                print(f"  (질문 번역: '{query}' → '{query_for_search}')")
            except Exception as e:
                print(f"\n[경고] 질문 번역 실패. 원본 질문으로 검색합니다. 오류: {e}")

        # 스피너로 즉시 “살아있음” 표시
        index_ready = False
        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            index_ready = self.retr.wait_until_ready(timeout=0.4)
            hits = self.retr.search(query_for_search, top_k=k)
        finally:
            spin.stop()
        dt = time.time() - t0

        # 히스토리 적재 (원본 query 기준)
        self.history.append(ChatTurn(role="user", text=query))

        self._augment_translations(hits)

        # 답변 생성(원본 query 기준)
        answer_lines = [f"‘{query}’에 대한 추천 문서 Top {len(hits)} (검색 {dt:.2f}s):"]
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
            answer_lines.append("관련 문서를 찾지 못했습니다. 표현을 바꿔보거나 더 구체적으로 적어주세요.")
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
        self.history.append(ChatTurn(role="assistant", text=answer, hits=hits))

        return {
            "answer": answer,
            "hits": hits,
            "suggestions": self._suggest_followups(query, hits),
        }

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
        for s in base:
            if s not in seen:
                out.append(s); seen.add(s)
        return out[:3]
