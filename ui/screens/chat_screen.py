import customtkinter as ctk
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import threading
from core.data_pipeline.policies.engine import PolicyEngine

# Core logic and helpers
from src.core.helpers import parse_query_and_filters, have_all_artifacts
from src.config import (
    CORPUS_PARQUET,
    CACHE_DIR,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    TOPIC_MODEL_PATH,
)
from src.core.retrieval import Retriever
from ui.smart_folder_context import SmartFolderContext
from ui.policy_cache import get_policy_engine


def _path_within(path: Path, root: Path) -> bool:
    try:
        # Python 3.9+: Path.is_relative_to
        return path.resolve().is_relative_to(root.resolve())  # type: ignore[attr-defined]
    except AttributeError:
        try:
            path_resolved = path.resolve()
            root_resolved = root.resolve()
        except Exception:
            return str(path).startswith(str(root))
        try:
            path_resolved.relative_to(root_resolved)
            return True
        except ValueError:
            return False
    except Exception:
        return False


@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path
    topk: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    retr: Optional[Retriever] = field(init=False, default=None)
    ready_done: bool = field(init=False, default=False)

    def ready(self, rebuild: bool = False):
        print("엔진 초기화 시작...")
        self.retr = Retriever(
            model_path=TOPIC_MODEL_PATH,
            corpus_path=self.corpus_path,
            cache_dir=self.cache_dir,
        )
        self.retr.ready(rebuild=rebuild)
        self.ready_done = True
        print("✅ LNP Chat 준비 완료")

    def ask(
        self,
        query: str,
        topk: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        path_prefix: Optional[Path] = None,
        policy_engine: Optional[PolicyEngine] = None,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.ready_done:
            self.ready(rebuild=False)

        k = topk or self.topk
        t0 = time.time()
        candidate_hits = self.retr.search(query, top_k=max(k * 2, 20))
        dt = time.time() - t0

        filtered_hits = [h for h in candidate_hits if h["similarity"] >= self.similarity_threshold]
        if path_prefix:
            prefix = path_prefix
            scoped_hits: List[Dict[str, Any]] = []
            for hit in filtered_hits:
                path_str = hit.get("path")
                if not path_str:
                    continue
                if _path_within(Path(path_str), prefix):
                    scoped_hits.append(hit)
            filtered_hits = scoped_hits

        blocked_hits = 0
        policy_active = policy_engine is not None and policy_engine.has_policies
        agent_name = (agent or self.policy_agent or "knowledge_search").strip()
        if policy_active:
            policy_hits: List[Dict[str, Any]] = []
            for hit in filtered_hits:
                path_str = hit.get("path")
                if not path_str:
                    blocked_hits += 1
                    continue
                hit_path = Path(path_str)
                try:
                    if policy_engine.allows(hit_path, agent=agent_name):
                        policy_hits.append(hit)
                    else:
                        blocked_hits += 1
                except Exception:
                    blocked_hits += 1
            filtered_hits = policy_hits

        final_hits = filtered_hits[:k]
        if not final_hits:
            if blocked_hits:
                answer_lines = [
                    f"정책에 의해 제외된 문서 {blocked_hits}건을 제외하면 ‘{query}’와 관련된 결과가 없습니다."
                ]
            else:
                answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [
                f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"
            ]
            for i, hit in enumerate(final_hits, 1):
                sim = f"{hit['similarity']:.3f}"
                answer_lines.append(f"{i}. {hit['path']}  (유사도: {sim})")
                if hit.get("summary"):
                    answer_lines.append(f"   요약: {hit['summary']}")
            if blocked_hits:
                answer_lines.append("")
                answer_lines.append(f"정책으로 제외된 문서 {blocked_hits}건이 목록에서 제거되었습니다.")

        return {
            "answer": "\n".join(answer_lines),
            "hits": final_hits,
            "policy_blocked": blocked_hits,
            "policy_enforced": policy_active,
        }


class ChatScreen(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app

        self.chat_engine = None
        self.active_context: Optional[SmartFolderContext] = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="지식·검색 비서",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=12, pady=(0, 4), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="자료를 자연어로 검색하고 결과 요약을 바로 확인하세요.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        self.scope_label = ctk.CTkLabel(
            self,
            text="🔍 검색 범위: 전체 (정책 미적용)",
            font=ctk.CTkFont(size=12),
            text_color=("#2f2f2f", "#cfcfcf"),
        )
        self.scope_label.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")

        self.context_warning_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#b3261e", "#ff9b9b"),
        )

        self.warning_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=15))
        self.train_button_redirect = ctk.CTkButton(
            self,
            text="🚀 전체 학습 실행",
            command=lambda: self.app.select_frame("train"),
        )

        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="예: 2024년 영업 보고서 요약 보여줘",
            height=40,
        )
        self.search_button = ctk.CTkButton(
            self.input_frame,
            text="검색",
            width=110,
            height=40,
            command=self.search_event,
        )

        self.results_textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(size=14),
            state="disabled",
        )

        self.refresh_state()

    def setup_ui(self):
        # This method is no longer directly called, its logic is integrated into refresh_state
        pass

    def refresh_state(self):
        # Clear previous state by forgetting grid layout
        self.warning_label.grid_forget()
        self.train_button_redirect.grid_forget()
        self.context_warning_label.grid_forget()
        self.input_frame.grid_forget()
        self.results_textbox.grid_forget()

        if not have_all_artifacts():
            self.warning_label.configure(text="⚠️ 학습 데이터가 없습니다. 먼저 전체 학습을 실행하세요.")
            self.warning_label.grid(row=3, column=0, pady=(60, 12))
            self.train_button_redirect.grid(row=4, column=0, pady=(0, 12))
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")
        else:
            # Re-create/show input_frame and results_textbox
            self.input_frame.grid(row=4, column=0, padx=12, pady=(0, 12), sticky="ew")
            self.search_entry.grid(row=0, column=0, sticky="ew")
            self.search_entry.bind("<Return>", self.search_event)
            self.search_button.grid(row=0, column=1, padx=(12, 0))
            self.results_textbox.grid(row=5, column=0, padx=12, pady=(0, 12), sticky="nsew")

            # Initialize chat engine if not already done
            if self.chat_engine is None or not self.chat_engine.ready_done:
                self.update_results("엔진을 초기화하는 중입니다... 잠시만 기다려주세요.")
                self.search_entry.configure(state="disabled")
                self.search_button.configure(state="disabled")
                threading.Thread(target=self.initialize_engine, daemon=True).start()
            else:
                self.update_results("질문을 입력하세요.")
                self.search_entry.configure(state="normal")
                self.search_button.configure(state="normal")

            self._apply_context_constraints()

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def initialize_engine(self):
        try:
            self.chat_engine = LNPChat(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
            self.chat_engine.ready()
            self.update_results("엔진 초기화 완료. 질문을 입력하세요.")
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")
            self._apply_context_constraints()
        except Exception as e:
            self.update_results(f"엔진 초기화 중 오류 발생: {e}")
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")

    def search_event(self, event=None):
        query = self.search_entry.get().strip()
        if not query or self.search_button.cget("state") == "disabled":
            return
        allowed, reason = self._context_allows_search()
        if not allowed:
            message = reason or "선택한 스마트 폴더에서는 지식·검색 비서를 사용할 수 없습니다. 다른 폴더를 선택하세요."
            self.update_results(message)
            if hasattr(self.app, "emit_work_center_event") and self.active_context is not None:
                try:
                    self.app.emit_work_center_event(
                        "knowledge.policy.blocked",
                        {"query": query, "reason": reason or "context_not_allowed"},
                        context=self.active_context,
                    )
                except Exception:
                    pass
            return

        self.search_entry.configure(state="disabled")
        self.search_button.configure(state="disabled")
        self.update_results(f"> {query}\n\n검색 중입니다...")

        # Run search in a thread to keep the UI responsive
        threading.Thread(target=self.run_search_thread, args=(query,), daemon=True).start()

    def run_search_thread(self, query):
        try:
            cleaned_query, filters = parse_query_and_filters(query)
            if self.chat_engine is None:
                raise RuntimeError("검색 엔진이 준비되지 않았습니다.")
            context_path = self.active_context.path if self.active_context and self.active_context.path else None
            policy_engine = self._policy_engine()
            result = self.chat_engine.ask(
                cleaned_query,
                filters=filters,
                path_prefix=context_path,
                policy_engine=policy_engine,
                agent="knowledge_search",
            )
            answer = result.get("answer", "오류가 발생했습니다.")
            self.update_results(f"> {query}\n\n{answer}")
            self._log_search_event(query, result)
        except Exception as e:
            self.update_results(f"> {query}\n\n검색 중 오류 발생: {e}")
        finally:
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")

    def update_results(self, text):
        self.after(0, self._do_update_results, text)

    def _do_update_results(self, text):
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", "end")
        self.results_textbox.insert("1.0", text)
        self.results_textbox.configure(state="disabled")

    # ------------------------------------------------------------------
    # Smart folder integration
    # ------------------------------------------------------------------
    def on_smart_folder_update(self, context: Optional[SmartFolderContext]) -> None:
        self.active_context = context
        if context is None:
            self.scope_label.configure(text="🔍 검색 범위: 전체 (정책 미적용)")
            self._apply_context_constraints()
            return

        scope = (context.scope or "").upper()
        parts = [f"🔍 검색 범위: {context.label}"]
        if scope:
            parts.append(f"· {scope}")
        if context.path:
            parts.append(f"· {context.path_display}")
        self.scope_label.configure(text=" ".join(parts))
        self._apply_context_constraints()

    def _log_search_event(self, query: str, result: Dict[str, Any]) -> None:
        if not hasattr(self.app, "emit_work_center_event"):
            return
        if self.active_context is None:
            return
        hits = result.get("hits") or []
        policy_blocked = int(result.get("policy_blocked") or 0)
        policy_enforced = bool(result.get("policy_enforced"))
        summary_hits = []
        for hit in hits[:3]:
            summary_hits.append(
                {
                    "path": hit.get("path"),
                    "similarity": hit.get("similarity"),
                }
            )
        try:
            self.app.emit_work_center_event(
                "knowledge.search.performed",
                {
                    "query": query,
                    "hit_count": len(hits),
                    "top_hits": summary_hits,
                    "policy_blocked": policy_blocked,
                    "policy_enforced": policy_enforced,
                },
                context=self.active_context,
            )
        except Exception:
            pass

    def _apply_context_constraints(self) -> None:
        self.context_warning_label.grid_forget()
        if not have_all_artifacts():
            return
        allowed, reason = self._context_allows_search()
        if not allowed:
            message = reason or "⚠️ 선택한 스마트 폴더에서는 지식·검색 비서를 사용할 수 없습니다."
            self.context_warning_label.configure(text=message)
            self.context_warning_label.grid(row=3, column=0, padx=12, pady=(0, 6), sticky="w")
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")
        else:
            if (
                have_all_artifacts()
                and self.chat_engine is not None
                and self.chat_engine.ready_done
                and self.search_button.cget("state") == "disabled"
            ):
                self.search_entry.configure(state="normal")
                self.search_button.configure(state="normal")

    def _context_allows_search(self) -> tuple[bool, Optional[str]]:
        if self.active_context is None:
            return True, None

        allowed = self.active_context.allows_agent("knowledge_search")
        reason: Optional[str] = None

        engine = self._policy_engine()
        if allowed and engine.has_policies and self.active_context.path is not None:
            try:
                allowed = engine.allows(self.active_context.path, agent="knowledge_search")
                if not allowed:
                    reason = "⚠️ 스마트 폴더 정책에 따라 지식·검색 비서가 제한되었습니다."
            except Exception:
                reason = "⚠️ 스마트 폴더 정책 확인 중 오류가 발생했습니다."
                allowed = False
        elif not allowed:
            reason = "⚠️ 이 스마트 폴더에서는 지식·검색 비서를 사용할 수 없습니다."

        return allowed, reason

    @staticmethod
    def _policy_engine() -> PolicyEngine:
        return get_policy_engine()
