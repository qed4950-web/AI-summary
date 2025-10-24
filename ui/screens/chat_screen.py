import customtkinter as ctk
import tkinter as tk
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import threading
from core.data_pipeline.policies.engine import PolicyEngine

# Core logic and helpers
from ui.utils import (
    parse_query_and_filters,
    have_all_artifacts,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    CORPUS_PARQUET,
    CACHE_DIR,
    TOPIC_MODEL_PATH,
)
from ui.smart_folder_context import SmartFolderContext
from ui.policy_cache import get_policy_engine
from core.conversation.lnp_chat import LNPChat as CoreLNPChat


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
            state="normal",
        )
        self.results_textbox.bind("<Key>", self._on_textbox_key)
        self.results_textbox.bind("<Button-1>", lambda e: None, add=True)
        self.results_textbox.bind("<Button-3>", self._show_textbox_menu)
        self.results_textbox_menu = tk.Menu(self, tearoff=0)
        self.results_textbox_menu.add_command(label="복사", command=self._copy_selection)
        self.results_textbox_menu.add_command(label="전체 선택", command=self._select_all)

        self.conversation_log: List[tuple[str, str]] = []

        self.refresh_state()

    def _ensure_chat_engine(self) -> None:
        if self.chat_engine is not None:
            return
        print("엔진 초기화 시작...")
        engine = CoreLNPChat(
            model_path=TOPIC_MODEL_PATH,
            corpus_path=CORPUS_PARQUET,
            cache_dir=CACHE_DIR,
            topk=DEFAULT_TOP_K,
            min_similarity=DEFAULT_SIMILARITY_THRESHOLD,
        )
        try:
            engine.ready(rebuild=False)
        except FileNotFoundError as exc:
            raise RuntimeError("지식 검색용 모델이 없습니다. 먼저 전체 학습을 실행하세요.") from exc
        except Exception as exc:
            raise RuntimeError(f"엔진 초기화에 실패했습니다: {exc}") from exc
        self.chat_engine = engine
        print("✅ LNP Chat 준비 완료")

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
            self.conversation_log.clear()
            self._refresh_conversation_display()
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
            if self.chat_engine is None or not getattr(self.chat_engine, "ready_done", False):
                self.conversation_log.clear()
                self._refresh_conversation_display()
                self._append_conversation("system", "엔진을 초기화하는 중입니다... 잠시만 기다려주세요.")
                self.search_entry.configure(state="disabled")
                self.search_button.configure(state="disabled")
                threading.Thread(target=self.initialize_engine, daemon=True).start()
            else:
                if not self.conversation_log:
                    self._append_conversation("system", "질문을 입력하세요.")
                self.search_entry.configure(state="normal")
                self.search_button.configure(state="normal")

            self._apply_context_constraints()

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def initialize_engine(self):
        try:
            self._ensure_chat_engine()
            self.conversation_log.clear()
            self._refresh_conversation_display()
            self._append_conversation("system", "엔진 초기화 완료. 질문을 입력하세요.")
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")
            self._apply_context_constraints()
        except Exception as e:
            self.conversation_log.clear()
            self._refresh_conversation_display()
            self._append_conversation("system", f"엔진 초기화 중 오류 발생: {e}")
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")

    def search_event(self, event=None):
        query = self.search_entry.get().strip()
        if not query or self.search_button.cget("state") == "disabled":
            return
        allowed, reason = self._context_allows_search()
        if not allowed:
            message = reason or "선택한 스마트 폴더에서는 지식·검색 비서를 사용할 수 없습니다. 다른 폴더를 선택하세요."
            self._append_conversation("system", message)
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
        self._append_conversation("user", query)

        # Run search in a thread to keep the UI responsive
        threading.Thread(target=self.run_search_thread, args=(query,), daemon=True).start()

    def run_search_thread(self, query):
        try:
            cleaned_query, _ = parse_query_and_filters(query)
            if self.chat_engine is None:
                raise RuntimeError("검색 엔진이 준비되지 않았습니다.")
            policy_engine = self._policy_engine()
            if policy_engine is not None:
                self.chat_engine.policy_engine = policy_engine
            result = self.chat_engine.ask(cleaned_query)
            hits = result.get("hits", [])
            scoped_hits = hits
            removed_by_scope = 0
            context_path = self.active_context.path if self.active_context and self.active_context.path else None
            if context_path:
                scoped_hits = [h for h in hits if _path_within(Path(str(h.get("path"))), context_path)]
                removed_by_scope = len(hits) - len(scoped_hits)
            formatted = self._compose_answer(
                query=query,
                hits=scoped_hits,
                llm_summary=result.get("llm_summary"),
                removed_by_scope=removed_by_scope,
                suggestions=result.get("suggestions"),
            )
            self._append_conversation("assistant", formatted)
            result_for_logging = dict(result)
            result_for_logging["hits"] = scoped_hits
            self._log_search_event(query, result_for_logging)
        except Exception as e:
            self._append_conversation("system", f"검색 중 오류 발생: {e}")
        finally:
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")

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
        if self.chat_engine is None and have_all_artifacts():
            try:
                self._ensure_chat_engine()
            except Exception as exc:
                self.conversation_log.clear()
                self._refresh_conversation_display()
                self._append_conversation("system", f"엔진 초기화 중 오류 발생: {exc}")
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

    def _compose_answer(
        self,
        *,
        query: str,
        hits: List[Dict[str, Any]],
        llm_summary: Optional[str],
        removed_by_scope: int,
        suggestions: Optional[List[str]],
    ) -> str:
        lines: List[str] = []
        summary_text = (llm_summary or "").strip()
        if summary_text:
            lines.append("🧠 요약")
            lines.append(summary_text)
            lines.append("")

        if not hits:
            lines.append(f"‘{query}’와 관련된 문서를 찾지 못했습니다.")
            if removed_by_scope:
                lines.append(f"(선택한 범위에서 {removed_by_scope}건이 제외되었습니다.)")
            return "\n".join(lines)

        header = f"‘{query}’ 관련 문서 Top {len(hits)}"
        if removed_by_scope:
            header += f" (범위 제외 {removed_by_scope}건)"
        lines.append(header)

        for idx, hit in enumerate(hits, 1):
            path = str(hit.get("path") or "")
            similarity = hit.get("similarity") or hit.get("vector_similarity") or 0.0
            try:
                sim_str = f"{float(similarity):.3f}"
            except Exception:
                sim_str = "-"
            lines.append(f"{idx}. {path} (유사도 {sim_str})")
            preview = str(hit.get("preview") or "").strip()
            if preview:
                snippet = preview.replace("\n", " ")[:200]
                if len(preview) > 200:
                    snippet += " …"
                lines.append(f"   {snippet}")

        if suggestions:
            cand = [s for s in suggestions if isinstance(s, str)]
            if cand:
                lines.append("")
                lines.append("다음 질문 추천:")
                for suggestion in cand[:3]:
                    lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def _append_conversation(self, role: str, text: str) -> None:
        def _do() -> None:
            content = (text or "").strip()
            prefix_map = {"user": "사용자", "assistant": "비서", "system": "시스템"}
            label = prefix_map.get(role, role)
            self.conversation_log.append((label, content))
            self._render_conversation_locked()

        self.after(0, _do)

    def _refresh_conversation_display(self) -> None:
        self.after(0, self._render_conversation_locked)

    def _render_conversation_locked(self) -> None:
        self.results_textbox.delete("1.0", "end")
        if not self.conversation_log:
            return
        for entry_label, entry_text in self.conversation_log:
            display_text = entry_text if entry_text else ""
            self.results_textbox.insert("end", f"{entry_label}: {display_text}\n\n")

    def _on_textbox_key(self, event):
        mods = event.state
        ctrl = bool(mods & 0x4)
        cmd = bool(mods & 0x10000)
        if (ctrl or cmd) and event.keysym.lower() in {"c", "x"}:
            return None
        if (ctrl or cmd) and event.keysym.lower() == "a":
            self._select_all()
            return "break"
        if event.keysym in {"Left", "Right", "Up", "Down", "Prior", "Next", "Home", "End"}:
            return None
        if event.keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R", "Command", "Option_L", "Option_R"}:
            return None
        return "break"

    def _copy_selection(self) -> None:
        try:
            text = self.results_textbox.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.results_textbox.get("1.0", "end-1c")
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _select_all(self) -> None:
        self.results_textbox.tag_add("sel", "1.0", "end-1c")

    def _show_textbox_menu(self, event: tk.Event) -> None:
        try:
            self.results_textbox_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.results_textbox_menu.grab_release()
