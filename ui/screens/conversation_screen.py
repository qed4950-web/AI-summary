import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import subprocess
import threading
import time

from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.agents.meeting import MeetingAgent
from core.agents.photo import PhotoAgent
from core.conversation.orchestrator import AssistantOrchestrator
from ui.utils import (
    CORPUS_PARQUET,
    CACHE_DIR,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    TOPIC_MODEL_PATH,
)
from ui.settings_manager import SettingsManager


SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "ui_settings.json"


class PathFormDialog(ctk.CTkToplevel):
    """Reusable dialog that lets users pick files or folders with history support."""

    def __init__(
        self,
        parent,
        *,
        title: str,
        message: str,
        mode: str = "file",
        allow_multiple: bool = False,
        history: Optional[List[str]] = None,
        filetypes: Optional[List[tuple[str, str]]] = None,
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.mode = mode
        self.allow_multiple = allow_multiple
        self.history = list(history or [])
        self.filetypes = filetypes
        self.result: Optional[str | List[str]] = None
        self._selected: List[str] = []

        self.title(title)
        self.geometry("520x360")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        label = ctk.CTkLabel(container, text=message, wraplength=460, justify="left")
        label.pack(fill="x", pady=(0, 12))

        if self.allow_multiple:
            self._listbox = ctk.CTkTextbox(container, height=130, state="disabled")
            self._listbox.pack(fill="both", expand=True)
            manual_frame = ctk.CTkFrame(container, fg_color="transparent")
            manual_frame.pack(fill="x", pady=(12, 8))
            self._manual_entry = ctk.CTkEntry(manual_frame, placeholder_text="경로를 입력하고 Enter를 누르면 추가됩니다.")
            self._manual_entry.pack(side="left", fill="x", expand=True)
            self._manual_entry.bind("<Return>", lambda _event: self._add_manual())
            ctk.CTkButton(manual_frame, text="추가", width=80, command=self._add_manual).pack(side="left", padx=(8, 0))
        else:
            self._value_var = ctk.StringVar()
            self._entry = ctk.CTkEntry(container, textvariable=self._value_var)
            self._entry.pack(fill="x", pady=(0, 12))
            self._entry.focus_set()

        if self.history:
            history_frame = ctk.CTkFrame(container, fg_color="transparent")
            history_frame.pack(fill="x", pady=(0, 12))
            history_label = ctk.CTkLabel(history_frame, text="최근 사용", font=ctk.CTkFont(size=12, weight="bold"))
            history_label.pack(anchor="w", pady=(0, 4))
            for item in self.history:
                btn = ctk.CTkButton(
                    history_frame,
                    text=item,
                    anchor="w",
                    command=lambda path=item: self._select_history(path),
                )
                btn.pack(fill="x", pady=2)

        button_row = ctk.CTkFrame(container, fg_color="transparent")
        button_row.pack(fill="x", pady=(8, 0))

        browse_text = "파일 선택" if mode == "file" else "폴더 선택"
        ctk.CTkButton(button_row, text=browse_text, width=110, command=self._browse).pack(side="left")

        right_group = ctk.CTkFrame(button_row, fg_color="transparent")
        right_group.pack(side="right")
        ctk.CTkButton(right_group, text="취소", width=90, command=self._on_cancel).pack(side="right")
        ctk.CTkButton(right_group, text="확인", width=90, command=self._on_confirm).pack(side="right", padx=(0, 8))

        self.after(10, self.focus_force)

    # ------------------------------------------------------------------
    def _select_history(self, value: str) -> None:
        if self.allow_multiple:
            self._add_path(value)
        else:
            self._value_var.set(value)

    def _add_manual(self) -> None:
        if not self.allow_multiple:
            return
        value = self._manual_entry.get().strip()
        if value:
            self._add_path(value)
        self._manual_entry.delete(0, "end")

    def _add_path(self, value: str) -> None:
        path = value.strip()
        if not path:
            return
        if self.allow_multiple:
            if path not in self._selected:
                self._selected.append(path)
                self._refresh_selected()
        else:
            self._value_var.set(path)

    def _refresh_selected(self) -> None:
        if not self.allow_multiple:
            return
        self._listbox.configure(state="normal")
        self._listbox.delete("1.0", "end")
        for item in self._selected:
            self._listbox.insert("end", item + "\n")
        self._listbox.configure(state="disabled")

    def _browse(self) -> None:
        if self.mode == "file":
            path = filedialog.askopenfilename(parent=self, filetypes=self.filetypes or [("All Files", "*.*")])
        else:
            path = filedialog.askdirectory(parent=self)
        if not path:
            return
        self._add_path(path)

    def _on_confirm(self) -> None:
        if self.allow_multiple:
            if not self._selected:
                self.bell()
                return
            self.result = list(self._selected)
        else:
            value = (self._value_var.get() if hasattr(self, "_value_var") else "").strip()
            if not value:
                self.bell()
                return
            self.result = value
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


class ConversationScreen(ctk.CTkFrame):
    """Full conversation assistant backed by LNPChat and optional local LLM."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.settings = SettingsManager(SETTINGS_PATH.resolve())
        backend_default = (self.settings.get("conversation", "llm_backend", default="") or "").strip()
        if not backend_default and shutil.which("ollama"):
            self.settings.set("ollama", "conversation", "llm_backend")
            model_default = (self.settings.get("conversation", "llm_model", default="") or "").strip()
            if not model_default:
                self.settings.set("llama3", "conversation", "llm_model")
        self.orchestrator: Optional[AssistantOrchestrator] = None
        self.history: List[tuple[str, str]] = []
        self.last_copyable_text: str = ""
        self._inflight = False
        self.recent_audio_files: List[str] = list(self.settings.get("agents", "meeting", "recent_audio_files", default=[]))
        self.recent_photo_roots: List[str] = list(self.settings.get("agents", "photo", "recent_roots", default=[]))
        self._active_cancel_event: Optional[threading.Event] = None
        self._pending_agent_label: Optional[str] = None
        self._task_started = False
        if sys.platform == "darwin":
            # Tk on macOS sets the Command/Option modifiers on the higher bits of event.state.
            # 0x100000 ≈ Command, 0x200000 ≈ Option/Alt. Keep a few aliases to cover layouts.
            self._command_masks = (0x100000, 0x200000, 0x10000)
        else:
            self._command_masks = ()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_footer()

        self.after(50, self._initialise_engine_async)
        self._reset_conversation()
        self._append_message("system", "엔진을 초기화하는 중입니다...")

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="대화 비서",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="문서를 찾아보고 요약까지 챗봇처럼 대화할 수 있습니다.",
            font=ctk.CTkFont(size=13),
            text_color=('#424242', '#c8c8c8'),
        )
        subtitle.grid(row=1, column=0, sticky="w")

        settings_btn = ctk.CTkButton(
            header,
            text="⚙️ 설정",
            width=96,
            command=self._open_settings_dialog,
        )
        settings_btn.grid(row=0, column=1, rowspan=2, padx=(12, 0))

    def _build_body(self) -> None:
        self.chat_display = ctk.CTkTextbox(self, state="normal", wrap="word")
        self.chat_display.grid(row=1, column=0, sticky="nsew")
        self.chat_display.tag_config("user", foreground="#1f5bb1")
        self.chat_display.tag_config("assistant", foreground="#2a9d8f")
        self.chat_display.tag_config("system", foreground="#b44141")
        self.chat_display.bind("<Key>", self._handle_display_key)
        self.chat_display.bind("<Button-3>", self._show_context_menu)
        self.chat_display.bind("<Control-c>", self._handle_copy_shortcut)
        self.chat_display.bind("<Control-C>", self._handle_copy_shortcut)
        self.chat_display.bind("<Command-c>", self._handle_copy_shortcut)
        self.chat_display.bind("<Command-C>", self._handle_copy_shortcut)
        self.chat_display.bind("<Meta-c>", self._handle_copy_shortcut)
        self.chat_display.bind("<Meta-C>", self._handle_copy_shortcut)
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="복사", command=self._copy_selection)
        self.menu.add_command(label="전체 선택", command=lambda: self.chat_display.tag_add("sel", "1.0", "end-1c"))

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        footer.grid_columnconfigure(0, weight=1)

        self.prompt_entry = ctk.CTkEntry(
            footer,
            placeholder_text="안녕하세요? 무엇을 도와드릴까요?",
        )
        self.prompt_entry.grid(row=0, column=0, sticky="ew")
        self.prompt_entry.bind("<Return>", self._handle_send)

        button_group = ctk.CTkFrame(footer, fg_color="transparent")
        button_group.grid(row=0, column=1, padx=(12, 0))
        button_group.grid_columnconfigure((0, 1, 2), weight=0)

        self.copy_button = ctk.CTkButton(button_group, text="복사", width=70, command=self._copy_last_response, state="disabled")
        self.copy_button.grid(row=0, column=0, padx=(0, 8))

        self.cancel_button = ctk.CTkButton(
            button_group,
            text="취소",
            width=80,
            command=self._request_cancel,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=(0, 8))

        send_btn = ctk.CTkButton(button_group, text="전송", width=90, command=self._handle_send)
        send_btn.grid(row=0, column=2)

    def _reset_conversation(self) -> None:
        self.history.clear()
        self.last_copyable_text = ""
        self.chat_display.delete("1.0", "end")
        if hasattr(self, "copy_button"):
            self.copy_button.configure(state="disabled")

    # ------------------------------------------------------------------
    def _initialise_engine_async(self) -> None:
        threading.Thread(target=self._ensure_engine, daemon=True).start()

    def _ensure_engine(self) -> None:
        if self.orchestrator is not None:
            return
        try:
            config = self._effective_settings()
            document_agent = DocumentAgent(
                DocumentAgentConfig(
                    model_path=TOPIC_MODEL_PATH,
                    corpus_path=CORPUS_PARQUET,
                    cache_dir=CACHE_DIR,
                    topk=config["top_k"],
                    min_similarity=config["min_similarity"],
                    llm_backend=config["llm_backend"] or None,
                    llm_model=config["llm_model"],
                    llm_host=config["llm_host"] or None,
                )
            )
            meeting_agent = MeetingAgent()
            photo_agent = PhotoAgent()
            self.orchestrator = AssistantOrchestrator(
                [document_agent, meeting_agent, photo_agent],
                llm_client=document_agent.llm_client,
            )
            if document_agent.llm_client is None:
                self._append_message(
                    "system",
                    "LLM이 설정되지 않아 문서 기반 안내만 제공됩니다. ⚙️ 설정에서 Ollama 등을 연결하면 자연어 답변을 받을 수 있습니다.",
                )
            else:
                self._append_message("system", "대화 비서가 준비되었습니다. 자유롭게 질문해 보세요.")
        except FileNotFoundError:
            self._append_message("system", "학습된 모델이 없어 대화 기능을 사용할 수 없습니다. 먼저 전체 학습을 실행하세요.")
        except Exception as exc:
            self._append_message("system", f"엔진 초기화 실패: {exc}")

    # ------------------------------------------------------------------
    def _handle_send(self, event=None) -> None:
        if self._inflight:
            return
        message = self.prompt_entry.get().strip()
        if not message:
            return
        self.prompt_entry.delete(0, "end")
        self._append_message("user", message)
        threading.Thread(target=self._run_conversation, args=(message,), daemon=True).start()

    def _run_conversation(self, message: str) -> None:
        if self.orchestrator is None:
            self._append_message("system", "엔진이 아직 준비되지 않았습니다.")
            return
        if self._inflight:
            return

        self._inflight = True
        t0 = time.time()
        try:
            response = self.orchestrator.handle(message)
            response = self._resolve_follow_up(message, response)
            elapsed = time.time() - t0
            config = self._effective_settings()
            response_text = response.message
            hits = []
            if response.agent == "document_search":
                hits = response.metadata.get("hits", []) if isinstance(response.metadata, dict) else []
                if config["include_references"] and hits:
                    ref_lines = ["", "관련 문서:"]
                    for hit in hits[:5]:
                        path = str(hit.get("path") or "")
                        score = hit.get("similarity") or hit.get("vector_similarity") or 0.0
                        try:
                            score_str = f"{float(score):.3f}"
                        except Exception:
                            score_str = "-"
                        ref_lines.append(f"- {path} (유사도 {score_str})")
                    response_text += "\n" + "\n".join(ref_lines)

            response_text += f"\n\n(응답 시간 {elapsed:.2f}s)"
            prefix_map = {
                "document_search": "비서",
                "meeting_summary": "회의 비서",
                "photo_manager": "사진 비서",
                "follow_up": "비서",
            }
            agent_label = prefix_map.get(response.agent, "비서")
            self._append_message("assistant", f"[{agent_label}] {response_text}")
        except Exception as exc:
            self._append_message("system", f"대화 중 오류가 발생했습니다: {exc}")
        finally:
            self._inflight = False

    # ------------------------------------------------------------------
    def _append_message(self, role: str, text: str) -> None:
        tag = role if role in {"user", "assistant", "system"} else "assistant"
        message = text.strip() if text else ""
        prefix_map = {"user": "사용자", "assistant": "비서", "system": "시스템"}
        label = prefix_map.get(tag, "비서")
        display = f"{label}: {message}\n\n"

        if tag in {"user", "assistant"} and message:
            self.history.append((tag, message))
            if len(self.history) > 20:
                self.history = self.history[-20:]
            if tag == "assistant":
                self.last_copyable_text = message
                self.after(0, lambda: self.copy_button.configure(state="normal"))
        elif tag == "system" and message:
            self.last_copyable_text = message
            self.after(0, lambda: self.copy_button.configure(state="normal"))

        def _do() -> None:
            self.chat_display.insert("end", display, tag)
            self.chat_display.see("end")
        self.after(0, _do)

    def _resolve_follow_up(self, query: str, response) -> "OrchestratorResponse":
        current = response
        while current.agent == "follow_up":
            follow_context = self._prompt_follow_up(current.reason, current.message)
            if not follow_context:
                break

            expects_long_task = current.reason in {"needs_audio", "needs_roots"}
            agent_label = self._agent_label_for_reason(current.reason)
            cancel_event: Optional[threading.Event] = None
            if expects_long_task:
                cancel_event = threading.Event()
                follow_context.setdefault("__progress_callback", self._make_progress_handler(agent_label))
                follow_context.setdefault("__cancel_event", cancel_event)
                if current.reason == "needs_audio":
                    follow_context.setdefault("enable_resume", True)
                self._active_cancel_event = cancel_event
                self._start_long_task(agent_label)

            try:
                current = self.orchestrator.handle(query, follow_context)
            finally:
                if expects_long_task:
                    self._finish_long_task()

        return current

    def _prompt_follow_up(self, reason: Optional[str], message: str) -> Optional[Dict[str, Any]]:
        self._append_message("system", message)
        if reason == "needs_audio":
            path = self._choose_audio_file()
            if not path:
                self._append_message("system", "오디오 파일을 선택하지 않아 요청이 취소되었습니다.")
                return None
            self._append_message("system", f"선택한 오디오 파일: {path}")
            self._remember_recent_audio(path)
            return {"audio_path": path, "enable_resume": True}
        if reason == "needs_roots":
            roots = self._choose_photo_roots()
            if not roots:
                self._append_message("system", "사진 폴더를 선택하지 않아 요청이 취소되었습니다.")
                return None
            for root in roots:
                self._append_message("system", f"선택한 폴더: {root}")
            self._remember_recent_roots(roots)
            return {"roots": roots}
        extra = self._call_in_ui_thread(lambda: simpledialog.askstring("추가 정보 필요", message, parent=self))
        if not extra:
            self._append_message("system", "추가 정보를 입력하지 않아 요청이 취소되었습니다.")
            return None
        return {"details": extra}

    def _agent_label_for_reason(self, reason: Optional[str]) -> str:
        if reason == "needs_audio":
            return "회의 비서"
        if reason == "needs_roots":
            return "사진 비서"
        return "비서"

    def _choose_audio_file(self) -> Optional[str]:
        history = list(self.recent_audio_files)
        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac"),
            ("All Files", "*.*"),
        ]

        def _open_dialog():
            dialog = PathFormDialog(
                self,
                title="회의 오디오 파일 선택",
                message="회의 요약을 실행하려면 오디오 파일을 선택하거나 직접 경로를 입력하세요.",
                mode="file",
                allow_multiple=False,
                history=history,
                filetypes=filetypes,
            )
            self.wait_window(dialog)
            return dialog.result

        result = self._call_in_ui_thread(_open_dialog)
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            return result[0]
        return None

    def _choose_photo_roots(self) -> Optional[List[str]]:
        history = list(self.recent_photo_roots)

        def _open_dialog():
            dialog = PathFormDialog(
                self,
                title="사진 폴더 선택",
                message="사진 정리를 실행하려면 분석할 폴더를 하나 이상 선택하거나 입력하세요.",
                mode="directory",
                allow_multiple=True,
                history=history,
            )
            self.wait_window(dialog)
            return dialog.result

        result = self._call_in_ui_thread(_open_dialog)
        if isinstance(result, list):
            return result
        if isinstance(result, str) and result:
            return [result]
        return None

    def _call_in_ui_thread(self, func: Callable[[], Any]) -> Any:
        event = threading.Event()
        payload: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                payload["value"] = func()
            finally:
                event.set()

        self.after(0, _runner)
        event.wait()
        return payload.get("value")

    def _remember_recent_audio(self, path: str) -> None:
        normalised = str(Path(path).expanduser())
        entries = [normalised] + [item for item in self.recent_audio_files if item != normalised]
        self.recent_audio_files = entries[:5]
        self.settings.set(self.recent_audio_files, "agents", "meeting", "recent_audio_files")

    def _remember_recent_roots(self, roots: List[str]) -> None:
        merged: List[str] = []
        for root in roots:
            normalised = str(Path(root).expanduser())
            if normalised and normalised not in merged:
                merged.append(normalised)
        for existing in self.recent_photo_roots:
            if existing not in merged:
                merged.append(existing)
        self.recent_photo_roots = merged[:5]
        self.settings.set(self.recent_photo_roots, "agents", "photo", "recent_roots")

    def _make_progress_handler(self, agent_label: str) -> Callable[[Dict[str, Any]], None]:
        def _handler(event: Dict[str, Any]) -> None:
            stage = event.get("stage")
            status = event.get("status")
            if not stage or not status:
                return
            if status == "running":
                self._update_status(f"{agent_label} · {stage} 단계 실행 중...")
            elif status == "completed":
                self._update_status(f"{agent_label} · {stage} 단계 완료")
            elif status == "failed":
                error = event.get("error")
                self._append_message(
                    "system",
                    f"{agent_label} 작업 중 '{stage}' 단계에서 오류가 발생했습니다: {error or '원인을 확인해 주세요.'}",
                )
                self._update_status(f"{agent_label} 작업이 실패했습니다.")
            elif status == "cancelled":
                self._append_message("system", f"{agent_label} 작업이 취소되었습니다.")
                self._update_status(f"{agent_label} 작업이 취소되었습니다.")
        return _handler

    def _start_long_task(self, agent_label: str) -> None:
        self._pending_agent_label = agent_label
        self._task_started = True
        self._update_status(f"{agent_label} 작업을 준비 중입니다...")
        self._set_cancel_enabled(True, label=f"{agent_label} 취소")

        def _kickoff() -> None:
            self.app.start_task(f"{agent_label} 작업이 실행 중입니다...")

        self.after(0, _kickoff)

    def _finish_long_task(self, message: Optional[str] = None) -> None:
        if not self._task_started:
            return
        self._set_cancel_enabled(False)
        if message:
            self._update_status(message)

        def _restore() -> None:
            current_status = self.app.status_var.get()
            self.app.end_task(message or current_status)

        self.after(0, _restore)
        self._pending_agent_label = None
        self._task_started = False
        self._active_cancel_event = None

    def _set_cancel_enabled(self, enabled: bool, label: Optional[str] = None) -> None:
        def _update() -> None:
            state = "normal" if enabled else "disabled"
            self.cancel_button.configure(state=state)
            if label:
                self.cancel_button.configure(text=label)
            elif not enabled:
                self.cancel_button.configure(text="취소")

        self.after(0, _update)

    def _request_cancel(self) -> None:
        event = self._active_cancel_event
        if event and not event.is_set():
            event.set()
            self._append_message("system", "현재 작업에 취소 요청을 보냈습니다. 잠시만 기다려 주세요.")
            self._update_status("취소 요청을 전송했습니다.")
            self._set_cancel_enabled(False)

    def _update_status(self, message: str) -> None:
        self.after(0, lambda: self.app.status_var.set(message))

    # ------------------------------------------------------------------
    def _handle_display_key(self, event) -> Optional[str]:
        mods = event.state
        ctrl = bool(mods & 0x4)
        cmd = any(mods & mask for mask in self._command_masks)
        if (ctrl or cmd) and event.keysym.lower() in {"c", "x"}:
            return None
        if (ctrl or cmd) and event.keysym.lower() == "a":
            self.chat_display.tag_add("sel", "1.0", "end-1c")
            return "break"
        if event.keysym in {"Left", "Right", "Up", "Down", "Prior", "Next", "Home", "End"}:
            return None
        if event.keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R", "Command", "Option_L", "Option_R"}:
            return None
        if ctrl or cmd:
            return None
        return "break"

    def _copy_selection(self) -> None:
        try:
            text = self.chat_display.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.chat_display.get("1.0", "end-1c")
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()
            self.after(0, lambda: self.app.status_var.set("선택한 내용을 복사했습니다."))

    def _handle_copy_shortcut(self, _event) -> str:
        self._copy_selection()
        return "break"

    def _copy_last_response(self) -> None:
        text = self.last_copyable_text.strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
        self.after(0, lambda: self.app.status_var.set("마지막 응답을 복사했습니다."))

    def _show_context_menu(self, event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    # ------------------------------------------------------------------
    def _effective_settings(self) -> Dict[str, any]:
        return {
            "llm_backend": self.settings.get("conversation", "llm_backend", default="").strip(),
            "llm_model": self.settings.get("conversation", "llm_model", default="").strip() or "llama3",
            "llm_host": self.settings.get("conversation", "llm_host", default="").strip(),
            "top_k": int(self.settings.get("conversation", "top_k", default=DEFAULT_TOP_K) or DEFAULT_TOP_K),
            "min_similarity": float(self.settings.get("conversation", "min_similarity", default=DEFAULT_SIMILARITY_THRESHOLD) or DEFAULT_SIMILARITY_THRESHOLD),
            "include_references": bool(self.settings.get("conversation", "include_references", default=True)),
        }

    # ------------------------------------------------------------------
    def _open_settings_dialog(self) -> None:
        dialog = ConversationSettingsDialog(self, self.settings)
        self.wait_window(dialog)
        if dialog.saved:
            self._reset_conversation()
            self._append_message("system", "설정이 저장되었습니다. 엔진을 다시 준비합니다.")
            self.orchestrator = None
            self._initialise_engine_async()


class ConversationSettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent: ConversationScreen, settings: SettingsManager):
        super().__init__(parent)
        self.title("대화 비서 설정")
        self.geometry("520x560")
        self.minsize(500, 480)
        self.resizable(True, True)
        self.settings = settings
        self.saved = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))
        self.content.grid_columnconfigure(0, weight=1)

        cfg = parent._effective_settings()

        row = 0
        self.backend = ctk.CTkComboBox(self.content, values=["", "ollama"], state="normal", command=self._on_backend_change)
        self.backend.set(cfg["llm_backend"])
        self._add_labeled_field("LLM 백엔드", self.backend, row)
        row += 1

        model_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        model_frame.grid(row=row, column=0, sticky="ew", padx=24, pady=(8, 8))
        model_frame.grid_columnconfigure(1, weight=1)

        model_label = ctk.CTkLabel(model_frame, text="모델 이름", font=ctk.CTkFont(size=13, weight="bold"))
        model_label.grid(row=0, column=0, sticky="w")

        self.model_combo = ctk.CTkComboBox(model_frame, values=[], state="normal")
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        self.model_combo.set(cfg["llm_model"])

        refresh_btn = ctk.CTkButton(model_frame, text="목록 갱신", width=90, command=self._refresh_model_list)
        refresh_btn.grid(row=0, column=2, padx=(12, 0))

        self.model_hint = ctk.CTkLabel(
            model_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b",
        )
        self.model_hint.grid(row=1, column=1, columnspan=2, sticky="w", pady=(4, 0))

        self._populate_model_values(cfg["llm_backend"], cfg["llm_model"], cfg["llm_host"], force_refresh=True)
        row += 1

        self.host_entry = ctk.CTkEntry(self.content)
        self.host_entry.insert(0, cfg["llm_host"])
        self._add_labeled_field("호스트 (선택)", self.host_entry, row)
        row += 1

        self.topk_slider = ctk.CTkSlider(self.content, from_=1, to=20, number_of_steps=19)
        self.topk_slider.set(cfg["top_k"])
        self._add_labeled_field("추천 문서 수", self.topk_slider, row, helper_label=lambda v: f"{int(float(v))}개")
        row += 1

        self.sim_slider = ctk.CTkSlider(self.content, from_=0.0, to=1.0, number_of_steps=20)
        self.sim_slider.set(cfg["min_similarity"])
        self._add_labeled_field("최소 유사도", self.sim_slider, row, helper_label=lambda v: f"{float(v):.2f}")
        row += 1

        self.ref_switch = ctk.CTkSwitch(self.content, text="답변에 참고 문서 포함", onvalue="on", offvalue="off")
        self.ref_switch.select() if cfg["include_references"] else self.ref_switch.deselect()
        self.ref_switch.grid(row=row, column=0, sticky="w", padx=24, pady=(12, 12))
        row += 1

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=1, column=0, pady=(4, 12), padx=12, sticky="ew")
        button_row.grid_columnconfigure((0, 1), weight=1)

        cancel_btn = ctk.CTkButton(button_row, text="취소", command=self.destroy)
        cancel_btn.grid(row=0, column=0, padx=12)

        save_btn = ctk.CTkButton(button_row, text="저장", command=self._save)
        save_btn.grid(row=0, column=1, padx=12)

    def _add_labeled_field(self, label_text: str, widget: ctk.CTkBaseClass, row: int, helper_label=None) -> None:
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=24, pady=(8, 8))
        frame.grid_columnconfigure(1, weight=1)

        label = ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(size=13, weight="bold"))
        label.grid(row=0, column=0, sticky="w")

        widget.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        if helper_label:
            helper = ctk.CTkLabel(frame, text=helper_label(widget.get()), font=ctk.CTkFont(size=11), text_color="#6b6b6b")

            def _update_label(value: float) -> None:
                helper.configure(text=helper_label(value))

            if isinstance(widget, ctk.CTkSlider):
                widget.configure(command=_update_label)
            helper.grid(row=1, column=1, sticky="w", pady=(4, 0))

    def _save(self) -> None:
        backend = self.backend.get().strip()
        model = self.model_combo.get().strip() or "llama3"
        host = self.host_entry.get().strip()
        top_k = int(self.topk_slider.get())
        min_sim = round(float(self.sim_slider.get()), 3)
        include_refs = self.ref_switch.get() == "on"

        self.settings.set(backend, "conversation", "llm_backend")
        self.settings.set(model, "conversation", "llm_model")
        self.settings.set(host, "conversation", "llm_host")
        self.settings.set(top_k, "conversation", "top_k")
        self.settings.set(min_sim, "conversation", "min_similarity")
        self.settings.set(include_refs, "conversation", "include_references")
        self.saved = True
        self.destroy()

    # ------------------------------------------------------------------
    def _on_backend_change(self, value: str) -> None:
        host = self.host_entry.get().strip()
        self._populate_model_values(value.strip(), self.model_combo.get().strip(), host)

    def _refresh_model_list(self) -> None:
        backend = self.backend.get().strip()
        current = self.model_combo.get().strip()
        host = self.host_entry.get().strip()
        self._populate_model_values(backend, current, host, force_refresh=True)

    def _populate_model_values(self, backend: str, current: str, host: str, force_refresh: bool = False) -> None:
        candidates = self._load_model_candidates(backend, host=host, force_refresh=force_refresh)
        if candidates:
            self.model_combo.configure(values=candidates)
            self.model_hint.configure(text="목록에서 선택하거나 직접 입력할 수 있습니다.")
            if current:
                self.model_combo.set(current)
            else:
                self.model_combo.set(candidates[0])
        else:
            self.model_combo.configure(values=[])
            if backend == "ollama":
                self.model_hint.configure(text="Ollama 모델 목록을 가져오지 못했습니다. 터미널에서 'ollama list'를 확인해 주세요.")
            else:
                self.model_hint.configure(text="사용할 모델 이름을 직접 입력하세요.")
            if current:
                self.model_combo.set(current)

    def _load_model_candidates(self, backend: str, *, host: str = "", force_refresh: bool = False) -> List[str]:
        backend = (backend or "").strip().lower()
        if backend != "ollama":
            return []
        env = None
        if host:
            env = os.environ.copy()
            env["OLLAMA_HOST"] = host

        def _run(cmd: List[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
                env=env,
            )

        try:
            result = _run(["ollama", "list", "--format", "json"])
            if result.returncode == 0:
                payload = json.loads(result.stdout or "[]")
                if isinstance(payload, list):
                    names = [str(item.get("name")) for item in payload if isinstance(item, dict) and item.get("name")]
                    if names:
                        return names
            else:
                if "unknown flag" not in (result.stderr or "").lower():
                    self.model_hint.configure(text=result.stderr.strip() or "ollama list 명령이 실패했습니다.")
        except Exception as exc:
            # Fall back to plain text parsing below
            self.model_hint.configure(text=f"ollama 모델 목록을 가져오는 중 경고: {exc}")

        try:
            result_plain = _run(["ollama", "list"])
        except Exception as exc:
            self.model_hint.configure(text=f"ollama 명령 실행 실패: {exc}")
            return []
        if result_plain.returncode != 0:
            self.model_hint.configure(text=result_plain.stderr.strip() or "ollama list 명령이 실패했습니다.")
            return []
        names: List[str] = []
        for line in (result_plain.stdout or "").strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0].lower() in {"name", "models"}:
                continue
            names.append(parts[0])
        return names
