"""Toplevel window for running MeetingAgent pipelines interactively."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, Optional

import customtkinter as ctk

from core.agents import AgentRequest
from core.agents.meeting import MeetingAgent, MeetingAgentConfig
from ui.utils import MEETING_OUTPUT_DIR


class MeetingPanel(ctk.CTkToplevel):
    """Desktop UI bridge for executing `MeetingAgent` jobs."""

    def __init__(
        self,
        master,
        *,
        on_activity: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:  # type: ignore[override]
        super().__init__(master)
        self.title("회의 비서")
        self.geometry("620x720")
        self.resizable(False, False)
        self.configure(fg_color="#11131c")

        self._on_activity = on_activity
        self._on_result = on_result
        self._task_thread: Optional[threading.Thread] = None

        self.meeting_agent = MeetingAgent(MeetingAgentConfig(output_root=MEETING_OUTPUT_DIR))
        try:
            self.meeting_agent.prepare()
        except Exception as exc:  # pragma: no cover - defensive path
            self._log(f"ERROR: 회의 비서 초기화 실패 - {exc}")

        self._build_layout()

    # ------------------------------------------------------------------
    # UI composition
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        header = ctk.CTkLabel(
            self,
            text="회의 요약 파이프라인",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#F5F5F5",
        )
        header.pack(anchor="w", padx=24, pady=(24, 6))

        sub = ctk.CTkLabel(
            self,
            text="오디오 / 전사 파일을 선택하고 요약·액션 아이템·결정 사항을 생성합니다.",
            font=ctk.CTkFont(size=13),
            text_color="#AAB0BE",
        )
        sub.pack(anchor="w", padx=24, pady=(0, 16))

        form = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        form.pack(fill="x", padx=20, pady=(0, 16))
        form.grid_columnconfigure(1, weight=1)

        # Audio file selector
        ctk.CTkLabel(form, text="오디오 / 전사 파일", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=16, pady=12, sticky="w"
        )
        audio_row = ctk.CTkFrame(form, fg_color="transparent")
        audio_row.grid(row=0, column=1, padx=16, pady=12, sticky="ew")
        audio_row.grid_columnconfigure(0, weight=1)
        self.audio_path_var = ctk.StringVar()
        self.audio_entry = ctk.CTkEntry(
            audio_row,
            textvariable=self.audio_path_var,
            placeholder_text="mp3 / wav / m4a / txt / md",
        )
        self.audio_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(audio_row, text="찾아보기", width=110, command=self._browse_audio).grid(
            row=0, column=1, padx=(12, 0)
        )

        # Output directory selector
        ctk.CTkLabel(form, text="출력 폴더", font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, padx=16, pady=12, sticky="w"
        )
        output_row = ctk.CTkFrame(form, fg_color="transparent")
        output_row.grid(row=1, column=1, padx=16, pady=12, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        self.output_dir_var = ctk.StringVar()
        self.output_entry = ctk.CTkEntry(
            output_row,
            textvariable=self.output_dir_var,
            placeholder_text=f"비워두면 {MEETING_OUTPUT_DIR.name}/<파일명>",
        )
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(output_row, text="폴더 지정", width=110, command=self._browse_output).grid(
            row=0, column=1, padx=(12, 0)
        )

        # Language selector
        ctk.CTkLabel(form, text="언어", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=16, pady=12, sticky="w"
        )
        self.language_var = ctk.StringVar(value="ko")
        self.language_menu = ctk.CTkOptionMenu(
            form,
            variable=self.language_var,
            values=["ko", "en", "ja", "zh"],
            width=140,
        )
        self.language_menu.grid(row=2, column=1, padx=16, pady=12, sticky="w")

        # Action buttons
        actions = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        actions.pack(fill="x", padx=20, pady=(0, 16))
        actions.grid_columnconfigure((0, 1), weight=1)

        self.run_button = ctk.CTkButton(
            actions,
            text="▶️ 회의 요약 실행",
            height=40,
            command=self._run_meeting_summary,
        )
        self.run_button.grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 16), sticky="ew")

        # Status + log
        self.status_label = ctk.CTkLabel(
            self,
            text="대기 중",
            font=ctk.CTkFont(size=12),
            text_color="#AAB0BE",
        )
        self.status_label.pack(fill="x", padx=24, pady=(0, 6))

        self.log_box = ctk.CTkTextbox(
            self,
            state="disabled",
            height=180,
            fg_color="#151a26",
            border_width=0,
            font=ctk.CTkFont(family="Menlo", size=12),
        )
        self.log_box.pack(fill="both", expand=False, padx=24, pady=(0, 12))

        self.summary_box = ctk.CTkTextbox(
            self,
            state="disabled",
            height=200,
            fg_color="#181c28",
            border_width=0,
            font=ctk.CTkFont(size=13),
        )
        self.summary_box.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="회의 오디오 / 전사 파일 선택",
            filetypes=[
                ("Audio / Text", "*.mp3 *.wav *.m4a *.txt *.md"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.audio_path_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(parent=self, title="출력 폴더 선택")
        if path:
            self.output_dir_var.set(path)

    def _run_meeting_summary(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요.")
            return

        audio_path = Path(self.audio_path_var.get().strip())
        if not audio_path.exists():
            self._log("ERROR: 오디오 파일을 선택해 주세요.")
            return

        output_dir = self.output_dir_var.get().strip() or ""
        language = self.language_var.get().strip() or "ko"

        context = {
            "audio_path": str(audio_path),
            "output_dir": output_dir,
            "language": language,
            "__progress_callback": self._handle_progress_event,
        }

        def _runner() -> None:
            self._set_status("회의 요약 실행 중...")
            self._toggle_controls(active=False)
            self._notify_activity(f"MEETING · started ({audio_path.name})")
            try:
                result = self.meeting_agent.run(AgentRequest(query="summarise_meeting", context=context))
            except Exception as exc:
                self._log(f"ERROR: 회의 요약 실패 - {exc}")
                self._notify_activity(f"ERROR · meeting failed: {exc}")
            else:
                self._display_summary(result.content)
                self._log("SUCCESS: 회의 요약이 완료되었습니다.")
                metadata = dict(result.metadata or {})
                metadata.setdefault("audio_path", str(audio_path))
                self._notify_activity("MEETING · completed")
                if self._on_result:
                    self._on_result(result.content, metadata)
            finally:
                self.after(0, self._task_finished)

        self._task_thread = threading.Thread(target=_runner, daemon=True)
        self._task_thread.start()

    # ------------------------------------------------------------------
    # Task lifecycle helpers
    # ------------------------------------------------------------------
    def _task_finished(self) -> None:
        self._set_status("대기 중")
        self._toggle_controls(active=True)
        self._task_thread = None

    def _toggle_controls(self, *, active: bool) -> None:
        state = "normal" if active else "disabled"
        self.run_button.configure(state=state)
        self.audio_entry.configure(state=state)
        self.output_entry.configure(state=state)
        self.language_menu.configure(state=state)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        def _append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.configure(state="disabled")
            self.log_box.see("end")

        self.after(0, _append)

    def _display_summary(self, text: str) -> None:
        def _update() -> None:
            self.summary_box.configure(state="normal")
            self.summary_box.delete("1.0", "end")
            self.summary_box.insert("end", text.strip() or "(요약 결과가 비어 있습니다)")
            self.summary_box.configure(state="disabled")
            self.summary_box.see("1.0")

        self.after(0, _update)

    def _handle_progress_event(self, event: Dict[str, object]) -> None:
        stage = str(event.get("stage") or "").upper()
        status = event.get("status") or ""
        message = event.get("message") or ""
        log_line = f"[{stage}] {status}"
        if message:
            log_line += f" · {message}"
        self._log(log_line)

    def _notify_activity(self, text: str) -> None:
        if self._on_activity:
            self._on_activity(text)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------
    def _handle_close(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 작업이 끝난 후 창이 닫힙니다.")
            return
        self.destroy()
