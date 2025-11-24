from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk
from tkinter import filedialog

from core.agents.meeting.models import MeetingJobConfig, StreamingSummarySnapshot
from core.agents.meeting.pipeline import (
    MeetingPipeline,
    StreamingMeetingSession,
    get_backend_diagnostics,
)
from ui.utils import MEETING_OUTPUT_DIR
from ui.smart_folder_context import SmartFolderContext
from ui.policy_cache import get_policy_engine
from ui.settings_manager import SettingsManager


DEFAULT_STREAM_INTERVAL = 60.0
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "ui_settings.json"


class MeetingScreen(ctk.CTkFrame):
    """UI bridge for the meeting agent MVP."""

    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.is_running = False
        self.last_output_dir: Optional[Path] = None
        self.settings = SettingsManager(SETTINGS_PATH.resolve())
        self._stt_traces_registered = False
        stt_prefs = self._load_stt_preferences()

        self.stt_backend_var = ctk.StringVar(value=stt_prefs["backend"])
        self.stt_model_var = ctk.StringVar(value=stt_prefs["model"])
        self.stt_device_var = ctk.StringVar(value=stt_prefs["device"])
        self.stt_compute_var = ctk.StringVar(value=stt_prefs["compute"])
        self.stt_download_var = ctk.StringVar(value=stt_prefs["download_dir"])
        self.live_mode_var = ctk.IntVar(value=0)
        self.live_interval_var = ctk.StringVar(value=str(int(DEFAULT_STREAM_INTERVAL)))
        self.live_speaker_var = ctk.StringVar()

        self.streaming_session: Optional[StreamingMeetingSession] = None
        self.streaming_job: Optional[MeetingJobConfig] = None
        self.streaming_log_path: Optional[Path] = None
        self.active_context: Optional[SmartFolderContext] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="회의 비서",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=(0, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="오디오/전사 파일에서 요약, 액션 아이템, 결정 사항을 추출합니다.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        self.context_hint_label = ctk.CTkLabel(
            self,
            text="📁 스마트 폴더: 전체 사용 가능",
            font=ctk.CTkFont(size=12),
            text_color=("#4f4f4f", "#cfcfcf"),
        )
        self.context_hint_label.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="w")

        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.grid(row=3, column=0, padx=16, pady=12, sticky="ew")
        self.form_frame.grid_columnconfigure(0, weight=0)
        self.form_frame.grid_columnconfigure(1, weight=1)

        # Input file selector
        self.audio_path_var = ctk.StringVar()
        ctk.CTkLabel(self.form_frame, text="입력 파일", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=8, sticky="w"
        )
        audio_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        audio_row.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
        audio_row.grid_columnconfigure(0, weight=1)
        self.audio_entry = ctk.CTkEntry(
            audio_row,
            textvariable=self.audio_path_var,
            placeholder_text="mp3 / wav / m4a / txt / md 파일을 선택하세요",
        )
        self.audio_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(audio_row, text="찾아보기", width=110, command=self.browse_audio).grid(
            row=0, column=1, padx=(12, 0)
        )

        # Output selector
        self.output_dir_var = ctk.StringVar()
        ctk.CTkLabel(self.form_frame, text="출력 폴더", font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, padx=12, pady=8, sticky="w"
        )
        output_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        output_row.grid(row=1, column=1, padx=12, pady=8, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(
            output_row,
            textvariable=self.output_dir_var,
            placeholder_text=f"비워두면 {MEETING_OUTPUT_DIR.name}/<날짜> 폴더가 생성됩니다",
        )
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(output_row, text="폴더 지정", width=110, command=self.browse_output_dir).grid(
            row=0, column=1, padx=(12, 0)
        )

        # Language + policy row
        ctk.CTkLabel(self.form_frame, text="언어", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=12, pady=8, sticky="w"
        )
        lang_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        lang_row.grid(row=2, column=1, padx=12, pady=8, sticky="ew")
        lang_row.grid_columnconfigure(1, weight=1)
        self.language_option = ctk.CTkOptionMenu(lang_row, values=["ko", "en", "ja", "zh"], width=120)
        self.language_option.set("ko")
        self.language_option.grid(row=0, column=0, padx=(0, 12), sticky="w")
        self.policy_var = ctk.StringVar()
        self.policy_entry = ctk.CTkEntry(
            lang_row,
            textvariable=self.policy_var,
            placeholder_text="스마트 폴더 정책 태그 (선택)",
        )
        self.policy_entry.grid(row=0, column=1, sticky="ew")

        # STT backend controls
        ctk.CTkLabel(self.form_frame, text="STT", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=12, pady=8, sticky="w"
        )
        stt_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        stt_row.grid(row=3, column=1, padx=12, pady=8, sticky="ew")
        stt_row.grid_columnconfigure(1, weight=1)
        self.stt_backend_menu = ctk.CTkOptionMenu(
            stt_row,
            values=["auto", "wav2vec2", "whisper", "off"],
            variable=self.stt_backend_var,
            command=self.on_stt_backend_change,
            width=130,
        )
        self.stt_backend_menu.set(stt_prefs["backend"])
        self.stt_backend_menu.grid(row=0, column=0, padx=(0, 12), sticky="w")

        advanced_row = ctk.CTkFrame(stt_row, fg_color="transparent")
        advanced_row.grid(row=0, column=1, sticky="ew")
        advanced_row.grid_columnconfigure((0, 1, 2), weight=1, uniform="stt")

        self.stt_model_entry = ctk.CTkEntry(
            advanced_row,
            textvariable=self.stt_model_var,
            placeholder_text="모델(ex: small)",
            state="disabled",
        )
        self.stt_model_entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.stt_device_entry = ctk.CTkEntry(
            advanced_row,
            textvariable=self.stt_device_var,
            placeholder_text="디바이스(auto)",
            state="disabled",
        )
        self.stt_device_entry.grid(row=0, column=1, padx=(0, 8), sticky="ew")

        self.stt_compute_entry = ctk.CTkEntry(
            advanced_row,
            textvariable=self.stt_compute_var,
            placeholder_text="연산 타입(int8)",
            state="disabled",
        )
        self.stt_compute_entry.grid(row=0, column=2, sticky="ew")

        self.stt_download_entry = ctk.CTkEntry(
            stt_row,
            textvariable=self.stt_download_var,
            placeholder_text="모델 다운로드 경로 (선택)",
            state="disabled",
        )
        self.stt_download_entry.grid(row=1, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        help_label = ctk.CTkLabel(
            stt_row,
            text="auto=환경 변수 사용, wav2vec2=HuggingFace(Wav2Vec2), whisper=faster-whisper. 설정 값은 자동 저장됩니다.",
            font=ctk.CTkFont(size=11),
            text_color=("#636363", "#bdbdbd"),
            wraplength=420,
            justify="left",
        )
        help_label.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="w")

        self._update_stt_entry_states()
        self._register_stt_traces()
        self._persist_stt_preferences()

        diag_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        diag_row.grid(row=4, column=1, padx=12, pady=(0, 8), sticky="ew")
        diag_row.grid_columnconfigure(0, weight=1)
        self.backend_status_label = ctk.CTkLabel(
            diag_row,
            text="백엔드 상태 확인 중...",
            anchor="w",
        )
        self.backend_status_label.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            diag_row,
            text="상태 새로고침",
            width=120,
            command=self.refresh_backend_status,
        ).grid(row=0, column=1, padx=(12, 0))

        self.on_stt_backend_change(stt_prefs["backend"])
        self.refresh_backend_status()

        # Diarisation controls
        diar_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        diar_row.grid(row=5, column=1, padx=12, pady=8, sticky="ew")
        diar_row.grid_columnconfigure(1, weight=1)
        self.diarize_switch = ctk.CTkSwitch(diar_row, text="화자 분리(실험적)", command=self.on_toggle_diarize)
        self.diarize_switch.grid(row=0, column=0, sticky="w")
        self.speaker_var = ctk.StringVar()
        self.speaker_entry = ctk.CTkEntry(
            diar_row,
            textvariable=self.speaker_var,
            placeholder_text="화자 수 (선택)",
            state="disabled",
            width=160,
        )
        self.speaker_entry.grid(row=0, column=1, padx=(12, 0), sticky="w")

        # Streaming mode controls
        live_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        live_row.grid(row=6, column=1, padx=12, pady=8, sticky="ew")
        live_row.grid_columnconfigure(0, weight=0)
        live_row.grid_columnconfigure(1, weight=1)
        self.live_mode_switch = ctk.CTkSwitch(
            live_row,
            text="실시간 요약 모드",
            variable=self.live_mode_var,
            command=self.on_toggle_live_mode,
        )
        self.live_mode_switch.grid(row=0, column=0, sticky="w")
        self.live_interval_entry = ctk.CTkEntry(
            live_row,
            textvariable=self.live_interval_var,
            placeholder_text="스냅샷 간격(초)",
            width=150,
            state="disabled",
        )
        self.live_interval_entry.grid(row=0, column=1, padx=(12, 0), sticky="w")
        self.live_hint_label = ctk.CTkLabel(
            live_row,
            text="실시간 모드에서는 발화를 추가하면 주기적으로 요약이 갱신됩니다.",
            font=ctk.CTkFont(size=11),
            text_color=("#636363", "#bdbdbd"),
        )
        self.live_hint_label.grid(row=1, column=0, columnspan=2, pady=(6, 0), sticky="w")

        self.live_controls_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.live_controls_frame.grid(row=7, column=1, padx=12, pady=8, sticky="ew")
        self.live_controls_frame.grid_columnconfigure(0, weight=0)
        self.live_controls_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.live_controls_frame,
            text="발화자",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=(0, 12), pady=(0, 6), sticky="w")
        self.live_speaker_entry = ctk.CTkEntry(
            self.live_controls_frame,
            textvariable=self.live_speaker_var,
            placeholder_text="발화자 라벨 (선택)",
        )
        self.live_speaker_entry.grid(row=0, column=1, pady=(0, 6), sticky="ew")

        self.live_textbox = ctk.CTkTextbox(
            self.live_controls_frame,
            height=90,
            font=ctk.CTkFont(family="monospace"),
        )
        self.live_textbox.grid(row=1, column=0, columnspan=2, sticky="ew")

        live_button_row = ctk.CTkFrame(self.live_controls_frame, fg_color="transparent")
        live_button_row.grid(row=2, column=0, columnspan=2, pady=8, sticky="ew")
        live_button_row.grid_columnconfigure(0, weight=1)
        self.live_add_button = ctk.CTkButton(
            live_button_row,
            text="발화 추가",
            command=self.add_live_segment,
            state="disabled",
        )
        self.live_add_button.grid(row=0, column=0, sticky="ew")
        self.live_finalize_button = ctk.CTkButton(
            live_button_row,
            text="실시간 요약 마무리",
            width=150,
            command=self.finalize_streaming_session,
            state="disabled",
        )
        self.live_finalize_button.grid(row=0, column=1, padx=(12, 0))

        self.live_status_label = ctk.CTkLabel(
            self.live_controls_frame,
            text="실시간 세션을 시작하면 요약이 여기에 표시됩니다.",
            anchor="w",
            text_color=("#636363", "#bdbdbd"),
        )
        self.live_status_label.grid(row=3, column=0, columnspan=2, sticky="w")

        self.live_controls_frame.grid_remove()

        # Action buttons
        button_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        button_row.grid(row=8, column=1, padx=12, pady=(8, 12), sticky="ew")
        button_row.grid_columnconfigure(0, weight=1)
        self.run_button = ctk.CTkButton(button_row, text="회의 요약 실행", command=self.start_meeting_job)
        self.run_button.grid(row=0, column=0, sticky="ew")
        self.open_folder_button = ctk.CTkButton(
            button_row,
            text="결과 폴더 열기",
            command=self.open_output_folder,
            state="disabled",
            width=140,
        )
        self.open_folder_button.grid(row=0, column=1, padx=(12, 0))

        self.log_textbox = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="monospace"),
        )
        self.log_textbox.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.append_log("회의 요약을 실행할 파일을 선택하세요.", reset=True)

        self._last_pipeline_info: Dict[str, object] = {}
        self.streaming_started_at: Optional[float] = None

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def append_log(self, message: str, reset: bool = False) -> None:
        def _update() -> None:
            self.log_textbox.configure(state="normal")
            if reset:
                self.log_textbox.delete("1.0", "end")
            if message:
                self.log_textbox.insert("end", message + "\n")
                self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")

        self.after(0, _update)

    def on_smart_folder_update(self, context: Optional[SmartFolderContext]) -> None:
        self.active_context = context
        if context and context.path:
            if not self.output_dir_var.get().strip():
                self.output_dir_var.set(str(context.path))
        if context:
            placeholder_source = context.policy or context.label
        else:
            placeholder_source = None
        placeholder = placeholder_source or "스마트 폴더 정책 태그 (선택)"
        try:
            self.policy_entry.configure(placeholder_text=placeholder)
        except Exception:
            if not self.policy_var.get() and context:
                self.policy_var.set(context.label)
        if context and not self.policy_var.get() and context.policy:
            self.policy_var.set(context.policy)
        self._apply_context_constraints()

    def _apply_context_constraints(self) -> None:
        allowed = self._context_allows_meeting()
        if self.active_context is None:
            hint = "📁 스마트 폴더: 전체 사용 가능"
        else:
            parts = [f"📁 스마트 폴더: {self.active_context.label}"]
            if self.active_context.path:
                parts.append(f"· {self.active_context.path_display}")
            parts.append("(허용)" if allowed else "(제한됨)")
            hint = " ".join(parts)
        self.context_hint_label.configure(text=hint)

        if allowed:
            self.live_mode_switch.configure(state="normal")
            if not self.is_running:
                self.run_button.configure(state="normal")
        else:
            self.run_button.configure(state="disabled")
            self.live_mode_switch.configure(state="disabled")
            self.live_add_button.configure(state="disabled")
            self.live_finalize_button.configure(state="disabled")

    def _context_allows_meeting(self) -> bool:
        if self.active_context is None:
            return True
        allowed = self.active_context.allows_agent("meeting")
        engine = get_policy_engine()
        if allowed and engine.has_policies and self.active_context.path is not None:
            try:
                allowed = engine.allows(self.active_context.path, agent="meeting")
            except Exception:
                allowed = False
        return allowed

    def _on_pipeline_error(self) -> None:
        self.run_button.configure(state="normal", text=self._default_run_button_label())
        self._apply_context_constraints()

    def _resolve_policy_tag(self) -> Optional[str]:
        explicit = self.policy_var.get().strip()
        if explicit:
            return explicit
        if self.active_context is None:
            return None
        return self.active_context.policy or self.active_context.folder_id

    def _meeting_context_dirs(self) -> List[Path]:
        if self.active_context and self.active_context.path:
            return [self.active_context.path]
        return []

    @staticmethod
    def _path_within(target: Path, root: Path) -> bool:
        try:
            target_resolved = target.resolve()
            root_resolved = root.resolve()
        except Exception:
            target_resolved = target
            root_resolved = root
        if target_resolved == root_resolved:
            return True
        return root_resolved in target_resolved.parents

    def browse_audio(self) -> None:
        file_path = filedialog.askopenfilename(
            title="회의 파일 선택",
            filetypes=[
                ("Audio / Transcript", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.txt *.md"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.audio_path_var.set(file_path)

    def browse_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="출력 폴더 선택")
        if directory:
            self.output_dir_var.set(directory)
        self._apply_context_constraints()

    def refresh_backend_status(self) -> None:
        try:
            diagnostics = get_backend_diagnostics()
        except Exception as exc:  # pragma: no cover - UI feedback
            self.backend_status_label.configure(text=f"상태 확인 실패: {exc}")
            return

        whisper_available = diagnostics.get("stt", {}).get("whisper", False)
        summary_status = diagnostics.get("summary", {})
        resource_status = diagnostics.get("resources", {})

        whisper_text = "사용 가능" if whisper_available else "미설치"
        summary_parts = []
        for name, available in sorted(summary_status.items()):
            status = "OK" if available else "X"
            summary_parts.append(f"{name}:{status}")
        summary_text = ", ".join(summary_parts) if summary_parts else "정보 없음"

        gpu_text = "GPU 사용 가능" if resource_status.get("gpu_available") else "GPU 없음"
        device_name = resource_status.get("cuda_device_name")
        if device_name:
            gpu_text += f" ({device_name})"

        self.backend_status_label.configure(
            text=f"Whisper: {whisper_text} | 요약: {summary_text} | 자원: {gpu_text}"
        )

    def on_toggle_diarize(self) -> None:
        enabled = self.diarize_switch.get() == 1
        state = "normal" if enabled else "disabled"
        self.speaker_entry.configure(state=state)
        if not enabled:
            self.speaker_var.set("")

    def on_toggle_live_mode(self) -> None:
        if self.streaming_session is not None:
            self.append_log("⚠️ 실시간 세션이 진행 중입니다. 먼저 마무리하세요.")
            self.live_mode_var.set(1)
            return

        enabled = self.live_mode_var.get() == 1
        if enabled:
            self.live_controls_frame.grid()
            self.live_interval_entry.configure(state="normal")
            self.live_add_button.configure(state="disabled")
            self.live_finalize_button.configure(state="disabled")
            self.run_button.configure(text=self._default_run_button_label())
            self.live_status_label.configure(text="실시간 세션을 시작하면 요약이 여기에 표시됩니다.")
        else:
            self.live_controls_frame.grid_remove()
            self.live_interval_entry.configure(state="disabled")
            self.live_add_button.configure(state="disabled")
            self.live_finalize_button.configure(state="disabled")
            self.live_textbox.delete("1.0", "end")
            self.live_speaker_var.set("")
            self.run_button.configure(text=self._default_run_button_label())

    def on_stt_backend_change(self, _: str) -> None:
        self._update_stt_entry_states()
        self._persist_stt_preferences()

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def start_meeting_job(self) -> None:
        if not self._context_allows_meeting():
            self.append_log("⚠️ 선택한 스마트 폴더에서는 회의 비서를 사용할 수 없습니다.")
            return

        if self.live_mode_var.get() == 1:
            if self.streaming_session is not None:
                self.append_log("⚠️ 이미 실시간 세션이 진행 중입니다.")
                return
            if self.is_running:
                return
            self.start_streaming_session()
            return

        if self.is_running:
            return

        audio_path_text = self.audio_path_var.get().strip()
        if not audio_path_text:
            self.append_log("⚠️ 요약할 파일을 먼저 선택하세요.")
            return

        audio_path = Path(audio_path_text)
        if not audio_path.exists():
            self.append_log("⚠️ 선택한 파일을 찾을 수 없습니다.")
            return

        context_path = self.active_context.path if self.active_context and self.active_context.path else None
        if context_path:
            try:
                context_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.append_log(f"⚠️ 스마트 폴더 경로를 준비할 수 없습니다: {exc}")
                return

        output_root_text = self.output_dir_var.get().strip()
        if output_root_text:
            output_dir = Path(output_root_text)
            if context_path is not None:
                if not self._path_within(output_dir, context_path):
                    self.append_log("⚠️ 출력 폴더는 선택한 스마트 폴더 경로 내부여야 합니다.")
                    return
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_name = audio_path.stem or "meeting"
            base_root = context_path if context_path is not None else MEETING_OUTPUT_DIR
            output_dir = base_root / safe_name / timestamp
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.append_log(f"⚠️ 출력 폴더를 생성할 수 없습니다: {exc}")
            return

        diarize = self.diarize_switch.get() == 1
        speaker_count = None
        if diarize:
            try:
                speaker_count = int(self.speaker_var.get()) if self.speaker_var.get().strip() else None
            except ValueError:
                self.append_log("⚠️ 화자 수는 숫자로 입력해주세요.")
                return

        policy_tag = self._resolve_policy_tag()
        job = MeetingJobConfig(
            audio_path=audio_path,
            output_dir=output_dir,
            language=self.language_option.get(),
            diarize=diarize,
            speaker_count=speaker_count,
            policy_tag=policy_tag,
            context_dirs=self._meeting_context_dirs(),
        )

        backend_display = self._describe_backend_choice()
        self.append_log(
            "\n".join(
                [
                    "회의 요약을 실행합니다...",
                    f"STT 설정: {backend_display}",
                ]
            ),
            reset=True,
        )
        self.is_running = True
        self.last_output_dir = output_dir
        self.open_folder_button.configure(state="disabled")
        self.run_button.configure(state="disabled", text="처리 중...")
        self.start_task_callback("⏳ 회의 요약을 실행 중입니다...")
        self._apply_context_constraints()

        pipeline = self._build_pipeline()
        self.streaming_started_at = None
        thread = threading.Thread(target=self._run_pipeline, args=(pipeline, job), daemon=True)
        thread.start()

    def _register_stt_traces(self) -> None:
        if getattr(self, "_stt_traces_registered", False):
            return
        self.stt_backend_var.trace_add("write", self._on_stt_var_change)
        for var in (
            self.stt_model_var,
            self.stt_device_var,
            self.stt_compute_var,
            self.stt_download_var,
        ):
            var.trace_add("write", self._on_stt_var_change)
        self._stt_traces_registered = True

    def _on_stt_var_change(self, *_: object) -> None:
        self._persist_stt_preferences()

    def _update_stt_entry_states(self) -> None:
        backend = self.stt_backend_var.get()
        is_whisper = backend == "whisper"
        is_wav2vec = backend == "wav2vec2"
        model_state = "normal" if is_whisper or is_wav2vec else "disabled"
        device_state = model_state
        compute_state = "normal" if is_whisper or is_wav2vec else "disabled"
        download_state = "normal" if is_whisper or is_wav2vec else "disabled"

        model_placeholder = "모델(ex: small)"
        device_placeholder = "디바이스(auto)"
        compute_placeholder = "연산 타입(int8)"
        download_placeholder = "모델 다운로드 경로 (선택)"

        if is_wav2vec:
            model_placeholder = "모델(ex: kresnik/wav2vec2-large-xlsr-korean)"
            compute_placeholder = "chunk(초, 기본 20)"
            download_placeholder = "stride(초, 기본 5)"

        self.stt_model_entry.configure(state=model_state, placeholder_text=model_placeholder)
        self.stt_device_entry.configure(state=device_state, placeholder_text=device_placeholder)
        self.stt_compute_entry.configure(state=compute_state, placeholder_text=compute_placeholder)
        self.stt_download_entry.configure(state=download_state, placeholder_text=download_placeholder)

    def _persist_stt_preferences(self) -> None:
        payload = {
            "backend": self.stt_backend_var.get(),
            "model": self.stt_model_var.get().strip(),
            "device": self.stt_device_var.get().strip(),
            "compute": self.stt_compute_var.get().strip(),
            "download_dir": self.stt_download_var.get().strip(),
        }
        self.settings.set(payload, "agents", "meeting", "stt")

        backend = payload["backend"]
        if not backend or backend == "auto":
            os.environ.pop("MEETING_STT_BACKEND", None)
        elif backend == "off":
            os.environ["MEETING_STT_BACKEND"] = "placeholder"
        else:
            os.environ["MEETING_STT_BACKEND"] = backend

        self._sync_optional_env("MEETING_STT_MODEL", payload["model"], backend == "whisper")
        self._sync_optional_env("MEETING_STT_DEVICE", payload["device"], backend == "whisper")
        self._sync_optional_env("MEETING_STT_COMPUTE", payload["compute"], backend == "whisper")
        self._sync_optional_env("MEETING_STT_MODEL_DIR", payload["download_dir"], backend == "whisper")
        self._sync_optional_env("MEETING_WAV2VEC2_MODEL", payload["model"], backend == "wav2vec2")
        self._sync_optional_env("MEETING_WAV2VEC2_DEVICE", payload["device"], backend == "wav2vec2")
        self._sync_optional_env("MEETING_WAV2VEC2_CHUNK", payload["compute"], backend == "wav2vec2")
        self._sync_optional_env("MEETING_WAV2VEC2_STRIDE", payload["download_dir"], backend == "wav2vec2")

    def _sync_optional_env(self, key: str, value: str, enabled: bool) -> None:
        if enabled and value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    def _load_stt_preferences(self) -> Dict[str, str]:
        prefs = self.settings.get("agents", "meeting", "stt", default={}) or {}
        return {
            "backend": str(prefs.get("backend") or "auto"),
            "model": str(prefs.get("model") or ""),
            "device": str(prefs.get("device") or ""),
            "compute": str(prefs.get("compute") or ""),
            "download_dir": str(prefs.get("download_dir") or ""),
        }

    def _build_pipeline(self) -> MeetingPipeline:
        backend_choice = self.stt_backend_var.get()
        backend: Optional[str]
        if backend_choice == "auto":
            backend = None
        elif backend_choice == "off":
            backend = "placeholder"
        else:
            backend = backend_choice

        stt_options = {}
        if backend == "whisper":
            model = self.stt_model_var.get().strip()
            if model:
                stt_options["model_size"] = model
            device = self.stt_device_var.get().strip()
            if device:
                stt_options["device"] = device
            compute = self.stt_compute_var.get().strip()
            if compute:
                stt_options["compute_type"] = compute
            download = self.stt_download_var.get().strip()
            if download:
                stt_options["download_root"] = download
        elif backend == "wav2vec2":
            model = self.stt_model_var.get().strip()
            if model:
                stt_options["model_id"] = model
            device = self.stt_device_var.get().strip()
            if device:
                stt_options["device"] = device
            chunk = self.stt_compute_var.get().strip()
            if chunk:
                try:
                    stt_options["chunk_length_s"] = float(chunk)
                except ValueError:
                    self.append_log("⚠️ chunk 길이는 숫자로 입력해주세요.")
            stride = self.stt_download_var.get().strip()
            if stride:
                try:
                    stt_options["stride_length_s"] = float(stride)
                except ValueError:
                    self.append_log("⚠️ stride 길이는 숫자로 입력해주세요.")

        pipeline = MeetingPipeline(stt_backend=backend, stt_options=stt_options)
        resource_info = getattr(pipeline, "_resource_info", {})
        self._last_pipeline_info = {
            "stt_backend": pipeline.stt_backend,
            "summary_backend": getattr(pipeline, "summary_backend", None),
            "resources": resource_info,
        }
        return pipeline

    def _run_pipeline(self, pipeline: MeetingPipeline, job: MeetingJobConfig) -> None:
        started_at = time.time()
        try:
            summary = pipeline.run(job)
            duration = max(time.time() - started_at, 0.0)
            metrics = {
                "duration_seconds": duration,
                "mode": "batch",
            }
            self._handle_summary_completion(
                summary,
                job,
                headline="✅ 회의 요약이 완료되었습니다!",
                completion_message="✅ 회의 요약이 완료되었습니다.",
                metrics=metrics,
            )
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.append_log(f"❌ 회의 요약 중 오류가 발생했습니다: {exc}")
            self.after(0, self._on_pipeline_error)
            self.after(0, lambda: self.end_task_callback("❌ 회의 요약 중 오류가 발생했습니다."))
        finally:
            self.is_running = False
            self.after(0, self._apply_context_constraints)

    def start_streaming_session(self) -> None:
        if not self._context_allows_meeting():
            self.append_log("⚠️ 선택한 스마트 폴더에서는 회의 비서를 사용할 수 없습니다.")
            return

        try:
            interval_str = self.live_interval_var.get().strip()
            if interval_str:
                interval = float(interval_str)
                if interval < 0:
                    raise ValueError
            else:
                interval = DEFAULT_STREAM_INTERVAL
        except ValueError:
            self.append_log("⚠️ 스냅샷 간격은 0 이상의 숫자로 입력해주세요.")
            return

        audio_path_text = self.audio_path_var.get().strip()
        safe_name = "live-session"
        audio_path: Optional[Path] = None
        if audio_path_text:
            audio_path = Path(audio_path_text)
            safe_name = audio_path.stem or safe_name

        context_path = self.active_context.path if self.active_context and self.active_context.path else None
        if context_path:
            try:
                context_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.append_log(f"⚠️ 스마트 폴더 경로를 준비할 수 없습니다: {exc}")
                return

        output_root_text = self.output_dir_var.get().strip()
        if output_root_text:
            output_dir = Path(output_root_text)
            if context_path is not None and not self._path_within(output_dir, context_path):
                self.append_log("⚠️ 출력 폴더는 선택한 스마트 폴더 경로 내부여야 합니다.")
                return
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_root = context_path if context_path is not None else MEETING_OUTPUT_DIR
            output_dir = base_root / safe_name / timestamp

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.append_log(f"⚠️ 출력 폴더를 생성할 수 없습니다: {exc}")
            return

        if audio_path is None or not audio_path.exists():
            audio_path = output_dir / "live_session.txt"
            try:
                audio_path.write_text("", encoding="utf-8")
            except Exception as exc:
                self.append_log(f"⚠️ 실시간 입력 파일을 준비할 수 없습니다: {exc}")
                return
            self.audio_path_var.set(str(audio_path))

        diarize = self.diarize_switch.get() == 1
        speaker_count = None
        if diarize:
            try:
                speaker_count = int(self.speaker_var.get()) if self.speaker_var.get().strip() else None
            except ValueError:
                self.append_log("⚠️ 화자 수는 숫자로 입력해주세요.")
                return

        job = MeetingJobConfig(
            audio_path=audio_path,
            output_dir=output_dir,
            language=self.language_option.get(),
            diarize=diarize,
            speaker_count=speaker_count,
            policy_tag=self.policy_var.get().strip() or None,
        )

        pipeline = self._build_pipeline()
        try:
            session = pipeline.start_streaming(job, update_interval=interval)
        except Exception as exc:  # pragma: no cover - defensive UI message
            self.append_log(f"❌ 실시간 세션을 시작할 수 없습니다: {exc}")
            return

        events_log = output_dir / "live_session_events.log"
        try:
            events_log.write_text("", encoding="utf-8")
        except Exception:
            # Non-fatal; continue without log file
            events_log = None

        self.streaming_session = session
        self.streaming_job = job
        self.streaming_log_path = events_log
        self.is_running = True
        self.last_output_dir = output_dir
        self.streaming_started_at = time.time()

        backend_display = self._describe_backend_choice()
        self.append_log(
            "\n".join(
                [
                    "실시간 요약 세션을 시작했습니다.",
                    f"STT 설정: {backend_display}",
                    "발화를 입력하고 '발화 추가' 버튼을 눌러 스냅샷을 갱신하세요.",
                ]
            ),
            reset=True,
        )

        self.open_folder_button.configure(state="disabled")
        self.run_button.configure(state="disabled", text="세션 진행 중...")
        self.live_add_button.configure(state="normal")
        self.live_finalize_button.configure(state="normal")
        self.live_status_label.configure(text="발화를 추가하면 요약이 업데이트됩니다.")
        self.live_textbox.delete("1.0", "end")
        self.start_task_callback("🟢 실시간 요약 세션이 활성화되었습니다.")
        self._apply_context_constraints()

    def add_live_segment(self) -> None:
        if self.streaming_session is None or self.streaming_job is None:
            self.append_log("⚠️ 실시간 세션을 먼저 시작하세요.")
            return

        text = self.live_textbox.get("1.0", "end").strip()
        if not text:
            self.append_log("⚠️ 추가할 발화를 입력하세요.")
            return

        speaker = self.live_speaker_var.get().strip() or None

        try:
            snapshot = self.streaming_session.ingest(text, speaker=speaker)
        except Exception as exc:  # pragma: no cover - streaming diagnostics
            self.append_log(f"⚠️ 발화를 처리하는 중 오류가 발생했습니다: {exc}")
            return

        if self.streaming_log_path is not None:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                speaker_label = speaker or "(unknown)"
                with self.streaming_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{timestamp}] {speaker_label}: {text}\n")
            except Exception:
                pass

        self.live_textbox.delete("1.0", "end")
        self.live_speaker_var.set("")

        if snapshot is not None:
            self._render_snapshot(snapshot)
        else:
            self.append_log("발화를 기록했습니다. 스냅샷은 곧 업데이트됩니다.")

    def _render_snapshot(self, snapshot: StreamingSummarySnapshot) -> None:
        elapsed = int(snapshot.elapsed_seconds)
        lines = [
            f"🟢 실시간 스냅샷 (경과 {elapsed}초)",
            "",
            "요약 하이라이트:",
        ]
        highlights = snapshot.highlights or []
        if highlights:
            lines.extend(f"- {item}" for item in highlights)
        else:
            lines.append("- (없음)")

        lines.append("")
        lines.append("액션 아이템:")
        actions = snapshot.action_items or []
        if actions:
            lines.extend(f"- {item}" for item in actions)
        else:
            lines.append("- (없음)")

        lines.append("")
        lines.append("결정 사항:")
        decisions = snapshot.decisions or []
        if decisions:
            lines.extend(f"- {item}" for item in decisions)
        else:
            lines.append("- (없음)")

        self.append_log("\n".join(lines), reset=True)
        self.live_status_label.configure(
            text=f"최근 스냅샷: {datetime.now().strftime('%H:%M:%S')} (경과 {elapsed}초)",
        )

    def finalize_streaming_session(self) -> None:
        if self.streaming_session is None or self.streaming_job is None:
            self.append_log("⚠️ 진행 중인 실시간 세션이 없습니다.")
            return

        if self.live_textbox.get("1.0", "end").strip():
            # 자동으로 남아있는 입력을 기록
            self.add_live_segment()

        self.live_add_button.configure(state="disabled")
        self.live_finalize_button.configure(state="disabled")
        self.run_button.configure(state="disabled", text="정리 중...")
        self.start_task_callback("⏳ 실시간 요약을 마무리하는 중입니다...")

        thread = threading.Thread(target=self._finalize_streaming_background, daemon=True)
        thread.start()

    def _finalize_streaming_background(self) -> None:
        session = self.streaming_session
        job = self.streaming_job
        if session is None or job is None:
            self.after(0, lambda: self.append_log("⚠️ 실시간 세션 정보를 찾을 수 없습니다."))
            return

        try:
            summary = session.finalize()
        except Exception as exc:  # pragma: no cover - streaming diagnostics
            self.after(0, lambda: self._handle_streaming_error(exc))
            return

        self.after(0, lambda: self._handle_streaming_completion(summary, job))

    def _handle_streaming_completion(self, summary, job) -> None:
        self.streaming_session = None
        self.streaming_job = None
        self.streaming_log_path = None
        self.is_running = False
        started = self.streaming_started_at
        self.streaming_started_at = None

        self.live_add_button.configure(state="disabled")
        self.live_finalize_button.configure(state="disabled")
        self.live_status_label.configure(text="실시간 세션을 시작하면 요약이 여기에 표시됩니다.")

        metrics: Optional[Dict[str, object]] = None
        if started is not None:
            metrics = {
                "duration_seconds": max(time.time() - started, 0.0),
                "mode": "streaming",
            }
        else:
            metrics = {"mode": "streaming"}

        self._handle_summary_completion(
            summary,
            job,
            headline="✅ 실시간 요약이 완료되었습니다!",
            completion_message="✅ 실시간 요약이 완료되었습니다.",
            metrics=metrics,
        )
        self._apply_context_constraints()

    def _handle_streaming_error(self, exc: Exception) -> None:
        self.streaming_session = None
        self.streaming_job = None
        self.streaming_log_path = None
        self.is_running = False
        self.streaming_started_at = None

        self.live_add_button.configure(state="disabled")
        self.live_finalize_button.configure(state="disabled")
        self.live_status_label.configure(text="실시간 세션을 시작하면 요약이 여기에 표시됩니다.")
        self.run_button.configure(text=self._default_run_button_label())
        self._apply_context_constraints()

        self.append_log(f"❌ 실시간 요약 중 오류가 발생했습니다: {exc}")
        self.end_task_callback("❌ 실시간 요약 중 오류가 발생했습니다.")

    def _handle_summary_completion(
        self,
        summary,
        job: MeetingJobConfig,
        *,
        headline: str,
        completion_message: str,
        metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        lines = [
            headline,
            f"출력 폴더: {job.output_dir}",
            "",
            "요약 하이라이트:",
        ]

        highlights = summary.highlights or []
        if highlights:
            lines.extend(f"- {item}" for item in highlights)
        else:
            lines.append("- (없음)")

        lines.append("")
        lines.append("액션 아이템:")
        actions = summary.action_items or []
        if actions:
            lines.extend(f"- {item}" for item in actions)
        else:
            lines.append("- (없음)")

        lines.append("")
        lines.append("결정 사항:")
        decisions = summary.decisions or []
        if decisions:
            lines.extend(f"- {item}" for item in decisions)
        else:
            lines.append("- (없음)")

        payload = {
            "output_dir": str(job.output_dir),
            "highlights": highlights,
            "action_items": actions,
            "decisions": decisions,
        }
        if job.policy_tag:
            payload["policy_tag"] = job.policy_tag
        diagnostics: Dict[str, object] = dict(self._last_pipeline_info or {})
        if metrics:
            if "mode" in metrics:
                payload.setdefault("mode", metrics["mode"])
            diagnostics.update(metrics)
        if diagnostics:
            payload["diagnostics"] = diagnostics
        payload["policy_enforced"] = get_policy_engine().has_policies
        if hasattr(self.app, "emit_work_center_event"):
            try:
                self.app.emit_work_center_event(
                    "meeting.summary.completed",
                    payload,
                    context=self.active_context,
                )
            except Exception:
                pass

        if summary.raw_summary:
            lines.append("")
            lines.append("자동 요약:")
            lines.append(summary.raw_summary)

        self.append_log("\n".join(lines), reset=True)
        self.last_output_dir = job.output_dir

        def _update_controls() -> None:
            self.open_folder_button.configure(state="normal")
            self.run_button.configure(text=self._default_run_button_label())
            self._apply_context_constraints()
            self.end_task_callback(completion_message)

        self.after(0, _update_controls)

    def _default_run_button_label(self) -> str:
        return "실시간 세션 시작" if self.live_mode_var.get() == 1 else "회의 요약 실행"

    def _describe_backend_choice(self) -> str:
        mapping = {
            "auto": "auto (환경 설정)",
            "whisper": "whisper (faster-whisper)",
            "wav2vec2": "wav2vec2 (HuggingFace)",
            "off": "off (비활성화)",
        }
        return mapping.get(self.stt_backend_var.get(), self.stt_backend_var.get())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def open_output_folder(self) -> None:
        if not self.last_output_dir or not self.last_output_dir.exists():
            self.append_log("⚠️ 열 수 있는 결과 폴더가 없습니다.")
            return

        path = self.last_output_dir
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[arg-type]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.append_log(f"⚠️ 폴더를 여는 중 오류가 발생했습니다: {exc}")
