from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog

from core.agents.meeting.models import MeetingJobConfig
from core.agents.meeting.pipeline import MeetingPipeline
from src.config import MEETING_OUTPUT_DIR


class MeetingScreen(ctk.CTkFrame):
    """UI bridge for the meeting agent MVP."""

    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.is_running = False
        self.last_output_dir: Optional[Path] = None
        self.stt_backend_var = ctk.StringVar(value="auto")
        self.stt_model_var = ctk.StringVar()
        self.stt_device_var = ctk.StringVar()
        self.stt_compute_var = ctk.StringVar()
        self.stt_download_var = ctk.StringVar()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

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

        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.grid(row=2, column=0, padx=16, pady=12, sticky="ew")
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
            values=["auto", "whisper", "off"],
            variable=self.stt_backend_var,
            command=self.on_stt_backend_change,
            width=130,
        )
        self.stt_backend_menu.set("auto")
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
            text="auto=환경 설정, whisper=faster-whisper, off=비활성화",
            font=ctk.CTkFont(size=11),
            text_color=("#636363", "#bdbdbd"),
        )
        help_label.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="w")

        self.on_stt_backend_change("auto")

        # Diarisation controls
        diar_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        diar_row.grid(row=4, column=1, padx=12, pady=8, sticky="ew")
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

        # Action buttons
        button_row = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        button_row.grid(row=5, column=1, padx=12, pady=(8, 12), sticky="ew")
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
        self.log_textbox.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.append_log("회의 요약을 실행할 파일을 선택하세요.", reset=True)

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

    def on_toggle_diarize(self) -> None:
        enabled = self.diarize_switch.get() == 1
        state = "normal" if enabled else "disabled"
        self.speaker_entry.configure(state=state)
        if not enabled:
            self.speaker_var.set("")

    def on_stt_backend_change(self, _: str) -> None:
        backend = self.stt_backend_var.get()
        is_whisper = backend == "whisper"
        state = "normal" if is_whisper else "disabled"
        for entry in (
            self.stt_model_entry,
            self.stt_device_entry,
            self.stt_compute_entry,
            self.stt_download_entry,
        ):
            entry.configure(state=state)
        if not is_whisper:
            self.stt_model_var.set("")
            self.stt_device_var.set("")
            self.stt_compute_var.set("")
            self.stt_download_var.set("")

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def start_meeting_job(self) -> None:
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

        output_root_text = self.output_dir_var.get().strip()
        if output_root_text:
            output_dir = Path(output_root_text)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_name = audio_path.stem or "meeting"
            output_dir = MEETING_OUTPUT_DIR / safe_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

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

        pipeline = self._build_pipeline()
        thread = threading.Thread(target=self._run_pipeline, args=(pipeline, job), daemon=True)
        thread.start()

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

        return MeetingPipeline(stt_backend=backend, stt_options=stt_options)

    def _run_pipeline(self, pipeline: MeetingPipeline, job: MeetingJobConfig) -> None:
        try:
            summary = pipeline.run(job)
            lines = [
                "✅ 회의 요약이 완료되었습니다!",
                f"출력 폴더: {job.output_dir}",
                "",
                "요약 하이라이트:",
            ]
            lines.extend(f"- {item}" for item in summary.highlights)
            lines.append("")
            lines.append("액션 아이템:")
            lines.extend(f"- {item}" for item in summary.action_items)
            lines.append("")
            lines.append("결정 사항:")
            lines.extend(f"- {item}" for item in summary.decisions)
            if summary.raw_summary:
                lines.append("")
                lines.append("자동 요약:")
                lines.append(summary.raw_summary)

            self.append_log("\n".join(lines), reset=True)
            self.after(0, lambda: self.open_folder_button.configure(state="normal"))
            self.after(0, lambda: self.run_button.configure(state="normal", text="회의 요약 실행"))
            self.after(0, lambda: self.end_task_callback("✅ 회의 요약이 완료되었습니다."))
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.append_log(f"❌ 회의 요약 중 오류가 발생했습니다: {exc}")
            self.after(0, lambda: self.run_button.configure(state="normal", text="회의 요약 실행"))
            self.after(0, lambda: self.end_task_callback("❌ 회의 요약 중 오류가 발생했습니다."))
        finally:
            self.is_running = False

    def _describe_backend_choice(self) -> str:
        mapping = {
            "auto": "auto (환경 설정)",
            "whisper": "whisper (faster-whisper)",
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
