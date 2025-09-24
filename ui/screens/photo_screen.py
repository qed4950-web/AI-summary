from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk
from tkinter import filedialog

from core.agents.photo.models import PhotoJobConfig
from core.agents.photo.pipeline import PhotoPipeline
from src.config import PHOTO_OUTPUT_DIR


class PhotoScreen(ctk.CTkFrame):
    """UI bridge for the photo organisation MVP."""

    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.pipeline = PhotoPipeline()
        self.selected_roots: List[Path] = []
        self.is_running = False
        self.last_output_dir: Optional[Path] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="사진 비서",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=(0, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="폴더를 훑어 태그를 붙이고 중복 그룹과 베스트샷을 추천합니다.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        # Folder selection block
        folder_frame = ctk.CTkFrame(self)
        folder_frame.grid(row=2, column=0, padx=16, pady=12, sticky="ew")
        folder_frame.grid_columnconfigure(0, weight=1)

        button_row = ctk.CTkFrame(folder_frame, fg_color="transparent")
        button_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        button_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(button_row, text="스캔할 폴더", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(button_row, text="폴더 추가", width=110, command=self.add_folder).grid(
            row=0, column=1, padx=(12, 0)
        )
        ctk.CTkButton(button_row, text="모두 제거", width=110, command=self.clear_folders).grid(
            row=0, column=2, padx=(12, 0)
        )

        self.roots_textbox = ctk.CTkTextbox(folder_frame, height=140, state="disabled")
        self.roots_textbox.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        self._update_roots_textbox()

        # Options frame
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=3, column=0, padx=16, pady=12, sticky="ew")
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options_frame, text="정책 태그", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=8, sticky="w"
        )
        self.policy_var = ctk.StringVar()
        self.policy_entry = ctk.CTkEntry(
            options_frame,
            textvariable=self.policy_var,
            placeholder_text="스마트 폴더 정책 태그 (선택)",
        )
        self.policy_entry.grid(row=0, column=1, padx=12, pady=8, sticky="ew")

        ctk.CTkLabel(options_frame, text="출력 폴더", font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, padx=12, pady=8, sticky="w"
        )
        output_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        output_row.grid(row=1, column=1, padx=12, pady=8, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        self.output_dir_var = ctk.StringVar()
        self.output_entry = ctk.CTkEntry(
            output_row,
            textvariable=self.output_dir_var,
            placeholder_text=f"비워두면 {PHOTO_OUTPUT_DIR.name}/<날짜> 폴더가 생성됩니다",
        )
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(output_row, text="폴더 지정", width=110, command=self.browse_output_dir).grid(
            row=0, column=1, padx=(12, 0)
        )

        self.gpu_switch = ctk.CTkSwitch(options_frame, text="GPU 선호 (사용 가능한 경우)")
        self.gpu_switch.grid(row=2, column=0, columnspan=2, padx=12, pady=8, sticky="w")

        button_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        button_row.grid(row=3, column=0, columnspan=2, padx=12, pady=(8, 12), sticky="ew")
        button_row.grid_columnconfigure(0, weight=1)
        self.run_button = ctk.CTkButton(button_row, text="사진 정리 실행", command=self.start_photo_job)
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
        self.append_log("정리할 사진 폴더를 추가하세요.", reset=True)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _update_roots_textbox(self) -> None:
        text = "\n".join(str(path) for path in self.selected_roots) or "(선택된 폴더가 없습니다)"
        self.roots_textbox.configure(state="normal")
        self.roots_textbox.delete("1.0", "end")
        self.roots_textbox.insert("1.0", text)
        self.roots_textbox.configure(state="disabled")

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

    def add_folder(self) -> None:
        directory = filedialog.askdirectory(title="사진 폴더 선택")
        if directory:
            path = Path(directory)
            if path not in self.selected_roots:
                self.selected_roots.append(path)
                self._update_roots_textbox()

    def clear_folders(self) -> None:
        self.selected_roots.clear()
        self._update_roots_textbox()

    def browse_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="출력 폴더 선택")
        if directory:
            self.output_dir_var.set(directory)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def start_photo_job(self) -> None:
        if self.is_running:
            return
        if not self.selected_roots:
            self.append_log("⚠️ 먼저 사진 폴더를 추가하세요.")
            return

        output_root_text = self.output_dir_var.get().strip()
        if output_root_text:
            output_dir = Path(output_root_text)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = PHOTO_OUTPUT_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        job = PhotoJobConfig(
            roots=self.selected_roots.copy(),
            output_dir=output_dir,
            policy_tag=self.policy_var.get().strip() or None,
            prefer_gpu=self.gpu_switch.get() == 1,
        )

        self.append_log("사진 정리를 실행합니다...", reset=True)
        self.is_running = True
        self.last_output_dir = output_dir
        self.run_button.configure(state="disabled", text="처리 중...")
        self.open_folder_button.configure(state="disabled")
        self.start_task_callback("⏳ 사진 정리를 실행 중입니다...")

        thread = threading.Thread(target=self._run_pipeline, args=(job,), daemon=True)
        thread.start()

    def _run_pipeline(self, job: PhotoJobConfig) -> None:
        try:
            recommendation = self.pipeline.run(job)
            best_preview = "\n".join(str(asset.path) for asset in recommendation.best_shots[:10])
            duplicate_preview = []
            for group in recommendation.duplicates[:5]:
                duplicate_preview.append("- " + "\n  ".join(str(asset.path) for asset in group))
            lines = [
                "✅ 사진 정리가 완료되었습니다!",
                f"결과 폴더: {job.output_dir}",
                "",
                "베스트샷 상위 10:",
                best_preview or "(베스트샷 후보가 없습니다)",
                "",
                "중복 그룹 일부:",
                "\n".join(duplicate_preview) or "(중복으로 감지된 그룹이 없습니다)",
                "",
                f"보고서 파일: {recommendation.report_path}",
            ]
            self.append_log("\n".join(lines), reset=True)
            self.after(0, lambda: self.open_folder_button.configure(state="normal"))
            self.after(0, lambda: self.run_button.configure(state="normal", text="사진 정리 실행"))
            self.after(0, lambda: self.end_task_callback("✅ 사진 정리가 완료되었습니다."))
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.append_log(f"❌ 사진 정리 중 오류가 발생했습니다: {exc}")
            self.after(0, lambda: self.run_button.configure(state="normal", text="사진 정리 실행"))
            self.after(0, lambda: self.end_task_callback("❌ 사진 정리 중 오류가 발생했습니다."))
        finally:
            self.is_running = False

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
