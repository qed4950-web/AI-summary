"""Toplevel window for running the photo organisation pipeline."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from core.agents.photo.models import PhotoJobConfig
from core.agents.photo.pipeline import PhotoPipeline
from ui.utils import PHOTO_OUTPUT_DIR


class PhotoPanel(ctk.CTkToplevel):
    """Desktop surface for the photo assistant pipeline."""

    def __init__(
        self,
        master,
        *,
        on_activity: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:  # type: ignore[override]
        super().__init__(master)
        self.title("사진 비서")
        self.geometry("620x720")
        self.resizable(False, False)
        self.configure(fg_color="#11131c")

        self._on_activity = on_activity
        self._on_result = on_result
        self._task_thread: Optional[threading.Thread] = None

        self.pipeline = PhotoPipeline()
        self.selected_roots: List[Path] = []
        self.last_output_dir: Optional[Path] = None

        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        header = ctk.CTkLabel(
            self,
            text="사진 정리 파이프라인",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#F5F5F5",
        )
        header.pack(anchor="w", padx=24, pady=(24, 6))

        sub = ctk.CTkLabel(
            self,
            text="사진 폴더를 훑어 태그를 붙이고 중복/베스트샷을 제안합니다.",
            font=ctk.CTkFont(size=13),
            text_color="#AAB0BE",
        )
        sub.pack(anchor="w", padx=24, pady=(0, 16))

        folders = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        folders.pack(fill="x", padx=20, pady=(0, 16))
        folders.grid_columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(folders, fg_color="transparent")
        header_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        header_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header_row, text="스캔할 폴더", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(header_row, text="폴더 추가", width=110, command=self._add_folder).grid(row=0, column=1, padx=(12, 0))
        ctk.CTkButton(header_row, text="모두 제거", width=110, command=self._clear_folders).grid(row=0, column=2, padx=(12, 0))

        self.roots_box = ctk.CTkTextbox(
            folders,
            state="disabled",
            height=140,
            fg_color="#151a26",
            border_width=0,
        )
        self.roots_box.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="ew")
        self._refresh_roots()

        options = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        options.pack(fill="x", padx=20, pady=(0, 16))
        options.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options, text="정책 태그", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=16, pady=10, sticky="w")
        self.policy_var = ctk.StringVar()
        self.policy_entry = ctk.CTkEntry(
            options,
            textvariable=self.policy_var,
            placeholder_text="스마트 폴더 정책 태그 (선택)",
        )
        self.policy_entry.grid(row=0, column=1, padx=16, pady=10, sticky="ew")

        ctk.CTkLabel(options, text="출력 폴더", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=16, pady=10, sticky="w")
        out_row = ctk.CTkFrame(options, fg_color="transparent")
        out_row.grid(row=1, column=1, padx=16, pady=10, sticky="ew")
        out_row.grid_columnconfigure(0, weight=1)
        self.output_var = ctk.StringVar()
        self.output_entry = ctk.CTkEntry(
            out_row,
            textvariable=self.output_var,
            placeholder_text=f"비워두면 {PHOTO_OUTPUT_DIR.name}/<날짜>",
        )
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(out_row, text="폴더 지정", width=110, command=self._browse_output).grid(row=0, column=1, padx=(12, 0))

        self.gpu_switch = ctk.CTkSwitch(options, text="GPU 선호 (가능한 경우)")
        self.gpu_switch.grid(row=2, column=0, columnspan=2, padx=16, pady=(0, 12), sticky="w")

        buttons = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        buttons.pack(fill="x", padx=20, pady=(0, 16))
        buttons.grid_columnconfigure((0, 1), weight=1)

        self.run_button = ctk.CTkButton(
            buttons,
            text="📸 사진 정리 실행",
            height=40,
            command=self._run_photo_job,
        )
        self.run_button.grid(row=0, column=0, padx=16, pady=(16, 16), sticky="ew")
        self.open_folder_button = ctk.CTkButton(
            buttons,
            text="결과 폴더 열기",
            height=40,
            state="disabled",
            command=self._open_output_dir,
        )
        self.open_folder_button.grid(row=0, column=1, padx=16, pady=(16, 16), sticky="ew")

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
            height=160,
            fg_color="#151a26",
            border_width=0,
            font=ctk.CTkFont(family="Menlo", size=12),
        )
        self.log_box.pack(fill="both", expand=False, padx=24, pady=(0, 12))

        self.report_box = ctk.CTkTextbox(
            self,
            state="disabled",
            height=200,
            fg_color="#181c28",
            border_width=0,
        )
        self.report_box.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    # ------------------------------------------------------------------
    def _refresh_roots(self) -> None:
        text = "\n".join(str(path) for path in self.selected_roots) or "(선택된 폴더가 없습니다)"
        self.roots_box.configure(state="normal")
        self.roots_box.delete("1.0", "end")
        self.roots_box.insert("1.0", text)
        self.roots_box.configure(state="disabled")

    def _add_folder(self) -> None:
        path = filedialog.askdirectory(parent=self, title="사진 폴더 선택")
        if path:
            resolved = Path(path).expanduser()
            if resolved not in self.selected_roots:
                self.selected_roots.append(resolved)
                self._refresh_roots()

    def _clear_folders(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 작업이 끝난 뒤 폴더 목록을 수정할 수 있습니다.")
            return
        self.selected_roots.clear()
        self._refresh_roots()

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(parent=self, title="출력 폴더 선택")
        if path:
            self.output_var.set(path)

    def _open_output_dir(self) -> None:
        if not self.last_output_dir:
            return
        path = self.last_output_dir
        try:
            import subprocess
            if Path("/usr/bin/open").exists():
                subprocess.Popen(["open", str(path)])
            elif Path("/usr/bin/xdg-open").exists():
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover - best effort
            self._log(f"INFO: 폴더를 자동으로 열 수 없습니다 ({exc}). 경로: {path}")

    # ------------------------------------------------------------------
    def _run_photo_job(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요.")
            return
        if not self.selected_roots:
            self._log("ERROR: 먼저 사진 폴더를 추가하세요.")
            return

        output_root_text = self.output_var.get().strip()
        if output_root_text:
            output_dir = Path(output_root_text).expanduser()
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = PHOTO_OUTPUT_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = output_dir

        job = PhotoJobConfig(
            roots=self.selected_roots.copy(),
            output_dir=output_dir,
            policy_tag=self.policy_var.get().strip() or None,
            prefer_gpu=self.gpu_switch.get() == 1,
        )

        def _runner() -> None:
            self._set_status("사진 정리 실행 중...")
            self._toggle_controls(active=False)
            self._notify_activity(f"PHOTO · started ({len(job.roots)} folders)")
            self._log("사진 정리를 시작합니다...")
            try:
                result = self.pipeline.run(
                    job,
                    progress_callback=self._handle_progress_event,
                )
            except Exception as exc:
                self._log(f"ERROR: 사진 정리 실패 - {exc}")
                self._notify_activity(f"ERROR · photo failed: {exc}")
            else:
                self._log("SUCCESS: 사진 정리가 완료되었습니다.")
                report = result.report_path.read_text(encoding="utf-8") if result.report_path.exists() else ""
                self._show_report(report or "결과 리포트를 찾을 수 없습니다.")
                metadata = {
                    "report_path": str(result.report_path),
                    "output_dir": str(job.output_dir),
                    "root_count": len(job.roots),
                }
                self._notify_activity("PHOTO · completed")
                if self._on_result:
                    self._on_result(self._format_summary(result), metadata)
            finally:
                self.after(0, self._task_finished)

        self._task_thread = threading.Thread(target=_runner, daemon=True)
        self._task_thread.start()

    def _format_summary(self, recommendation) -> str:
        best = "\n".join(str(asset.path) for asset in recommendation.best_shots[:5])
        dup = len(recommendation.duplicates)
        return (
            "📸 사진 비서 결과\n"
            f"- 대표 사진 상위 5개:\n{best or '(없음)'}\n"
            f"- 중복 그룹 수: {dup}\n"
            f"- 리포트: {recommendation.report_path}"
        )

    def _handle_progress_event(self, event: Dict[str, object]) -> None:
        stage = event.get("stage") or "stage"
        status = event.get("status") or ""
        message = event.get("message") or ""
        text = f"[{stage}] {status}"
        if message:
            text += f" · {message}"
        self._log(text)

    def _task_finished(self) -> None:
        self._set_status("대기 중")
        self._toggle_controls(active=True)
        self.open_folder_button.configure(state="normal" if self.last_output_dir else "disabled")
        self._task_thread = None

    def _toggle_controls(self, *, active: bool) -> None:
        state = "normal" if active else "disabled"
        self.run_button.configure(state=state)
        self.policy_entry.configure(state=state)
        self.output_entry.configure(state=state)
        self.gpu_switch.configure(state=state)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _log(self, message: str) -> None:
        def _append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.configure(state="disabled")
            self.log_box.see("end")

        self.after(0, _append)

    def _show_report(self, report: str) -> None:
        def _update() -> None:
            self.report_box.configure(state="normal")
            self.report_box.delete("1.0", "end")
            self.report_box.insert("end", report)
            self.report_box.configure(state="disabled")
            self.report_box.see("1.0")

        self.after(0, _update)

    def _notify_activity(self, text: str) -> None:
        if self._on_activity:
            self._on_activity(text)

    def _handle_close(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 작업이 끝난 후 창이 닫힙니다.")
            return
        self.destroy()
