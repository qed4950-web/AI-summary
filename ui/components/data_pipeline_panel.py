"""Toplevel window exposing data pipeline actions (scan/train/index)."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional, Sequence

import customtkinter as ctk

from ui.utils import (
    CACHE_DIR,
    CORPUS_PARQUET,
    FOUND_FILES_CSV,
    SUPPORTED_EXTS,
    TOPIC_MODEL_PATH,
)
from ui.cli_runner import run_infopilot_cli


def _parse_extensions(raw: str) -> List[str]:
    exts: List[str] = []
    for token in (raw or "").split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        exts.append(cleaned)
    return exts


class DataPipelinePanel(ctk.CTkToplevel):
    """Desktop surface for file scanning and index maintenance."""

    def __init__(
        self,
        master,
        *,
        on_activity: Optional[Callable[[str], None]] = None,
        on_pipeline_complete: Optional[Callable[[], None]] = None,
    ) -> None:  # type: ignore[override]
        super().__init__(master)
        self.title("데이터 파이프라인")
        self.geometry("720x860")
        self.minsize(640, 720)
        self.resizable(True, True)
        self.configure(fg_color="#11131c")

        self._on_activity = on_activity
        self._on_pipeline_complete = on_pipeline_complete
        self._roots: List[str] = []
        self._task_thread: Optional[threading.Thread] = None

        self._build_layout()
        self._refresh_roots_list()

    # ------------------------------------------------------------------
    # UI composition
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        header = ctk.CTkLabel(
            self,
            text="문서 스캔 & 인덱스 관리",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#F5F5F5",
        )
        header.pack(anchor="w", padx=20, pady=(20, 8))

        sub = ctk.CTkLabel(
            self,
            text="루트 폴더와 확장자를 지정해 스캔/학습/재인덱스를 실행합니다.",
            font=ctk.CTkFont(size=13),
            text_color="#AAB0BE",
        )
        sub.pack(anchor="w", padx=20, pady=(0, 12))

        roots_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        roots_frame.pack(fill="x", padx=20, pady=(0, 12))

        title_row = ctk.CTkFrame(roots_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=14, pady=(14, 6))
        ctk.CTkLabel(
            title_row,
            text="스캔 루트",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F5F5F5",
        ).pack(side="left")
        ctk.CTkButton(
            title_row,
            text="폴더 추가",
            width=110,
            height=32,
            command=self._select_root,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            title_row,
            text="모두 지우기",
            width=110,
            height=32,
            command=self._clear_roots,
        ).pack(side="right")

        self.roots_list = ctk.CTkTextbox(
            roots_frame,
            state="disabled",
            height=120,
            fg_color="#151a26",
            border_width=0,
        )
        self.roots_list.pack(fill="x", padx=14, pady=(0, 14))

        options_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        options_frame.pack(fill="x", padx=20, pady=(0, 12))
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            options_frame,
            text="확장자 (콤마 구분)",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=14, pady=10, sticky="w")
        self.ext_entry = ctk.CTkEntry(options_frame)
        self.ext_entry.insert(0, ",".join(sorted(SUPPORTED_EXTS)))
        self.ext_entry.grid(row=0, column=1, padx=14, pady=10, sticky="ew")

        self.roots_only_switch = ctk.CTkSwitch(
            options_frame,
            text="선택한 루트만 스캔 (꺼두면 전체 드라이브)",
        )
        self.roots_only_switch.grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 12), sticky="w")
        self.roots_only_switch.configure(command=self._on_roots_mode_changed)

        self.ignore_policy_switch = ctk.CTkSwitch(
            options_frame,
            text="스마트 폴더 정책 무시",
            onvalue=1,
            offvalue=0,
        )
        self.ignore_policy_switch.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 12), sticky="w")
        self.ignore_policy_switch.select()
        self._on_roots_mode_changed()

        buttons_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        buttons_frame.pack(fill="x", padx=20, pady=(0, 12))
        buttons_frame.grid_columnconfigure((0, 1), weight=1)

        self.pipeline_button = ctk.CTkButton(
            buttons_frame,
            text="🚀 전체 파이프라인 실행",
            height=38,
            command=self._run_pipeline_all,
        )
        self.pipeline_button.grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="ew")

        self.scan_button = ctk.CTkButton(
            buttons_frame,
            text="🔍 스캔만 실행",
            height=34,
            command=self._run_scan,
        )
        self.scan_button.grid(row=1, column=0, padx=14, pady=6, sticky="ew")

        self.train_button = ctk.CTkButton(
            buttons_frame,
            text="🧠 학습 + 재인덱스",
            height=34,
            command=self._run_train,
        )
        self.train_button.grid(row=1, column=1, padx=14, pady=6, sticky="ew")

        self.index_button = ctk.CTkButton(
            buttons_frame,
            text="🧱 재인덱스만",
            height=34,
            command=self._run_index,
        )
        self.index_button.grid(row=2, column=0, columnspan=2, padx=14, pady=(6, 14), sticky="ew")

        self.status_label = ctk.CTkLabel(
            self,
            text="대기 중",
            font=ctk.CTkFont(size=12),
            text_color="#AAB0BE",
        )
        self.status_label.pack(fill="x", padx=20, pady=(0, 6))

        log_controls = ctk.CTkFrame(self, fg_color="transparent")
        log_controls.pack(fill="x", padx=20, pady=(0, 4))
        self.copy_log_button = ctk.CTkButton(
            log_controls,
            text="로그 복사",
            width=120,
            height=30,
            command=self._copy_log,
        )
        self.copy_log_button.pack(side="right")

        self.log_box = ctk.CTkTextbox(
            self,
            state="disabled",
            height=360,
            fg_color="#151a26",
            border_width=0,
            font=ctk.CTkFont(family="Menlo", size=12),
        )
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    # ------------------------------------------------------------------
    # Root management
    # ------------------------------------------------------------------
    def _select_root(self) -> None:
        path = filedialog.askdirectory(parent=self)
        if not path:
            return
        normalized = str(Path(path).expanduser())
        if normalized not in self._roots:
            self._roots.append(normalized)
            self._refresh_roots_list()

    def _clear_roots(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("WARNING: 실행 중에는 루트를 변경할 수 없습니다.")
            return
        self._roots.clear()
        self._refresh_roots_list()

    def _refresh_roots_list(self) -> None:
        text = "\n".join(self._roots) if self._roots else "(선택된 루트가 없습니다)"
        self.roots_list.configure(state="normal")
        self.roots_list.delete("1.0", "end")
        self.roots_list.insert("end", text)
        self.roots_list.configure(state="disabled")

    def _on_roots_mode_changed(self) -> None:
        if self.roots_only_switch.get():
            # 스마트 폴더 모드에서는 정책을 기본값(사용)으로 돌린다.
            self.ignore_policy_switch.deselect()
        else:
            # 전역 스캔 모드에서는 정책을 무시해 전체 드라이브를 탐색한다.
            self.ignore_policy_switch.select()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _run_scan(self) -> None:
        if self._task_inflight():
            return
        self._start_task("스캔 실행 중...", self._execute_scan)

    def _run_train(self) -> None:
        if self._task_inflight():
            return
        self._start_task("학습/재인덱스 실행 중...", self._execute_train)

    def _run_index(self) -> None:
        if self._task_inflight():
            return
        self._start_task("재인덱스 실행 중...", self._execute_index)

    def _run_pipeline_all(self) -> None:
        if self._task_inflight():
            return
        self._start_task("전체 파이프라인 실행 중...", self._execute_pipeline_all)

    # ------------------------------------------------------------------
    # Task orchestration
    # ------------------------------------------------------------------
    def _start_task(self, status: str, target: Callable[[], None]) -> None:
        self._set_status(status)
        self._toggle_buttons(active=False)
        self._log(f"=== {status}")

        def _runner() -> None:
            try:
                target()
            finally:
                self.after(0, self._task_finished)

        self._task_thread = threading.Thread(target=_runner, daemon=True)
        self._task_thread.start()

    def _task_finished(self) -> None:
        self._set_status("대기 중")
        self._toggle_buttons(active=True)
        self._task_thread = None
        if self._on_pipeline_complete:
            self._on_pipeline_complete()

    def _task_inflight(self) -> bool:
        if self._task_thread and self._task_thread.is_alive():
            self._log("WARNING: 다른 작업이 완료될 때까지 기다려 주세요.")
            return True
        return False

    def _toggle_buttons(self, *, active: bool) -> None:
        state = "normal" if active else "disabled"
        for btn in (self.pipeline_button, self.scan_button, self.train_button, self.index_button):
            btn.configure(state=state)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    # ------------------------------------------------------------------
    # CLI invocations
    # ------------------------------------------------------------------
    def _execute_scan(self) -> None:
        try:
            args = self._build_scan_args()
            self._run_cli(["run", "scan", *args], label="scan")
        except Exception as exc:
            self._log(f"ERROR: 스캔 실패 - {exc}")
            self._notify_activity(f"ERROR · scan failed: {exc}")
        else:
            self._log("SUCCESS: 스캔이 완료되었습니다.")
            self._notify_activity("SCAN · completed")

    def _execute_train(self) -> None:
        try:
            train_args = self._build_train_args()
            self._run_cli(["run", "train", *train_args], label="train")
            index_args = self._build_index_args()
            self._run_cli(["run", "index", *index_args], label="index")
        except Exception as exc:
            self._log(f"ERROR: 학습/재인덱스 실패 - {exc}")
            self._notify_activity(f"ERROR · train/index failed: {exc}")
        else:
            self._log("SUCCESS: 학습 및 재인덱스가 완료되었습니다.")
            self._notify_activity("TRAIN · completed")

    def _execute_index(self) -> None:
        try:
            args = self._build_index_args()
            self._run_cli(["run", "index", *args], label="index")
        except Exception as exc:
            self._log(f"ERROR: 재인덱스 실패 - {exc}")
            self._notify_activity(f"ERROR · index failed: {exc}")
        else:
            self._log("SUCCESS: 재인덱스가 완료되었습니다.")
            self._notify_activity("INDEX · completed")

    def _execute_pipeline_all(self) -> None:
        try:
            pipeline_args = self._build_pipeline_args()
            self._run_cli(["run", "pipeline", *pipeline_args], label="pipeline")
        except Exception as exc:
            self._log(f"ERROR: 전체 파이프라인 실패 - {exc}")
            self._notify_activity(f"ERROR · pipeline failed: {exc}")
        else:
            self._log("SUCCESS: 전체 파이프라인이 완료되었습니다.")
            self._notify_activity("PIPELINE · completed")

    def _run_cli(self, args: Sequence[str], *, label: str) -> None:
        self._log(f"$ python infopilot.py {' '.join(args)}")

        def _stream(line: str) -> None:
            if not line:
                return
            self._log(line)

        run_infopilot_cli(args, log_callback=_stream)

    # ------------------------------------------------------------------
    # Argument builders
    # ------------------------------------------------------------------
    def _build_scan_args(self) -> List[str]:
        args: List[str] = [
            "--out",
            str(FOUND_FILES_CSV),
        ]
        extensions = _parse_extensions(self.ext_entry.get())
        for ext in extensions:
            args.extend(["--ext", ext])
        if self.ignore_policy_switch.get():
            args.extend(["--policy", "none"])

        if self.roots_only_switch.get():
            if not self._roots:
                raise RuntimeError("스마트 폴더 모드를 켰다면 루트를 하나 이상 지정해야 합니다.")
            for root in self._roots:
                args.extend(["--root", root])
        return args

    def _build_train_args(self) -> List[str]:
        args: List[str] = [
            "--scan_csv",
            str(FOUND_FILES_CSV),
            "--corpus",
            str(CORPUS_PARQUET),
            "--model",
            str(TOPIC_MODEL_PATH),
            "--state-file",
            str(Path(CACHE_DIR) / "scan_state.json"),
            "--chunk-cache",
            str(Path(CACHE_DIR) / "chunk_cache.json"),
            "--async-embed",
            "--embedding-concurrency",
            "2",
        ]
        if self.ignore_policy_switch.get():
            args.extend(["--policy", "none"])
        return args

    def _build_index_args(self) -> List[str]:
        args = [
            "--model",
            str(TOPIC_MODEL_PATH),
            "--corpus",
            str(CORPUS_PARQUET),
            "--cache",
            str(CACHE_DIR),
        ]
        if self.ignore_policy_switch.get():
            args.extend(["--policy", "none"])
        return args

    def _build_pipeline_args(self) -> List[str]:
        args: List[str] = [
            "--out",
            str(FOUND_FILES_CSV),
            "--corpus",
            str(CORPUS_PARQUET),
            "--model",
            str(TOPIC_MODEL_PATH),
            "--cache",
            str(CACHE_DIR),
            "--async-embed",
            "--embedding-concurrency",
            "2",
        ]
        extensions = _parse_extensions(self.ext_entry.get())
        for ext in extensions:
            args.extend(["--ext", ext])
        if self.ignore_policy_switch.get():
            args.extend(["--policy", "none"])
        if self.roots_only_switch.get():
            if not self._roots:
                raise RuntimeError("스마트 폴더 모드를 켰다면 루트를 하나 이상 지정해야 합니다.")
            for root in self._roots:
                args.extend(["--roots", root])
        return args

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        def _append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.configure(state="disabled")
            self.log_box.see("end")

        self.after(0, _append)

    def _copy_log(self) -> None:
        text = self.log_box.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("로그가 클립보드에 복사되었습니다.")

    def _notify_activity(self, text: str) -> None:
        if self._on_activity:
            self._on_activity(text)

    # ------------------------------------------------------------------
    # Shutdown handling
    # ------------------------------------------------------------------
    def _handle_close(self) -> None:
        if self._task_thread and self._task_thread.is_alive():
            self._log("INFO: 작업이 끝난 후 창이 닫힙니다.")
            return
        self.destroy()
