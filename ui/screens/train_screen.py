
import customtkinter as ctk
import threading

from ui.api_client import (
    APIClientError,
    DEFAULT_API_BASE,
    cancel_pipeline,
    fetch_pipeline_status,
    trigger_pipeline_run,
)
from ui.utils import (
    DATA_DIR,
    CACHE_DIR,
    CORPUS_PARQUET,
    FOUND_FILES_CSV,
    SUPPORTED_EXTS,
    TOPIC_MODEL_PATH,
)
from ui.cli_runner import run_infopilot_cli

SCAN_STATE_PATH = DATA_DIR / "scan_state.json"
CHUNK_CACHE_PATH = CACHE_DIR / "chunk_cache.json"


def _parse_extensions(raw: str) -> list[str]:
    values: list[str] = []
    for token in (raw or "").split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        values.append(normalized)
    return values


def _run_full_train_logic(exts_text, do_scan, log_callback, done_callback):
    try:
        log_callback("INFO: CLI 파이프라인을 준비합니다.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        extensions = _parse_extensions(exts_text)
        if do_scan:
            log_callback("INFO: CLI 기반 스캔을 실행합니다...")
            scan_args = [
                "run",
                "scan",
                "--out",
                str(FOUND_FILES_CSV),
            ]
            for ext in extensions:
                scan_args.extend(["--ext", ext])
            run_infopilot_cli(scan_args, log_callback=log_callback)
        elif not FOUND_FILES_CSV.exists():
            log_callback("ERROR: 기존 스캔 결과가 없어 스캔 단계를 먼저 실행해야 합니다.")
            return

        log_callback("INFO: CLI 학습 단계를 실행합니다...")
        train_args = [
            "run",
            "train",
            "--scan_csv",
            str(FOUND_FILES_CSV),
            "--corpus",
            str(CORPUS_PARQUET),
            "--model",
            str(TOPIC_MODEL_PATH),
            "--state-file",
            str(SCAN_STATE_PATH),
            "--chunk-cache",
            str(CHUNK_CACHE_PATH),
            "--async-embed",
            "--embedding-concurrency",
            "2",
        ]
        run_infopilot_cli(train_args, log_callback=log_callback)

        log_callback("INFO: 벡터 인덱스를 재생성합니다...")
        index_args = [
            "run",
            "index",
            "--model",
            str(TOPIC_MODEL_PATH),
            "--corpus",
            str(CORPUS_PARQUET),
            "--cache",
            str(CACHE_DIR),
        ]
        run_infopilot_cli(index_args, log_callback=log_callback)

        log_callback("🎉 SUCCESS: CLI 파이프라인이 완료되었습니다!")

    except Exception as e:
        log_callback(f"FATAL: CLI 실행 중 오류 발생 - {e}")
    finally:
        done_callback()

class TrainScreen(ctk.CTkFrame):
    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="전체 학습 파이프라인",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=(0, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="모든 문서를 스캔하고 코퍼스·인덱스를 새로 생성합니다.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=2, column=0, padx=16, pady=12, sticky="ew")
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options_frame, text="검색할 확장자", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=12, pady=10)
        self.exts_entry = ctk.CTkEntry(options_frame)
        self.exts_entry.insert(0, ",".join(SUPPORTED_EXTS))
        self.exts_entry.grid(row=0, column=1, padx=12, pady=10, sticky="ew")

        self.scan_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="PC 전체 드라이브 스캔 실행 (시간이 오래 걸릴 수 있습니다)",
        )
        self.scan_checkbox.select()
        self.scan_checkbox.grid(row=1, column=0, columnspan=2, padx=12, pady=8, sticky="w")

        self.start_button = ctk.CTkButton(options_frame, text="▶️ 전체 학습 시작", command=self.start_training)
        self.start_button.grid(row=2, column=0, columnspan=2, padx=12, pady=(8, 10), sticky="ew")

        # FastAPI Pipeline control
        api_frame = ctk.CTkFrame(self)
        api_frame.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        api_frame.grid_columnconfigure(1, weight=1)
        api_frame.grid_columnconfigure(2, weight=0)

        ctk.CTkLabel(api_frame, text="FastAPI 제어 (선택)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )
        self.api_url_entry = ctk.CTkEntry(api_frame)
        self.api_url_entry.insert(0, DEFAULT_API_BASE)
        self.api_url_entry.grid(row=0, column=1, columnspan=2, padx=12, pady=(10, 6), sticky="ew")

        self.api_trigger_button = ctk.CTkButton(
            api_frame,
            text="🌐 API 실행",
            width=120,
            command=self.trigger_api_pipeline,
        )
        self.api_trigger_button.grid(row=1, column=0, padx=12, pady=6, sticky="ew")

        self.api_refresh_button = ctk.CTkButton(
            api_frame,
            text="📡 상태 새로고침",
            width=140,
            command=self.refresh_api_status,
        )
        self.api_refresh_button.grid(row=1, column=1, padx=12, pady=6, sticky="ew")

        self.api_cancel_button = ctk.CTkButton(
            api_frame,
            text="⛔ API 취소",
            width=120,
            fg_color="#b3261e",
            hover_color="#a01f18",
            command=self.cancel_api_pipeline,
        )
        self.api_cancel_button.grid(row=1, column=2, padx=12, pady=6, sticky="ew")

        self.api_status_label = ctk.CTkLabel(
            api_frame,
            text="API 서버 상태: 미연결",
            font=ctk.CTkFont(size=12),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.api_status_label.grid(row=2, column=0, columnspan=3, padx=12, pady=(6, 12), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="monospace"),
        )
        self.log_textbox.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="nsew")

    def log_message(self, message):
        self.after(0, self._insert_log, message)

    def _insert_log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def training_done(self):
        self.after(0, self._enable_button)
        self.end_task_callback("✅ 전체 학습이 완료되었습니다.")

    def _enable_button(self):
        self.start_button.configure(state="normal", text="▶️ 전체 학습 시작")

    def _set_api_status(self, message: str, *, error: bool = False) -> None:
        def _apply():
            color = ("#b3261e", "#ff9b9b") if error else ("#1f5bb1", "#c8e1ff")
            self.api_status_label.configure(text=f"API 서버 상태: {message}", text_color=color)

        self.after(0, _apply)

    def start_training(self):
        self.start_task_callback("⏳ 전체 학습 파이프라인을 실행 중입니다...")
        self.start_button.configure(state="disabled", text="학습 진행 중...")
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

        exts_text = self.exts_entry.get()
        do_scan = self.scan_checkbox.get() == 1

        train_thread = threading.Thread(
            target=_run_full_train_logic,
            args=(exts_text, do_scan, self.log_message, self.training_done)
        )
        train_thread.daemon = True
        train_thread.start()

    # ------------------------------------------------------------------
    # FastAPI helpers
    # ------------------------------------------------------------------
    def _api_payload(self) -> dict:
        return {
            "scan_csv": str(FOUND_FILES_CSV),
            "corpus": str(CORPUS_PARQUET),
            "model": str(TOPIC_MODEL_PATH),
            "cache": str(CACHE_DIR),
            "roots": [],
            "exts": list(_parse_extensions(self.exts_entry.get())),
        }

    def trigger_api_pipeline(self) -> None:
        base_url = self.api_url_entry.get().strip() or DEFAULT_API_BASE

        def _worker():
            try:
                resp = trigger_pipeline_run(base_url, self._api_payload())
            except APIClientError as exc:
                self._set_api_status(str(exc), error=True)
                return
            message = resp.get("message", "요청 전송 완료")
            self._set_api_status(message, error=False)

        threading.Thread(target=_worker, daemon=True).start()

    def refresh_api_status(self) -> None:
        base_url = self.api_url_entry.get().strip() or DEFAULT_API_BASE

        def _worker():
            try:
                status = fetch_pipeline_status(base_url)
            except APIClientError as exc:
                self._set_api_status(str(exc), error=True)
                return
            state = status.get("state")
            stage = status.get("stage") or "-"
            self._set_api_status(f"{state} (stage={stage})", error=False)

        threading.Thread(target=_worker, daemon=True).start()

    def cancel_api_pipeline(self) -> None:
        base_url = self.api_url_entry.get().strip() or DEFAULT_API_BASE

        def _worker():
            try:
                cancel_pipeline(base_url)
            except APIClientError as exc:
                self._set_api_status(str(exc), error=True)
                return
            self._set_api_status("취소 요청 전송 완료", error=False)

        threading.Thread(target=_worker, daemon=True).start()
