
import customtkinter as ctk
import os
import pandas as pd
import time
import threading
from pathlib import Path

# Core logic and helpers
from ui.utils import (
    get_drives,
    EXCLUDE_DIRS,
    SUPPORTED_EXTS,
    DATA_DIR,
    CACHE_DIR,
    CORPUS_PARQUET,
    FOUND_FILES_CSV,
    TOPIC_MODEL_PATH,
    rebuild_index,
)
from core.data_pipeline.pipeline import TrainConfig, run_step2

def _run_full_train_logic(exts_text, do_scan, log_callback, done_callback):
    try:
        log_callback("INFO: 필요 디렉토리 생성 중...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        rows = None
        if do_scan:
            log_callback("INFO: 드라이브 스캔 시작...")
            current_supported_exts = {e.strip() for e in exts_text.split(",") if e.strip()}
            file_list = []
            for drive in get_drives():
                log_callback(f"INFO: {drive} 스캔 중...")
                for root, dirs, files in os.walk(drive, topdown=True):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for file in files:
                        try:
                            p_file = Path(root) / file
                            if p_file.suffix.lower() in current_supported_exts:
                                if not any(part in EXCLUDE_DIRS for part in p_file.parts):
                                    stat = p_file.stat()
                                    file_list.append(
                                        {
                                            'path': str(p_file),
                                            'size': stat.st_size,
                                            'mtime': stat.st_mtime,
                                            'ext': p_file.suffix.lower(),
                                        }
                                    )
                        except (FileNotFoundError, PermissionError):
                            continue
            rows = file_list
            pd.DataFrame(rows).to_csv(FOUND_FILES_CSV, index=False, encoding="utf-8")
            log_callback(f"SUCCESS: 스캔 완료. {len(rows):,}개 파일 발견.")

        if rows is None and FOUND_FILES_CSV.exists():
            rows = pd.read_csv(FOUND_FILES_CSV).to_dict("records")

        if not rows:
            log_callback("ERROR: 처리할 파일 목록이 없습니다. 스캔을 먼저 수행하세요.")
            done_callback()
            return

        log_callback("INFO: 텍스트 추출 및 임베딩 학습을 실행합니다... (콘솔에 진행률 표시)")
        for record in rows:
            record.setdefault('ext', Path(record['path']).suffix.lower())
        try:
            cfg = TrainConfig(use_sentence_transformer=False)
            run_step2(
                rows,
                out_corpus=CORPUS_PARQUET,
                out_model=TOPIC_MODEL_PATH,
                cfg=cfg,
                use_tqdm=False,
                translate=False,
            )
            log_callback("SUCCESS: 코퍼스와 토픽 모델 학습 완료.")
        except Exception as exc:
            log_callback(f"FATAL: 학습 단계에서 오류가 발생했습니다: {exc}")
            done_callback()
            return

        log_callback("INFO: 벡터 인덱스를 재생성합니다... (잠시 기다려주세요)")
        rebuild_index(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
        log_callback("SUCCESS: 인덱싱 완료.")

        log_callback("🎉 SUCCESS: 모든 학습 과정이 완료되었습니다!")

    except Exception as e:
        log_callback(f"FATAL: 학습 중 오류 발생 - {e}")
    finally:
        done_callback()

class TrainScreen(ctk.CTkFrame):
    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

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

        self.log_textbox = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="monospace"),
        )
        self.log_textbox.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")

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
