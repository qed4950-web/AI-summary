
import customtkinter as ctk
import os
import pandas as pd
import time
import joblib
import shutil
import threading
from pathlib import Path

# Core logic and helpers
from ui.utils import (
    get_drives,
    have_all_artifacts,
    EXCLUDE_DIRS,
    SUPPORTED_EXTS,
    DATA_DIR,
    MODELS_DIR,
    CACHE_DIR,
    CORPUS_PARQUET,
    FOUND_FILES_CSV,
    TOPIC_MODEL_PATH,
    CorpusBuilder,
    rebuild_index,
)

def _run_update_index_logic(log_callback, done_callback):
    try:
        # 1. 기존 코퍼스 로드 및 유효성 검사
        if not CORPUS_PARQUET.exists():
            log_callback("ERROR: 기존 학습 데이터가 없습니다. 전체 학습을 먼저 실행해주세요.")
            done_callback()
            return

        log_callback("INFO: 기존 데이터 로드 중...")
        old_df = pd.read_parquet(CORPUS_PARQUET)
        if old_df.empty or 'mtime' not in old_df.columns or 'size' not in old_df.columns:
            log_callback("ERROR: 기존 데이터에 파일 메타정보(mtime, size)가 없습니다.")
            log_callback("INFO: 정확한 업데이트를 위해 [전체 학습]을 먼저 실행해주세요.")
            done_callback()
            return
        log_callback(f"SUCCESS: 기존 데이터 로드 완료. ({len(old_df)}개 파일)")

        # 2. 현재 PC 파일 스캔
        log_callback("INFO: PC 전체 파일 스캔 중...")
        current_files_list = []
        for drive in get_drives():
            log_callback(f"INFO: {drive} 스캔 중...")
            for root, dirs, files in os.walk(drive, topdown=True):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    try:
                        p_file = Path(root) / file
                        if p_file.suffix.lower() in SUPPORTED_EXTS:
                                if not any(part in EXCLUDE_DIRS for part in p_file.parts):
                                    stat = p_file.stat()
                                    current_files_list.append(
                                        {
                                            'path': str(p_file),
                                            'size': stat.st_size,
                                            'mtime': stat.st_mtime,
                                            'ext': p_file.suffix.lower(),
                                        }
                                    )
                    except (FileNotFoundError, PermissionError): continue
        current_df = pd.DataFrame(current_files_list)
        log_callback(f"SUCCESS: PC 스캔 완료. ({len(current_df)}개 파일 발견)")

        # 3. 변경사항 감지
        log_callback("INFO: 변경사항 감지 중...")
        old_df_for_merge = old_df[['path', 'size', 'mtime']].copy().rename(columns={'size': 'size_old', 'mtime': 'mtime_old'})
        current_df_for_merge = current_df[['path', 'size', 'mtime']].copy().rename(columns={'size': 'size_new', 'mtime': 'mtime_new'})
        merged_df = pd.merge(old_df_for_merge, current_df_for_merge, on='path', how='outer')

        new_files_info = merged_df[merged_df['mtime_old'].isna()][['path', 'size_new', 'mtime_new']].rename(columns={'size_new': 'size', 'mtime_new': 'mtime'}).to_dict('records')
        deleted_paths = merged_df[merged_df['mtime_new'].isna()]['path'].tolist()
        modified_files_df = merged_df[merged_df['mtime_old'].notna() & merged_df['mtime_new'].notna() & ((merged_df['mtime_new'] > merged_df['mtime_old']) | (merged_df['size_new'] != merged_df['size_old']))]
        modified_files_info = modified_files_df[['path', 'size_new', 'mtime_new']].rename(columns={'size_new': 'size', 'mtime_new': 'mtime'}).to_dict('records')
        log_callback(f"INFO: 신규 {len(new_files_info)}개, 수정 {len(modified_files_info)}개, 삭제 {len(deleted_paths)}개")

        # 4. 코퍼스 업데이트
        def _with_ext(records):
            enriched = []
            for record in records:
                path = record.get('path')
                try:
                    ext = Path(path).suffix.lower()
                except Exception:
                    ext = ''
                enriched.append({**record, 'ext': ext})
            return enriched

        files_to_process = _with_ext(new_files_info + modified_files_info)
        new_extracted_df = pd.DataFrame()
        if files_to_process:
            log_callback(f"INFO: {len(files_to_process)}개 파일 텍스트 추출 중... (진행률은 콘솔 창에 표시됩니다)")
            cb = CorpusBuilder(progress=True)
            new_extracted_df = cb.build(files_to_process)
        
        paths_to_remove = set(deleted_paths) | {f['path'] for f in modified_files_info}
        updated_old_df = old_df[~old_df['path'].isin(paths_to_remove)].copy()
        final_corpus_df = pd.concat([updated_old_df, new_extracted_df], ignore_index=True)

        if not final_corpus_df.empty:
            CorpusBuilder.save(final_corpus_df, CORPUS_PARQUET)
            log_callback("SUCCESS: 코퍼스 업데이트 완료.")
        else:
            log_callback("WARNING: 업데이트 후 코퍼스가 비어있습니다.")
            if CORPUS_PARQUET.exists(): CORPUS_PARQUET.unlink()

        # 5. 인덱스 재생성
        if CORPUS_PARQUET.exists() and not pd.read_parquet(CORPUS_PARQUET).empty:
            log_callback("INFO: 벡터 인덱스 재생성 중... (진행률은 콘솔 창에 표시됩니다)")
            rebuild_index(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
            log_callback("SUCCESS: 인덱스 재생성 완료.")
        else:
            log_callback("WARNING: 코퍼스가 비어있어 인덱싱을 건너뜁니다.")
            if CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR); CACHE_DIR.mkdir(parents=True, exist_ok=True)

        log_callback("INFO: 메타 정보 저장 중...")
        joblib.dump({'indexed_at': time.strftime("%Y-%m-%d %H:%M:%S")}, TOPIC_MODEL_PATH)
        log_callback("🎉 SUCCESS: 모든 업데이트 과정 완료!")

    except Exception as e:
        log_callback(f"FATAL: 업데이트 중 오류 발생 - {e}")
    finally:
        done_callback()

class UpdateScreen(ctk.CTkFrame):
    def __init__(self, master, app, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="증분 업데이트",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=(0, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="기존 코퍼스를 기준으로 신규·수정·삭제된 파일만 반영합니다.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        self.warning_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=15))
        self.train_button_redirect = ctk.CTkButton(
            self,
            text="🚀 전체 학습 실행",
            command=lambda: self.app.select_frame("train"),
        )

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.grid_columnconfigure(0, weight=1)
        self.start_button = ctk.CTkButton(self.options_frame, text="▶️ 업데이트 시작", command=self.start_update)
        self.log_textbox = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="monospace"),
        )

        self.refresh_state()

    def setup_ui(self):
        # This method is no longer directly called, its logic is integrated into refresh_state
        pass

    def refresh_state(self):
        # Clear previous state
        self.warning_label.grid_forget()
        self.train_button_redirect.grid_forget()
        self.options_frame.grid_forget()
        self.log_textbox.grid_forget()

        if not have_all_artifacts():
            self.warning_label.configure(text="⚠️ 학습 데이터가 없어 업데이트를 수행할 수 없습니다.")
            self.warning_label.grid(row=2, column=0, pady=(60, 12))
            self.train_button_redirect.grid(row=3, column=0, pady=(0, 12))
        else:
            # Re-create/show options_frame and log_textbox
            self.options_frame.grid(row=2, column=0, padx=16, pady=12, sticky="ew")
            ctk.CTkLabel(
                self.options_frame,
                text="새로 추가되거나 수정된 파일만 효율적으로 업데이트합니다.",
                justify="left",
            ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
            self.start_button.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

            self.log_textbox.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def log_message(self, message):
        self.after(0, self._insert_log, message)

    def _insert_log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def update_done(self):
        self.after(0, self._enable_button)
        self.end_task_callback("✅ 증분 업데이트가 완료되었습니다.")

    def _enable_button(self):
        self.start_button.configure(state="normal", text="▶️ 업데이트 시작")

    def start_update(self):
        self.start_task_callback("⏳ 증분 업데이트를 실행 중입니다...")
        self.start_button.configure(state="disabled", text="업데이트 진행 중...")
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

        update_thread = threading.Thread(target=_run_update_index_logic, args=(self.log_message, self.update_done), daemon=True)
        update_thread.start()
