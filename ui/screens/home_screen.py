import customtkinter as ctk
from typing import Dict

from src.config import (
    CORPUS_PARQUET,
    TOPIC_MODEL_PATH,
    DATA_DIR,
    MODELS_DIR,
    CACHE_DIR,
)
from src.core.helpers import have_all_artifacts


class HomeScreen(ctk.CTkFrame):
    """Landing page that surfaces quick actions and status chips."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text="환영합니다!",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=12, pady=(4, 2), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="InfoPilot이 문서, 회의, 사진을 자동으로 정리하도록 설정하세요.",
            font=ctk.CTkFont(size=14),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")

        # --- Status + quick action cards ---
        self.status_badge = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=8,
            padx=14,
            pady=6,
            fg_color=("#ddebf9", "#1f2b3e"),
            text_color=("#0f4aa3", "#cde1ff"),
        )
        self.status_badge.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")

        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.grid(row=3, column=0, padx=6, pady=(0, 12), sticky="nsew")
        self.cards_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="card")

        card_specs = [
            {
                "key": "chat",
                "emoji": "💡",
                "title": "지식·검색",
                "desc": "학습한 문서를 의미 기반으로 검색하고 답을 찾습니다.",
                "command": lambda: self.app.select_frame("chat"),
                "requires_artifacts": True,
            },
            {
                "key": "update",
                "emoji": "⚡",
                "title": "증분 업데이트",
                "desc": "새로나 바뀐 파일만 빠르게 인덱스에 반영합니다.",
                "command": lambda: self.app.select_frame("update"),
                "requires_artifacts": True,
            },
            {
                "key": "train",
                "emoji": "🚀",
                "title": "전체 학습",
                "desc": "처음부터 모든 문서를 스캔하고 인덱스를 재생성합니다.",
                "command": lambda: self.app.select_frame("train"),
                "requires_artifacts": False,
            },
            {
                "key": "meeting",
                "emoji": "📝",
                "title": "회의 요약",
                "desc": "오디오 또는 전사 파일에서 요약과 액션 아이템을 추출합니다.",
                "command": lambda: self.app.select_frame("meeting"),
                "requires_artifacts": False,
            },
            {
                "key": "photos",
                "emoji": "📸",
                "title": "사진 정리",
                "desc": "중복 사진을 묶고 베스트샷을 추천합니다.",
                "command": lambda: self.app.select_frame("photos"),
                "requires_artifacts": False,
            },
        ]

        self.card_widgets: Dict[str, Dict[str, ctk.CTkBaseClass]] = {}
        for idx, spec in enumerate(card_specs):
            row, col = divmod(idx, 3)
            card = ctk.CTkFrame(self.cards_frame, corner_radius=12)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            card.grid_rowconfigure(2, weight=1)

            icon = ctk.CTkLabel(card, text=spec["emoji"], font=ctk.CTkFont(size=32))
            icon.grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")

            title = ctk.CTkLabel(card, text=spec["title"], font=ctk.CTkFont(size=18, weight="bold"))
            title.grid(row=1, column=0, padx=16, pady=(0, 4), sticky="w")

            desc = ctk.CTkLabel(
                card,
                text=spec["desc"],
                wraplength=260,
                justify="left",
                font=ctk.CTkFont(size=13),
                text_color=("#4e4e4e", "#cfcfcf"),
            )
            desc.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="nsew")

            button = ctk.CTkButton(card, text="바로가기", command=spec["command"], height=36)
            button.grid(row=3, column=0, padx=16, pady=(6, 16), sticky="ew")

            self.card_widgets[spec["key"]] = {"frame": card, "button": button, "desc": desc}

        # --- Path info block ---
        self.path_info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.path_info_frame.grid(row=4, column=0, padx=12, pady=(8, 0), sticky="ew")
        self.path_info_frame.grid_columnconfigure(0, weight=1)

        path_label = ctk.CTkLabel(
            self.path_info_frame,
            text="프로젝트 경로",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        path_label.grid(row=0, column=0, sticky="w")

        path_text = (
            f"데이터 디렉토리 : {DATA_DIR}\n"
            f"모델 디렉토리  : {MODELS_DIR}\n"
            f"인덱스 캐시    : {CACHE_DIR}\n"
            f"코퍼스 파일    : {CORPUS_PARQUET} (존재: {CORPUS_PARQUET.exists()})\n"
            f"토픽 모델      : {TOPIC_MODEL_PATH} (존재: {TOPIC_MODEL_PATH.exists()})"
        )

        self.path_textbox = ctk.CTkTextbox(
            self.path_info_frame,
            height=120,
            font=ctk.CTkFont(family="monospace", size=12),
        )
        self.path_textbox.insert("1.0", path_text)
        self.path_textbox.configure(state="disabled")
        self.path_textbox.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.refresh_state()

    def refresh_state(self) -> None:
        artifacts_ready = have_all_artifacts()
        if artifacts_ready:
            self.status_badge.configure(text="✅ 코퍼스와 모델이 준비되었습니다.")
        else:
            self.status_badge.configure(text="⚠️ 학습 데이터가 없습니다. 먼저 전체 학습을 실행하세요.")

        for key, widgets in self.card_widgets.items():
            requires = key in {"chat", "update"}
            button: ctk.CTkButton = widgets["button"]  # type: ignore[assignment]
            if requires and not artifacts_ready:
                button.configure(state="disabled")
            else:
                button.configure(state="normal")

        # refresh path existence info
        path_text = (
            f"데이터 디렉토리 : {DATA_DIR}\n"
            f"모델 디렉토리  : {MODELS_DIR}\n"
            f"인덱스 캐시    : {CACHE_DIR}\n"
            f"코퍼스 파일    : {CORPUS_PARQUET} (존재: {CORPUS_PARQUET.exists()})\n"
            f"토픽 모델      : {TOPIC_MODEL_PATH} (존재: {TOPIC_MODEL_PATH.exists()})"
        )
        self.path_textbox.configure(state="normal")
        self.path_textbox.delete("1.0", "end")
        self.path_textbox.insert("1.0", path_text)
        self.path_textbox.configure(state="disabled")
