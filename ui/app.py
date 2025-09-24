import customtkinter as ctk
import sys
import os
from typing import Dict

# 프로젝트 루트 경로를 sys.path에 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure infopilot_core import aliases point to local core package (runtime parity with tests)
if "infopilot_core" not in sys.modules:
    try:
        import infopilot_core as _infopilot_core  # type: ignore
    except ImportError:
        import core as _core

        sys.modules["infopilot_core"] = _core
        for _name in ("agents", "conversation", "data_pipeline", "infra", "search", "utils"):
            module = __import__(f"core.{_name}", fromlist=[_name])
            sys.modules[f"infopilot_core.{_name}"] = module
            setattr(_core, _name, module)
    else:  # pragma: no cover - already available as installed package
        sys.modules.setdefault("infopilot_core", _infopilot_core)

# Import screen modules
from ui.screens.home_screen import HomeScreen
from ui.screens.chat_screen import ChatScreen
from ui.screens.train_screen import TrainScreen
from ui.screens.update_screen import UpdateScreen
from ui.screens.meeting_screen import MeetingScreen
from ui.screens.photo_screen import PhotoScreen


class App(ctk.CTk):
    """InfoPilot desktop shell that stitches every agent UI together."""

    NAV_DEFS = (
        ("home", "홈 대시보드"),
        ("chat", "지식·검색"),
        ("train", "전체 학습"),
        ("update", "증분 업데이트"),
        ("meeting", "회의 비서"),
        ("photos", "사진 정리"),
    )

    def __init__(self) -> None:
        super().__init__()

        self.title("InfoPilot 데스크톱 비서")
        self.geometry("1280x800")
        self.minsize(1100, 720)
        self.is_task_running = False
        self.active_frame_key: str = "home"

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar navigation ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(len(self.NAV_DEFS) + 2, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="InfoPilot",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=24, pady=(30, 4), sticky="w")

        self.tagline_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="문서 · 회의 · 사진 한 번에",
            font=ctk.CTkFont(size=13),
            text_color=("#444", "#d3d3d3"),
        )
        self.tagline_label.grid(row=1, column=0, padx=24, pady=(0, 20), sticky="w")

        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        for index, (key, label) in enumerate(self.NAV_DEFS, start=2):
            button = ctk.CTkButton(
                self.sidebar_frame,
                text=label,
                height=44,
                corner_radius=8,
                anchor="w",
                fg_color=("#2b7de9", "#225aa1") if key == "home" else "transparent",
                hover_color=("#2d83f3", "#1f5bb0"),
                text_color=("white", "white") if key == "home" else None,
                command=lambda name=key: self.select_frame(name),
            )
            button.grid(row=index, column=0, padx=18, pady=4, sticky="ew")
            self.nav_buttons[key] = button

        # Appearance mode switcher anchored to bottom
        self.appearance_switch = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode,
        )
        self.appearance_switch.set("System")
        self.appearance_switch.grid(row=len(self.NAV_DEFS) + 2, column=0, padx=24, pady=(20, 16), sticky="ew")

        # --- Header ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=1, padx=(24, 24), pady=(28, 8), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.header_title = ctk.CTkLabel(
            self.header_frame,
            text="",  # updated in select_frame
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.header_title.grid(row=0, column=0, sticky="w")

        self.header_subtitle = ctk.CTkLabel(
            self.header_frame,
            text="",  # updated in select_frame
            font=ctk.CTkFont(size=14),
            text_color=("#525252", "#d0d0d0"),
        )
        self.header_subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # --- Main frames ---
        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.grid(row=1, column=1, padx=(24, 24), pady=(0, 16), sticky="nsew")
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        self.frames: Dict[str, ctk.CTkFrame] = {
            "home": HomeScreen(self.content_container, app=self, corner_radius=0, fg_color="transparent"),
            "chat": ChatScreen(self.content_container, app=self, corner_radius=0, fg_color="transparent"),
            "train": TrainScreen(
                self.content_container,
                app=self,
                start_task_callback=self.start_task,
                end_task_callback=self.end_task,
                corner_radius=0,
                fg_color="transparent",
            ),
            "update": UpdateScreen(
                self.content_container,
                app=self,
                start_task_callback=self.start_task,
                end_task_callback=self.end_task,
                corner_radius=0,
                fg_color="transparent",
            ),
            "meeting": MeetingScreen(
                self.content_container,
                app=self,
                start_task_callback=self.start_task,
                end_task_callback=self.end_task,
                corner_radius=0,
                fg_color="transparent",
            ),
            "photos": PhotoScreen(
                self.content_container,
                app=self,
                start_task_callback=self.start_task,
                end_task_callback=self.end_task,
                corner_radius=0,
                fg_color="transparent",
            ),
        }

        # --- Status bar ---
        self.status_var = ctk.StringVar(value="Ready.")
        self.status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            anchor="w",
            height=28,
            corner_radius=6,
            padx=16,
            fg_color=("#e9f2ff", "#1c1c1c"),
            text_color=("#1f5bb1", "#e0e0e0"),
        )
        self.status_bar.grid(row=2, column=0, columnspan=2, padx=24, pady=(0, 20), sticky="ew")

        self.select_frame("home")

    # ------------------------------------------------------------------
    # Appearance and navigation helpers
    # ------------------------------------------------------------------
    def change_appearance_mode(self, new_mode: str) -> None:
        ctk.set_appearance_mode(new_mode)

    def select_frame(self, name: str) -> None:
        if self.is_task_running:
            self.status_var.set("⚙️ 작업이 완료될 때까지 다른 화면으로 이동할 수 없습니다.")
            return

        if name not in self.frames:
            raise KeyError(f"Unknown frame: {name}")

        # Hide previous frame
        for frame in self.frames.values():
            frame.grid_forget()

        # Update navigation button colors
        for key, button in self.nav_buttons.items():
            is_active = key == name
            button.configure(fg_color=("#2b7de9", "#225aa1") if is_active else "transparent",
                             text_color=("white", "white") if is_active else ("gray", "gray"),)


        # Show requested frame
        frame = self.frames[name]
        frame.grid(row=0, column=0, sticky="nsew")

        # Update header text
        self.active_frame_key = name
        header_texts = {
            "home": ("대시보드", "현재 상태를 확인하고 주요 작업으로 이동하세요."),
            "chat": ("지식·검색 비서", "학습된 문서를 대상으로 의미 검색을 수행합니다."),
            "train": ("전체 학습", "새로운 코퍼스를 만들고 인덱스를 재구축합니다."),
            "update": ("증분 업데이트", "변경된 파일만 빠르게 반영합니다."),
            "meeting": ("회의 비서", "오디오 또는 전사 파일을 요약하고 액션 아이템을 추출합니다."),
            "photos": ("사진 비서", "폴더를 정리하고 중복·베스트샷을 추천합니다."),
        }
        title, subtitle = header_texts.get(name, (name.title(), ""))
        self.header_title.configure(text=title)
        self.header_subtitle.configure(text=subtitle)

        # Allow frames to refresh themselves when shown
        if hasattr(frame, "on_show"):
            frame.on_show()

        self.status_var.set("Ready.")

    # ------------------------------------------------------------------
    # Long running task coordination
    # ------------------------------------------------------------------
    def start_task(self, status_message: str | None = None) -> None:
        """Disable navigation when a long task starts."""
        self.is_task_running = True
        for button in self.nav_buttons.values():
            button.configure(state="disabled")
        if status_message:
            self.status_var.set(status_message)
        else:
            self.status_var.set("⏳ 작업이 진행 중입니다...")

    def end_task(self, status_message: str | None = None) -> None:
        """Re-enable navigation once the long task ends."""
        self.is_task_running = False
        for button in self.nav_buttons.values():
            button.configure(state="normal")
        if status_message:
            self.status_var.set(status_message)
        else:
            self.status_var.set("작업이 완료되었습니다.")


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("dark-blue")

    app = App()
    app.mainloop()
