🧭 Atlas 스타일 데스크톱 UI (CustomTkinter 버전) 세트를 바로 만들어드리겠습니다.

이 세트는 기존 launch_desktop.py → ui/app.py 실행 흐름을 그대로 사용하면서,
아래처럼 Compact → Expanded 전환형 UI로 작동합니다 👇

📁 생성 구조
AI-summary/ui/
├── app.py
├── components/
│   ├── sidebar.py
│   ├── chat_panel.py
│   ├── input_dock.py
│   └── work_center_panel.py  # Work Center & 리소스 로그 뷰어
└── themes/
    └── dark.json

🧠 app.py — 메인 앱 (Compact ↔ Expanded 자동 전환)
# ui/app.py
import customtkinter as ctk
from components.sidebar import SideBar
from components.chat_panel import ChatPanel
from components.input_dock import InputDock
from components.settings_panel import SettingsPanel
from components.work_center_panel import WorkCenterPanel
from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.agents.meeting import MeetingAgent
from core.agents.photo import PhotoAgent
from core.conversation.orchestrator import AssistantOrchestrator

class AISummaryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI-summary Desktop")
        self.geometry("1100x720")
        self.configure(fg_color="#0E0E0E")
        ctk.set_appearance_mode("dark")

        # 상태
        self.expanded = False

        # 구성요소
        self.sidebar = SideBar(self)
        self.sidebar.pack(side="left", fill="y")

        self.main_surface = ctk.CTkFrame(self, fg_color="#181c28")
        self.main_surface.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.chat_panel = ChatPanel(self.main_surface, fg_color="#11131c")
        self.chat_panel.pack(side="top", fill="both", expand=True)
        self.chat_panel.hide()

        self.input_dock = InputDock(self.main_surface, on_send=self.on_send)
        self.input_dock.pack(side="bottom", fill="x", pady=16)

        # ⚙️ Settings → SettingsPanel (backend/model/API key 입력)
        self.settings_panel = None
        self.settings = SettingsManager(SETTINGS_PATH)
        self.orchestrator = None
        self._initialise_engine()

    def _initialise_engine(self):
        document_agent = DocumentAgent(DocumentAgentConfig(...))
        meeting_agent = MeetingAgent()
        photo_agent = PhotoAgent()
        self.orchestrator = AssistantOrchestrator(
            [document_agent, meeting_agent, photo_agent],
            llm_client=document_agent.llm_client,
        )

    def on_send(self, text):
        if not text.strip():
            return
        # compact → expanded 전환
        if not self.expanded:
            self.chat_panel.show()
            self.expanded = True

        # 사용자 메시지 출력
        self.chat_panel.add_message("user", text)

        # 여기서 CLI 연동 (예시)
        self.after(300, lambda: self.chat_panel.add_message("assistant", "응답 생성 중..."))
        # 실제론 subprocess 로 infopilot.py 호출 가능

if __name__ == "__main__":
    app = AISummaryApp()
    app.mainloop()

💬 chat_panel.py — 대화 영역
# ui/components/chat_panel.py
import customtkinter as ctk

class ChatPanel(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="#0E0E0E")
        self.row_index = 0
        self.visible = False

    def add_message(self, role, content):
        color = "#FFFFFF" if role == "user" else "#A0A0A0"
        text = f"{'🧑 ' if role=='user' else '🤖 '} {content}"
        label = ctk.CTkLabel(self, text=text, justify="left", anchor="w", wraplength=850, text_color=color)
        label.grid(row=self.row_index, column=0, sticky="w", padx=20, pady=6)
        self.row_index += 1

    def show(self):
        if not self.visible:
            self.pack(side="top", fill="both", expand=True)
            self.visible = True

    def hide(self):
        if self.visible:
            self.pack_forget()
            self.visible = False

⌨️ input_dock.py — “무엇이든 부탁하세요” 입력창
# ui/components/input_dock.py
import customtkinter as ctk

class InputDock(ctk.CTkFrame):
    def __init__(self, master, on_send):
        super().__init__(master, fg_color="#121212", corner_radius=0)
        self.on_send = on_send

        self.entry = ctk.CTkEntry(
            self,
            placeholder_text="무엇이든 부탁하세요...",
            height=40,
            fg_color="#1A1A1A",
            text_color="#EAEAEA",
            border_width=1,
            border_color="#222",
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=16, pady=12)
        self.entry.bind("<Return>", lambda e: self.send())

        self.button = ctk.CTkButton(
            self,
            text="➤",
            width=50,
            height=40,
            fg_color="#2A2A2A",
            hover_color="#333",
            command=self.send,
        )
        self.button.pack(side="right", padx=16, pady=12)

    def send(self):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.on_send(text)

🧭 sidebar.py — 좌측 툴바 (홈 / 채팅 / 설정)
# ui/components/sidebar.py
import customtkinter as ctk

class SideBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=70, fg_color="#141414", corner_radius=0)
        self.pack_propagate(False)

        self.home_btn = ctk.CTkButton(self, text="🏠", width=50, height=50, command=self.on_home)
        self.home_btn.pack(pady=(30,10))
        self.chat_btn = ctk.CTkButton(self, text="💬", width=50, height=50, command=self.on_chat)
        self.chat_btn.pack(pady=10)
        self.settings_btn = ctk.CTkButton(self, text="⚙️", width=50, height=50, command=self.on_settings)
        self.settings_btn.pack(side="bottom", pady=20)

    def on_home(self): pass
    def on_chat(self): pass
    def on_settings(self): pass

🎨 themes/dark.json — Atlas 다크 테마
{
  "CTk": {
    "fg_color": ["#0E0E0E", "#0E0E0E"]
  },
  "CTkButton": {
    "corner_radius": 12,
    "fg_color": ["#1E1E1E", "#1E1E1E"],
    "hover_color": ["#2A2A2A", "#2A2A2A"],
    "text_color": ["#EAEAEA", "#EAEAEA"]
  },
  "CTkEntry": {
    "corner_radius": 10,
    "fg_color": ["#1A1A1A", "#1A1A1A"],
    "text_color": ["#EAEAEA", "#EAEAEA"],
    "border_color": ["#2C2C2C", "#2C2C2C"]
  }
}

⚙️ 실행 방법
# 개발 중
python ui/app.py

# 혹은
python scripts/launch_desktop.py

🧩 통합 포인트 (CLI 연결 예시)

app.py → on_send() 함수 안에 다음을 추가하면,
CLI 명령(infopilot.py)을 UI에서 직접 실행 가능 👇

import subprocess

def on_send(self, text):
    if not text.strip():
        return
    if not self.expanded:
        self.chat_panel.show()
        self.expanded = True

    self.chat_panel.add_message("user", text)

    # CLI 실행
    result = subprocess.run(
        ["python", "scripts/infopilot.py", "run", "chat", "--input", text],
        capture_output=True, text=True
    )
    self.chat_panel.add_message("assistant", result.stdout or "(응답 없음)")
