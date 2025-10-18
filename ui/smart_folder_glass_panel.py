import customtkinter as ctk


APP_BG = "#0f1115"


class SmartFolderGlassPanel(ctk.CTk):
    """Floating command palette styled with a glassmorphism theme."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Folder")
        self.geometry("520x200")
        self.resizable(False, False)
        self.configure(fg_color=APP_BG)

        self._build_shell()
        self._build_canvas_layer()
        self._build_content()

    def _build_shell(self) -> None:
        self.shell = ctk.CTkFrame(self, fg_color="transparent")
        self.shell.pack(fill="both", expand=True, padx=18, pady=18)

    def _build_canvas_layer(self) -> None:
        self.canvas = ctk.CTkCanvas(self.shell, highlightthickness=0, bg=APP_BG)
        self.canvas.pack(fill="both", expand=True)

        # Main glass rectangle
        self.canvas.create_rectangle(
            6,
            6,
            514,
            194,
            fill="#1b2331aa",
            outline="#3b4a62",
            width=1,
        )

        # Accent line
        self.canvas.create_line(
            120,
            12,
            400,
            12,
            fill="#6fa8ff55",
            width=3,
        )

        # Glow halo
        self.canvas.create_oval(
            460,
            150,
            540,
            230,
            fill="#36c2ff22",
            outline="",
        )

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self.canvas, fg_color="transparent")
        content.place(relx=0.5, rely=0.5, anchor="center")

        title = ctk.CTkLabel(
            content,
            text="Smart Folder Command",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#dfe9ff",
        )
        title.pack(pady=(4, 2))

        subtitle = ctk.CTkLabel(
            content,
            text="선택된 정책 범위에서 명령을 입력하세요.",
            font=ctk.CTkFont(size=13),
            text_color="#8fa1c6",
        )
        subtitle.pack(pady=(0, 12))

        self.entry = ctk.CTkEntry(
            content,
            placeholder_text="무엇이든 부탁하세요",
            justify="center",
            width=340,
            height=42,
            corner_radius=18,
            fg_color="#141925",
            border_color="#425b7e",
            border_width=2,
            text_color="#f0f4ff",
        )
        self.entry.pack()

        button_row = ctk.CTkFrame(content, fg_color="transparent")
        button_row.pack(pady=(16, 0))

        buttons = [
            ("＋", "추가 폴더", self._handle_add_folder, 46, 46, 16, "#202836", "#2f3b4f"),
            ("🌐", "전역 검색", self._handle_global_search, 48, 48, 18, "#1e2a38", "#2d3c50"),
            ("🎙", "음성 명령", self._handle_voice_command, 52, 52, 20, "#253349", "#345069"),
            ("📄", "회의 요약", self._handle_meeting_summary, 46, 46, 16, "#202836", "#2f3b4f"),
            ("⚙️", "설정", self._handle_open_settings, 42, 42, 14, "#1c2534", "#2a3647"),
        ]

        for icon, tooltip, handler, width, height, radius, color, hover in buttons:
            button = ctk.CTkButton(
                button_row,
                text=icon,
                width=width,
                height=height,
                corner_radius=radius,
                fg_color=color,
                hover_color=hover,
                text_color="#dbe7ff",
                command=lambda cb=handler: cb(),
            )
            button.pack(side="left", padx=6)
            button._tooltip = tooltip  # type: ignore[attr-defined]

    def _handle_add_folder(self) -> None:
        self._show_status("새 스마트 폴더를 추가합니다.")

    def _handle_global_search(self) -> None:
        query = self.entry.get().strip()
        msg = f"전역 검색 실행: {query}" if query else "전역 검색을 시작합니다."
        self._show_status(msg)

    def _handle_voice_command(self) -> None:
        self._show_status("음성 명령을 준비합니다.")

    def _handle_meeting_summary(self) -> None:
        self._show_status("회의 요약 에이전트를 호출합니다.")

    def _handle_open_settings(self) -> None:
        self._show_status("스마트 폴더 설정을 엽니다.")

    def _show_status(self, message: str) -> None:
        if hasattr(self, "_status_label"):
            self._status_label.destroy()

        label = ctk.CTkLabel(
            self.shell,
            text=message,
            fg_color="#121722cc",
            text_color="#75c2ff",
            font=ctk.CTkFont(size=12),
            corner_radius=12,
            padx=16,
            pady=8,
        )
        label.place(relx=0.85, rely=0.2, anchor="center")
        self._status_label = label
        self.after(2200, label.destroy)


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    SmartFolderGlassPanel().mainloop()


if __name__ == "__main__":
    main()
