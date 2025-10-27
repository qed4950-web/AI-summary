"""Scrollable chat surface that reveals on demand."""

from __future__ import annotations

import customtkinter as ctk


class ChatPanel(ctk.CTkFrame):
    """Conversation log container with helper methods."""

    def __init__(self, master, *, fg_color: str = "#11131c"):  # type: ignore[override]
        super().__init__(master, fg_color=fg_color)
        self.visible = False

        self.toolbar = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar.pack(side="top", fill="x", padx=12, pady=(12, 0))

        self.title_label = ctk.CTkLabel(
            self.toolbar,
            text="대화 로그",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#F5F5F5",
        )
        self.title_label.pack(side="left")

        self.copy_all_button = ctk.CTkButton(
            self.toolbar,
            text="전체 복사",
            width=100,
            height=28,
            command=self._copy_all,
        )
        self.copy_all_button.pack(side="right")

        self.textbox = ctk.CTkTextbox(
            self,
            fg_color=fg_color,
            border_width=0,
            wrap="word",
            state="normal",
        )
        self.textbox.pack(side="top", fill="both", expand=True, padx=12, pady=12)
        self.textbox.tag_config("user", foreground="#F5F5F5")
        self.textbox.tag_config("assistant", foreground="#B0B0B0")
        self.textbox.bind("<Button-1>", self._focus_textbox)
        self.textbox.bind("<Key>", self._block_edit)
        self.textbox.bind("<Control-c>", self._copy_selection)
        self.textbox.bind("<Command-c>", self._copy_selection)
        self.textbox.bind("<Control-a>", self._select_all)
        self.textbox.bind("<Command-a>", self._select_all)
        self.textbox.bind("<Control-v>", self._block_clipboard)
        self.textbox.bind("<Command-v>", self._block_clipboard)
        self.textbox.bind("<Control-x>", self._block_clipboard)
        self.textbox.bind("<Command-x>", self._block_clipboard)

    def add_message(self, role: str, content: str) -> None:
        prefix = "🧑 " if role == "user" else "🤖 "
        text = f"{prefix}{content}"

        if self.textbox.index("end-1c") != "1.0":
            self.textbox.insert("end", "\n")
        start_index = self.textbox.index("end")
        self.textbox.insert("end", text + "\n")
        end_index = f"{start_index}+{len(text)}c"
        self.textbox.tag_add(role, start_index, end_index)
        self.textbox.insert("end", "\n")
        self._scroll_to_end()

    def show(self) -> None:
        if not self.visible:
            self.pack(side="top", fill="both", expand=True, padx=(0, 12), pady=(16, 0))
            self.visible = True

    def hide(self) -> None:
        if self.visible:
            self.pack_forget()
            self.visible = False

    def clear(self) -> None:
        self.textbox.delete("1.0", "end")

    def _scroll_to_end(self) -> None:
        self.textbox.see("end")

    def _focus_textbox(self, *_args) -> None:
        self.textbox.focus_set()

    def _block_edit(self, event) -> str | None:
        navigation_keys = {
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Prior",
            "Next",
            "Page_Up",
            "Page_Down",
            "Tab",
        }
        modifier_keys = {
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Meta_L",
            "Meta_R",
            "Command",
        }
        if event.keysym in navigation_keys or event.keysym in modifier_keys:
            return None

        control_mask = 0x0004
        alt_mask = 0x0008
        meta_mask = 0x0010
        control_pressed = bool(event.state & control_mask)
        meta_pressed = bool(event.state & meta_mask)
        alt_pressed = bool(event.state & alt_mask)

        if control_pressed or meta_pressed:
            return None
        if alt_pressed:
            navigation_with_alt = {"Left", "Right"}
            if event.keysym in navigation_with_alt:
                return None
            return "break"
        return "break"

    def _copy_selection(self, _event) -> str:
        try:
            selection = self.textbox.get("sel.first", "sel.last")
        except Exception:
            return self._copy_all()
        if not selection:
            return self._copy_all()
        self.textbox.clipboard_clear()
        self.textbox.clipboard_append(selection)
        return "break"

    def _select_all(self, _event) -> str:
        self.textbox.tag_add("sel", "1.0", "end")
        return "break"

    def _block_clipboard(self, _event) -> str:
        return "break"

    def _copy_all(self) -> str:
        content = self.textbox.get("1.0", "end").strip()
        if not content:
            return "break"
        self.textbox.clipboard_clear()
        self.textbox.clipboard_append(content)
        return "break"
