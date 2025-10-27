"""Lower dock with text entry + send button."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk


class InputDock(ctk.CTkFrame):
    """Single-line prompt box with send button."""

    def __init__(self, master, *, on_send: Callable[[str], None]):  # type: ignore[override]
        super().__init__(master, fg_color="#121212", corner_radius=0)
        self._on_send = on_send

        self.entry = ctk.CTkEntry(
            self,
            placeholder_text="무엇이든 부탁하세요...",
            height=44,
            fg_color="#1A1A1A",
            text_color="#EAEAEA",
            border_width=1,
            border_color="#292929",
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=18, pady=16)
        self.entry.bind("<Return>", lambda event: self.send())

        self.button = ctk.CTkButton(
            self,
            text="➤",
            width=48,
            height=44,
            fg_color="#282828",
            hover_color="#333333",
            command=self.send,
        )
        self.button.pack(side="right", padx=(0, 18), pady=16)

    def send(self) -> None:
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self._on_send(text)

    def set_text(self, text: str) -> None:
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
