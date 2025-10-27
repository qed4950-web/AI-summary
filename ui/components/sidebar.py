"""Vertical toolbar used by the Atlas desktop shell."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class SideBar(ctk.CTkFrame):
    """Compact navigation strip with emoji buttons."""

    def __init__(self, master, *, on_select: Optional[Callable[[str], None]] = None) -> None:  # type: ignore[override]
        super().__init__(master, width=72, fg_color="#151515", corner_radius=0)
        self.pack_propagate(False)
        self._on_select = on_select or (lambda *_: None)

        self.home_btn = ctk.CTkButton(
            self,
            text="🏠",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("home"),
        )
        self.home_btn.pack(pady=(28, 12))

        self.chat_btn = ctk.CTkButton(
            self,
            text="💬",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("chat"),
        )
        self.chat_btn.pack(pady=12)

        self.work_btn = ctk.CTkButton(
            self,
            text="📋",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("work"),
        )
        self.work_btn.pack(pady=12)

        self.meeting_btn = ctk.CTkButton(
            self,
            text="🎙️",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("meeting"),
        )
        self.meeting_btn.pack(pady=12)

        self.photo_btn = ctk.CTkButton(
            self,
            text="📸",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("photo"),
        )
        self.photo_btn.pack(pady=12)

        self.data_btn = ctk.CTkButton(
            self,
            text="🗃️",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("data"),
        )
        self.data_btn.pack(pady=12)

        self.settings_btn = ctk.CTkButton(
            self,
            text="⚙️",
            width=50,
            height=50,
            fg_color="#1E1E1E",
            hover_color="#2A2A2A",
            command=lambda: self._select("settings"),
        )
        self.settings_btn.pack(side="bottom", pady=24)

    def _select(self, key: str) -> None:
        self._on_select(key)
