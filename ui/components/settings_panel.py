"""Settings dialog for configuring conversation backends."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from ui.settings_manager import SettingsManager


class SettingsPanel(ctk.CTkToplevel):
    def __init__(
        self,
        master,
        settings: SettingsManager,
        *,
        on_save: Optional[Callable[[], None]] = None,
    ):  # type: ignore[override]
        super().__init__(master)
        self.settings = settings
        self._on_save = on_save
        self.title("Settings")
        self.geometry("420x360")
        self.resizable(False, False)
        self.grab_set()

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        container.grid_columnconfigure(1, weight=1)

        self.backend_var = ctk.StringVar(value=self._get_setting("llm_backend"))
        self.model_var = ctk.StringVar(value=self._get_setting("llm_model"))
        self.host_var = ctk.StringVar(value=self._get_setting("llm_host"))
        self.api_var = ctk.StringVar(value=self._get_setting("llm_api_key"))

        ctk.CTkLabel(container, text="LLM Backend").grid(row=0, column=0, sticky="w", pady=(0, 8))
        backend_menu = ctk.CTkOptionMenu(
            container,
            values=["", "ollama", "openai", "local"],
            variable=self.backend_var,
        )
        backend_menu.grid(row=0, column=1, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(container, text="Model Name").grid(row=1, column=0, sticky="w", pady=8)
        ctk.CTkEntry(container, textvariable=self.model_var).grid(row=1, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(container, text="Host/Base URL").grid(row=2, column=0, sticky="w", pady=8)
        ctk.CTkEntry(container, textvariable=self.host_var).grid(row=2, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(container, text="API Key").grid(row=3, column=0, sticky="w", pady=8)
        ctk.CTkEntry(container, textvariable=self.api_var, show="•").grid(row=3, column=1, sticky="ew", pady=8)

        button_row = ctk.CTkFrame(container, fg_color="transparent")
        button_row.grid(row=4, column=0, columnspan=2, sticky="e", pady=(20, 0))
        ctk.CTkButton(button_row, text="취소", width=90, command=self._cancel).pack(side="right", padx=(8, 0))
        ctk.CTkButton(button_row, text="저장", width=90, command=self._save).pack(side="right")

    def _get_setting(self, key: str) -> str:
        return str(
            self.settings.get("conversation", key, default="") or ""
        ).strip()

    def _save(self) -> None:
        self.settings.set(self.backend_var.get().strip(), "conversation", "llm_backend")
        self.settings.set(self.model_var.get().strip(), "conversation", "llm_model")
        self.settings.set(self.host_var.get().strip(), "conversation", "llm_host")
        self.settings.set(self.api_var.get().strip(), "conversation", "llm_api_key")
        self._close()
        if self._on_save:
            self._on_save()

    def _cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        self.grab_release()
        self.destroy()
