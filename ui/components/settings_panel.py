"""Settings dialog for configuring conversation backends."""

from __future__ import annotations

import json
import subprocess
import os
from typing import Callable, Optional

import customtkinter as ctk

from ui.settings_manager import SettingsManager

_STATIC_MODEL_CHOICES = {
    "openai": ["", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "local": ["", "llama3", "phi3", "mistral"],
}


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
        self.geometry("420x420")
        self.resizable(False, False)
        self.grab_set()

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        container.grid_columnconfigure(1, weight=1)

        self.backend_var = ctk.StringVar(value=self._get_setting("llm_backend"))
        self.model_var = ctk.StringVar(value=self._get_setting("llm_model"))
        self.host_var = ctk.StringVar(value=self._get_setting("llm_host"))
        self.api_var = ctk.StringVar(value=self._get_setting("llm_api_key"))
        self.health_timeout_var = ctk.StringVar(value=self._get_setting("llm_health_timeout") or "20")

        ctk.CTkLabel(container, text="LLM Backend").grid(row=0, column=0, sticky="w", pady=(0, 8))
        backend_menu = ctk.CTkOptionMenu(
            container,
            values=["", "ollama", "openai", "local"],
            variable=self.backend_var,
            command=self._on_backend_change,
        )
        backend_menu.grid(row=0, column=1, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(container, text="Model Name").grid(row=1, column=0, sticky="w", pady=8)
        model_frame = ctk.CTkFrame(container, fg_color="transparent")
        model_frame.grid(row=1, column=1, sticky="ew", pady=8)
        model_frame.grid_columnconfigure(0, weight=1)

        self.model_combo = ctk.CTkComboBox(model_frame, values=[], variable=self.model_var)
        self.model_combo.grid(row=0, column=0, sticky="ew")

        refresh_btn = ctk.CTkButton(model_frame, text="목록 갱신", width=80, command=self._refresh_model_list)
        refresh_btn.grid(row=0, column=1, padx=(8, 0))

        self.model_hint = ctk.CTkLabel(
            model_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b",
        )
        self.model_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ctk.CTkLabel(container, text="Host/Base URL").grid(row=2, column=0, sticky="w", pady=8)
        host_entry = ctk.CTkEntry(container, textvariable=self.host_var)
        host_entry.grid(row=2, column=1, sticky="ew", pady=8)
        self.host_entry = host_entry

        ctk.CTkLabel(container, text="API Key").grid(row=3, column=0, sticky="w", pady=8)
        ctk.CTkEntry(container, textvariable=self.api_var, show="•").grid(row=3, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(container, text="Health Timeout (seconds)").grid(row=4, column=0, sticky="w", pady=8)
        timeout_entry = ctk.CTkEntry(container, textvariable=self.health_timeout_var)
        timeout_entry.grid(row=4, column=1, sticky="ew", pady=8)
        timeout_hint = ctk.CTkLabel(
            container,
            text="초기 Ollama 로딩이 길면 값을 늘려 주세요 (기본 20초).",
            font=ctk.CTkFont(size=11),
            text_color="#6b6b6b",
        )
        timeout_hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 12))

        button_row = ctk.CTkFrame(container, fg_color="transparent")
        button_row.grid(row=6, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ctk.CTkButton(button_row, text="취소", width=90, command=self._cancel).pack(side="right", padx=(8, 0))
        ctk.CTkButton(button_row, text="저장", width=90, command=self._save).pack(side="right")

        self._populate_model_values(self.backend_var.get(), host=self.host_var.get(), force_refresh=True)

    def _get_setting(self, key: str) -> str:
        return str(
            self.settings.get("conversation", key, default="") or ""
        ).strip()

    def _save(self) -> None:
        self.settings.set(self.backend_var.get().strip(), "conversation", "llm_backend")
        self.settings.set(self.model_var.get().strip(), "conversation", "llm_model")
        self.settings.set(self.host_var.get().strip(), "conversation", "llm_host")
        self.settings.set(self.api_var.get().strip(), "conversation", "llm_api_key")
        self.settings.set(self._parse_timeout(self.health_timeout_var.get()), "conversation", "llm_health_timeout")
        self._close()
        if self._on_save:
            self._on_save()

    def _cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        self.grab_release()
        self.destroy()

    def _on_backend_change(self, value: str) -> None:
        backend = (value or "").strip()
        self.backend_var.set(backend)
        self._populate_model_values(backend, host=self.host_var.get().strip())

    def _refresh_model_list(self) -> None:
        self._populate_model_values(self.backend_var.get(), host=self.host_var.get().strip(), force_refresh=True)

    def _populate_model_values(self, backend: str, *, host: str, force_refresh: bool = False) -> None:
        candidates = self._load_model_candidates(backend, host=host, force_refresh=force_refresh)
        if candidates:
            self.model_combo.configure(values=candidates)
            current = self.model_var.get().strip()
            if current and current in candidates:
                self.model_combo.set(current)
            else:
                self.model_combo.set(candidates[0])
            self.model_hint.configure(text="목록에서 선택하거나 직접 입력할 수 있습니다.")
        else:
            self.model_combo.configure(values=[])
            current = self.model_var.get().strip()
            if current:
                self.model_combo.set(current)
            self.model_hint.configure(text="사용할 모델 이름을 직접 입력하세요.")

    def _load_model_candidates(self, backend: str, *, host: str = "", force_refresh: bool = False) -> list[str]:
        backend_key = (backend or "").strip().lower()
        if not backend_key:
            return []
        if backend_key == "ollama":
            env = None
            if host:
                env = os.environ.copy()
                env["OLLAMA_HOST"] = host

            def _run(cmd: list[str]) -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5.0,
                    env=env,
                )

            try:
                result = _run(["ollama", "list", "--format", "json"])
                if result.returncode == 0:
                    payload = json.loads(result.stdout or "[]")
                    if isinstance(payload, list):
                        names = [
                            str(item.get("name"))
                            for item in payload
                            if isinstance(item, dict) and item.get("name")
                        ]
                        if names:
                            return names
            except Exception:
                # Fallback to plain text parsing below.
                pass

            try:
                result_plain = _run(["ollama", "list"])
            except Exception:
                return []
            if result_plain.returncode != 0:
                self.model_hint.configure(text=result_plain.stderr.strip() or "ollama list 명령이 실패했습니다.")
                return []
            models: list[str] = []
            for line in (result_plain.stdout or "").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0].lower() in {"name", "models"}:
                    continue
                models.append(parts[0])
            return models

        if backend_key in _STATIC_MODEL_CHOICES:
            return _STATIC_MODEL_CHOICES[backend_key]
        return []

    def _parse_timeout(self, raw: str) -> float:
        value = (raw or "").strip()
        if not value:
            return 20.0
        try:
            parsed = float(value)
        except ValueError:
            return 20.0
        return max(1.0, parsed)
