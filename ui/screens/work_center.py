from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import customtkinter as ctk

from ui.smart_folder_context import SmartFolderContext
from ui.policy_cache import get_policy_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "core" / "config" / "smart_folders.json"
EVENT_LOG_PATH = REPO_ROOT / "data" / "work_center" / "events.jsonl"


def _expand_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    expanded = Path(os.path.expanduser(raw))
    try:
        return expanded.resolve(strict=False)
    except OSError:
        return expanded


@dataclass(frozen=True)
class SmartFolderItem:
    folder_id: str
    label: str
    agent_type: str
    scope: str
    raw_path: str
    policy: Optional[str]

    @property
    def resolved_path(self) -> Optional[Path]:
        return _expand_path(self.raw_path)


class WorkCenterScreen(ctk.CTkFrame):
    """Unified hub that connects smart folders with Knowledge and Meeting agents."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.policy_engine = get_policy_engine()

        self.title_label = ctk.CTkLabel(
            self,
            text="작업 센터",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(0, 4), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="스마트 폴더를 선택해 지식·검색과 회의 비서를 빠르게 실행하세요.",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#d0d0d0"),
        )
        self.subtitle_label.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        # Left: Smart folder list
        self.folder_list = ctk.CTkScrollableFrame(self, width=260)
        self.folder_list.grid(row=2, column=0, rowspan=2, padx=(12, 8), pady=(4, 12), sticky="ns")
        self.folder_list.grid_columnconfigure(0, weight=1)

        # Right: Detail & activity
        self.detail_frame = ctk.CTkFrame(self)
        self.detail_frame.grid(row=2, column=1, padx=(8, 12), pady=(4, 6), sticky="nsew")
        self.detail_frame.grid_columnconfigure(0, weight=1)

        self.detail_title = ctk.CTkLabel(
            self.detail_frame,
            text="폴더를 선택하세요.",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.detail_title.grid(row=0, column=0, padx=16, pady=(16, 4), sticky="w")

        self.detail_scope_label = ctk.CTkLabel(
            self.detail_frame,
            text="",
            font=ctk.CTkFont(size=13),
            text_color=("#4f4f4f", "#cfcfcf"),
        )
        self.detail_scope_label.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")

        self.detail_path_label = ctk.CTkLabel(
            self.detail_frame,
            text="경로: -",
            font=ctk.CTkFont(size=13),
            justify="left",
            text_color=("#4f4f4f", "#cfcfcf"),
        )
        self.detail_path_label.grid(row=2, column=0, padx=16, pady=(0, 6), sticky="w")

        self.detail_policy_label = ctk.CTkLabel(
            self.detail_frame,
            text="정책: -",
            font=ctk.CTkFont(size=13),
            justify="left",
            text_color=("#4f4f4f", "#cfcfcf"),
        )
        self.detail_policy_label.grid(row=3, column=0, padx=16, pady=(0, 6), sticky="w")

        self.quick_actions_frame = ctk.CTkFrame(self.detail_frame, fg_color="transparent")
        self.quick_actions_frame.grid(row=4, column=0, padx=16, pady=(8, 12), sticky="ew")
        self.quick_actions_frame.grid_columnconfigure((0, 1), weight=1, uniform="actions")
        self.chat_button = ctk.CTkButton(
            self.quick_actions_frame,
            text="지식·검색 비서 열기",
            command=self.open_chat,
            state="disabled",
            height=38,
        )
        self.chat_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.meeting_button = ctk.CTkButton(
            self.quick_actions_frame,
            text="회의 비서 열기",
            command=self.open_meeting,
            state="disabled",
            height=38,
        )
        self.meeting_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        self.activity_frame = ctk.CTkFrame(self)
        self.activity_frame.grid(row=3, column=1, padx=(8, 12), pady=(0, 12), sticky="nsew")
        self.activity_frame.grid_columnconfigure(0, weight=1)
        self.activity_frame.grid_rowconfigure(1, weight=1)

        activity_title = ctk.CTkLabel(
            self.activity_frame,
            text="최근 활동",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        activity_title.grid(row=0, column=0, padx=16, pady=(12, 8), sticky="w")

        self.activity_textbox = ctk.CTkTextbox(
            self.activity_frame,
            state="disabled",
            font=ctk.CTkFont(size=12, family="monospace"),
        )
        self.activity_textbox.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")

        self.load_error_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#b3261e", "#ff9b9b"),
        )
        self.load_error_label.grid(row=4, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        self.items: List[SmartFolderItem] = []
        self.selected_item: Optional[SmartFolderItem] = None
        self.folder_buttons: Dict[str, ctk.CTkButton] = {}
        self.events: List[Dict[str, object]] = []

        self.reload_config()
        self.refresh_events()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self.policy_engine = get_policy_engine()
        self.reload_config()
        self.refresh_events()
        self._highlight_selected_button()
        self._render_activity()

    def on_work_center_event(self, record: Dict[str, object]) -> None:
        self.events.append(record)
        if self.selected_item is None:
            return
        ctx = record.get("context") or {}
        if ctx.get("folder_id") == self.selected_item.folder_id:
            self._render_activity()

    # ------------------------------------------------------------------
    # Smart folder handling
    # ------------------------------------------------------------------
    def reload_config(self) -> None:
        self.policy_engine = get_policy_engine()
        if not CONFIG_PATH.exists():
            self.load_error_label.configure(text=f"⚠️ 스마트 폴더 구성 파일을 찾을 수 없습니다: {CONFIG_PATH}")
            self.items = []
            self._rebuild_folder_buttons()
            self._update_detail(None)
            return

        try:
            raw_entries = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw_entries, Iterable):
                raise ValueError("구성 파일은 리스트 형태여야 합니다.")
        except Exception as exc:
            self.load_error_label.configure(text=f"⚠️ 스마트 폴더 구성을 읽는 중 오류: {exc}")
            self.items = []
            self._rebuild_folder_buttons()
            self._update_detail(None)
            return

        self.load_error_label.configure(text="")
        parsed_items: List[SmartFolderItem] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            folder_id = str(entry.get("id") or "").strip()
            label = str(entry.get("label") or "").strip() or folder_id
            if not folder_id:
                continue
            parsed_items.append(
                SmartFolderItem(
                    folder_id=folder_id,
                    label=label,
                    agent_type=str(entry.get("type") or "documents"),
                    scope=str(entry.get("scope") or "policy"),
                    raw_path=str(entry.get("path") or ""),
                    policy=str(entry.get("policy") or "") or None,
                )
            )

        self.items = parsed_items
        self._rebuild_folder_buttons()

        if self.selected_item and self.selected_item.folder_id not in self.folder_buttons:
            self.selected_item = None
            self._update_detail(None)

    def _rebuild_folder_buttons(self) -> None:
        for child in list(self.folder_list.winfo_children()):
            child.destroy()
        self.folder_buttons.clear()

        if not self.items:
            placeholder = ctk.CTkLabel(
                self.folder_list,
                text="등록된 스마트 폴더가 없습니다.\n구성 파일을 업데이트하세요.",
                font=ctk.CTkFont(size=12),
                justify="center",
            )
            placeholder.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
            return

        for index, item in enumerate(self.items):
            button = ctk.CTkButton(
                self.folder_list,
                text=item.label,
                anchor="w",
                height=40,
                command=lambda it=item: self._select_item(it),
            )
            button.grid(row=index, column=0, padx=6, pady=4, sticky="ew")
            self.folder_buttons[item.folder_id] = button

    def _select_item(self, item: SmartFolderItem) -> None:
        self.selected_item = item
        self._highlight_selected_button()
        context = self._build_context(item)
        self.app.set_smart_folder_context(context)
        self._update_detail(item)
        self._render_activity()
        self.app.status_var.set(f"📂 '{item.label}' 컨텍스트가 적용되었습니다.")

    def _highlight_selected_button(self) -> None:
        for folder_id, button in self.folder_buttons.items():
            if self.selected_item and folder_id == self.selected_item.folder_id:
                button.configure(fg_color=("#2b7de9", "#225aa1"))
            else:
                button.configure(fg_color="transparent")

    def _build_context(self, item: SmartFolderItem) -> SmartFolderContext:
        path = item.resolved_path
        return SmartFolderContext(
            folder_id=item.folder_id,
            label=item.label,
            path=path,
            scope=item.scope,
            policy=item.policy,
            agent_type=item.agent_type,
            allowed_agents=self._allowed_agents_for_item(item),
        )

    def _update_detail(self, item: Optional[SmartFolderItem]) -> None:
        if item is None:
            self.detail_title.configure(text="폴더를 선택하세요.")
            self.detail_scope_label.configure(text="")
            self.detail_path_label.configure(text="경로: -")
            self.detail_policy_label.configure(text="정책: -")
            self.chat_button.configure(state="disabled")
            self.meeting_button.configure(state="disabled")
            return

        scope_text = f"스코프: {item.scope.upper()}"
        type_text = f" · 유형: {item.agent_type}"
        self.detail_title.configure(text=item.label)
        self.detail_scope_label.configure(text=scope_text + type_text)

        resolved = item.resolved_path
        path_display = str(resolved) if resolved else "(경로 미지정)"
        self.detail_path_label.configure(text=f"경로: {path_display}")

        policy_display = item.policy or "(정책 파일 미지정)"
        self.detail_policy_label.configure(text=f"정책: {policy_display}")

        self.chat_button.configure(state="normal" if self._supports_chat(item) else "disabled")
        self.meeting_button.configure(state="normal" if self._supports_meeting(item) else "disabled")

    # ------------------------------------------------------------------
    # Activity log handling
    # ------------------------------------------------------------------
    def refresh_events(self) -> None:
        if not EVENT_LOG_PATH.exists():
            self.events = []
            self._render_activity()
            return

        records: List[Dict[str, object]] = []
        try:
            with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            records = []
        self.events = records
        self._render_activity()

    def _render_activity(self) -> None:
        self.activity_textbox.configure(state="normal")
        self.activity_textbox.delete("1.0", "end")

        if not self.selected_item:
            self.activity_textbox.insert("1.0", "활동 로그는 스마트 폴더를 선택하면 표시됩니다.")
            self.activity_textbox.configure(state="disabled")
            return

        folder_id = self.selected_item.folder_id
        filtered = [
            event for event in reversed(self.events)
            if (event.get("context") or {}).get("folder_id") == folder_id
        ][:20]

        if not filtered:
            self.activity_textbox.insert("1.0", "아직 기록된 활동이 없습니다.")
            self.activity_textbox.configure(state="disabled")
            return

        lines: List[str] = []
        for event in filtered:
            timestamp = event.get("timestamp")
            try:
                display_ts = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "-"
            except Exception:
                display_ts = timestamp or "-"
            event_type = event.get("type", "event")
            lines.append(f"[{display_ts}] {event_type}")

            payload = event.get("payload") or {}
            if event_type.startswith("knowledge"):
                query = payload.get("query", "")
                hit_count = payload.get("hit_count", 0)
                lines.append(f"  · 질문: {query}")
                lines.append(f"  · 결과 수: {hit_count}")
                policy_blocked = payload.get("policy_blocked")
                if policy_blocked:
                    lines.append(f"  · 정책 제외 문서: {policy_blocked}")
                top_hits = payload.get("top_hits") or []
                for idx, hit in enumerate(top_hits, start=1):
                    path = hit.get("path")
                    sim = hit.get("similarity")
                    lines.append(f"    {idx}. {path} (유사도 {sim})")
            elif event_type.startswith("meeting"):
                output_dir = payload.get("output_dir")
                lines.append(f"  · 출력: {output_dir}")
                mode = payload.get("mode")
                if mode:
                    lines.append(f"  · 모드: {mode}")
                highlights = payload.get("highlights") or []
                if highlights:
                    lines.append("  · 하이라이트:")
                    for item in highlights[:3]:
                        lines.append(f"    - {item}")
                actions = payload.get("action_items") or []
                if actions:
                    lines.append("  · 액션:")
                    for item in actions[:3]:
                        lines.append(f"    - {item}")
                policy_tag = payload.get("policy_tag")
                if policy_tag:
                    lines.append(f"  · 정책: {policy_tag}")
                policy_blocked = payload.get("policy_blocked")
                if policy_blocked:
                    lines.append(f"  · 정책 제외 문서: {policy_blocked}")
            elif event_type == "knowledge.policy.blocked":
                reason = payload.get("reason", "context_not_allowed")
                lines.append("  · 정책 차단: 지식·검색 비서를 사용할 수 없습니다.")
                lines.append(f"  · 사유: {reason}")
            lines.append("")

            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict) and diagnostics:
                lines.append("  · 진단:")
                for key, value in diagnostics.items():
                    lines.append(f"    - {key}: {value}")
                lines.append("")

        self.activity_textbox.insert("1.0", "\n".join(lines).strip())
        self.activity_textbox.configure(state="disabled")

    # ------------------------------------------------------------------
    # Quick actions
    # ------------------------------------------------------------------
    def open_chat(self) -> None:
        if not self.selected_item:
            return
        self.app.set_smart_folder_context(self._build_context(self.selected_item))
        self.app.select_frame("chat")

    def open_meeting(self) -> None:
        if not self.selected_item:
            return
        self.app.set_smart_folder_context(self._build_context(self.selected_item))
        self.app.select_frame("meeting")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _supports_chat(item: SmartFolderItem) -> bool:
        agent_type = item.agent_type.lower()
        return agent_type in {"documents", "global", "meeting"}

    @staticmethod
    def _supports_meeting(item: SmartFolderItem) -> bool:
        agent_type = item.agent_type.lower()
        return agent_type in {"meeting", "global"}

    def _allowed_agents_for_item(self, item: SmartFolderItem) -> frozenset[str]:
        agent_type = (item.agent_type or "").lower()
        allowed = set()
        policy_engine = self.policy_engine
        resolved_path = item.resolved_path

        if policy_engine and resolved_path is not None and policy_engine.has_policies:
            policy = policy_engine.policy_for_path(resolved_path)
            if policy:
                allowed.update(str(agent) for agent in policy.agents)

        # Fallback heuristics when 정책 파일이 없거나 항목이 미지정
        if not allowed:
            if self._supports_chat(item):
                allowed.add("knowledge_search")
            if self._supports_meeting(item):
                allowed.add("meeting")
            if agent_type:
                allowed.add(agent_type)
            if agent_type == "global":
                allowed.update({"knowledge_search", "meeting", "photo"})
            if agent_type == "photos":
                allowed.add("photo")
        return frozenset(allowed)
