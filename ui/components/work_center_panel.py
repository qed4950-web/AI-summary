"""Work Center panel showing recent actions, quick shortcuts and resource logs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import customtkinter as ctk

from ui.utils import CORPUS_PARQUET, TOPIC_MODEL_PATH, have_all_artifacts

RESOURCE_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "resource_log.jsonl"


class WorkCenterPanel(ctk.CTkToplevel):
    def __init__(
        self,
        master,
        *,
        on_quick_query: Optional[Callable[[str], None]] = None,
        on_rebuild_index: Optional[Callable[[], None]] = None,
        on_open_pipeline: Optional[Callable[[], None]] = None,
        on_open_meeting: Optional[Callable[[], None]] = None,
        on_open_photo: Optional[Callable[[], None]] = None,
    ):  # type: ignore[override]
        super().__init__(master)
        self.title("Work Center")
        self.geometry("480x600")
        self.resizable(False, False)
        self.configure(fg_color="#11131c")

        self._on_quick_query = on_quick_query
        self._on_rebuild_index = on_rebuild_index
        self._on_open_pipeline = on_open_pipeline
        self._on_open_meeting = on_open_meeting
        self._on_open_photo = on_open_photo

        self.quick_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        self.quick_frame.pack(fill="x", padx=20, pady=(20, 12))

        ctk.CTkLabel(
            self.quick_frame,
            text="빠른 동작",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F5F5F5",
        ).grid(row=0, column=0, sticky="w", pady=(12, 8), columnspan=2)

        self.quick_frame.grid_columnconfigure((0, 1), weight=1)
        quick_actions: List[Tuple[str, Callable[[], None]]] = [
            ("ML 문서 찾아보기", lambda: self._fire_quick_query("머신러닝 관련 문서를 찾아줘")),
            ("최근 업데이트 문서", lambda: self._fire_quick_query("최근에 수정된 문서를 보여줘")),
            ("인덱스 재구축", self._handle_rebuild_index),
            ("데이터 파이프라인 열기", self._open_pipeline_panel),
            ("회의 비서 열기", self._open_meeting_panel),
            ("사진 비서 열기", self._open_photo_panel),
        ]
        for idx, (label, callback) in enumerate(quick_actions):
            button = ctk.CTkButton(
                self.quick_frame,
                text=label,
                height=36,
                command=callback,
            )
            button.grid(row=1 + idx // 2, column=idx % 2, padx=8, pady=6, sticky="ew")

        self.status_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(
            self.status_frame,
            text="인덱스 상태",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F5F5F5",
        ).pack(anchor="w", padx=16, pady=(12, 6))

        self.status_labels = {
            "corpus": self._create_status_row(self.status_frame, "문서 코퍼스"),
            "model": self._create_status_row(self.status_frame, "토픽 모델"),
            "artifacts": self._create_status_row(self.status_frame, "통합 상태"),
            "resource": self._create_status_row(self.status_frame, "최근 리소스 로그"),
        }
        self._refresh_status()

        self.activity_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        self.activity_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        ctk.CTkLabel(
            self.activity_frame,
            text="최근 활동",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F5F5F5",
        ).pack(anchor="w", padx=16, pady=(12, 6))
        self.activity_box = ctk.CTkTextbox(
            self.activity_frame,
            state="disabled",
            height=160,
            fg_color="#151a26",
            border_width=0,
        )
        self.activity_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.resource_frame = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=10)
        self.resource_frame.pack(fill="both", expand=False, padx=20, pady=(0, 20))
        header = ctk.CTkFrame(self.resource_frame, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(
            header,
            text="리소스 로그",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F5F5F5",
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="새로고침",
            width=90,
            height=30,
            command=self.refresh_resource_log,
        ).pack(side="right")

        self.resource_box = ctk.CTkTextbox(
            self.resource_frame,
            state="disabled",
            height=140,
            fg_color="#151a26",
            border_width=0,
        )
        self.resource_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.refresh_resource_log()

    def update_activity(self, entries: Iterable[str]) -> None:
        entries = list(entries)
        self.activity_box.configure(state="normal")
        self.activity_box.delete("1.0", "end")
        for item in entries:
            self.activity_box.insert("end", item + "\n")
        self.activity_box.configure(state="disabled")

    def refresh_resource_log(self) -> None:
        lines = _tail_json(RESOURCE_LOG_PATH, 20)
        self.resource_box.configure(state="normal")
        self.resource_box.delete("1.0", "end")
        for entry in lines:
            cpu = entry.get("cpu")
            mem = entry.get("mem")
            ts = entry.get("ts") or entry.get("timestamp")
            stamp = ts or "-"
            cpu_str = f"{cpu:.1f}%" if isinstance(cpu, (int, float)) else "-"
            mem_str = f"{mem:.1f}%" if isinstance(mem, (int, float)) else "-"
            self.resource_box.insert("end", f"[{stamp}] CPU {cpu_str} · RAM {mem_str}\n")
        if not lines:
            self.resource_box.insert("end", "(리소스 로그 없음)\n")
        self.resource_box.configure(state="disabled")
        self._refresh_status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_status_row(self, parent: ctk.CTkFrame, label: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(
            row,
            text=label,
            width=140,
            anchor="w",
            text_color="#AAB0BE",
        ).pack(side="left")
        value = ctk.CTkLabel(row, text="-", anchor="w", text_color="#F5F5F5")
        value.pack(side="left", fill="x", expand=True)
        return value

    def _refresh_status(self) -> None:
        corpus_ready = CORPUS_PARQUET.exists()
        model_ready = TOPIC_MODEL_PATH.exists()
        artifacts_ready = have_all_artifacts()
        recent_resource = self._latest_resource_timestamp()

        self.status_labels["corpus"].configure(
            text="준비됨" if corpus_ready else "없음",
            text_color="#7CFFB0" if corpus_ready else "#FF8383",
        )
        self.status_labels["model"].configure(
            text="준비됨" if model_ready else "없음",
            text_color="#7CFFB0" if model_ready else "#FF8383",
        )
        self.status_labels["artifacts"].configure(
            text="모두 준비됨" if artifacts_ready else "점검 필요",
            text_color="#7CFFB0" if artifacts_ready else "#FFC483",
        )
        self.status_labels["resource"].configure(
            text=recent_resource,
            text_color="#F5F5F5",
        )

    def _latest_resource_timestamp(self) -> str:
        entries = _tail_json(RESOURCE_LOG_PATH, 1)
        if not entries:
            return "기록 없음"
        ts = entries[0].get("ts") or entries[0].get("timestamp")
        if not ts:
            return "기록 없음"
        try:
            parsed = datetime.fromisoformat(str(ts))
            return parsed.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)

    def _fire_quick_query(self, text: str) -> None:
        if self._on_quick_query:
            self._on_quick_query(text)

    def _handle_rebuild_index(self) -> None:
        if self._on_rebuild_index:
            self._on_rebuild_index()
        else:
            self._fire_quick_query("인덱스 재구축을 실행할 수 없습니다.")

    def _open_pipeline_panel(self) -> None:
        if self._on_open_pipeline:
            self._on_open_pipeline()
        else:
            self._fire_quick_query("데이터 파이프라인 인터페이스가 연결되지 않았습니다.")

    def _open_meeting_panel(self) -> None:
        if self._on_open_meeting:
            self._on_open_meeting()
        else:
            self._fire_quick_query("회의 비서 인터페이스가 연결되지 않았습니다.")

    def _open_photo_panel(self) -> None:
        if self._on_open_photo:
            self._on_open_photo()
        else:
            self._fire_quick_query("사진 비서 인터페이스가 연결되지 않았습니다.")


def _tail_json(path: Path, limit: int) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = f.readlines()
    except OSError:
        return []
    picked: List[dict] = []
    for raw in rows[-limit:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            picked.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return picked
