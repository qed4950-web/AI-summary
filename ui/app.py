"""Atlas-style CustomTkinter shell with compact ↔ expanded chat state."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional

import customtkinter as ctk

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.agents.meeting import MeetingAgent
from core.agents.photo import PhotoAgent
from core.conversation.orchestrator import AssistantOrchestrator

from ui.components.chat_panel import ChatPanel
from ui.components.input_dock import InputDock
from ui.components.sidebar import SideBar
from ui.components.data_pipeline_panel import DataPipelinePanel
from ui.components.meeting_panel import MeetingPanel
from ui.components.photo_panel import PhotoPanel
from ui.components.work_center_panel import WorkCenterPanel
from ui.components.settings_panel import SettingsPanel
from ui.settings_manager import SettingsManager
from ui.utils import (
    CACHE_DIR,
    CORPUS_PARQUET,
    TOPIC_MODEL_PATH,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    rebuild_index,
)

THEME_PATH = Path(__file__).resolve().parent / "themes" / "dark.json"
SETTINGS_PATH = REPO_ROOT / "data" / "ui_settings.json"


def _configure_theme() -> None:
    ctk.set_appearance_mode("dark")
    theme_override = os.getenv("INFOPILOT_CUSTOM_THEME")
    if theme_override and THEME_PATH.exists():
        try:
            ctk.set_default_color_theme(str(THEME_PATH))
            return
        except Exception:
            pass
    ctk.set_default_color_theme("dark-blue")


class AISummaryApp(ctk.CTk):
    """Compact input-first UI that expands into a chat surface on demand."""

    def __init__(self) -> None:
        super().__init__()
        _configure_theme()

        self.title("AI-summary Desktop")
        self.geometry("1100x720")
        self.minsize(980, 640)
        self.configure(fg_color="#11131c")

        self.settings = SettingsManager(SETTINGS_PATH)
        self.expanded = False
        self.activity_log: List[str] = []
        self.work_center_panel: Optional[WorkCenterPanel] = None
        self.settings_panel: Optional[SettingsPanel] = None
        self.meeting_panel: Optional[MeetingPanel] = None
        self.photo_panel: Optional[PhotoPanel] = None
        self.data_pipeline_panel: Optional[DataPipelinePanel] = None
        self.orchestrator: Optional[AssistantOrchestrator] = None
        self._inflight = False
        self._engine_thread: Optional[threading.Thread] = None
        self._reindex_thread: Optional[threading.Thread] = None

        self.sidebar = SideBar(self, on_select=self._handle_nav)
        self.sidebar.pack(side="left", fill="y")

        self.main_surface = ctk.CTkFrame(self, fg_color="#181c28", corner_radius=0)
        self.main_surface.pack(side="left", fill="both", expand=True, padx=(0, 0), pady=(12, 12))

        self.chat_panel = ChatPanel(self.main_surface, fg_color="#11131c")
        self.chat_panel.pack(side="top", fill="both", expand=True, padx=16, pady=(16, 0))
        self.chat_panel.hide()

        self.input_dock = InputDock(self.main_surface, on_send=self.on_send)
        self.input_dock.pack(side="bottom", fill="x", padx=16, pady=(0, 16))

        self._initialise_engine()

    def _handle_nav(self, key: str) -> None:
        if key == "home":
            self.chat_panel.clear()
            self.chat_panel.hide()
            self.expanded = False
        elif key == "chat" and not self.expanded:
            self.chat_panel.show()
            self.expanded = True
        elif key == "meeting":
            self._open_meeting_panel()
        elif key == "photo":
            self._open_photo_panel()
        elif key == "data":
            self._open_data_pipeline()
        elif key == "work":
            self._open_work_center()
        elif key == "settings":
            self._open_settings_dialog()

    def on_send(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        if not self.expanded:
            self.chat_panel.show()
            self.expanded = True

        if self.orchestrator is None:
            self._post_message("assistant", "엔진이 준비되는 중입니다. 잠시 후 다시 시도해 주세요.")
            self._log_activity("SYSTEM · orchestrator warming up")
            self._initialise_engine()
            return
        if self._inflight:
            self._post_message("assistant", "이전 요청을 처리 중입니다. 잠시만 기다려 주세요.")
            return

        self.chat_panel.add_message("user", cleaned)
        self._log_activity(f"USER · {cleaned}")
        threading.Thread(target=self._run_conversation, args=(cleaned,), daemon=True).start()

    def _run_conversation(self, message: str) -> None:
        if self.orchestrator is None:
            return
        self._inflight = True
        started = time.time()
        try:
            response = self.orchestrator.handle(message)
        except Exception as exc:  # pragma: no cover - defensive
            text = f"요청 처리 중 오류가 발생했습니다: {exc}"
            self._post_message("assistant", text)
            self._log_activity(f"ERROR · {exc}")
            return
        finally:
            self._inflight = False

        self._handle_orchestrator_response(response, elapsed=time.time() - started)

    def _handle_orchestrator_response(self, response, *, elapsed: float) -> None:
        config = self._effective_settings()
        agent_map = {
            "document_search": "비서",
            "meeting_summary": "회의 비서",
            "photo_manager": "사진 비서",
            "follow_up": "요청",
        }
        label = agent_map.get(response.agent, response.agent)
        text = response.message.strip() if response.message else "결과가 없습니다."
        if response.agent == "follow_up":
            body = f"[{label}] {text}"
            self._post_message("assistant", body)
            self._log_activity(f"FOLLOW_UP · {text}")
            return

        hits = []
        if isinstance(response.metadata, dict):
            hits = response.metadata.get("hits", []) or []
        if config["include_references"] and hits:
            ref_lines = ["", "관련 문서:"]
            for hit in hits[:5]:
                path = str(hit.get("path") or "")
                score = hit.get("similarity") or hit.get("vector_similarity")
                try:
                    score_str = f"{float(score):.3f}" if score is not None else "-"
                except Exception:
                    score_str = "-"
                ref_lines.append(f"- {path} (유사도 {score_str})")
            text += "\n" + "\n".join(ref_lines)

        body = f"[{label}] {text}\n\n(응답 시간 {elapsed:.2f}s)"
        self._post_message("assistant", body)
        self._log_activity(f"{label.upper()} · {text[:120]}")

    def _initialise_engine(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            return
        self._engine_thread = threading.Thread(target=self._ensure_engine, daemon=True)
        self._engine_thread.start()

    def _ensure_engine(self) -> None:
        try:
            config = self._effective_settings()
            llm_options = self._build_llm_options(config)
            os.environ.setdefault("LNPCHAT_LLM_TIMEOUT", "30")
            os.environ.setdefault("LNPCHAT_LLM_HEALTH_TIMEOUT", "5")
            document_agent = DocumentAgent(
                DocumentAgentConfig(
                    model_path=TOPIC_MODEL_PATH,
                    corpus_path=CORPUS_PARQUET,
                    cache_dir=CACHE_DIR,
                    topk=config["top_k"],
                    min_similarity=config["min_similarity"],
                    lexical_weight=config["lexical_weight"],
                    llm_backend=config["llm_backend"] or None,
                    llm_model=config["llm_model"],
                    llm_host=config["llm_host"],
                    llm_options=llm_options,
                    llm_health_timeout=config["llm_health_timeout"],
                    llm_timeout=6.0,
                    auto_search=bool(config["auto_search"]),
                    rerank=False,
                )
            )
            meeting_agent = MeetingAgent()
            photo_agent = PhotoAgent()
            orchestrator = AssistantOrchestrator(
                [document_agent, meeting_agent, photo_agent],
                llm_client=document_agent.llm_client if config["use_router_llm"] else None,
            )
            self.orchestrator = orchestrator
            if document_agent.llm_client is None:
                message = "LLM이 설정되지 않아 문서 검색 응답만 제공합니다. ⚙️ 설정에서 Ollama 등을 연결하세요."
            else:
                message = "대화 비서가 준비되었습니다. 먼저 자유롭게 대화해 보고, 문서 검색이 필요하면 '/search 질문'처럼 입력해 보세요."
            try:
                document_agent.prepare()
            except Exception as exc:  # pragma: no cover - defensive preload
                message = f"대화 비서 초기화 중 경고: {exc}"
        except FileNotFoundError:
            message = "학습된 모델을 찾을 수 없습니다. 먼저 전체 학습을 실행하세요."
        except Exception as exc:  # pragma: no cover - defensive
            message = f"엔진 초기화 실패: {exc}"
        self._post_message("assistant", message)
        self._log_activity(f"SYSTEM · {message}")

    def _effective_settings(self) -> Dict[str, object]:
        convo = self.settings.get("conversation", default={}) or {}
        backend = str(convo.get("llm_backend") or "").strip()
        model = str(convo.get("llm_model") or "llama3").strip() or "llama3"
        host = str(convo.get("llm_host") or "").strip()
        api_key = str(convo.get("llm_api_key") or "").strip()
        try:
            top_k = max(1, int(convo.get("top_k") or DEFAULT_TOP_K))
        except Exception:
            top_k = DEFAULT_TOP_K
        try:
            min_sim = float(convo.get("min_similarity") or DEFAULT_SIMILARITY_THRESHOLD)
        except Exception:
            min_sim = DEFAULT_SIMILARITY_THRESHOLD
        try:
            lexical_weight = max(0.0, min(1.0, float(convo.get("lexical_weight", 0.35))))
        except Exception:
            lexical_weight = 0.35
        include_refs = bool(convo.get("include_references", True))
        auto_search = bool(convo.get("auto_search", False))
        use_router_llm = bool(convo.get("use_router_llm", False))
        health_timeout = convo.get("llm_health_timeout")
        try:
            if health_timeout in ("", None):
                parsed_timeout = None
            else:
                parsed_timeout = float(health_timeout)
        except Exception:
            parsed_timeout = None
        if parsed_timeout is None:
            parsed_timeout = 5.0
        parsed_timeout = max(1.0, float(parsed_timeout))
        return {
            "llm_backend": backend,
            "llm_model": model,
            "llm_host": host,
            "llm_api_key": api_key,
            "top_k": top_k,
            "min_similarity": min_sim,
            "lexical_weight": lexical_weight,
            "include_references": include_refs,
            "auto_search": auto_search,
            "llm_health_timeout": parsed_timeout,
            "use_router_llm": use_router_llm,
        }

    def _build_llm_options(self, config: Dict[str, object]) -> Dict[str, str]:
        options: Dict[str, str] = {}
        api_key = str(config.get("llm_api_key") or "").strip()
        if api_key:
            options["api_key"] = api_key
        return options

    def _handle_settings_saved(self) -> None:
        self._post_message("assistant", "설정을 반영하는 중입니다. 잠시만 기다려 주세요.")
        self._log_activity("SYSTEM · settings saved, reinitialising orchestrator")
        self.orchestrator = None
        self._initialise_engine()

    def _post_message(self, role: str, text: str) -> None:
        self.after(0, lambda: self.chat_panel.add_message(role, text))

    def _log_activity(self, entry: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        record = f"[{stamp}] {entry}"
        self.activity_log.append(record)
        self.activity_log = self.activity_log[-100:]
        if self.work_center_panel and self.work_center_panel.winfo_exists():
            self.work_center_panel.update_activity(list(reversed(self.activity_log)))

    def _open_work_center(self) -> None:
        if self.work_center_panel and self.work_center_panel.winfo_exists():
            self.work_center_panel.update_activity(list(reversed(self.activity_log)))
            self.work_center_panel.refresh_resource_log()
            self.work_center_panel.lift()
            return
        self.work_center_panel = WorkCenterPanel(
            self,
            on_quick_query=self._handle_quick_action,
            on_rebuild_index=self._trigger_rebuild_index,
            on_open_pipeline=self._open_data_pipeline,
            on_open_meeting=self._open_meeting_panel,
            on_open_photo=self._open_photo_panel,
        )
        self.work_center_panel.update_activity(list(reversed(self.activity_log)))
        self.work_center_panel.refresh_resource_log()

    def _open_data_pipeline(self) -> None:
        if self.data_pipeline_panel and self.data_pipeline_panel.winfo_exists():
            self.data_pipeline_panel.lift()
            return
        self.data_pipeline_panel = DataPipelinePanel(
            self,
            on_activity=self._log_activity,
            on_pipeline_complete=self._handle_pipeline_complete,
        )

    def _handle_pipeline_complete(self) -> None:
        if self.work_center_panel and self.work_center_panel.winfo_exists():
            self.work_center_panel.refresh_resource_log()

    def _open_meeting_panel(self) -> None:
        if self.meeting_panel and self.meeting_panel.winfo_exists():
            self.meeting_panel.lift()
            return
        self.meeting_panel = MeetingPanel(
            self,
            on_activity=self._log_activity,
            on_result=self._handle_meeting_result,
        )

    def _handle_meeting_result(self, summary: str, metadata: Dict[str, object]) -> None:
        message = summary.strip()
        output_dir = metadata.get("output_dir")
        audio_path = metadata.get("audio_path")
        header_lines = ["[회의 비서]"]
        if audio_path:
            header_lines.append(f"입력: {audio_path}")
        if output_dir:
            header_lines.append(f"산출물 폴더: {output_dir}")
        header_lines.append("")
        header = "\n".join(header_lines)
        full_message = f"{header}{message}"
        if not self.expanded:
            self.chat_panel.show()
            self.expanded = True
        self._post_message("assistant", full_message)
        self._log_activity("MEETING · summary delivered")

    def _open_photo_panel(self) -> None:
        if self.photo_panel and self.photo_panel.winfo_exists():
            self.photo_panel.lift()
            return
        self.photo_panel = PhotoPanel(
            self,
            on_activity=self._log_activity,
            on_result=self._handle_photo_result,
        )

    def _handle_photo_result(self, summary: str, metadata: Dict[str, object]) -> None:
        header_lines = ["[사진 비서]"]
        if metadata.get("root_count"):
            header_lines.append(f"대상 폴더 수: {metadata['root_count']}")
        if metadata.get("output_dir"):
            header_lines.append(f"결과 폴더: {metadata['output_dir']}")
        if metadata.get("report_path"):
            header_lines.append(f"리포트: {metadata['report_path']}")
        header_lines.append("")
        if not self.expanded:
            self.chat_panel.show()
            self.expanded = True
        self._post_message("assistant", "\n".join(header_lines) + summary.strip())
        self._log_activity("PHOTO · recommendation delivered")

    def _handle_quick_action(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        self.input_dock.set_text(cleaned)
        self.input_dock.send()

    def _trigger_rebuild_index(self) -> None:
        if self._reindex_thread and self._reindex_thread.is_alive():
            self._post_message("assistant", "인덱스 재구축이 이미 진행 중입니다. 잠시만 기다려 주세요.")
            return

        def _worker() -> None:
            started = time.time()
            self._log_activity("SYSTEM · index rebuild started")
            self._post_message("assistant", "문서 인덱스를 재구축하는 중입니다. 잠시만 기다려 주세요.")
            try:
                rebuild_index(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
            except Exception as exc:  # pragma: no cover - defensive
                self._log_activity(f"ERROR · index rebuild failed: {exc}")
                self._post_message("assistant", f"인덱스 재구축 중 오류가 발생했습니다: {exc}")
            else:
                elapsed = time.time() - started
                self._log_activity("SYSTEM · index rebuild completed")
                self._post_message("assistant", f"문서 인덱스 재구축이 완료되었습니다. ({elapsed:.1f}s)")
            finally:
                self._reindex_thread = None
                if self.work_center_panel and self.work_center_panel.winfo_exists():
                    self.after(0, self.work_center_panel.refresh_resource_log)

        self._reindex_thread = threading.Thread(target=_worker, daemon=True)
        self._reindex_thread.start()

    def _open_settings_dialog(self) -> None:
        if self.settings_panel and self.settings_panel.winfo_exists():
            self.settings_panel.lift()
            return
        self.settings_panel = SettingsPanel(
            self,
            self.settings,
            on_save=self._handle_settings_saved,
        )


# Backwards compatibility for scripts that import App from ui.app
App = AISummaryApp


if __name__ == "__main__":
    app = AISummaryApp()
    app.mainloop()
