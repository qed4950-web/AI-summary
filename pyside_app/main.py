from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.config.paths import CACHE_DIR, CORPUS_PATH, TOPIC_MODEL_PATH
from pyside_app.chat_worker import ChatWorker


class EngineLoader(QtCore.QThread):
    ready = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, agent: DocumentAgent, parent=None) -> None:
        super().__init__(parent)
        self.agent = agent

    def run(self) -> None:
        try:
            self.agent.prepare()
            self.ready.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class RebuildWorker(QtCore.QThread):
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, agent: DocumentAgent, parent=None) -> None:
        super().__init__(parent)
        self.agent = agent

    def run(self) -> None:
        try:
            self.agent.rebuild_index()
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AI Summary (PySide)")
        self.setStyleSheet(Path(__file__).with_name("styles.qss").read_text(encoding="utf-8"))
        self.resize(980, 720)
        self.setMinimumSize(900, 640)
        self._fit_to_screen()

        self.agent = self._build_agent()
        self.agent_ready = False

        self._build_ui()
        self._log_system("엔진을 준비하는 중입니다...")
        self._load_engine()

    def _build_agent(self) -> DocumentAgent:
        llm_backend = (os.getenv("UI_LLM_BACKEND") or "ollama").strip()
        llm_model = (os.getenv("UI_LLM_MODEL") or "gemma3:4b").strip()
        return DocumentAgent(
            DocumentAgentConfig(
                model_path=TOPIC_MODEL_PATH,
                corpus_path=CORPUS_PATH,
                cache_dir=CACHE_DIR,
                topk=5,
                min_similarity=0.35,
                auto_search=False,
                llm_backend=llm_backend,
                llm_model=llm_model,
                rerank=False,
                llm_timeout=6.0,
            )
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header 영역
        header_box = QtWidgets.QHBoxLayout()
        header_text = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("AI Summary")
        title.setProperty("title", True)
        subtitle = QtWidgets.QLabel("로컬 엔진과 직접 통신하는 오프라인 챗")
        subtitle.setProperty("subtitle", True)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header_box.addLayout(header_text, 1)

        self.status_label = QtWidgets.QLabel("준비 중")
        self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_label.setProperty("status", True)
        header_box.addWidget(self.status_label, 0)
        layout.addLayout(header_box)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.list_widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.list_widget, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QTextEdit()
        self.input.setPlaceholderText("대화를 입력하세요. 검색은 /search, 종료는 /quit, 인덱스 재구성은 /reindex")
        self.input.setFixedHeight(80)
        mono_font = QtGui.QFont("Segoe UI", 11)
        self.input.setFont(mono_font)
        input_row.addWidget(self.input, 1)

        self.send_btn = QtWidgets.QPushButton("전송")
        self.send_btn.clicked.connect(self._handle_send)
        input_row.addWidget(self.send_btn, 0)

        layout.addLayout(input_row)

        self.setCentralWidget(central)

    def _load_engine(self) -> None:
        self.loader = EngineLoader(self.agent)
        self.loader.ready.connect(self._on_agent_ready)
        self.loader.failed.connect(self._on_agent_failed)
        self.loader.start()

    def _on_agent_ready(self) -> None:
        self.agent_ready = True
        self.status_label.setText("준비 완료")
        self._log_system("엔진 준비가 완료되었습니다.")

    def _on_agent_failed(self, msg: str) -> None:
        self.status_label.setText("엔진 오류")
        self._log_system(f"엔진 초기화 실패: {msg}")

    def _handle_send(self) -> None:
        QtGui.QGuiApplication.inputMethod().commit()
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._append_message("나", text, role="user")

        lower_text = text.strip().lower()
        if lower_text in {"/reindex", "/rebuild", "/rebuild_index"}:
            self._log_system("인덱스를 다시 구성할게요. 잠시만 기다려 주세요.")
            self._start_rebuild()
            return

        context: dict | None = None
        if not text.startswith(("/search", "/doc")):
            context = {"force_action": "dialogue"}
        if not self.agent_ready:
            self._log_system("엔진 준비 중입니다. 잠시 후 다시 시도하세요.")
            return
        self.send_btn.setEnabled(False)
        self.status_label.setText("요청 중...")
        self.worker = ChatWorker(self.agent, text, context=context)
        self.worker.finished.connect(self._on_answer)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_answer(self, payload: dict) -> None:
        answer = payload.get("answer") or "결과가 없습니다."
        hits: List[dict] = payload.get("hits") or []
        self._append_message("비서", answer, role="assistant", hits=hits)
        self.status_label.setText("준비 완료")
        self.send_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self._append_message("시스템", f"요청 실패: {msg}", role="system")
        self.status_label.setText("오류")
        self.send_btn.setEnabled(True)

    def _append_message(self, who: str, text: str, role: str = "assistant", hits: List[dict] | None = None) -> None:
        item = QtWidgets.QListWidgetItem()
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setContentsMargins(12, 8, 12, 8)
        badge = QtWidgets.QLabel(who)
        badge.setProperty("badge", role)
        badge.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(badge)
        header.addStretch(1)
        vbox.addLayout(header)

        body = QtWidgets.QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        body.setProperty("bubble", role)
        vbox.addWidget(body)

        if hits:
            refs = QtWidgets.QLabel("관련 문서:")
            refs.setProperty("refs", True)
            vbox.addWidget(refs)
            for hit in hits[:5]:
                ref_label = QtWidgets.QLabel(f"- {hit.get('path')} (유사도 {hit.get('similarity')})")
                ref_label.setProperty("refitem", True)
                vbox.addWidget(ref_label)

        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        self._update_item_size(item, widget)
        self.list_widget.scrollToBottom()

    def _log_system(self, text: str) -> None:
        self._append_message("시스템", text, role="system")

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        size = self.geometry()
        x = geo.x() + (geo.width() - size.width()) // 2
        y = geo.y() + (geo.height() - size.height()) // 2
        self.move(x, y)

    def _fit_to_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        width = min(self.width(), geo.width() - 40)
        height = min(self.height(), geo.height() - 80)
        self.resize(max(width, self.minimumWidth()), max(height, self.minimumHeight()))
        self._center_on_screen()

    def _start_rebuild(self) -> None:
        if getattr(self, "rebuild_worker", None) and self.rebuild_worker.isRunning():
            self._log_system("이미 인덱스를 다시 구성 중입니다.")
            return
        self.send_btn.setEnabled(False)
        self.status_label.setText("인덱스 재구성 중...")
        self.rebuild_worker = RebuildWorker(self.agent)
        self.rebuild_worker.finished.connect(self._on_rebuild_finished)
        self.rebuild_worker.failed.connect(self._on_rebuild_failed)
        self.rebuild_worker.start()

    def _on_rebuild_finished(self) -> None:
        self.status_label.setText("준비 완료")
        self.send_btn.setEnabled(True)
        self._log_system("인덱스 재구성이 완료되었습니다.")

    def _on_rebuild_failed(self, message: str) -> None:
        self.status_label.setText("오류")
        self.send_btn.setEnabled(True)
        self._log_system(f"인덱스 재구성 실패: {message}")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_items_size()

    def _refresh_items_size(self) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if not item:
                continue
            widget = self.list_widget.itemWidget(item)
            if widget:
                self._update_item_size(item, widget)

    def _update_item_size(self, item: QtWidgets.QListWidgetItem, widget: QtWidgets.QWidget) -> None:
        """Keep chat bubbles wide enough to wrap text and avoid clipping."""
        viewport_width = self.list_widget.viewport().width()
        frame = self.list_widget.frameWidth()
        margins = widget.layout().contentsMargins() if widget.layout() else QtCore.QMargins(0, 0, 0, 0)
        wrap_width = max(0, viewport_width - frame * 2 - 8)
        widget.setMinimumWidth(wrap_width)
        widget.setMaximumWidth(wrap_width)
        bubble_width = max(0, wrap_width - margins.left() - margins.right())
        for label in widget.findChildren(QtWidgets.QLabel):
            if label.property("bubble") and label.wordWrap():
                label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, label.sizePolicy().verticalPolicy())
                label.setMaximumWidth(bubble_width)
        widget.adjustSize()
        item.setSizeHint(widget.sizeHint())


def main() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
