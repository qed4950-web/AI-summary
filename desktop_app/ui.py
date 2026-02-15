from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse
from uuid import uuid4

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config.desktop_runtime_policy import (
    DEFAULT_DESKTOP_RUNTIME_POLICY,
    load_desktop_runtime_policy,
    load_desktop_runtime_policy_history,
    save_desktop_runtime_policy,
)
from core.config.mode_profiles import MODE_ORDER, load_mode_profiles, save_mode_profiles
from core.policy.registry import SmartFolderRegistry
from desktop_app.gallery_ui import PhotoGalleryDialog
from desktop_app.tasks_ui import TaskManagerWindow

try:
    from core.agents.meeting.pii import mask_text as _mask_pii_text
except Exception:  # pragma: no cover - optional privacy helper import
    def _mask_pii_text(text: str) -> str:
        return text


class EnhancedInput(QTextEdit):
    submit = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._min_height = 44
        self._max_height = 168
        self.setFixedHeight(self._min_height)
        self.textChanged.connect(self.adjust_height)

    def adjust_height(self):
        doc_height = int(self.document().size().height()) + 10
        new_height = max(self._min_height, min(self._max_height, doc_height))
        self.setFixedHeight(new_height)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.submit.emit()
            return
        super().keyPressEvent(event)


class ActionRecoveryCard(QFrame):
    def __init__(self, *, file_path: str, shortcut_mod: str, action_callback: Callable[[str, str], None], parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self.setObjectName("ActionRecoveryCard")
        file_name = Path(file_path).name
        self.setAccessibleName("File recovery actions")
        self.setAccessibleDescription(f"Recovery actions for {file_name}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel(f"복구 액션: {file_name}")
        title.setObjectName("ActionRecoveryTitle")
        title.setToolTip(file_path)
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        btn_retry = QPushButton(f"{shortcut_mod}+O 다시 열기")
        btn_retry.setObjectName("RecoveryRetryButton")
        btn_retry.setAccessibleName("Retry open file")
        btn_retry.setAccessibleDescription(f"Try opening {file_name} again")
        btn_retry.setDefault(True)
        btn_parent = QPushButton(f"{shortcut_mod}+Shift+P 상위 폴더")
        btn_parent.setObjectName("RecoveryParentButton")
        btn_parent.setAccessibleName("Open parent folder")
        btn_parent.setAccessibleDescription(f"Open parent folder of {file_name}")
        btn_reveal = QPushButton(f"{shortcut_mod}+Shift+R 위치 열기")
        btn_reveal.setObjectName("RecoveryRevealButton")
        btn_reveal.setAccessibleName("Reveal in Finder")
        btn_reveal.setAccessibleDescription(f"Reveal location for {file_name}")
        btn_copy = QPushButton(f"{shortcut_mod}+Shift+O 경로 복사")
        btn_copy.setObjectName("RecoveryCopyButton")
        btn_copy.setAccessibleName("Copy file path")
        btn_copy.setAccessibleDescription(f"Copy full path for {file_name}")
        btn_retry.clicked.connect(lambda: action_callback("retry_open", self._file_path))
        btn_parent.clicked.connect(lambda: action_callback("open_parent", self._file_path))
        btn_reveal.clicked.connect(lambda: action_callback("reveal_in_finder", self._file_path))
        btn_copy.clicked.connect(lambda: action_callback("copy_path", self._file_path))
        row.addWidget(btn_retry)
        row.addWidget(btn_parent)
        row.addWidget(btn_reveal)
        row.addWidget(btn_copy)
        row.addStretch()
        layout.addLayout(row)
        hint = QLabel("Tab 순서: 다시 열기 → 상위 폴더 → 위치 열기 → 경로 복사")
        hint.setObjectName("ActionRecoveryHint")
        layout.addWidget(hint)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocusProxy(btn_retry)
        QWidget.setTabOrder(btn_retry, btn_parent)
        QWidget.setTabOrder(btn_parent, btn_reveal)
        QWidget.setTabOrder(btn_reveal, btn_copy)

        self.setStyleSheet(
            """
            QFrame#ActionRecoveryCard {
                background: #dbeafe;
                border: 1px solid #93c5fd;
                border-radius: 10px;
            }
            QLabel#ActionRecoveryTitle {
                color: #1e3a8a;
                font-weight: 700;
                font-size: 12px;
            }
            QLabel#ActionRecoveryHint {
                color: #1e40af;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#RecoveryRetryButton {
                border: 1px solid #1d4ed8;
                border-radius: 8px;
                background: #1d4ed8;
                color: #ffffff;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 800;
            }
            QPushButton#RecoveryParentButton,
            QPushButton#RecoveryRevealButton,
            QPushButton#RecoveryCopyButton {
                border: 1px solid #93c5fd;
                border-radius: 8px;
                background: #eff6ff;
                color: #1d4ed8;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#RecoveryRetryButton:hover {
                background: #1e3a8a;
            }
            QPushButton#RecoveryParentButton:hover,
            QPushButton#RecoveryRevealButton:hover,
            QPushButton#RecoveryCopyButton:hover {
                background: #dbeafe;
            }
            """
        )


class FailureGuideCard(QFrame):
    _button_name_map = {
        "open_permission_guide": "GuidePermissionButton",
        "open_app_association_guide": "GuideAssociationButton",
        "search_similar_files": "GuideSimilarButton",
        "reveal_in_finder": "GuideFinderButton",
    }

    def __init__(
        self,
        *,
        file_path: str,
        actions: list[tuple[str, str]],
        action_callback: Callable[[str, str], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("FailureGuideCard")
        file_name = Path(file_path).name
        self.setAccessibleName("File open guidance")
        self.setAccessibleDescription(f"Guidance actions for {file_name}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel(f"열기 실패 가이드: {file_name}")
        title.setObjectName("FailureGuideTitle")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        buttons: list[QPushButton] = []
        for label, action_code in actions:
            button = QPushButton(label)
            button.setObjectName(self._button_name_map.get(action_code, "GuideActionButton"))
            button.setAccessibleName(label)
            button.setAccessibleDescription(f"{label} for {file_name}")
            button.clicked.connect(lambda _checked=False, code=action_code: action_callback(code, file_path))
            row.addWidget(button)
            buttons.append(button)
        row.addStretch()
        layout.addLayout(row)
        hint_labels = [label for label, _ in actions if str(label).strip()]
        hint_text = " → ".join(hint_labels) if hint_labels else "가이드 액션"
        hint = QLabel(f"Tab 순서: {hint_text}")
        hint.setObjectName("FailureGuideHint")
        layout.addWidget(hint)

        if buttons:
            buttons[0].setDefault(True)
            self.setFocusPolicy(Qt.StrongFocus)
            self.setFocusProxy(buttons[0])
            for idx in range(len(buttons) - 1):
                QWidget.setTabOrder(buttons[idx], buttons[idx + 1])

        self.setStyleSheet(
            """
            QFrame#FailureGuideCard {
                background: #fff7ed;
                border: 1px solid #fdba74;
                border-radius: 10px;
            }
            QLabel#FailureGuideTitle {
                color: #9a3412;
                font-weight: 700;
                font-size: 12px;
            }
            QLabel#FailureGuideHint {
                color: #9a3412;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#GuidePermissionButton,
            QPushButton#GuideAssociationButton,
            QPushButton#GuideSimilarButton,
            QPushButton#GuideFinderButton,
            QPushButton#GuideActionButton {
                border: 1px solid #fdba74;
                border-radius: 8px;
                background: #ffffff;
                color: #9a3412;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#GuidePermissionButton:hover,
            QPushButton#GuideAssociationButton:hover,
            QPushButton#GuideSimilarButton:hover,
            QPushButton#GuideFinderButton:hover,
            QPushButton#GuideActionButton:hover {
                background: #ffedd5;
            }
            QPushButton#GuidePermissionButton:focus,
            QPushButton#GuideAssociationButton:focus,
            QPushButton#GuideSimilarButton:focus,
            QPushButton#GuideFinderButton:focus,
            QPushButton#GuideActionButton:focus {
                border: 2px solid #ea580c;
            }
            """
        )


class SmartFolderManagerDialog(QDialog):
    def __init__(self, registry, parent=None, mode_callback=None, runtime_policy_callback=None):
        super().__init__(parent)
        self.registry = registry
        self.mode_callback = mode_callback
        self.runtime_policy_callback = runtime_policy_callback
        self.setWindowTitle("Smart Folders")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Manage Smart Folders")
        title.setObjectName("DialogTitle")
        layout.addWidget(title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("FolderScroll")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        controls = QHBoxLayout()
        if self.mode_callback is not None:
            btn_mode = QPushButton("Mode Presets")
            btn_mode.clicked.connect(self.mode_callback)
            btn_mode.setObjectName("DialogModeBtn")
            controls.addWidget(btn_mode)
        if self.runtime_policy_callback is not None:
            btn_policy = QPushButton("Runtime Policy")
            btn_policy.clicked.connect(self.runtime_policy_callback)
            btn_policy.setObjectName("DialogModeBtn")
            controls.addWidget(btn_policy)
        controls.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setObjectName("DialogCloseBtn")
        controls.addWidget(btn_close)
        layout.addLayout(controls)

        self.setStyleSheet(
            """
            QDialog { background: #f4f6f9; }
            QLabel#DialogTitle { font-size: 20px; font-weight: 700; color: #1e2430; }
            QScrollArea#FolderScroll { border: 1px solid #d5dae3; border-radius: 10px; background: white; }
            QPushButton#DialogCloseBtn {
                background: #1f2937; color: #ffffff; border: none; border-radius: 8px;
                padding: 8px 14px; font-weight: 600;
            }
            QPushButton#DialogCloseBtn:hover { background: #111827; }
            QPushButton#DialogModeBtn {
                background: #ffffff; color: #1f2937; border: 1px solid #cfd6e1; border-radius: 8px;
                padding: 8px 14px; font-weight: 600;
            }
            QPushButton#DialogModeBtn:hover { background: #f3f4f6; }
            """
        )

        self.refresh_list()

    def refresh_list(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        folders = self.registry.list_folders()
        if not folders:
            lbl = QLabel("No smart folders registered.\nDrop a folder into the launcher to add one.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #6b7280; padding: 28px;")
            self.scroll_layout.addWidget(lbl)
            return

        for folder in folders:
            row = QFrame()
            row.setStyleSheet(
                "background: #ffffff; border: 1px solid #d8dde7; border-radius: 10px; padding: 8px;"
            )
            row_layout = QHBoxLayout(row)

            info_layout = QVBoxLayout()
            lbl_name = QLabel(folder.get("label", "Unknown"))
            lbl_name.setStyleSheet("font-weight: 700; color: #111827;")
            lbl_path = QLabel(folder.get("path", ""))
            lbl_path.setStyleSheet("color: #6b7280; font-size: 12px;")
            info_layout.addWidget(lbl_name)
            info_layout.addWidget(lbl_path)

            btn_remove = QPushButton("Remove")
            btn_remove.setCursor(Qt.PointingHandCursor)
            btn_remove.setStyleSheet(
                """
                QPushButton {
                    background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;
                    border-radius: 6px; padding: 6px 10px; font-weight: 600;
                }
                QPushButton:hover { background: #fecaca; }
                """
            )
            btn_remove.clicked.connect(lambda checked=False, p=folder.get("path"): self.remove_folder(p))

            row_layout.addLayout(info_layout, stretch=1)
            row_layout.addWidget(btn_remove, stretch=0)

            self.scroll_layout.addWidget(row)

    def remove_folder(self, path):
        if self.registry.remove_folder(Path(path)):
            self.refresh_list()


class ModeProfileDialog(QDialog):
    def __init__(self, profiles: dict[str, dict[str, object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mode Presets")
        self.resize(760, 520)
        self._profiles = profiles
        self._editors: dict[str, dict[str, QWidget]] = {}
        self._desc_fields: dict[str, QLineEdit] = {}
        self._status_fields: dict[str, QLineEdit] = {}
        self._topk_fields: dict[str, QLineEdit] = {}
        self._desc_error_labels: dict[str, QLabel] = {}
        self._status_error_labels: dict[str, QLabel] = {}
        self._topk_error_labels: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("Customize Response Mode Presets")
        title.setObjectName("ModeDialogTitle")
        root.addWidget(title)

        subtitle = QLabel("각 모드의 action/top-k/tokens/temperature를 조정할 수 있습니다.")
        subtitle.setObjectName("ModeDialogSubtitle")
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(10)
        body_layout.setAlignment(Qt.AlignTop)

        for mode in MODE_ORDER:
            row = QFrame()
            row.setObjectName("ModePresetRow")
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(10, 10, 10, 10)
            row_layout.setSpacing(6)

            profile = self._profiles.get(mode, {})
            heading = QLabel(mode)
            heading.setObjectName("ModePresetHeading")
            row_layout.addWidget(heading)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight)

            desc = QLineEdit(str(profile.get("description", "")).strip())
            desc.setPlaceholderText("1-48 chars")
            desc.textChanged.connect(self._on_text_fields_changed)
            self._desc_fields[mode] = desc
            status = QLineEdit(str(profile.get("thinking_status", "")).strip())
            status.setPlaceholderText("1-24 chars")
            status.textChanged.connect(self._on_text_fields_changed)
            self._status_fields[mode] = status

            action = QComboBox()
            action.addItems(["auto", "chat", "search"])
            force_action = profile.get("force_action")
            action_val = "auto" if force_action in (None, "", "auto") else str(force_action)
            index = action.findText(action_val)
            action.setCurrentIndex(index if index >= 0 else 0)

            desc_error = QLabel("Description은 1-48자여야 합니다.")
            desc_error.setObjectName("ModeInlineError")
            desc_error.setWordWrap(True)
            desc_error.hide()
            self._desc_error_labels[mode] = desc_error

            status_error = QLabel("Status는 1-24자여야 합니다.")
            status_error.setObjectName("ModeInlineError")
            status_error.setWordWrap(True)
            status_error.hide()
            self._status_error_labels[mode] = status_error

            topk = QLineEdit()
            topk_val = profile.get("topk")
            topk.setText("auto" if topk_val in (None, "", "auto") else str(topk_val))
            topk.setPlaceholderText("auto or number")
            topk.textChanged.connect(self._on_topk_changed)
            self._topk_fields[mode] = topk

            topk_error = QLabel("Top-k는 auto/none 또는 양의 정수만 허용됩니다.")
            topk_error.setObjectName("ModeInlineError")
            topk_error.setWordWrap(True)
            topk_error.hide()
            self._topk_error_labels[mode] = topk_error

            tokens = QSpinBox()
            tokens.setRange(64, 4096)
            tokens.setSingleStep(64)
            tokens.setValue(int(profile.get("llm_max_new_tokens", 512) or 512))

            temp = QDoubleSpinBox()
            temp.setRange(0.0, 2.0)
            temp.setSingleStep(0.01)
            temp.setDecimals(2)
            temp.setValue(float(profile.get("llm_temperature", 0.0) or 0.0))

            form.addRow("Description", desc)
            form.addRow("", desc_error)
            form.addRow("Status", status)
            form.addRow("", status_error)
            form.addRow("Action", action)
            form.addRow("Top-k", topk)
            form.addRow("", topk_error)
            form.addRow("Max tokens", tokens)
            form.addRow("Temperature", temp)
            row_layout.addLayout(form)

            self._editors[mode] = {
                "description": desc,
                "thinking_status": status,
                "force_action": action,
                "topk": topk,
                "llm_max_new_tokens": tokens,
                "llm_temperature": temp,
            }
            body_layout.addWidget(row)

        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save")
        btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save_and_close)
        actions.addWidget(btn_cancel)
        actions.addWidget(self.btn_save)
        root.addLayout(actions)

        self.setStyleSheet(
            """
            QDialog { background: #f4f6f9; }
            QLabel#ModeDialogTitle { font-size: 18px; font-weight: 700; color: #111827; }
            QLabel#ModeDialogSubtitle { font-size: 12px; color: #6b7280; }
            QLabel#ModeInlineError { font-size: 11px; color: #b91c1c; padding: 1px 2px; }
            QFrame#ModePresetRow {
                background: #ffffff;
                border: 1px solid #d8dde7;
                border-radius: 10px;
            }
            QLabel#ModePresetHeading { font-weight: 700; color: #111827; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #d0d7e4;
                border-radius: 7px;
                padding: 4px 6px;
                background: #ffffff;
                min-height: 22px;
            }
            QLineEdit[invalid="true"] {
                border: 1px solid #dc2626;
                background: #fef2f2;
                color: #991b1b;
            }
            QPushButton {
                border: 1px solid #cfd6e1;
                border-radius: 8px;
                background: #ffffff;
                color: #1f2937;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f3f4f6; }
            """
        )
        self._validate_form()

    @staticmethod
    def _is_valid_description(raw: str) -> bool:
        normalized = (raw or "").strip()
        return 1 <= len(normalized) <= 48

    @staticmethod
    def _is_valid_status(raw: str) -> bool:
        normalized = (raw or "").strip()
        return 1 <= len(normalized) <= 24

    @staticmethod
    def _is_valid_topk(raw: str) -> bool:
        normalized = (raw or "").strip().lower()
        if normalized in {"", "auto", "none"}:
            return True
        try:
            value = int(normalized)
        except ValueError:
            return False
        return value > 0

    @staticmethod
    def _parse_topk(raw: str) -> int | None:
        normalized = (raw or "").strip().lower()
        if normalized in {"", "auto", "none"}:
            return None
        try:
            value = int(normalized)
        except ValueError:
            return None
        return value if value > 0 else None

    @staticmethod
    def _set_invalid_state(widget: QWidget, invalid: bool) -> None:
        widget.setProperty("invalid", invalid)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _on_topk_changed(self, _text: str) -> None:
        self._validate_form()

    def _on_text_fields_changed(self, _text: str) -> None:
        self._validate_form()

    def _validate_form(self) -> bool:
        has_error = False
        for mode in MODE_ORDER:
            desc = self._desc_fields.get(mode)
            if desc is not None:
                desc_invalid = not self._is_valid_description(desc.text())
                self._set_invalid_state(desc, desc_invalid)
                desc_error = self._desc_error_labels.get(mode)
                if desc_error is not None:
                    desc_error.setVisible(desc_invalid)
                has_error = has_error or desc_invalid

            status = self._status_fields.get(mode)
            if status is not None:
                status_invalid = not self._is_valid_status(status.text())
                self._set_invalid_state(status, status_invalid)
                status_error = self._status_error_labels.get(mode)
                if status_error is not None:
                    status_error.setVisible(status_invalid)
                has_error = has_error or status_invalid

            topk = self._topk_fields.get(mode)
            if topk is not None:
                topk_invalid = not self._is_valid_topk(topk.text())
                self._set_invalid_state(topk, topk_invalid)
                topk_error = self._topk_error_labels.get(mode)
                if topk_error is not None:
                    topk_error.setVisible(topk_invalid)
                has_error = has_error or topk_invalid

        self.btn_save.setEnabled(not has_error)
        return not has_error

    def _build_profiles(self) -> dict[str, dict[str, object]]:
        profiles: dict[str, dict[str, object]] = {}
        for mode, controls in self._editors.items():
            desc_widget = controls["description"]
            status_widget = controls["thinking_status"]
            action_widget = controls["force_action"]
            topk_widget = controls["topk"]
            tokens_widget = controls["llm_max_new_tokens"]
            temp_widget = controls["llm_temperature"]
            assert isinstance(desc_widget, QLineEdit)
            assert isinstance(status_widget, QLineEdit)
            assert isinstance(action_widget, QComboBox)
            assert isinstance(topk_widget, QLineEdit)
            assert isinstance(tokens_widget, QSpinBox)
            assert isinstance(temp_widget, QDoubleSpinBox)
            action_val = action_widget.currentText().strip().lower()
            profiles[mode] = {
                "description": desc_widget.text().strip(),
                "thinking_status": status_widget.text().strip(),
                "force_action": None if action_val == "auto" else action_val,
                "topk": self._parse_topk(topk_widget.text()),
                "llm_max_new_tokens": int(tokens_widget.value()),
                "llm_temperature": float(temp_widget.value()),
            }
        return profiles

    def _save_and_close(self):
        if not self._validate_form():
            return
        profiles = self._build_profiles()
        save_mode_profiles(profiles)
        self.accept()


class RuntimePolicyDialog(QDialog):
    def __init__(self, policy: dict[str, object], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Runtime Policy")
        self.resize(520, 360)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("Runtime Policy")
        title.setObjectName("RuntimeDialogTitle")
        subtitle = QLabel("응답/링크/마스킹 정책을 설정 파일로 저장합니다.")
        subtitle.setObjectName("RuntimeDialogSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        effective = self._normalize_policy(policy)

        panel = QFrame()
        panel.setObjectName("RuntimePolicyPanel")
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight)

        self.privacy_mask = QCheckBox("민감정보를 마스킹하여 표시")
        self.privacy_mask.setChecked(bool(effective["privacy_mask_enabled"]))

        self.max_file_links = QSpinBox()
        self.max_file_links.setRange(1, 64)
        self.max_file_links.setSingleStep(1)
        self.max_file_links.setValue(int(effective["max_file_links"]))

        self.max_reference_links = QSpinBox()
        self.max_reference_links.setRange(1, 64)
        self.max_reference_links.setSingleStep(1)
        self.max_reference_links.setValue(int(effective["max_reference_links"]))

        self.max_response_chars = QSpinBox()
        self.max_response_chars.setRange(1200, 120000)
        self.max_response_chars.setSingleStep(200)
        self.max_response_chars.setValue(int(effective["max_response_chars"]))

        self.max_suggestion_chars = QSpinBox()
        self.max_suggestion_chars.setRange(24, 1024)
        self.max_suggestion_chars.setSingleStep(8)
        self.max_suggestion_chars.setValue(int(effective["max_suggestion_chars"]))

        form.addRow("Privacy", self.privacy_mask)
        form.addRow("Max file links", self.max_file_links)
        form.addRow("Max reference links", self.max_reference_links)
        form.addRow("Max response chars", self.max_response_chars)
        form.addRow("Max suggestion chars", self.max_suggestion_chars)
        root.addWidget(panel, stretch=1)

        note = QLabel("저장 후 재시작 없이 다음 질의부터 적용됩니다.")
        note.setObjectName("RuntimeDialogNote")
        root.addWidget(note)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_save = QPushButton("Save")
        btn_cancel.clicked.connect(self.reject)
        btn_save.clicked.connect(self._save_and_close)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_save)
        root.addLayout(actions)

        self.setStyleSheet(
            """
            QDialog { background: #f4f6f9; }
            QLabel#RuntimeDialogTitle { font-size: 18px; font-weight: 700; color: #111827; }
            QLabel#RuntimeDialogSubtitle { font-size: 12px; color: #6b7280; }
            QLabel#RuntimeDialogNote { font-size: 12px; color: #334155; }
            QFrame#RuntimePolicyPanel {
                background: #ffffff;
                border: 1px solid #d8dde7;
                border-radius: 10px;
                padding: 6px;
            }
            QCheckBox { color: #111827; font-weight: 600; }
            QSpinBox {
                border: 1px solid #d0d7e4;
                border-radius: 7px;
                padding: 4px 6px;
                background: #ffffff;
                min-height: 22px;
            }
            QPushButton {
                border: 1px solid #cfd6e1;
                border-radius: 8px;
                background: #ffffff;
                color: #1f2937;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f3f4f6; }
            """
        )

    @staticmethod
    def _normalize_policy(raw_policy: dict[str, object]) -> dict[str, object]:
        defaults = dict(DEFAULT_DESKTOP_RUNTIME_POLICY)
        if isinstance(raw_policy, dict):
            defaults.update(raw_policy)
        return defaults

    def _build_policy(self) -> dict[str, object]:
        return {
            "privacy_mask_enabled": bool(self.privacy_mask.isChecked()),
            "max_file_links": int(self.max_file_links.value()),
            "max_reference_links": int(self.max_reference_links.value()),
            "max_response_chars": int(self.max_response_chars.value()),
            "max_suggestion_chars": int(self.max_suggestion_chars.value()),
        }

    def _save_and_close(self):
        save_desktop_runtime_policy(self._build_policy(), source="runtime_policy_dialog_save")
        self.accept()


class SettingsHubDialog(QDialog):
    _session_history_filter_state: dict[str, str] = {
        "source": "all",
        "period": "all_time",
    }

    def __init__(
        self,
        *,
        smart_folder_count: int,
        current_mode: str,
        runtime_policy: dict[str, object],
        open_folders_callback: Callable[[], None],
        open_mode_callback: Callable[[], None],
        open_runtime_callback: Callable[[], None],
        on_runtime_policy_applied: Callable[[], None] | None = None,
        on_mode_profile_applied: Callable[[], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings Hub")
        self.resize(680, 540)
        self._smart_folder_count = smart_folder_count
        self._current_mode = current_mode
        self._open_folders_callback = open_folders_callback
        self._open_mode_callback = open_mode_callback
        self._open_runtime_callback = open_runtime_callback
        self._on_runtime_policy_applied = on_runtime_policy_applied
        self._on_mode_profile_applied = on_mode_profile_applied
        self._history_entries: list[dict[str, object]] = []
        self._history_entries_all: list[dict[str, object]] = []
        self._hub_status_reset_timer = QTimer(self)
        self._hub_status_reset_timer.setSingleShot(True)
        self._hub_status_reset_timer.timeout.connect(self._reset_hub_status)
        self._hub_status_last_key: tuple[str, str] | None = None
        self._hub_status_last_at: datetime | None = None
        self._hub_status_throttle_ms = 700
        self._hub_status_events: list[dict[str, str]] = []
        self._hub_status_event_limit = 8
        self._runtime_policy_snapshot = dict(DEFAULT_DESKTOP_RUNTIME_POLICY)
        if isinstance(runtime_policy, dict):
            self._runtime_policy_snapshot.update(runtime_policy)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("Settings Hub")
        title.setObjectName("SettingsHubTitle")
        subtitle = QLabel("프로젝트 엔진/모드/UI 정책을 한 곳에서 관리합니다.")
        subtitle.setObjectName("SettingsHubSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        self.summary_folders = QLabel("")
        self.summary_mode = QLabel("")
        self.summary_policy = QLabel("")
        self.summary_folders.setObjectName("SettingsHubSummary")
        self.summary_mode.setObjectName("SettingsHubSummary")
        self.summary_policy.setObjectName("SettingsHubSummary")

        summary_card = QFrame()
        summary_card.setObjectName("SettingsHubCard")
        card_layout = QVBoxLayout(summary_card)
        card_layout.setSpacing(8)
        card_layout.addWidget(self.summary_folders)
        card_layout.addWidget(self.summary_mode)
        card_layout.addWidget(self.summary_policy)
        root.addWidget(summary_card)

        self.hub_status = QLabel("Settings ready")
        self.hub_status.setObjectName("SettingsHubStatusBanner")
        self.hub_status.setProperty("tone", "info")
        self.hub_status.setWordWrap(True)
        root.addWidget(self.hub_status)

        status_log_title = QLabel("Status timeline (recent)")
        status_log_title.setObjectName("SettingsHubInlineStatus")
        root.addWidget(status_log_title)
        status_log_controls = QHBoxLayout()
        status_log_controls.setSpacing(6)
        status_log_filter_label = QLabel("Filter")
        status_log_filter_label.setObjectName("SettingsHubInlineStatus")
        self.hub_status_filter = QComboBox()
        self.hub_status_filter.addItem("All", "all")
        self.hub_status_filter.addItem("Errors", "error")
        self.hub_status_filter.addItem("Warnings", "warning")
        self.hub_status_filter.addItem("Success", "success")
        self.hub_status_filter.addItem("Info", "info")
        self.hub_status_filter.currentIndexChanged.connect(lambda _index: self._render_hub_status_log())
        status_log_range_label = QLabel("Range")
        status_log_range_label.setObjectName("SettingsHubInlineStatus")
        self.hub_status_time_filter = QComboBox()
        self.hub_status_time_filter.addItem("All", "all")
        self.hub_status_time_filter.addItem("Last 10m", "10m")
        self.hub_status_time_filter.addItem("Last 1h", "1h")
        self.hub_status_time_filter.addItem("Last 24h", "24h")
        self.hub_status_time_filter.currentIndexChanged.connect(lambda _index: self._render_hub_status_log())
        self.btn_copy_status_log = QPushButton("Copy")
        self.btn_copy_status_log.clicked.connect(self._copy_hub_status_log)
        self.btn_export_status_log = QPushButton("Export")
        self.btn_export_status_log.clicked.connect(self._export_hub_status_log)
        status_log_controls.addWidget(status_log_filter_label)
        status_log_controls.addWidget(self.hub_status_filter)
        status_log_controls.addWidget(status_log_range_label)
        status_log_controls.addWidget(self.hub_status_time_filter)
        status_log_controls.addStretch()
        status_log_controls.addWidget(self.btn_copy_status_log)
        status_log_controls.addWidget(self.btn_export_status_log)
        root.addLayout(status_log_controls)
        self.hub_status_log = QTextEdit()
        self.hub_status_log.setObjectName("SettingsHubStatusLog")
        self.hub_status_log.setReadOnly(True)
        self.hub_status_log.setFixedHeight(88)
        self.hub_status_log.setPlainText("No status events yet.")
        root.addWidget(self.hub_status_log)

        mode_card = QFrame()
        mode_card.setObjectName("SettingsHubInlineCard")
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.setSpacing(8)
        mode_title = QLabel("Inline Mode Preset")
        mode_title.setObjectName("SettingsHubInlineTitle")
        mode_layout.addWidget(mode_title)

        mode_form = QFormLayout()
        mode_form.setLabelAlignment(Qt.AlignRight)
        self.inline_mode_selector = QComboBox()
        self.inline_mode_selector.addItems(MODE_ORDER)
        self.inline_mode_selector.setCurrentText(current_mode if current_mode in MODE_ORDER else "Auto")
        self.inline_mode_action = QComboBox()
        self.inline_mode_action.addItems(["auto", "chat", "search"])
        self.inline_mode_topk = QLineEdit()
        self.inline_mode_topk.setPlaceholderText("auto or number")
        self.inline_mode_tokens = QSpinBox()
        self.inline_mode_tokens.setRange(64, 4096)
        self.inline_mode_tokens.setSingleStep(64)
        self.inline_mode_temp = QDoubleSpinBox()
        self.inline_mode_temp.setRange(0.0, 2.0)
        self.inline_mode_temp.setSingleStep(0.01)
        self.inline_mode_temp.setDecimals(2)
        mode_form.addRow("Mode", self.inline_mode_selector)
        mode_form.addRow("Action", self.inline_mode_action)
        mode_form.addRow("Top-k", self.inline_mode_topk)
        mode_form.addRow("Max tokens", self.inline_mode_tokens)
        mode_form.addRow("Temperature", self.inline_mode_temp)
        mode_layout.addLayout(mode_form)

        mode_actions = QHBoxLayout()
        self.btn_apply_inline_mode = QPushButton("Apply Inline Mode")
        self.inline_mode_status = QLabel("")
        self.inline_mode_error = QLabel("")
        self.inline_mode_status.setObjectName("SettingsHubInlineStatus")
        self.inline_mode_error.setObjectName("SettingsHubInlineError")
        self.inline_mode_error.setWordWrap(True)
        self.btn_apply_inline_mode.clicked.connect(self._apply_inline_mode_profile)
        mode_actions.addWidget(self.btn_apply_inline_mode)
        mode_actions.addWidget(self.inline_mode_status, stretch=1)
        mode_layout.addLayout(mode_actions)
        mode_layout.addWidget(self.inline_mode_error)
        root.addWidget(mode_card)

        inline_card = QFrame()
        inline_card.setObjectName("SettingsHubInlineCard")
        inline_layout = QVBoxLayout(inline_card)
        inline_layout.setSpacing(8)
        inline_title = QLabel("Inline Runtime Policy")
        inline_title.setObjectName("SettingsHubInlineTitle")
        inline_layout.addWidget(inline_title)

        inline_form = QFormLayout()
        inline_form.setLabelAlignment(Qt.AlignRight)
        self.inline_privacy_mask = QCheckBox("민감정보 마스킹 사용")
        self.inline_max_refs = QSpinBox()
        self.inline_max_refs.setRange(1, 64)
        self.inline_max_links = QSpinBox()
        self.inline_max_links.setRange(1, 64)
        self.inline_max_response_chars = QSpinBox()
        self.inline_max_response_chars.setRange(1200, 120000)
        self.inline_max_response_chars.setSingleStep(200)
        self.inline_max_suggestion_chars = QSpinBox()
        self.inline_max_suggestion_chars.setRange(24, 1024)
        self.inline_max_suggestion_chars.setSingleStep(8)
        inline_form.addRow("Privacy", self.inline_privacy_mask)
        inline_form.addRow("Max reference links", self.inline_max_refs)
        inline_form.addRow("Max file links", self.inline_max_links)
        inline_form.addRow("Max response chars", self.inline_max_response_chars)
        inline_form.addRow("Max suggestion chars", self.inline_max_suggestion_chars)
        inline_layout.addLayout(inline_form)

        inline_actions = QHBoxLayout()
        self.btn_apply_inline_policy = QPushButton("Apply Inline Policy")
        self.inline_status = QLabel("")
        self.inline_status.setObjectName("SettingsHubInlineStatus")
        self.btn_apply_inline_policy.clicked.connect(self._apply_inline_runtime_policy)
        inline_actions.addWidget(self.btn_apply_inline_policy)
        inline_actions.addWidget(self.inline_status, stretch=1)
        inline_layout.addLayout(inline_actions)
        root.addWidget(inline_card)

        history_card = QFrame()
        history_card.setObjectName("SettingsHubHistoryCard")
        history_layout = QVBoxLayout(history_card)
        history_layout.setSpacing(6)
        history_title = QLabel("Recent Runtime Policy Changes")
        history_title.setObjectName("SettingsHubInlineTitle")
        history_filter_row = QHBoxLayout()
        history_filter_row.setSpacing(6)
        history_source_label = QLabel("History source")
        history_source_label.setObjectName("SettingsHubInlineStatus")
        self.history_source_filter = QComboBox()
        self.history_source_filter.addItem("All sources", "all")
        self.history_source_filter.addItem("Manual save", "save_desktop_runtime_policy")
        self.history_source_filter.addItem("Runtime dialog", "runtime_policy_dialog_save")
        self.history_source_filter.addItem("Settings inline apply", "settings_inline_policy_apply")
        self.history_source_filter.addItem("Settings history restore", "settings_history_restore")
        history_period_label = QLabel("History period")
        history_period_label.setObjectName("SettingsHubInlineStatus")
        self.history_period_filter = QComboBox()
        self.history_period_filter.addItem("All time", "all_time")
        self.history_period_filter.addItem("24h", "last_24h")
        self.history_period_filter.addItem("7 days", "last_7d")
        self.history_period_filter.addItem("30 days", "last_30d")
        for label, token in self._build_history_absolute_period_presets():
            self.history_period_filter.addItem(label, token)
        history_custom_label = QLabel("Custom from")
        history_custom_label.setObjectName("SettingsHubInlineStatus")
        self.history_custom_period_input = QLineEdit()
        self.history_custom_period_input.setPlaceholderText("YYYY-MM-DD HH:MM")
        self.btn_apply_history_custom_period = QPushButton("Apply custom")
        self.btn_apply_history_custom_period.clicked.connect(self._apply_custom_history_period)
        self.history_custom_period_input.returnPressed.connect(self._apply_custom_history_period)
        history_filter_row.addWidget(history_source_label)
        history_filter_row.addWidget(self.history_source_filter, stretch=1)
        history_filter_row.addWidget(history_period_label)
        history_filter_row.addWidget(self.history_period_filter, stretch=1)
        history_filter_row.addWidget(history_custom_label)
        history_filter_row.addWidget(self.history_custom_period_input, stretch=1)
        history_filter_row.addWidget(self.btn_apply_history_custom_period)
        self.history_selector = QComboBox()
        self.btn_restore_history = QPushButton("Restore Selected Policy")
        self.history_confirm = QCheckBox("Preview 확인 후 복원")
        self.history_status = QLabel("")
        self.history_status.setObjectName("SettingsHubInlineStatus")
        self.btn_restore_history.clicked.connect(self._restore_selected_history_policy)
        self.history_box = QTextEdit()
        self.history_box.setObjectName("SettingsHubHistory")
        self.history_box.setReadOnly(True)
        self.history_box.setFixedHeight(118)
        self.history_preview = QTextEdit()
        self.history_preview.setObjectName("SettingsHubHistory")
        self.history_preview.setReadOnly(True)
        self.history_preview.setFixedHeight(80)
        history_layout.addWidget(history_title)
        history_layout.addLayout(history_filter_row)
        history_layout.addWidget(self.history_selector)
        history_layout.addWidget(self.history_confirm)
        history_layout.addWidget(self.history_box)
        history_layout.addWidget(self.history_preview)
        history_actions = QHBoxLayout()
        history_actions.addWidget(self.btn_restore_history)
        history_actions.addWidget(self.history_status, stretch=1)
        history_layout.addLayout(history_actions)
        root.addWidget(history_card)

        actions = QHBoxLayout()
        self.btn_folders = QPushButton("Smart Folders")
        self.btn_mode = QPushButton("Mode Presets")
        self.btn_runtime = QPushButton("Runtime Policy")
        self.btn_close = QPushButton("Close")
        self.btn_folders.clicked.connect(self._open_folders)
        self.btn_mode.clicked.connect(self._open_mode)
        self.btn_runtime.clicked.connect(self._open_runtime)
        self.btn_close.clicked.connect(self.accept)
        actions.addWidget(self.btn_folders)
        actions.addWidget(self.btn_mode)
        actions.addWidget(self.btn_runtime)
        actions.addStretch()
        actions.addWidget(self.btn_close)
        root.addLayout(actions)

        self._update_summary(
            smart_folder_count=smart_folder_count,
            current_mode=current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )
        self._inline_mode_profiles = load_mode_profiles()
        self.inline_mode_selector.currentTextChanged.connect(self._sync_inline_mode_controls)
        self.inline_mode_action.currentTextChanged.connect(lambda _value: self._validate_inline_mode_inputs())
        self.inline_mode_topk.textChanged.connect(lambda _value: self._validate_inline_mode_inputs())
        self.history_selector.currentIndexChanged.connect(lambda _index: self._update_history_preview())
        self.history_confirm.toggled.connect(lambda _checked: self._sync_history_restore_state())
        self.history_source_filter.currentIndexChanged.connect(self._on_history_filters_changed)
        self.history_period_filter.currentIndexChanged.connect(self._on_history_filters_changed)
        self._restore_history_filter_state()
        self._sync_inline_mode_controls(self.inline_mode_selector.currentText())
        self._hydrate_inline_controls(self._runtime_policy_snapshot)
        self._reload_history()

        self.setStyleSheet(
            """
            QDialog { background: #f4f6f9; }
            QLabel#SettingsHubTitle { font-size: 18px; font-weight: 700; color: #111827; }
            QLabel#SettingsHubSubtitle { font-size: 12px; color: #6b7280; }
            QFrame#SettingsHubCard {
                background: #ffffff;
                border: 1px solid #d8dde7;
                border-radius: 10px;
                padding: 10px;
            }
            QFrame#SettingsHubInlineCard, QFrame#SettingsHubHistoryCard {
                background: #ffffff;
                border: 1px solid #d8dde7;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel#SettingsHubInlineTitle { color: #111827; font-size: 13px; font-weight: 700; }
            QLabel#SettingsHubInlineStatus { color: #334155; font-size: 12px; font-weight: 600; }
            QLabel#SettingsHubInlineError { color: #b91c1c; font-size: 12px; font-weight: 600; }
            QLabel#SettingsHubSummary { color: #111827; font-size: 13px; font-weight: 600; }
            QLabel#SettingsHubStatusBanner {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#SettingsHubStatusBanner[tone="info"] {
                background: #eff6ff;
                border-color: #bfdbfe;
                color: #1d4ed8;
            }
            QLabel#SettingsHubStatusBanner[tone="success"] {
                background: #ecfdf3;
                border-color: #86efac;
                color: #166534;
            }
            QLabel#SettingsHubStatusBanner[tone="warning"] {
                background: #fffbeb;
                border-color: #fde68a;
                color: #92400e;
            }
            QLabel#SettingsHubStatusBanner[tone="error"] {
                background: #fef2f2;
                border-color: #fecaca;
                color: #b91c1c;
            }
            QTextEdit#SettingsHubHistory {
                border: 1px solid #d0d7e4;
                border-radius: 8px;
                background: #f8fafc;
                color: #0f172a;
                padding: 6px;
                font-size: 12px;
            }
            QTextEdit#SettingsHubStatusLog {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background: #f8fafc;
                color: #0f172a;
                padding: 6px;
                font-size: 11px;
            }
            QLineEdit[invalid="true"] {
                border: 1px solid #dc2626;
                background: #fef2f2;
                color: #991b1b;
            }
            QPushButton {
                border: 1px solid #cfd6e1;
                border-radius: 8px;
                background: #ffffff;
                color: #1f2937;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f3f4f6; }
            """
        )

    @staticmethod
    def _format_runtime_policy_summary(runtime_policy: dict[str, object]) -> str:
        mask_enabled = bool(runtime_policy.get("privacy_mask_enabled", True))
        refs = int(runtime_policy.get("max_reference_links", 5) or 5)
        links = int(runtime_policy.get("max_file_links", 8) or 8)
        chars = int(runtime_policy.get("max_response_chars", 24000) or 24000)
        return (
            f"Runtime policy: privacy={'mask' if mask_enabled else 'raw'} / "
            f"refs<={refs} / file-links<={links} / response<={chars}"
        )

    def _update_summary(self, *, smart_folder_count: int, current_mode: str, runtime_policy: dict[str, object]) -> None:
        self.summary_folders.setText(f"Smart folders: {smart_folder_count}")
        self.summary_mode.setText(f"Current mode: {current_mode}")
        self.summary_policy.setText(self._format_runtime_policy_summary(runtime_policy))

    def _set_hub_status(
        self,
        text: str,
        tone: str = "info",
        *,
        auto_reset_ms: int | None = None,
        legacy_label: QLabel | None = None,
        allow_throttle: bool = True,
        record_event: bool = True,
    ) -> None:
        normalized_tone = tone if tone in {"info", "success", "warning", "error"} else "info"
        status_text = str(text or "").strip() or "Settings ready"
        status_key = (normalized_tone, status_text)
        now = datetime.now(timezone.utc)
        if (
            allow_throttle
            and self._hub_status_last_key == status_key
            and self._hub_status_last_at is not None
            and (now - self._hub_status_last_at).total_seconds() * 1000 < self._hub_status_throttle_ms
        ):
            if legacy_label is not None:
                legacy_label.setText(status_text)
            return

        self._hub_status_reset_timer.stop()
        self._hub_status_last_key = status_key
        self._hub_status_last_at = now
        self.hub_status.setText(status_text)
        self.hub_status.setProperty("tone", normalized_tone)
        self.hub_status.style().unpolish(self.hub_status)
        self.hub_status.style().polish(self.hub_status)
        self.hub_status.update()
        if legacy_label is not None:
            legacy_label.setText(status_text)
        if record_event:
            self._append_hub_status_event(status_text, normalized_tone, now)
        if auto_reset_ms is None:
            auto_reset_ms = 4200 if normalized_tone in {"success", "warning"} else 0
        if auto_reset_ms > 0:
            self._hub_status_reset_timer.start(auto_reset_ms)

    def _reset_hub_status(self) -> None:
        self._set_hub_status(
            "Settings ready",
            "info",
            auto_reset_ms=0,
            allow_throttle=False,
            record_event=False,
        )

    def _append_hub_status_event(self, text: str, tone: str, at: datetime) -> None:
        stamp = at.astimezone().strftime("%H:%M:%S")
        self._hub_status_events.append(
            {
                "stamp": stamp,
                "tone": tone,
                "text": text,
                "at_utc": at.astimezone(timezone.utc).isoformat(),
            }
        )
        if len(self._hub_status_events) > self._hub_status_event_limit:
            self._hub_status_events = self._hub_status_events[-self._hub_status_event_limit :]
        self._render_hub_status_log()

    def _selected_hub_status_filter(self) -> str:
        value = self.hub_status_filter.currentData()
        normalized = str(value or "all").strip().lower()
        return normalized or "all"

    def _selected_hub_status_time_filter(self) -> str:
        value = self.hub_status_time_filter.currentData()
        normalized = str(value or "all").strip().lower()
        return normalized or "all"

    def _filtered_hub_status_events(self) -> list[dict[str, str]]:
        tone_filter = self._selected_hub_status_filter()
        time_filter = self._selected_hub_status_time_filter()
        now_utc = datetime.now(timezone.utc)
        seconds_window = 0
        if time_filter == "10m":
            seconds_window = 10 * 60
        elif time_filter == "1h":
            seconds_window = 60 * 60
        elif time_filter == "24h":
            seconds_window = 24 * 60 * 60

        filtered: list[dict[str, str]] = []
        for event in self._hub_status_events:
            tone = str(event.get("tone", "info"))
            if tone_filter != "all" and tone != tone_filter:
                continue
            if seconds_window > 0:
                at_raw = str(event.get("at_utc", ""))
                at_dt = self._parse_history_timestamp(at_raw)
                if at_dt is None:
                    continue
                if (now_utc - at_dt).total_seconds() > seconds_window:
                    continue
            filtered.append(event)
        return filtered

    def _render_hub_status_log(self) -> None:
        lines: list[str] = []
        for event in self._filtered_hub_status_events():
            tone = str(event.get("tone", "info"))
            stamp = str(event.get("stamp", ""))
            text = str(event.get("text", ""))
            lines.append(f"{stamp} [{tone}] {text}")
        self.hub_status_log.setPlainText("\n".join(lines) if lines else "No status events for current filter.")

    def _copy_hub_status_log(self) -> None:
        text = self.hub_status_log.toPlainText().strip()
        if not text:
            self._set_hub_status("Status log is empty", "warning", auto_reset_ms=1800, record_event=False)
            return
        QApplication.clipboard().setText(text)
        self._set_hub_status("Status log copied", "success", auto_reset_ms=1800, record_event=False)

    @staticmethod
    def _default_status_log_export_path() -> Path:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        return Path.home() / f"ai-summary-status-log-{stamp}.txt"

    def _export_hub_status_log(self) -> None:
        events = self._filtered_hub_status_events()
        if not events:
            self._set_hub_status("Status log is empty", "warning", auto_reset_ms=1800, record_event=False)
            return
        default_path = str(self._default_status_log_export_path())
        target, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export status log",
            default_path,
            "Text Files (*.txt);;JSON Files (*.json);;CSV Files (*.csv);;All Files (*)",
        )
        if not target:
            self._set_hub_status("Status log export canceled", "info", auto_reset_ms=1800, record_event=False)
            return
        target_path = Path(target)
        export_json = str(selected_filter or "").startswith("JSON") or target_path.suffix.lower() == ".json"
        export_csv = str(selected_filter or "").startswith("CSV") or target_path.suffix.lower() == ".csv"
        try:
            if export_json:
                payload = [
                    {
                        "timestamp_utc": str(event.get("at_utc", "")),
                        "timestamp_local": str(event.get("stamp", "")),
                        "tone": str(event.get("tone", "info")),
                        "text": str(event.get("text", "")),
                    }
                    for event in events
                ]
                target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            elif export_csv:
                with target_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["timestamp_utc", "timestamp_local", "tone", "text"])
                    for event in events:
                        writer.writerow(
                            [
                                str(event.get("at_utc", "")),
                                str(event.get("stamp", "")),
                                str(event.get("tone", "info")),
                                str(event.get("text", "")),
                            ]
                        )
            else:
                lines = [
                    f"{str(event.get('stamp', ''))} [{str(event.get('tone', 'info'))}] {str(event.get('text', ''))}"
                    for event in events
                ]
                target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            self._set_hub_status(f"Status log export failed: {exc}", "error", auto_reset_ms=2600, record_event=False)
            return
        self._set_hub_status(f"Status log exported: {target_path.name}", "success", auto_reset_ms=2200, record_event=False)

    @staticmethod
    def _build_history_absolute_period_presets() -> list[tuple[str, str]]:
        now_local = datetime.now().astimezone().replace(second=0, microsecond=0)
        today_start_local = now_local.replace(hour=0, minute=0)
        yesterday_start_local = (today_start_local - timedelta(days=1)).replace(hour=0, minute=0)
        last_hour_local = now_local - timedelta(hours=1)
        presets: list[tuple[str, str]] = []
        for prefix, anchor_local in (
            ("Since today 00:00", today_start_local),
            ("Since yesterday 00:00", yesterday_start_local),
            ("Since 1 hour ago", last_hour_local),
        ):
            anchor_utc = anchor_local.astimezone(timezone.utc)
            token = f"absolute:{anchor_utc.isoformat()}"
            label = f"{prefix} ({anchor_local.strftime('%Y-%m-%d %H:%M')})"
            presets.append((label, token))
        return presets

    @staticmethod
    def _format_history_source(source: object) -> str:
        normalized = str(source or "").strip()
        if not normalized:
            return "manual update"
        mapping = {
            "save_desktop_runtime_policy": "manual save",
            "runtime_policy_dialog_save": "runtime dialog",
            "settings_inline_policy_apply": "settings inline apply",
            "settings_history_restore": "settings history restore",
        }
        return mapping.get(normalized, normalized)

    @staticmethod
    def _parse_inline_topk_input(raw: str) -> tuple[int | None, bool]:
        normalized = (raw or "").strip().lower()
        if normalized in {"", "auto", "none"}:
            return None, True
        try:
            value = int(normalized)
        except ValueError:
            return None, False
        if value <= 0:
            return None, False
        return value, True

    def _validate_inline_mode_inputs(self) -> bool:
        topk_value, topk_valid = self._parse_inline_topk_input(self.inline_mode_topk.text())
        action_value = self.inline_mode_action.currentText().strip().lower()

        error_message = ""
        if not topk_valid:
            error_message = "Top-k는 auto/none 또는 양의 정수만 허용됩니다."
        elif action_value == "search" and topk_value is None:
            error_message = "Action이 search인 경우 Top-k를 숫자로 지정해야 합니다."

        has_error = bool(error_message)
        self.inline_mode_topk.setProperty("invalid", has_error)
        self.inline_mode_topk.style().unpolish(self.inline_mode_topk)
        self.inline_mode_topk.style().polish(self.inline_mode_topk)
        self.inline_mode_topk.update()
        self.inline_mode_error.setText(error_message)
        if has_error:
            self._set_hub_status(error_message, "error", legacy_label=self.inline_mode_error)
        self.btn_apply_inline_mode.setEnabled(not has_error)
        return not has_error

    def _refresh_inline_mode_profiles(self) -> None:
        self._inline_mode_profiles = load_mode_profiles()

    def _sync_inline_mode_controls(self, mode: str) -> None:
        self._refresh_inline_mode_profiles()
        effective_mode = mode if mode in MODE_ORDER else "Auto"
        profile = self._inline_mode_profiles.get(effective_mode, {})
        force_action = profile.get("force_action")
        action_value = "auto" if force_action in (None, "", "auto") else str(force_action)
        action_index = self.inline_mode_action.findText(action_value)
        self.inline_mode_action.setCurrentIndex(action_index if action_index >= 0 else 0)
        topk_value = profile.get("topk")
        self.inline_mode_topk.setText("auto" if topk_value in (None, "", "auto") else str(topk_value))
        self.inline_mode_tokens.setValue(int(profile.get("llm_max_new_tokens", 512) or 512))
        self.inline_mode_temp.setValue(float(profile.get("llm_temperature", 0.0) or 0.0))
        self._validate_inline_mode_inputs()
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )

    def _apply_inline_mode_profile(self) -> None:
        if not self._validate_inline_mode_inputs():
            return
        self._refresh_inline_mode_profiles()
        selected_mode = self.inline_mode_selector.currentText().strip()
        if selected_mode not in MODE_ORDER:
            selected_mode = "Auto"
        profile = dict(self._inline_mode_profiles.get(selected_mode, {}))
        action_value = self.inline_mode_action.currentText().strip().lower()
        profile["force_action"] = None if action_value == "auto" else action_value
        parsed_topk, _ = self._parse_inline_topk_input(self.inline_mode_topk.text())
        profile["topk"] = parsed_topk
        profile["llm_max_new_tokens"] = int(self.inline_mode_tokens.value())
        profile["llm_temperature"] = float(self.inline_mode_temp.value())
        self._inline_mode_profiles[selected_mode] = profile
        save_mode_profiles(self._inline_mode_profiles)
        self._set_hub_status(f"Mode preset saved ({selected_mode})", "success", legacy_label=self.inline_mode_status)
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=load_desktop_runtime_policy(),
        )
        if self._on_mode_profile_applied is not None:
            self._on_mode_profile_applied()

    def _hydrate_inline_controls(self, runtime_policy: dict[str, object]) -> None:
        defaults = dict(DEFAULT_DESKTOP_RUNTIME_POLICY)
        if isinstance(runtime_policy, dict):
            defaults.update(runtime_policy)
        self.inline_privacy_mask.setChecked(bool(defaults["privacy_mask_enabled"]))
        self.inline_max_refs.setValue(int(defaults["max_reference_links"]))
        self.inline_max_links.setValue(int(defaults["max_file_links"]))
        self.inline_max_response_chars.setValue(int(defaults["max_response_chars"]))
        self.inline_max_suggestion_chars.setValue(int(defaults["max_suggestion_chars"]))

    def _build_inline_policy(self) -> dict[str, object]:
        return {
            "privacy_mask_enabled": bool(self.inline_privacy_mask.isChecked()),
            "max_reference_links": int(self.inline_max_refs.value()),
            "max_file_links": int(self.inline_max_links.value()),
            "max_response_chars": int(self.inline_max_response_chars.value()),
            "max_suggestion_chars": int(self.inline_max_suggestion_chars.value()),
        }

    @staticmethod
    def _parse_history_timestamp(raw_value: object) -> datetime | None:
        text = str(raw_value or "").strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_custom_history_period_input(raw: str) -> datetime | None:
        normalized = str(raw or "").strip()
        if not normalized:
            return None
        try:
            parsed_local = datetime.strptime(normalized, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc
        return parsed_local.replace(tzinfo=local_tz).astimezone(timezone.utc)

    def _selected_history_source_filter(self) -> str:
        value = self.history_source_filter.currentData()
        normalized = str(value or "all").strip()
        return normalized or "all"

    def _selected_history_period_token(self) -> str:
        value = self.history_period_filter.currentData()
        normalized = str(value or "all_time").strip()
        return normalized or "all_time"

    def _selected_history_period_days(self) -> int:
        token = self._selected_history_period_token()
        mapping = {
            "last_24h": 1,
            "last_7d": 7,
            "last_30d": 30,
        }
        return mapping.get(token, 0)

    def _selected_history_period_threshold(self) -> datetime | None:
        token = self._selected_history_period_token()
        now = datetime.now(timezone.utc)
        if token == "all_time":
            return None
        if token == "last_24h":
            return now - timedelta(hours=24)
        if token == "last_7d":
            return now - timedelta(days=7)
        if token == "last_30d":
            return now - timedelta(days=30)
        if token.startswith("absolute:"):
            return self._parse_history_timestamp(token.split(":", 1)[1])
        return None

    def _is_history_entry_visible(self, row: dict[str, object]) -> bool:
        source_filter = self._selected_history_source_filter()
        source_raw = str(row.get("source", "save_desktop_runtime_policy")).strip() or "save_desktop_runtime_policy"
        if source_filter != "all" and source_raw != source_filter:
            return False
        threshold = self._selected_history_period_threshold()
        if threshold is None:
            return True
        parsed_time = self._parse_history_timestamp(row.get("updated_at"))
        if parsed_time is None:
            return False
        return parsed_time >= threshold

    def _capture_history_filter_state(self) -> None:
        type(self)._session_history_filter_state = {
            "source": self._selected_history_source_filter(),
            "period": self._selected_history_period_token(),
        }

    def _ensure_history_period_option(self, token: str) -> int:
        existing_index = self.history_period_filter.findData(token)
        if existing_index >= 0:
            return existing_index
        if not token.startswith("absolute:"):
            return -1
        parsed_time = self._parse_history_timestamp(token.split(":", 1)[1])
        if parsed_time is None:
            return -1
        local_time = parsed_time.astimezone()
        label = f"Since custom ({local_time.strftime('%Y-%m-%d %H:%M')})"
        self.history_period_filter.addItem(label, token)
        return self.history_period_filter.count() - 1

    @staticmethod
    def _set_invalid_state(widget: QWidget, invalid: bool) -> None:
        widget.setProperty("invalid", invalid)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _restore_history_filter_state(self) -> None:
        state = dict(type(self)._session_history_filter_state)
        source_value = str(state.get("source", "all") or "all")
        period_value = str(state.get("period", "all_time") or "all_time")
        self.history_source_filter.blockSignals(True)
        self.history_period_filter.blockSignals(True)
        source_index = self.history_source_filter.findData(source_value)
        period_index = self._ensure_history_period_option(period_value)
        self.history_source_filter.setCurrentIndex(source_index if source_index >= 0 else 0)
        self.history_period_filter.setCurrentIndex(period_index if period_index >= 0 else 0)
        self.history_source_filter.blockSignals(False)
        self.history_period_filter.blockSignals(False)
        self._capture_history_filter_state()

    def _on_history_filters_changed(self, _index: int) -> None:
        self._capture_history_filter_state()
        self._apply_history_filters()

    def _apply_custom_history_period(self) -> None:
        parsed_utc = self._parse_custom_history_period_input(self.history_custom_period_input.text())
        if parsed_utc is None:
            self._set_invalid_state(self.history_custom_period_input, True)
            self._set_hub_status(
                "Custom datetime format: YYYY-MM-DD HH:MM",
                "error",
                legacy_label=self.history_status,
            )
            return
        self._set_invalid_state(self.history_custom_period_input, False)
        token = f"absolute:{parsed_utc.isoformat()}"
        period_index = self._ensure_history_period_option(token)
        if period_index < 0:
            self._set_hub_status(
                "Custom datetime parse failed",
                "error",
                legacy_label=self.history_status,
            )
            return
        if self.history_period_filter.currentIndex() == period_index:
            self._on_history_filters_changed(period_index)
        else:
            self.history_period_filter.setCurrentIndex(period_index)
        has_result = bool(self._history_entries)
        self._set_hub_status(
            "Custom period applied" if has_result else "Custom period applied (no matching history)",
            "success" if has_result else "warning",
            legacy_label=self.history_status,
        )

    def _apply_history_filters(self) -> None:
        self._history_entries = [
            row
            for row in self._history_entries_all
            if isinstance(row, dict) and self._is_history_entry_visible(row)
        ]
        self.history_selector.clear()
        if not self._history_entries:
            self.history_box.setPlainText("No runtime policy history for current filters.")
            self.history_preview.setPlainText("No diff preview available.")
            self._set_hub_status(
                "No restorable history (filter result)",
                "warning",
                legacy_label=self.history_status,
            )
            self._sync_history_restore_state()
            return
        lines: list[str] = []
        for idx, row in enumerate(self._history_entries):
            updated_at = str(row.get("updated_at", "")).replace("T", " ").replace("Z", "")
            source_raw = str(row.get("source", "manual_update")).strip() or "manual_update"
            source_label = self._format_history_source(source_raw)
            policy = row.get("policy", {})
            if not isinstance(policy, dict):
                policy = {}
            self.history_selector.addItem(f"{idx + 1}. {updated_at[:19]} ({source_label})")
            lines.append(
                (
                    f"{updated_at[:19]} | {source_label} [{source_raw}] | "
                    f"privacy={'mask' if bool(policy.get('privacy_mask_enabled', True)) else 'raw'} | "
                    f"refs<={int(policy.get('max_reference_links', 5) or 5)} | "
                    f"file-links<={int(policy.get('max_file_links', 8) or 8)}"
                )
            )
        self.history_box.setPlainText("\n".join(lines))
        self._set_hub_status(
            "Select a version and confirm preview to restore",
            "info",
            legacy_label=self.history_status,
        )
        self._capture_history_filter_state()
        if self.history_selector.count() > 0:
            self.history_selector.setCurrentIndex(0)
        self._update_history_preview()
        self._sync_history_restore_state()

    def _reload_history(self) -> None:
        history = load_desktop_runtime_policy_history(limit=60)
        self._history_entries_all = [row for row in history if isinstance(row, dict)]
        self._history_entries = []
        self.history_confirm.setChecked(False)
        if not self._history_entries_all:
            self.history_selector.clear()
            self.history_box.setPlainText("No runtime policy history yet.")
            self.history_preview.setPlainText("No diff preview available.")
            self.history_status.setText("No restorable history")
            self._sync_history_restore_state()
            return
        self._apply_history_filters()

    @staticmethod
    def _build_policy_diff_lines(current_policy: dict[str, object], target_policy: dict[str, object]) -> list[str]:
        keys = (
            "privacy_mask_enabled",
            "max_reference_links",
            "max_file_links",
            "max_response_chars",
            "max_suggestion_chars",
        )
        lines: list[str] = []
        for key in keys:
            current_value = current_policy.get(key)
            target_value = target_policy.get(key)
            if current_value == target_value:
                continue
            lines.append(f"{key}: {current_value} -> {target_value}")
        return lines

    def _update_history_preview(self) -> None:
        target_policy = self._selected_history_policy()
        if target_policy is None:
            self.history_preview.setPlainText("No diff preview available.")
            self._sync_history_restore_state()
            return
        current_policy = load_desktop_runtime_policy()
        diff_lines = self._build_policy_diff_lines(current_policy, target_policy)
        if not diff_lines:
            self.history_preview.setPlainText("현재 정책과 동일합니다.")
            self._sync_history_restore_state()
            return
        self.history_preview.setPlainText("Preview diff:\n" + "\n".join(diff_lines))
        self._sync_history_restore_state()

    def _selected_history_policy(self) -> dict[str, object] | None:
        index = self.history_selector.currentIndex()
        if index < 0 or index >= len(self._history_entries):
            return None
        row = self._history_entries[index]
        policy = row.get("policy", {})
        if not isinstance(policy, dict):
            return None
        return policy

    def _sync_history_restore_state(self) -> None:
        policy = self._selected_history_policy()
        if policy is None:
            self.btn_restore_history.setEnabled(False)
            return
        current_policy = load_desktop_runtime_policy()
        has_diff = bool(self._build_policy_diff_lines(current_policy, policy))
        self.btn_restore_history.setEnabled(bool(self.history_confirm.isChecked()) and has_diff)

    def _apply_inline_runtime_policy(self) -> None:
        save_desktop_runtime_policy(self._build_inline_policy(), source="settings_inline_policy_apply")
        current_policy = load_desktop_runtime_policy()
        self._runtime_policy_snapshot = dict(current_policy)
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )
        self._reload_history()
        self._set_hub_status("Inline policy saved", "success", legacy_label=self.inline_status)
        if self._on_runtime_policy_applied is not None:
            self._on_runtime_policy_applied()

    def _restore_selected_history_policy(self) -> None:
        if not self.history_confirm.isChecked():
            self._set_hub_status(
                "복원 전 Preview 확인 체크가 필요합니다.",
                "warning",
                legacy_label=self.history_status,
            )
            self._sync_history_restore_state()
            return
        policy = self._selected_history_policy()
        if policy is None:
            self._set_hub_status("No history selected", "warning", legacy_label=self.history_status)
            self._sync_history_restore_state()
            return
        current_policy = load_desktop_runtime_policy()
        if not self._build_policy_diff_lines(current_policy, policy):
            self._set_hub_status("현재 정책과 동일합니다.", "warning", legacy_label=self.history_status)
            self._sync_history_restore_state()
            return
        save_desktop_runtime_policy(policy, source="settings_history_restore")
        current_policy = load_desktop_runtime_policy()
        self._runtime_policy_snapshot = dict(current_policy)
        self._hydrate_inline_controls(current_policy)
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )
        self._reload_history()
        self.history_confirm.setChecked(False)
        self._set_hub_status("Selected policy restored", "success", legacy_label=self.history_status)
        if self._on_runtime_policy_applied is not None:
            self._on_runtime_policy_applied()

    def _open_folders(self):
        self._open_folders_callback()

    def _refresh_current_mode_from_parent(self) -> None:
        parent = self.parent()
        if parent is None:
            return
        candidate = getattr(parent, "_response_mode", None)
        if isinstance(candidate, str) and candidate.strip():
            self._current_mode = candidate.strip()

    def _open_mode(self):
        self._open_mode_callback()
        self._refresh_current_mode_from_parent()
        self._sync_inline_mode_controls(self.inline_mode_selector.currentText())
        self._runtime_policy_snapshot = load_desktop_runtime_policy()
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )

    def _open_runtime(self):
        self._open_runtime_callback()
        self._refresh_current_mode_from_parent()
        current_policy = load_desktop_runtime_policy()
        self._runtime_policy_snapshot = dict(current_policy)
        self._hydrate_inline_controls(current_policy)
        self._update_summary(
            smart_folder_count=self._smart_folder_count,
            current_mode=self._current_mode,
            runtime_policy=self._runtime_policy_snapshot,
        )
        self._reload_history()


class LauncherWindow(QWidget):
    query_requested = Signal(str, str)
    runtime_policy_refresh_requested = Signal()
    _THREAD_STORE_MAX = 120
    _THREAD_PAGE_SIZE = 5
    _THREAD_TIMELINE_MAX = 400
    _FILE_RESOLUTION_CACHE_MAX = 512
    _SIMILAR_LOOKUP_CACHE_MAX = 256
    _SIMILAR_SCAN_LIMIT_DEFAULT = 600
    _OPEN_EVENT_LOG_MAX_BYTES = 2 * 1024 * 1024
    _OPEN_COMMAND_TIMEOUT_DEFAULT = 8.0

    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend
        self.policy_registry = SmartFolderRegistry()

        self._query_in_flight = False
        self.streaming_item: Optional[QListWidgetItem] = None
        self._streaming_buffer = ""
        self._last_system_message = ""
        self._model_chip_base = "GPT-5.3-Codex  •  Extra High"
        self._response_mode = "Auto"
        self._runtime_policy: dict[str, object] = {}
        self._privacy_mask_enabled = True
        self._max_file_links = 8
        self._max_reference_links = 5
        self._max_response_chars = 24000
        self._max_suggestion_chars = 120
        self._reload_runtime_policy()
        self._mode_actions: dict[str, QAction] = {}
        self._response_mode_profiles: dict[str, dict[str, object]] = {}
        self._response_mode_descriptions: dict[str, str] = {}
        self._response_mode_runtime: dict[str, dict[str, object]] = {}
        self._reload_mode_profiles()
        self._thread_entries: list[dict[str, str]] = []
        self._thread_timelines: dict[str, list[dict[str, object]]] = {}
        self._file_resolution_cache: dict[str, str] = {}
        self._similar_lookup_cache: dict[str, dict[str, object]] = {}
        self._thread_visible_limit = 0
        self._active_thread_id = ""
        self._load_thread_timelines()
        self._load_thread_entries()
        self._load_file_resolution_cache()
        self._load_similar_lookup_cache()

        self.setAcceptDrops(True)
        self.setWindowTitle("AI-summary")
        self.resize(1500, 980)
        self.setMinimumSize(1080, 700)

        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self.update_thinking_text)
        self.thinking_dots = 0

        self.setup_ui()
        self.setup_styles()
        self._populate_sidebar_threads()

        if self.backend:
            self.query_requested.connect(self.backend.handle_query)
            if hasattr(self.backend, "refresh_runtime_policy"):
                self.runtime_policy_refresh_requested.connect(self.backend.refresh_runtime_policy)
            self.backend.response_ready.connect(self.handle_response)
            self.backend.stream_update.connect(self.handle_stream_update)
            self.backend.status_msg.connect(self.update_status_msg)
            self.backend.error_occurred.connect(self.handle_error)
            if hasattr(self.backend, "ready"):
                self.backend.ready.connect(self._on_backend_ready)
            self._set_status("Initializing AI Core...", "busy")
            self._set_query_controls_enabled(False)
        else:
            self._set_status("Local mode", "neutral")

    def setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(286)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 14, 14, 12)
        sidebar_layout.setSpacing(10)

        app_title = QLabel("AI-summary")
        app_title.setObjectName("AppTitle")
        sidebar_layout.addWidget(app_title)

        self.btn_new_thread = self._create_sidebar_button("↻ New thread", self._start_new_thread)
        self.btn_automations = self._create_sidebar_button("⌚ Automations", self._open_automations)
        self.btn_skills = self._create_sidebar_button("⌘ Skills", self._open_skills)

        sidebar_layout.addWidget(self.btn_new_thread)
        sidebar_layout.addWidget(self.btn_automations)
        sidebar_layout.addWidget(self.btn_skills)

        thread_header = QLabel("Threads")
        thread_header.setObjectName("SidebarSection")
        sidebar_layout.addWidget(thread_header)

        self.thread_search = QLineEdit()
        self.thread_search.setObjectName("ThreadSearch")
        self.thread_search.setPlaceholderText("검색 (⌘/Ctrl+K)")
        self.thread_search.textChanged.connect(self._apply_thread_filter)
        self.thread_search.returnPressed.connect(self._activate_thread_from_search)
        sidebar_layout.addWidget(self.thread_search)

        self.thread_list = QListWidget()
        self.thread_list.setObjectName("SidebarThreads")
        self.thread_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thread_list.currentRowChanged.connect(self._on_thread_selection_changed)
        sidebar_layout.addWidget(self.thread_list, stretch=1)

        self.thread_empty_state = QLabel("검색 결과가 없습니다.")
        self.thread_empty_state.setObjectName("ThreadEmptyState")
        self.thread_empty_state.setAlignment(Qt.AlignCenter)
        self.thread_empty_state.setWordWrap(True)
        self.thread_empty_state.hide()
        sidebar_layout.addWidget(self.thread_empty_state)

        self.btn_show_more_threads = self._create_sidebar_button("Show more", self._show_more_threads)
        self.btn_show_more_threads.setObjectName("SidebarSecondaryButton")
        sidebar_layout.addWidget(self.btn_show_more_threads)

        self.btn_sidebar_settings = self._create_sidebar_button("⚙ Settings", self.open_settings)
        sidebar_layout.addWidget(self.btn_sidebar_settings)

        self.main_panel = QFrame()
        self.main_panel.setObjectName("MainPanel")
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(22, 18, 22, 18)
        main_layout.setSpacing(14)

        self.header_bar = QFrame()
        self.header_bar.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(10)

        title_col = QVBoxLayout()
        self.header_title = QLabel("Reply to 0f greeting")
        self.header_title.setObjectName("HeaderTitle")
        self.header_subtitle = QLabel("AI-summary")
        self.header_subtitle.setObjectName("HeaderSubtitle")
        title_col.addWidget(self.header_title)
        title_col.addWidget(self.header_subtitle)

        right_col = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusBadge")
        self.header_open_btn = self._create_header_button("⌂  Open", self.open_tasks)
        self.header_tools_btn = self._create_header_button("⋯", self.open_settings)

        right_col.addWidget(self.status_label)
        right_col.addWidget(self.header_open_btn)
        right_col.addWidget(self.header_tools_btn)

        header_layout.addLayout(title_col, stretch=1)
        header_layout.addLayout(right_col, stretch=0)

        self.result_list = QListWidget()
        self.result_list.setObjectName("ChatList")
        self.result_list.setWordWrap(True)
        self.result_list.itemClicked.connect(self.on_result_item_clicked)

        self.chat_empty_state = QLabel("새 대화를 시작해보세요.")
        self.chat_empty_state.setObjectName("ChatEmptyState")
        self.chat_empty_state.setAlignment(Qt.AlignCenter)
        self.chat_empty_state.setWordWrap(True)

        composer_shell = QWidget()
        composer_shell_layout = QVBoxLayout(composer_shell)
        composer_shell_layout.setContentsMargins(0, 6, 0, 0)

        self.composer_panel = QFrame()
        self.composer_panel.setObjectName("ComposerPanel")
        self.composer_panel.setMinimumWidth(720)
        self.composer_panel.setMaximumWidth(960)
        self.composer_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        composer_layout = QVBoxLayout(self.composer_panel)
        composer_layout.setContentsMargins(16, 12, 16, 12)
        composer_layout.setSpacing(10)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        self.input_field = EnhancedInput()
        self.input_field.setObjectName("ComposerInput")
        self.input_field.setPlaceholderText("무엇이든 부탁하세요")
        self.input_field.submit.connect(self.on_submit)

        self.btn_send = QPushButton("▲")
        self.btn_send.setObjectName("SendButton")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.clicked.connect(self.on_submit)
        self.btn_send.setFixedSize(40, 40)

        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.btn_send, stretch=0)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_attach = self._create_action_button("＋", "첨부", self._insert_attachment_hint)
        self.btn_web = self._create_action_button("◎", "웹검색", self._insert_web_hint)
        self.btn_photo = self._create_action_button("▣", "사진", self.open_photo_dialog)
        self.btn_mic = self._create_action_button("◉", "회의", self.open_meeting_dialog)
        self.btn_tasks = self._create_action_button("☰", "태스크", self.open_tasks)
        self.btn_settings = self._create_action_button("⚙", "설정", self.open_settings)
        self.btn_shortcuts = self._create_action_button("⌨", "단축키 도움말", self._show_shortcuts_help)

        action_row.addWidget(self.btn_attach)
        action_row.addWidget(self.btn_web)
        action_row.addWidget(self.btn_photo)
        action_row.addWidget(self.btn_mic)
        action_row.addWidget(self.btn_tasks)
        action_row.addWidget(self.btn_settings)
        action_row.addWidget(self.btn_shortcuts)
        action_row.addStretch()

        self.mode_button = QPushButton(self._response_mode)
        self.mode_button.setObjectName("ModeButton")
        self.mode_button.setCursor(Qt.PointingHandCursor)
        self.mode_button.setMenu(self._build_mode_menu())
        self.mode_button.setToolTip("응답 모드")

        self.model_chip = QLabel(f"{self._model_chip_base}  •  {self._response_mode}")
        self.model_chip.setObjectName("ModelChip")
        action_row.addWidget(self.mode_button)
        action_row.addWidget(self.model_chip)

        self.shortcut_hint = QLabel("")
        self.shortcut_hint.setObjectName("ShortcutHint")
        self.shortcut_hint.setWordWrap(True)
        self.mode_hint = QLabel("")
        self.mode_hint.setObjectName("ModeHint")
        self.mode_hint.setWordWrap(True)

        composer_layout.addLayout(input_row)
        composer_layout.addLayout(action_row)
        composer_layout.addWidget(self.shortcut_hint)
        composer_layout.addWidget(self.mode_hint)
        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(0, 0, 0, 0)
        composer_row.addStretch()
        composer_row.addWidget(self.composer_panel, stretch=1)
        composer_row.addStretch()
        composer_shell_layout.addLayout(composer_row)

        composer_shadow = QGraphicsDropShadowEffect(self)
        composer_shadow.setBlurRadius(54)
        composer_shadow.setColor(QColor(20, 24, 32, 148))
        composer_shadow.setOffset(0, 12)
        self.composer_panel.setGraphicsEffect(composer_shadow)

        main_layout.addWidget(self.header_bar, stretch=0)
        main_layout.addWidget(self.chat_empty_state, stretch=0)
        main_layout.addWidget(self.result_list, stretch=1)
        main_layout.addWidget(composer_shell, stretch=0)

        root.addWidget(self.sidebar, stretch=0)
        root.addWidget(self.main_panel, stretch=1)

        self._query_controls = [
            self.input_field,
            self.btn_send,
            self.btn_attach,
            self.btn_web,
            self.btn_photo,
            self.btn_mic,
            self.btn_tasks,
            self.btn_settings,
            self.btn_shortcuts,
            self.mode_button,
        ]
        self._sync_chat_empty_state()
        self._refresh_shortcut_hint()
        self._set_response_mode(self._response_mode, announce=False)

    def setup_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Apple SD Gothic Neo", "Pretendard", "Noto Sans KR", "Helvetica Neue", sans-serif;
                font-size: 14px;
                color: #111827;
                background: #f5f6f8;
            }

            QFrame#Sidebar {
                background: #eceff3;
                border-right: 1px solid #d8dde7;
            }

            QLabel#AppTitle {
                font-size: 18px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 8px;
            }

            QLabel#SidebarSection {
                font-size: 12px;
                font-weight: 700;
                color: #6b7280;
                margin-top: 8px;
                margin-bottom: 2px;
                text-transform: uppercase;
            }

            QLineEdit#ThreadSearch {
                border: 1px solid #d0d7e4;
                border-radius: 9px;
                background: #f8fafc;
                color: #1f2937;
                padding: 7px 9px;
                font-size: 13px;
            }
            QLineEdit#ThreadSearch:focus {
                border: 1px solid #91a9d4;
                background: #ffffff;
            }

            QLabel#ThreadEmptyState {
                color: #6b7280;
                font-size: 12px;
                padding: 10px 8px;
                border-radius: 8px;
                background: #f4f7fb;
                border: 1px dashed #d3dbe8;
            }

            QPushButton#SidebarButton {
                border: none;
                border-radius: 8px;
                background: transparent;
                text-align: left;
                padding: 8px 10px;
                color: #1f2937;
                font-weight: 500;
            }
            QPushButton#SidebarButton:hover {
                background: #dce2eb;
            }
            QPushButton#SidebarSecondaryButton {
                border: none;
                background: transparent;
                text-align: left;
                padding: 6px 10px;
                color: #6b7280;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#SidebarSecondaryButton:hover {
                color: #111827;
                background: #e4e9f2;
                border-radius: 8px;
            }

            QListWidget#SidebarThreads {
                border: none;
                border-radius: 10px;
                background: #f6f7f9;
                padding: 6px;
            }
            QListWidget#SidebarThreads::item {
                border: none;
                margin: 2px 0;
                padding: 0;
            }
            QFrame#ThreadRow {
                border-radius: 10px;
                background: transparent;
                border: 1px solid transparent;
            }
            QFrame#ThreadRow[active=\"true\"] {
                background: #dce5f4;
                border: 1px solid #cbd8ec;
            }
            QLabel#ThreadTitle {
                color: #111827;
                font-size: 14px;
                font-weight: 550;
            }
            QLabel#ThreadMeta {
                color: #6b7280;
                font-size: 12px;
                font-weight: 500;
            }

            QFrame#MainPanel {
                background: #f8fafc;
            }

            QFrame#HeaderBar {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid #d8dde7;
                border-radius: 12px;
            }

            QLabel#HeaderTitle {
                font-size: 18px;
                font-weight: 700;
                color: #111827;
            }

            QLabel#HeaderSubtitle {
                font-size: 13px;
                color: #6b7280;
            }

            QPushButton#HeaderButton {
                border: 1px solid #cfd6e1;
                border-radius: 8px;
                background: #ffffff;
                color: #1f2937;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton#HeaderButton:hover {
                background: #f3f4f6;
            }

            QLabel#StatusBadge {
                border-radius: 999px;
                padding: 4px 10px;
                background: #e5e7eb;
                color: #374151;
                font-weight: 700;
            }

            QListWidget#ChatList {
                border: none;
                background: transparent;
                outline: none;
                padding-right: 6px;
            }
            QListWidget#ChatList::item {
                margin: 4px 0;
                padding: 10px 12px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #e5e9f1;
            }
            QListWidget#ChatList::item:hover {
                background: #f2f6fd;
                border: 1px solid #ccd9ef;
            }
            QListWidget#ChatList::item:selected {
                background: #e7f0fe;
                border: 1px solid #b9ccef;
                color: #0f172a;
            }

            QLabel#ChatEmptyState {
                color: #6b7280;
                font-size: 13px;
                border: 1px dashed #d6dde9;
                border-radius: 10px;
                background: #f7f9fc;
                padding: 12px 14px;
            }

            QFrame#ComposerPanel {
                background: #383b43;
                border: 1px solid #545b68;
                border-radius: 26px;
            }

            QTextEdit#ComposerInput {
                border: none;
                background: transparent;
                color: #f3f4f6;
                font-size: 18px;
                padding-top: 2px;
            }

            QPushButton#ActionButton {
                border: none;
                border-radius: 8px;
                background: transparent;
                color: #d6d8de;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton#ActionButton:hover {
                background: rgba(255, 255, 255, 0.12);
                color: #ffffff;
            }

            QPushButton#ModeButton {
                border: 1px solid #5a6170;
                border-radius: 999px;
                background: rgba(0, 0, 0, 0.18);
                color: #d9dde6;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 650;
                min-width: 86px;
            }
            QPushButton#ModeButton:hover {
                border-color: #7f8795;
                color: #ffffff;
            }
            QPushButton#ModeButton::menu-indicator {
                image: none;
                width: 0px;
            }

            QLabel#ModelChip {
                color: #c5c9d3;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 999px;
                border: 1px solid #5a6170;
                background: rgba(0, 0, 0, 0.18);
            }

            QLabel#ShortcutHint {
                color: #aeb6c7;
                font-size: 11px;
                padding: 1px 2px 0 2px;
            }

            QLabel#ModeHint {
                color: #b6bfd2;
                font-size: 11px;
                padding: 0 2px 2px 2px;
            }

            QPushButton#SendButton {
                border: none;
                border-radius: 20px;
                color: #e5e7eb;
                background: #5b6474;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#SendButton:hover {
                background: #6d7688;
            }
            QPushButton#SendButton:disabled {
                background: #4b4f58;
                color: #8b8f99;
            }

            QScrollBar:vertical {
                width: 10px;
                border: none;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                border-radius: 5px;
                background: #c7cfdd;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    def _create_sidebar_button(self, text: str, callback) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("SidebarButton")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _create_header_button(self, text: str, callback) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("HeaderButton")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumWidth(70)
        btn.clicked.connect(callback)
        return btn

    def _create_action_button(self, icon: str, tooltip: str, callback) -> QPushButton:
        btn = QPushButton(icon)
        btn.setObjectName("ActionButton")
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _reload_runtime_policy(self):
        policy = load_desktop_runtime_policy()
        self._runtime_policy = policy
        self._privacy_mask_enabled = bool(policy.get("privacy_mask_enabled", True))
        self._max_file_links = max(1, self._as_int(policy.get("max_file_links"), 8))
        self._max_reference_links = max(1, self._as_int(policy.get("max_reference_links"), 5))
        self._max_response_chars = max(1200, self._as_int(policy.get("max_response_chars"), 24000))
        self._max_suggestion_chars = max(24, self._as_int(policy.get("max_suggestion_chars"), 120))

    def _reload_mode_profiles(self):
        self._response_mode_profiles = load_mode_profiles()
        self._response_mode_descriptions = {}
        self._response_mode_runtime = {}
        for mode in MODE_ORDER:
            profile = self._response_mode_profiles.get(mode, {})
            description = str(profile.get("description", "")).strip() or mode
            force_action = profile.get("force_action")
            action = "auto" if force_action in (None, "", "auto") else str(force_action)
            topk = profile.get("topk")
            if topk in (None, "", "auto"):
                runtime_topk: object = "auto"
            else:
                try:
                    runtime_topk = int(topk)
                except (TypeError, ValueError):
                    runtime_topk = "auto"
            runtime = {
                "action": action,
                "topk": runtime_topk,
                "tokens": int(profile.get("llm_max_new_tokens", 512) or 512),
                "temp": float(profile.get("llm_temperature", 0.0) or 0.0),
            }
            self._response_mode_descriptions[mode] = description
            self._response_mode_runtime[mode] = runtime

    def _rebuild_mode_menu(self):
        self._mode_actions.clear()
        self.mode_button.setMenu(self._build_mode_menu())

    def _open_mode_profile_editor(self):
        dialog = ModeProfileDialog(self._response_mode_profiles, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._on_mode_profile_updated()

    def _on_mode_profile_updated(self):
        self._reload_mode_profiles()
        self._rebuild_mode_menu()
        self._set_response_mode(self._response_mode, announce=False)
        self._set_status("Mode preset updated", "ok")

    def _open_runtime_policy_editor(self):
        dialog = RuntimePolicyDialog(self._runtime_policy, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._on_runtime_policy_updated()

    def _on_runtime_policy_updated(self):
        self._reload_runtime_policy()
        self.runtime_policy_refresh_requested.emit()
        self._refresh_mode_hint()
        self._set_status("Runtime policy updated", "ok")

    def _build_mode_menu(self) -> QMenu:
        menu = QMenu(self)
        for label in MODE_ORDER:
            desc = self._response_mode_descriptions.get(label, label)
            action = QAction(f"{label}  -  {desc}", self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, mode=label: self._set_response_mode(mode))
            menu.addAction(action)
            self._mode_actions[label] = action
        menu.addSeparator()
        customize = QAction("Customize Presets...", self)
        customize.triggered.connect(self._open_mode_profile_editor)
        menu.addAction(customize)
        menu.aboutToShow.connect(self._sync_mode_menu_checks)
        return menu

    def _sync_mode_menu_checks(self):
        for label, action in self._mode_actions.items():
            action.setChecked(label == self._response_mode)

    def _set_response_mode(self, mode: str, *, announce: bool = True):
        if mode not in self._response_mode_descriptions:
            mode = "Auto"
        self._response_mode = mode
        self.mode_button.setText(mode)
        self.model_chip.setText(f"{self._model_chip_base}  •  {mode}")
        self.mode_button.setToolTip(f"{mode}: {self._response_mode_descriptions[mode]}")
        self._sync_mode_menu_checks()
        self._refresh_mode_hint()
        if announce:
            self._set_status(f"Mode: {mode}", "neutral")

    def _refresh_mode_hint(self):
        runtime = self._response_mode_runtime.get(self._response_mode, self._response_mode_runtime.get("Auto", {}))
        description = self._response_mode_descriptions.get(self._response_mode, self._response_mode)
        self.mode_hint.setText(
            (
                f"{self._response_mode}: {description}  •  "
                f"action={runtime.get('action', 'auto')}  •  top-k={runtime.get('topk', 'auto')}  •  "
                f"tokens={runtime.get('tokens', 512)}  •  temp={float(runtime.get('temp', 0.0)):.2f}  •  "
                f"privacy={'mask' if self._privacy_mask_enabled else 'raw'}  •  "
                f"refs<={self._max_reference_links}  •  "
                f"file-links<={self._max_file_links}"
            )
        )

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _thread_history_path() -> Path:
        custom = os.getenv("DESKTOP_THREAD_HISTORY_PATH", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".ai-summary" / "desktop_thread_history.json"

    @staticmethod
    def _thread_timeline_path() -> Path:
        custom = os.getenv("DESKTOP_THREAD_TIMELINE_PATH", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".ai-summary" / "desktop_thread_timelines.json"

    @staticmethod
    def _file_resolution_cache_path() -> Path:
        custom = os.getenv("DESKTOP_FILE_RESOLUTION_CACHE_PATH", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".ai-summary" / "desktop_file_resolution_cache.json"

    @staticmethod
    def _similar_lookup_cache_path() -> Path:
        custom = os.getenv("DESKTOP_SIMILAR_LOOKUP_CACHE_PATH", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".ai-summary" / "desktop_similar_lookup_cache.json"

    @staticmethod
    def _open_event_log_path() -> Path:
        custom = os.getenv("DESKTOP_OPEN_EVENT_LOG_PATH", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".ai-summary" / "desktop_file_open_events.jsonl"

    def _record_open_event(
        self,
        *,
        event: str,
        path: Path | None,
        success: bool,
        detail: str = "",
        resolution: str = "",
        source_path: Path | None = None,
        category: str | None = None,
    ) -> None:
        target = self._open_event_log_path()
        normalized_detail = str(detail or "").strip()
        if category is None:
            if success:
                category = "ok"
            elif normalized_detail:
                category = self._classify_open_error(normalized_detail)
            else:
                category = "generic"
        payload = {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "event": str(event).strip() or "unknown",
            "success": bool(success),
            "category": str(category).strip() or "generic",
            "resolution": str(resolution).strip(),
            "path": str(path) if path is not None else "",
            "source_path": str(source_path) if source_path is not None else "",
            "detail": normalized_detail[:600],
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > self._OPEN_EVENT_LOG_MAX_BYTES:
                rotated = target.with_suffix(target.suffix + ".1")
                try:
                    rotated.unlink(missing_ok=True)
                except OSError:
                    pass
                target.replace(rotated)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _load_file_resolution_cache(self) -> None:
        target = self._file_resolution_cache_path()
        self._file_resolution_cache = {}
        if not target.exists():
            return
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        normalized: dict[str, str] = {}
        for source_raw, resolved_raw in raw.items():
            source_text = str(source_raw).strip()
            resolved_text = str(resolved_raw).strip()
            if not source_text or not resolved_text:
                continue
            source_path = Path(source_text).expanduser()
            resolved_path = Path(resolved_text).expanduser()
            if not resolved_path.exists():
                continue
            source_key = self._path_cache_key(source_path)
            try:
                normalized[source_key] = str(resolved_path.resolve(strict=False))
            except OSError:
                normalized[source_key] = str(resolved_path)
        if len(normalized) > self._FILE_RESOLUTION_CACHE_MAX:
            # Keep latest loaded entries by insertion order.
            trimmed = list(normalized.items())[-self._FILE_RESOLUTION_CACHE_MAX :]
            normalized = {key: value for key, value in trimmed}
        self._file_resolution_cache = normalized

    def _save_file_resolution_cache(self) -> None:
        target = self._file_resolution_cache_path()
        if len(self._file_resolution_cache) > self._FILE_RESOLUTION_CACHE_MAX:
            trimmed = list(self._file_resolution_cache.items())[-self._FILE_RESOLUTION_CACHE_MAX :]
            self._file_resolution_cache = {key: value for key, value in trimmed}
        payload = dict(self._file_resolution_cache)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return

    def _load_similar_lookup_cache(self) -> None:
        target = self._similar_lookup_cache_path()
        self._similar_lookup_cache = {}
        if not target.exists():
            return
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        normalized: dict[str, dict[str, object]] = {}
        for key_raw, value_raw in raw.items():
            key = str(key_raw).strip()
            value = value_raw if isinstance(value_raw, dict) else {}
            if not key:
                continue
            folder_mtime = float(value.get("folder_mtime") or 0.0)
            paths_raw = value.get("paths", [])
            paths = paths_raw if isinstance(paths_raw, list) else []
            cleaned_paths: list[str] = []
            for item in paths[:128]:
                text = str(item).strip()
                if text:
                    cleaned_paths.append(text)
            if not cleaned_paths:
                continue
            normalized[key] = {"folder_mtime": folder_mtime, "paths": cleaned_paths}
        if len(normalized) > self._SIMILAR_LOOKUP_CACHE_MAX:
            trimmed = list(normalized.items())[-self._SIMILAR_LOOKUP_CACHE_MAX :]
            normalized = {key: value for key, value in trimmed}
        self._similar_lookup_cache = normalized

    def _save_similar_lookup_cache(self) -> None:
        target = self._similar_lookup_cache_path()
        if len(self._similar_lookup_cache) > self._SIMILAR_LOOKUP_CACHE_MAX:
            overflow = len(self._similar_lookup_cache) - self._SIMILAR_LOOKUP_CACHE_MAX
            for stale_key in list(self._similar_lookup_cache.keys())[:overflow]:
                self._similar_lookup_cache.pop(stale_key, None)
        payload = dict(self._similar_lookup_cache)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _similar_lookup_cache_key(folder: Path, target_stem: str) -> str:
        try:
            folder_text = str(folder.resolve(strict=False))
        except OSError:
            folder_text = str(folder)
        return f"{folder_text}::{target_stem.casefold()}"

    @classmethod
    def _similar_lookup_scan_limit(cls) -> int:
        raw = os.getenv("DESKTOP_SIMILAR_SCAN_MAX", "").strip()
        if not raw:
            return cls._SIMILAR_SCAN_LIMIT_DEFAULT
        try:
            parsed = int(raw)
        except ValueError:
            return cls._SIMILAR_SCAN_LIMIT_DEFAULT
        return max(1, min(5000, parsed))

    @classmethod
    def _open_command_timeout_seconds(cls) -> float:
        raw = os.getenv("DESKTOP_OPEN_CMD_TIMEOUT_SEC", "").strip()
        if not raw:
            return cls._OPEN_COMMAND_TIMEOUT_DEFAULT
        try:
            parsed = float(raw)
        except ValueError:
            return cls._OPEN_COMMAND_TIMEOUT_DEFAULT
        return max(1.0, min(30.0, parsed))

    def _run_open_command(self, args: list[str]) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._open_command_timeout_seconds(),
            )
        except subprocess.TimeoutExpired:
            return 124, "open command timed out"
        except OSError as exc:
            return 127, str(exc)
        detail = (proc.stderr or proc.stdout or "").strip()
        return int(proc.returncode), detail

    def _load_similar_candidates_for_folder(
        self,
        *,
        folder: Path,
        target_stem: str,
        candidate_patterns: list[str],
    ) -> list[Path]:
        key = self._similar_lookup_cache_key(folder, target_stem)
        folder_mtime = 0.0
        try:
            folder_mtime = float(folder.stat().st_mtime)
        except OSError:
            folder_mtime = 0.0

        cached = self._similar_lookup_cache.get(key)
        if cached is not None:
            cached_mtime = float(cached.get("folder_mtime") or 0.0)
            if abs(cached_mtime - folder_mtime) <= 1.0:
                cached_paths_raw = cached.get("paths", [])
                cached_paths = cached_paths_raw if isinstance(cached_paths_raw, list) else []
                valid_paths: list[Path] = []
                for raw in cached_paths:
                    text = str(raw).strip()
                    if not text:
                        continue
                    candidate = Path(text)
                    if not candidate.exists() or not candidate.is_file():
                        continue
                    if candidate.stem.casefold() != target_stem:
                        continue
                    valid_paths.append(candidate)
                if valid_paths:
                    return valid_paths
            self._similar_lookup_cache.pop(key, None)
            self._save_similar_lookup_cache()

        seen: set[str] = set()
        discovered: list[Path] = []
        scan_limit = self._similar_lookup_scan_limit()
        truncated = False
        try:
            for pattern in candidate_patterns:
                for matched in folder.rglob(pattern):
                    if not matched.is_file():
                        continue
                    if matched.stem.casefold() != target_stem:
                        continue
                    resolved = str(matched.resolve(strict=False))
                    dedupe = resolved.casefold() if os.name == "nt" else resolved
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    discovered.append(Path(resolved))
                    if len(discovered) >= scan_limit:
                        truncated = True
                        break
                if truncated:
                    break
        except (OSError, RuntimeError):
            return []

        self._similar_lookup_cache[key] = {
            "folder_mtime": folder_mtime,
            "paths": [str(path) for path in discovered[:128]],
        }
        if truncated:
            self._record_open_event(
                event="similar_scan_capped",
                path=folder,
                success=True,
                detail=f"scan_limit={scan_limit}, discovered={len(discovered)}",
                category="scan_cap",
            )
        if len(self._similar_lookup_cache) > self._SIMILAR_LOOKUP_CACHE_MAX:
            overflow = len(self._similar_lookup_cache) - self._SIMILAR_LOOKUP_CACHE_MAX
            for stale_key in list(self._similar_lookup_cache.keys())[:overflow]:
                self._similar_lookup_cache.pop(stale_key, None)
        self._save_similar_lookup_cache()
        return discovered

    @staticmethod
    def _normalize_timeline_row(row: object) -> dict[str, object] | None:
        if not isinstance(row, dict):
            return None
        sender = str(row.get("sender", "")).strip()
        text = str(row.get("text", "")).strip()
        if not sender or not text:
            return None
        payload: dict[str, object] = {
            "sender": sender,
            "text": text,
        }
        file_path = str(row.get("file_path", "")).strip()
        if file_path:
            payload["file_path"] = file_path
            payload["file_missing"] = bool(row.get("file_missing", False))
        action_code = str(row.get("action_code", "")).strip()
        if action_code:
            payload["action_code"] = action_code
            action_target = row.get("action_target")
            if isinstance(action_target, str) and action_target.strip():
                payload["action_target"] = action_target.strip()
        return payload

    def _load_thread_timelines(self) -> None:
        target = self._thread_timeline_path()
        self._thread_timelines = {}
        if not target.exists():
            return
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        normalized: dict[str, list[dict[str, object]]] = {}
        for key, value in raw.items():
            thread_id = str(key).strip()
            if not thread_id or not isinstance(value, list):
                continue
            rows: list[dict[str, object]] = []
            for row in value:
                normalized_row = self._normalize_timeline_row(row)
                if normalized_row is None:
                    continue
                rows.append(normalized_row)
            if rows:
                normalized[thread_id] = rows[-self._THREAD_TIMELINE_MAX :]
        self._thread_timelines = normalized

    def _save_thread_timelines(self) -> None:
        target = self._thread_timeline_path()
        payload: dict[str, list[dict[str, object]]] = {}
        for thread_id, rows in self._thread_timelines.items():
            normalized_thread_id = str(thread_id).strip()
            if not normalized_thread_id:
                continue
            normalized_rows: list[dict[str, object]] = []
            for row in rows[-self._THREAD_TIMELINE_MAX :]:
                normalized_row = self._normalize_timeline_row(row)
                if normalized_row is None:
                    continue
                normalized_rows.append(normalized_row)
            if normalized_rows:
                payload[normalized_thread_id] = normalized_rows
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return

    def _serialize_timeline_item(self, item: QListWidgetItem) -> dict[str, object] | None:
        if bool(item.data(Qt.UserRole + 12)) or bool(item.data(Qt.UserRole + 13)):
            return None
        sender = str(item.data(Qt.UserRole + 2) or "").strip()
        text = str(item.data(Qt.UserRole + 5) or "").strip()
        if not sender or not text:
            return None
        payload: dict[str, object] = {
            "sender": sender,
            "text": text,
        }
        file_path = item.data(Qt.UserRole)
        if isinstance(file_path, str) and file_path.strip():
            payload["file_path"] = file_path.strip()
            payload["file_missing"] = bool(item.data(Qt.UserRole + 8))
        action_code = item.data(Qt.UserRole + 10)
        if isinstance(action_code, str) and action_code.strip():
            payload["action_code"] = action_code.strip()
            action_target = item.data(Qt.UserRole + 11)
            if isinstance(action_target, str) and action_target.strip():
                payload["action_target"] = action_target.strip()
        return payload

    def _capture_thread_timeline(self, thread_id: str) -> None:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return
        rows: list[dict[str, object]] = []
        for idx in range(self.result_list.count()):
            item = self.result_list.item(idx)
            if item is None:
                continue
            row = self._serialize_timeline_item(item)
            if row is None:
                continue
            rows.append(row)
        self._thread_timelines[normalized_thread_id] = rows[-self._THREAD_TIMELINE_MAX :]
        self._save_thread_timelines()

    def _restore_thread_timeline(self, thread_id: str) -> None:
        normalized_thread_id = str(thread_id or "").strip()
        rows = self._thread_timelines.get(normalized_thread_id, [])
        self.result_list.clear()
        self.streaming_item = None
        self._streaming_buffer = ""
        self._sync_chat_empty_state()
        if not rows:
            return
        for row in rows:
            sender = str(row.get("sender", "")).strip()
            text = str(row.get("text", "")).strip()
            if not sender or not text:
                continue
            file_path_raw = row.get("file_path")
            file_path = str(file_path_raw).strip() if isinstance(file_path_raw, str) else ""
            action_code_raw = row.get("action_code")
            action_code = str(action_code_raw).strip() if isinstance(action_code_raw, str) else ""
            action_target_raw = row.get("action_target")
            action_target = str(action_target_raw).strip() if isinstance(action_target_raw, str) else ""
            self.add_message(
                sender,
                text,
                file_path=file_path or None,
                file_missing=bool(row.get("file_missing", False)),
                action_code=action_code or None,
                action_target=action_target or None,
                persist_thread=False,
            )
        self.result_list.scrollToBottom()

    @staticmethod
    def _parse_thread_timestamp(raw_value: object) -> datetime | None:
        if raw_value in (None, ""):
            return None
        value = str(raw_value).strip()
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _format_thread_relative_time(updated_at_raw: object) -> str:
        parsed = LauncherWindow._parse_thread_timestamp(updated_at_raw)
        if parsed is None:
            return ""
        now = datetime.now(timezone.utc)
        delta = now - parsed
        if delta.total_seconds() <= 0:
            return "now"
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{max(1, minutes)}m"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h"
        days = hours // 24
        if days < 7:
            return f"{days}d"
        if days < 30:
            return f"{max(1, days // 7)}w"
        return f"{max(1, days // 30)}mo"

    @staticmethod
    def _normalize_thread_title(raw_title: object) -> str:
        title = " ".join(str(raw_title or "").split()).strip()
        if not title:
            return "New thread"
        if len(title) > 56:
            return title[:53].rstrip() + "..."
        return title

    def _default_thread_entries(self) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        samples = (
            ("Reply to 0f greeting", timedelta(hours=18)),
            ("검증 로직 점검하고 결과 정리", timedelta(days=6)),
            ("프로젝트 완성도평가해", timedelta(days=7)),
            ("5.2 개발과 5.10 마이그레이션", timedelta(days=30)),
            ("Analyze 프로젝트 원판구성", timedelta(days=30, hours=6)),
        )
        entries: list[dict[str, str]] = []
        for idx, (title, age) in enumerate(samples, start=1):
            entries.append(
                {
                    "id": f"seed-{idx}",
                    "title": title,
                    "updated_at": (now - age).isoformat(),
                }
            )
        return entries

    def _normalize_thread_entry(self, row: object, *, fallback_index: int) -> dict[str, str] | None:
        if not isinstance(row, dict):
            return None
        thread_id = str(row.get("id", "")).strip() or f"thread-{fallback_index}"
        title = self._normalize_thread_title(row.get("title", ""))
        parsed = self._parse_thread_timestamp(row.get("updated_at"))
        if parsed is None:
            parsed = datetime.now(timezone.utc) - timedelta(minutes=fallback_index)
        return {
            "id": thread_id,
            "title": title,
            "updated_at": parsed.isoformat(),
        }

    def _sort_thread_entries(self) -> None:
        def _sort_key(entry: dict[str, str]) -> float:
            parsed = self._parse_thread_timestamp(entry.get("updated_at"))
            if parsed is None:
                return 0.0
            return parsed.timestamp()

        self._thread_entries.sort(key=_sort_key, reverse=True)

    def _save_thread_entries(self) -> None:
        target = self._thread_history_path()
        payload = [
            {
                "id": str(row.get("id", "")),
                "title": self._normalize_thread_title(row.get("title", "")),
                "updated_at": str(row.get("updated_at", "")),
            }
            for row in self._thread_entries[: self._THREAD_STORE_MAX]
            if isinstance(row, dict)
        ]
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return

    def _load_thread_entries(self) -> None:
        target = self._thread_history_path()
        raw_entries: list[object] = []
        if target.exists():
            try:
                loaded = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    raw_entries = loaded
            except (OSError, ValueError):
                raw_entries = []
        if not raw_entries:
            raw_entries = self._default_thread_entries()
        normalized: list[dict[str, str]] = []
        for idx, row in enumerate(raw_entries, start=1):
            entry = self._normalize_thread_entry(row, fallback_index=idx)
            if entry is None:
                continue
            normalized.append(entry)
        if not normalized:
            normalized = self._default_thread_entries()
        self._thread_entries = normalized[: self._THREAD_STORE_MAX]
        self._sort_thread_entries()
        if self._thread_visible_limit <= 0:
            self._thread_visible_limit = min(max(self._THREAD_PAGE_SIZE, 5), len(self._thread_entries))
        if not self._active_thread_id and self._thread_entries:
            self._active_thread_id = str(self._thread_entries[0].get("id", ""))
        known_ids = {str(row.get("id", "")).strip() for row in self._thread_entries if isinstance(row, dict)}
        if known_ids:
            self._thread_timelines = {
                key: value for key, value in self._thread_timelines.items() if str(key).strip() in known_ids
            }
            self._save_thread_timelines()
        self._save_thread_entries()

    @staticmethod
    def _new_thread_id() -> str:
        return f"thread-{uuid4().hex[:12]}"

    def _entry_for_thread_id(self, thread_id: str) -> dict[str, str] | None:
        for entry in self._thread_entries:
            if str(entry.get("id", "")) == thread_id:
                return entry
        return None

    def _upsert_thread_entry(self, *, thread_id: str, title: str | None = None, touch: bool = True) -> None:
        entry = self._entry_for_thread_id(thread_id)
        if entry is None:
            entry = {
                "id": thread_id,
                "title": "New thread",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._thread_entries.append(entry)
        if title is not None:
            entry["title"] = self._normalize_thread_title(title)
        if touch:
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._sort_thread_entries()
        self._thread_entries = self._thread_entries[: self._THREAD_STORE_MAX]
        self._thread_timelines.setdefault(thread_id, [])
        self._save_thread_entries()

    def _sync_active_thread_for_query(self, query: str) -> None:
        thread_id = str(self._active_thread_id or "").strip()
        if not thread_id:
            thread_id = self._new_thread_id()
            self._active_thread_id = thread_id
        query_title = self._normalize_thread_title(query)
        current = self._entry_for_thread_id(thread_id)
        if current is None:
            self._upsert_thread_entry(thread_id=thread_id, title=query_title, touch=True)
        else:
            current_title = self._normalize_thread_title(current.get("title", ""))
            update_title = query_title if current_title == "New thread" else None
            self._upsert_thread_entry(thread_id=thread_id, title=update_title, touch=True)
        if self._thread_visible_limit < self._THREAD_PAGE_SIZE:
            self._thread_visible_limit = self._THREAD_PAGE_SIZE
        self._populate_sidebar_threads(select_thread_id=thread_id)

    def _shortcut_mod_label(self) -> str:
        return "⌘" if sys.platform == "darwin" else "Ctrl"

    def _refresh_shortcut_hint(self):
        mod = self._shortcut_mod_label()
        self.shortcut_hint.setText(
            (
                f"{mod}K 검색  •  {mod}L 입력  •  {mod}J 타임라인  •  "
                f"{mod}M 모드  •  {mod}⇧↑/↓ 이동  •  {mod}⇧C 인용  •  "
                f"{mod}O 열기  •  {mod}⇧P 상위폴더  •  {mod}⇧R 위치열기  •  {mod}⇧O 경로복사"
            )
        )

    def _show_shortcuts_help(self):
        mod = self._shortcut_mod_label()
        help_text = (
            f"Shortcuts: {mod}+K(검색), {mod}+L(입력), {mod}+J(타임라인), "
            f"{mod}+M(모드), {mod}+Shift+↑/↓(메시지 이동), {mod}+Shift+C(인용), "
            f"{mod}+O(파일 열기), {mod}+Shift+P(상위 폴더), {mod}+Shift+R(위치 열기), "
            f"{mod}+Shift+O(파일 경로 복사)"
        )
        self._append_system_message_once(help_text)

    def _cycle_response_mode(self):
        modes = [mode for mode in MODE_ORDER if mode in self._response_mode_descriptions]
        if not modes:
            return
        current_index = modes.index(self._response_mode) if self._response_mode in modes else 0
        self._set_response_mode(modes[(current_index + 1) % len(modes)])

    def _populate_sidebar_threads(self, *, select_thread_id: str | None = None):
        self.thread_list.clear()
        if not self._thread_entries:
            self._load_thread_entries()
        visible_limit = min(max(self._thread_visible_limit, self._THREAD_PAGE_SIZE), len(self._thread_entries))
        self._thread_visible_limit = visible_limit
        visible_entries = self._thread_entries[:visible_limit]
        for row in visible_entries:
            title = self._normalize_thread_title(row.get("title", ""))
            time_label = self._format_thread_relative_time(row.get("updated_at", ""))
            thread_id = str(row.get("id", "")).strip()
            updated_at = str(row.get("updated_at", "")).strip()
            self._add_thread_item(title, time_label, thread_id=thread_id, updated_at=updated_at)

        has_more = len(self._thread_entries) > visible_limit
        self.btn_show_more_threads.setEnabled(has_more)
        self.btn_show_more_threads.setText("Show more" if has_more else "No more")

        if self.thread_list.count() <= 0:
            self._apply_thread_filter(self.thread_search.text())
            return

        requested_id = str(select_thread_id or self._active_thread_id).strip()
        selected_row = 0
        if requested_id:
            for idx in range(self.thread_list.count()):
                item = self.thread_list.item(idx)
                if str(item.data(Qt.UserRole + 20) or "") == requested_id:
                    selected_row = idx
                    break
        self.thread_list.setCurrentRow(selected_row)
        self._on_thread_selection_changed(selected_row)
        self._apply_thread_filter(self.thread_search.text())

    def _add_thread_item(self, title: str, time_label: str, *, thread_id: str, updated_at: str):
        item = QListWidgetItem()
        item.setSizeHint(QSize(230, 42))
        item.setData(Qt.UserRole + 1, title.casefold())
        item.setData(Qt.UserRole + 20, thread_id)
        item.setData(Qt.UserRole + 21, updated_at)
        self.thread_list.addItem(item)

        row = QFrame()
        row.setObjectName("ThreadRow")
        row.setProperty("active", False)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 8, 10, 8)
        row_layout.setSpacing(8)

        title_widget = QLabel(title)
        title_widget.setObjectName("ThreadTitle")
        meta_widget = QLabel(time_label)
        meta_widget.setObjectName("ThreadMeta")

        row_layout.addWidget(title_widget, stretch=1)
        row_layout.addWidget(meta_widget, stretch=0)
        self.thread_list.setItemWidget(item, row)

    def _on_thread_selection_changed(self, current_row: int):
        previous_thread_id = str(self._active_thread_id or "").strip()
        next_thread_id = ""
        if 0 <= current_row < self.thread_list.count():
            selected_item = self.thread_list.item(current_row)
            next_thread_id = str(selected_item.data(Qt.UserRole + 20) or "").strip()
        if next_thread_id and previous_thread_id != next_thread_id:
            if previous_thread_id:
                self._capture_thread_timeline(previous_thread_id)
            self._active_thread_id = next_thread_id
            self._restore_thread_timeline(next_thread_id)
        elif next_thread_id:
            self._active_thread_id = next_thread_id
        for idx in range(self.thread_list.count()):
            item = self.thread_list.item(idx)
            row = self.thread_list.itemWidget(item)
            if row is None:
                continue
            row.setProperty("active", idx == current_row)
            row.style().unpolish(row)
            row.style().polish(row)
            row.update()
            if idx == current_row:
                title_widget = row.findChild(QLabel, "ThreadTitle")
                if title_widget and title_widget.text():
                    self.header_title.setText(title_widget.text())

    def _show_more_threads(self):
        if self._thread_visible_limit >= len(self._thread_entries):
            self._append_system_message_once("모든 스레드를 표시 중입니다.")
            return
        self._thread_visible_limit = min(len(self._thread_entries), self._thread_visible_limit + self._THREAD_PAGE_SIZE)
        self._populate_sidebar_threads(select_thread_id=self._active_thread_id)
        self._set_status(f"Threads: {self._thread_visible_limit}/{len(self._thread_entries)}", "neutral")

    def _focus_thread_search(self):
        self.thread_search.setFocus()
        self.thread_search.selectAll()

    def _focus_composer_input(self):
        self.input_field.setFocus()
        self.input_field.selectAll()

    def _focus_chat_timeline(self):
        if self.result_list.count() <= 0:
            return
        self.result_list.setFocus()
        self.result_list.setCurrentRow(self.result_list.count() - 1)
        self.result_list.scrollToBottom()
        self._set_status(f"Timeline: {self.result_list.currentRow() + 1}/{self.result_list.count()}", "neutral")

    def _move_chat_selection(self, delta: int):
        total = self.result_list.count()
        if total <= 0:
            return

        current_row = self.result_list.currentRow()
        if current_row < 0:
            current_row = total - 1 if delta < 0 else 0
        next_row = (current_row + delta) % total

        self.result_list.setFocus()
        self.result_list.setCurrentRow(next_row)
        self.result_list.scrollToItem(self.result_list.item(next_row))
        self._set_status(f"Timeline: {next_row + 1}/{total}", "neutral")

    def _open_selected_timeline_item(self):
        item = self.result_list.currentItem()
        if item is None:
            self._append_system_message_once("열 수 있는 메시지가 없습니다.")
            return
        if not item.data(Qt.UserRole):
            self._append_system_message_once("선택된 메시지에는 파일 링크가 없습니다.")
            return
        self.on_result_item_clicked(item)

    def _selected_timeline_file_path(self) -> Path | None:
        item = self.result_list.currentItem()
        if item is None:
            self._append_system_message_once("선택된 타임라인 항목이 없습니다.")
            return None
        raw_path = item.data(Qt.UserRole)
        if raw_path in (None, ""):
            raw_path = item.data(Qt.UserRole + 11)
        file_path = self._normalize_local_file_path(raw_path)
        if file_path is None:
            self._append_system_message_once("선택된 항목에는 파일 경로가 없습니다.")
            return None
        return file_path

    def _open_selected_timeline_parent(self):
        file_path = self._selected_timeline_file_path()
        if file_path is None:
            return
        parent = file_path.parent
        if not parent.exists():
            self.add_message("System", f"상위 폴더를 찾을 수 없습니다: {parent}")
            return
        opened, info = self._open_local_file(parent)
        if opened:
            self._set_status(f"Opened parent: {parent.name}", "ok")
            self.add_message("System", f"상위 폴더를 열었습니다: {parent}")
            if info:
                self.add_message("System", info)
            return
        self._append_open_failure_guidance(
            file_path=file_path,
            error_message=info,
            parent_attempt=True,
            parent_path=parent,
        )

    def _copy_selected_timeline_file_path(self):
        file_path = self._selected_timeline_file_path()
        if file_path is None:
            return
        QApplication.clipboard().setText(str(file_path))
        self._set_status("File path copied", "ok")
        self.add_message("System", f"파일 경로를 복사했습니다: {file_path}")

    def _reveal_selected_timeline_item(self):
        file_path = self._selected_timeline_file_path()
        if file_path is None:
            return
        revealed, info = self._reveal_in_finder(file_path)
        if revealed:
            self._set_status("Revealed in Finder", "ok")
            self.add_message("System", info or f"Finder에서 위치를 열었습니다: {file_path}")
            return
        self._append_open_failure_guidance(file_path=file_path, error_message=info)

    def _append_open_recovery_actions(self, file_path: Path):
        mod = self._shortcut_mod_label()
        self.add_message(
            "System",
            (
                f"다음 동작: {mod}+O(다시 열기) / "
                f"{mod}+Shift+P(상위 폴더) / {mod}+Shift+R(위치 열기) / {mod}+Shift+O(경로 복사) - {file_path}"
            ),
        )
        self._add_recovery_action_card(file_path)

    def _append_open_failure_cta_actions(self, *, file_path: Path, category: str) -> None:
        ctas: list[tuple[str, str]] = []
        if category in {"permission", "canceled"}:
            ctas.append(("권한 설정 가이드 열기", "open_permission_guide"))
        if category in {"association", "canceled"}:
            ctas.append(("기본 앱 연결 가이드 열기", "open_app_association_guide"))
        if sys.platform == "darwin":
            ctas.append(("Finder에서 위치 열기", "reveal_in_finder"))
        if category in {"not_found", "generic", "canceled", "association"}:
            ctas.append(("이름 유사 문서 찾기", "search_similar_files"))
        deduped: list[tuple[str, str]] = []
        seen_codes: set[str] = set()
        for label, code in ctas:
            action_code = str(code).strip()
            if not action_code or action_code in seen_codes:
                continue
            seen_codes.add(action_code)
            deduped.append((label, action_code))
        ctas = deduped
        if not ctas:
            return
        labels = ", ".join(label for label, _ in ctas)
        self.add_message("System", f"권장 조치: {labels}")
        self._add_failure_guide_card(file_path=file_path, actions=ctas)

    @staticmethod
    def _path_similarity_score(candidate_dir: Path, target_dir: Path) -> int:
        candidate_parts = [part.casefold() for part in candidate_dir.parts if part]
        target_parts = [part.casefold() for part in target_dir.parts if part]
        score = 0
        idx = 1
        while idx <= len(candidate_parts) and idx <= len(target_parts):
            if candidate_parts[-idx] != target_parts[-idx]:
                break
            score += 1
            idx += 1
        return score

    @staticmethod
    def _path_cache_key(path: Path) -> str:
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        text = str(resolved)
        return text.casefold() if os.name == "nt" else text

    def _cached_resolved_file_path(self, file_path: Path) -> Path | None:
        key = self._path_cache_key(file_path)
        cached_raw = str(self._file_resolution_cache.get(key, "")).strip()
        if not cached_raw:
            return None
        candidate = Path(cached_raw).expanduser()
        if not candidate.exists():
            self._file_resolution_cache.pop(key, None)
            self._save_file_resolution_cache()
            return None
        try:
            return candidate.resolve(strict=False)
        except OSError:
            return candidate

    def _remember_resolved_file_path(self, source_path: Path, resolved_path: Path) -> None:
        source_key = self._path_cache_key(source_path)
        try:
            normalized = resolved_path.resolve(strict=False)
        except OSError:
            normalized = resolved_path
        self._file_resolution_cache[source_key] = str(normalized)
        self._save_file_resolution_cache()

    def _resolve_click_target_file(self, file_path: Path) -> tuple[Path | None, str, list[Path]]:
        if file_path.exists():
            return file_path, "exact", []
        cached = self._cached_resolved_file_path(file_path)
        if cached is not None:
            return cached, "cached", []
        candidates = self._find_similar_file_candidates(file_path, limit=3)
        if len(candidates) == 1:
            candidate = candidates[0]
            self._remember_resolved_file_path(file_path, candidate)
            return candidate, "similar", candidates
        if len(candidates) >= 2:
            return None, "ambiguous", candidates
        return None, "missing", []

    def _find_similar_file_candidates(self, file_path: Path, *, limit: int = 5) -> list[Path]:
        target_name = file_path.name
        target_stem = file_path.stem.casefold()
        target_suffix = file_path.suffix.casefold()
        target_parent = file_path.parent
        if not target_name or not target_stem:
            return []
        ranked: list[tuple[int, int, int, float, str, Path]] = []
        seen: set[str] = set()
        candidate_patterns: list[str] = [target_name]
        wildcard_pattern = f"{file_path.stem}.*".strip()
        if wildcard_pattern and wildcard_pattern not in candidate_patterns:
            candidate_patterns.append(wildcard_pattern)
        for folder_priority, entry in enumerate(self.policy_registry.list_folders()):
            folder_raw = str(entry.get("path", "")).strip()
            if not folder_raw:
                continue
            folder = Path(folder_raw).expanduser()
            if not folder.exists() or not folder.is_dir():
                continue
            candidates = self._load_similar_candidates_for_folder(
                folder=folder,
                target_stem=target_stem,
                candidate_patterns=candidate_patterns,
            )
            for matched in candidates:
                resolved = str(matched.resolve(strict=False))
                key = resolved.casefold() if os.name == "nt" else resolved
                if key in seen:
                    continue
                seen.add(key)
                mtime = 0.0
                try:
                    mtime = matched.stat().st_mtime
                except OSError:
                    mtime = 0.0
                ext_penalty = 0 if matched.suffix.casefold() == target_suffix else 1
                path_similarity = self._path_similarity_score(matched.parent, target_parent)
                ranked.append((folder_priority, ext_penalty, -path_similarity, -mtime, resolved, Path(resolved)))
        ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
        return [row[5] for row in ranked[:limit]]

    def _append_similar_file_candidates(self, file_path: Path, *, candidates: list[Path] | None = None) -> None:
        resolved_candidates = candidates if candidates is not None else self._find_similar_file_candidates(file_path)
        seen: set[str] = set()
        unique_candidates: list[Path] = []
        for candidate in resolved_candidates:
            key = str(candidate).casefold() if os.name == "nt" else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        candidates = unique_candidates
        if not candidates:
            self.add_message("System", f"유사 문서를 찾지 못했습니다: {file_path.name}")
            return
        self.add_message("System", f"유사 문서 후보 {len(candidates)}개를 찾았습니다: {file_path.name}")
        for idx, candidate in enumerate(candidates, start=1):
            label = f"후보 문서 열기 {idx}: {candidate.name}"
            self.add_message("Action", label, action_code="open_candidate_file", action_target=str(candidate))

    def _add_failure_guide_card(self, *, file_path: Path, actions: list[tuple[str, str]]) -> None:
        item = QListWidgetItem("FailureGuideCard")
        item.setData(Qt.UserRole, str(file_path))
        item.setData(Qt.UserRole + 13, True)
        item.setSizeHint(QSize(0, 86))
        self.result_list.addItem(item)
        card = FailureGuideCard(
            file_path=str(file_path),
            actions=actions,
            action_callback=lambda code, target: self._run_timeline_action(code, target),
            parent=self.result_list,
        )
        self.result_list.setItemWidget(item, card)
        self.result_list.scrollToBottom()
        self._sync_chat_empty_state()

    def _append_open_failure_guidance(
        self,
        *,
        file_path: Path,
        error_message: str,
        parent_attempt: bool = False,
        parent_path: Path | None = None,
    ) -> None:
        prefix = "상위 폴더 열기 실패" if parent_attempt else "파일 열기 실패"
        self.add_message("System", f"{prefix}: {error_message}")
        category = self._classify_open_error(error_message)
        if category == "permission":
            self.add_message(
                "System",
                "권한 점검: 시스템 설정 > 개인정보 보호 및 보안 > 파일 및 폴더에서 앱 권한을 허용하세요.",
            )
        elif category == "canceled":
            self.add_message(
                "System",
                "취소 원인 점검: 1) Finder에서 위치 열기 2) 기본 앱 연결 확인 3) 파일 및 폴더 권한 확인",
            )
        elif category == "association":
            self.add_message(
                "System",
                "기본 앱 점검: Finder > 정보 가져오기 > 다음으로 열기에서 기본 앱을 지정하세요.",
            )
        elif category == "not_found":
            self.add_message(
                "System",
                "경로 점검: 파일이 이동/삭제되었는지 확인하고 최신 문서를 다시 첨부하세요.",
            )
        else:
            self.add_message(
                "System",
                "오류 점검: 경로/앱 연결 상태를 확인한 뒤 다시 시도하세요.",
            )
        if parent_attempt and parent_path is not None:
            self.add_message("System", f"열기 대상 상위 폴더: {parent_path}")
        self._append_open_failure_cta_actions(file_path=file_path, category=category)
        self._append_open_recovery_actions(file_path)

    def _add_recovery_action_card(self, file_path: Path) -> None:
        item = QListWidgetItem("ActionCard")
        item.setData(Qt.UserRole, str(file_path))
        item.setData(Qt.UserRole + 10, "action_card")
        item.setData(Qt.UserRole + 11, str(file_path))
        item.setData(Qt.UserRole + 12, True)
        item.setSizeHint(QSize(0, 78))
        self.result_list.addItem(item)
        card = ActionRecoveryCard(
            file_path=str(file_path),
            shortcut_mod=self._shortcut_mod_label(),
            action_callback=lambda code, target: self._run_timeline_action(code, target),
            parent=self.result_list,
        )
        self.result_list.setItemWidget(item, card)
        self.result_list.scrollToBottom()
        self._sync_chat_empty_state()

    def _run_timeline_action(self, action_code: str, action_target: object) -> None:
        file_path = self._normalize_local_file_path(action_target)
        if action_code in {"open_permission_guide", "open_app_association_guide"}:
            guide_url = ""
            if action_code == "open_permission_guide":
                guide_url = "https://support.apple.com/guide/mac-help/control-access-to-files-and-folders-on-mac-mchlccb25729/mac"
            elif action_code == "open_app_association_guide":
                guide_url = "https://support.apple.com/guide/mac-help/choose-an-app-to-open-a-file-on-mac-mh35597/mac"
            opened = QDesktopServices.openUrl(QUrl(guide_url))
            if opened:
                self._set_status("Opened guide", "ok")
                self.add_message("System", f"가이드를 열었습니다: {guide_url}")
            else:
                self._set_status("Guide open failed", "error")
                self.add_message("System", f"가이드 열기 실패: {guide_url}")
            return

        if file_path is None:
            self.add_message("System", "액션에 필요한 파일 경로를 확인할 수 없습니다.")
            return
        if action_code == "retry_open":
            item = QListWidgetItem("Action: retry")
            item.setData(Qt.UserRole, str(file_path))
            self.on_result_item_clicked(item)
            return
        if action_code == "open_parent":
            parent = file_path.parent
            if not parent.exists():
                self.add_message("System", f"상위 폴더를 찾을 수 없습니다: {parent}")
                return
            opened, info = self._open_local_file(parent)
            if opened:
                self._set_status(f"Opened parent: {parent.name}", "ok")
                self.add_message("System", f"상위 폴더를 열었습니다: {parent}")
                if info:
                    self.add_message("System", info)
                return
            self._append_open_failure_guidance(
                file_path=file_path,
                error_message=info,
                parent_attempt=True,
                parent_path=parent,
            )
            return
        if action_code == "copy_path":
            QApplication.clipboard().setText(str(file_path))
            self._set_status("File path copied", "ok")
            self.add_message("System", f"파일 경로를 복사했습니다: {file_path}")
            return
        if action_code == "search_similar_files":
            self._append_similar_file_candidates(file_path)
            return
        if action_code == "reveal_in_finder":
            revealed, info = self._reveal_in_finder(file_path)
            if revealed:
                self._set_status("Revealed in Finder", "ok")
                self.add_message("System", info or f"Finder에서 위치를 열었습니다: {file_path}")
                return
            self._append_open_failure_guidance(file_path=file_path, error_message=info)
            return
        if action_code == "open_candidate_file":
            item = QListWidgetItem("Action: open candidate")
            item.setData(Qt.UserRole, str(file_path))
            self.on_result_item_clicked(item)
            return
        self.add_message("System", f"지원되지 않는 액션입니다: {action_code}")

    def _cite_selected_timeline_item(self):
        item = self.result_list.currentItem()
        if item is None:
            self._append_system_message_once("인용할 메시지가 없습니다.")
            return

        quote = item.text().strip()
        if not quote:
            self._append_system_message_once("인용할 메시지가 없습니다.")
            return

        current = self.input_field.toPlainText().rstrip()
        prefix = f"{current}\n\n" if current else ""
        self.input_field.setPlainText(f"{prefix}> {quote}\n")
        self._focus_composer_input()
        self._set_status("Quoted selected message", "ok")

    def _apply_thread_filter(self, text: str):
        needle = text.strip().casefold()
        visible_rows: list[int] = []
        for idx in range(self.thread_list.count()):
            item = self.thread_list.item(idx)
            haystack = item.data(Qt.UserRole + 1) or ""
            is_visible = not needle or needle in haystack
            item.setHidden(not is_visible)
            if is_visible:
                visible_rows.append(idx)

        if not visible_rows:
            self.thread_list.setCurrentRow(-1)
            self.thread_empty_state.setText("검색 결과가 없습니다.")
            self.thread_empty_state.show()
            self._on_thread_selection_changed(-1)
            return

        self.thread_empty_state.hide()
        if self.thread_list.currentRow() not in visible_rows:
            self.thread_list.setCurrentRow(visible_rows[0])
        self._on_thread_selection_changed(self.thread_list.currentRow())

    def _activate_thread_from_search(self):
        visible_rows = [idx for idx in range(self.thread_list.count()) if not self.thread_list.item(idx).isHidden()]
        if not visible_rows:
            return
        row_index = self.thread_list.currentRow()
        if row_index not in visible_rows:
            row_index = visible_rows[0]
            self.thread_list.setCurrentRow(row_index)
        self.thread_list.scrollToItem(self.thread_list.item(row_index))
        self.thread_list.setFocus()

    def _move_thread_selection(self, delta: int):
        visible_rows = [idx for idx in range(self.thread_list.count()) if not self.thread_list.item(idx).isHidden()]
        if not visible_rows:
            return

        current_row = self.thread_list.currentRow()
        if current_row not in visible_rows:
            next_row = visible_rows[0] if delta >= 0 else visible_rows[-1]
        else:
            current_index = visible_rows.index(current_row)
            next_index = (current_index + delta) % len(visible_rows)
            next_row = visible_rows[next_index]

        self.thread_list.setCurrentRow(next_row)
        self.thread_list.scrollToItem(self.thread_list.item(next_row))

    def _set_query_controls_enabled(self, enabled: bool):
        for control in self._query_controls:
            control.setEnabled(enabled)

    def _sync_chat_empty_state(self):
        self.chat_empty_state.setVisible(self.result_list.count() <= 0)

    def _set_status(self, text: str, tone: str = "neutral"):
        tone_styles = {
            "neutral": "background:#e5e7eb; color:#374151;",
            "busy": "background:#e0ecff; color:#1d4ed8;",
            "error": "background:#fee2e2; color:#991b1b;",
            "ok": "background:#dcfce7; color:#166534;",
        }
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "border-radius: 999px; padding: 4px 10px; font-weight: 700; " + tone_styles.get(tone, tone_styles["neutral"])
        )

    def _append_system_message_once(self, text: str):
        if self._last_system_message == text:
            return
        self._last_system_message = text
        self.add_message("System", text)

    def _start_new_thread(self):
        previous_thread_id = str(self._active_thread_id or "").strip()
        if previous_thread_id:
            self._capture_thread_timeline(previous_thread_id)
        self.result_list.clear()
        self._sync_chat_empty_state()
        self.input_field.clear()
        self.streaming_item = None
        self._streaming_buffer = ""
        new_thread_id = self._new_thread_id()
        self._active_thread_id = new_thread_id
        self._upsert_thread_entry(thread_id=new_thread_id, title="New thread", touch=True)
        self._thread_timelines[new_thread_id] = []
        self._save_thread_timelines()
        if self._thread_visible_limit < self._THREAD_PAGE_SIZE:
            self._thread_visible_limit = self._THREAD_PAGE_SIZE
        self._populate_sidebar_threads(select_thread_id=new_thread_id)
        self.header_title.setText("New thread")
        self._set_status("Ready", "neutral")

    def _open_automations(self):
        self._append_system_message_once("Automations 화면은 준비 중입니다.")

    def _open_skills(self):
        self._append_system_message_once("Skills 화면은 준비 중입니다.")

    def _insert_attachment_hint(self):
        self.input_field.insertPlainText(" [첨부] ")
        self.input_field.setFocus()

    def _insert_web_hint(self):
        self.input_field.insertPlainText("@검색 ")
        self.input_field.setFocus()

    def _on_backend_ready(self):
        self._set_status("Ready", "ok")
        self._set_query_controls_enabled(True)

    def on_submit(self):
        if self._query_in_flight:
            self._append_system_message_once("이전 응답을 생성 중입니다. 잠시만 기다려주세요.")
            return

        query = self.input_field.toPlainText().strip()
        if not query:
            return

        self._sync_active_thread_for_query(query)
        self._last_system_message = ""
        self.input_field.clear()
        self.input_field.setFixedHeight(self.input_field._min_height)

        self.add_message("Me", query)

        self._query_in_flight = True
        self._set_query_controls_enabled(False)
        self.thinking_dots = 0
        self._set_status(f"Thinking ({self._response_mode})", "busy")
        self.thinking_timer.start(420)
        self.streaming_item = None
        self._streaming_buffer = ""
        self.query_requested.emit(query, self._response_mode)

    def update_thinking_text(self):
        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "·" * (self.thinking_dots + 1)
        self._set_status(f"Thinking ({self._response_mode}) {dots}", "busy")

    def handle_stream_update(self, chunk: str):
        if not chunk:
            return
        if self.streaming_item is None:
            self._streaming_buffer = ""
            self.streaming_item = self._create_stream_item("Assistant")
        self._streaming_buffer += chunk
        display_text, _ = self._mask_display_text(self._streaming_buffer)
        self._set_message_item_text(self.streaming_item, "Assistant", display_text or self._streaming_buffer)
        self.result_list.scrollToBottom()
        self._set_status("Typing", "busy")

    def _mask_display_text(self, raw_text: str) -> tuple[str, bool]:
        text = (raw_text or "").strip()
        if not text:
            return "", False
        if not self._privacy_mask_enabled:
            return text, False
        masked = _mask_pii_text(text)
        return masked, masked != text

    def _file_open_shortcut_hint(self) -> str:
        mod = self._shortcut_mod_label()
        return f"{mod}+O"

    @staticmethod
    def _format_file_item_label(file_name: str, *, is_missing: bool) -> str:
        if is_missing:
            return f"[missing] {file_name} (open parent folder)"
        return file_name

    def _extract_file_links(
        self, response: str, *, limit: int | None = None
    ) -> tuple[str, list[str], int, int, int, int]:
        pattern = re.compile(r"\[FILE_LINK:([^\]]+)\]")
        raw_links = pattern.findall(response or "")
        clean_response = pattern.sub("", response or "").strip()
        normalized_links: list[str] = []
        seen: set[str] = set()
        overflow = 0
        invalid_count = 0
        merged_duplicate_count = 0
        legacy_converted_count = 0
        effective_limit = self._max_file_links if limit is None else max(1, int(limit))
        for raw in raw_links:
            raw_token = raw.strip()
            if len(raw_token) >= 2 and raw_token[0] == raw_token[-1] and raw_token[0] in {"'", '"'}:
                raw_token = raw_token[1:-1].strip()
            is_file_uri = raw_token.startswith("file://")
            is_legacy_absolute_path = raw_token.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", raw_token))
            if not is_file_uri and not is_legacy_absolute_path:
                invalid_count += 1
                continue
            if is_legacy_absolute_path and not is_file_uri:
                legacy_converted_count += 1
            path = self._normalize_local_file_path(raw_token)
            if path is None:
                invalid_count += 1
                continue
            value = str(path)
            key = value.casefold() if os.name == "nt" else value
            if key in seen:
                merged_duplicate_count += 1
                continue
            seen.add(key)
            if len(normalized_links) >= effective_limit:
                overflow += 1
                continue
            normalized_links.append(value)
        return clean_response, normalized_links, overflow, invalid_count, merged_duplicate_count, legacy_converted_count

    def handle_response(self, response: str):
        self.thinking_timer.stop()

        clean_response, file_links, overflow_count, invalid_count, merged_duplicate_count, legacy_converted_count = (
            self._extract_file_links(response)
        )
        masked_response, response_masked = self._mask_display_text(clean_response)
        if not masked_response and file_links:
            masked_response = "참조 문서만 반환되었습니다. 아래 파일을 확인하세요."
        if response_masked and "민감정보 일부 마스킹됨" not in masked_response:
            masked_response = masked_response.rstrip() + "\n\n(보안: UI 표시 단계에서 민감정보 일부 마스킹됨)"

        if self.streaming_item is not None:
            masked_stream, _ = self._mask_display_text(self._streaming_buffer.strip())
            final_text = masked_response or masked_stream or self._streaming_buffer.strip()
            if final_text:
                self._set_message_item_text(self.streaming_item, "Assistant", final_text)
            else:
                row = self.result_list.row(self.streaming_item)
                if row >= 0:
                    self.result_list.takeItem(row)
            self.streaming_item = None
            self._streaming_buffer = ""
        else:
            self.add_message("Assistant", masked_response)

        missing_count = 0
        for file_path in file_links:
            file_name = os.path.basename(file_path)
            is_missing = False
            if not Path(file_path).exists():
                missing_count += 1
                is_missing = True
            file_name = self._format_file_item_label(file_name, is_missing=is_missing)
            masked_file_name, _ = self._mask_display_text(file_name)
            self.add_message("File", masked_file_name or file_name, file_path=file_path, file_missing=is_missing)
        note_parts: list[str] = []
        if overflow_count > 0:
            note_parts.append(f"총 {len(file_links) + overflow_count}개 중 {len(file_links)}개 표시")
        if missing_count > 0:
            note_parts.append(f"{len(file_links)}개 중 {missing_count}개는 현재 경로에 없습니다")
        if invalid_count > 0:
            note_parts.append(f"유효하지 않은 링크 {invalid_count}개는 제외했습니다")
        if merged_duplicate_count > 0:
            note_parts.append(f"중복 링크 {merged_duplicate_count}개는 병합했습니다")
        if legacy_converted_count > 0:
            note_parts.append(f"레거시 경로 링크 {legacy_converted_count}개를 표준 경로로 변환했습니다")
        if note_parts:
            self.add_message("System", "참조 문서 요약: " + " / ".join(note_parts))

        self._query_in_flight = False
        self._set_query_controls_enabled(True)
        self._capture_thread_timeline(str(self._active_thread_id or "").strip())
        self._set_status("Ready", "ok")

    def update_status_msg(self, msg: str):
        lower = msg.lower()
        if "error" in lower or "실패" in msg:
            self._set_status(msg, "error")
            return
        if any(token in lower for token in ("loading", "thinking", "initial", "busy")) or "준비" in msg:
            self._set_status(msg, "busy")
            return
        if "ready" in lower:
            self._set_status(msg, "ok")
            return
        self._set_status(msg, "neutral")

    def handle_error(self, msg: str):
        self.thinking_timer.stop()
        self._query_in_flight = False
        self._set_query_controls_enabled(True)
        if self.streaming_item is not None and not self._streaming_buffer.strip():
            row = self.result_list.row(self.streaming_item)
            if row >= 0:
                self.result_list.takeItem(row)
        self.streaming_item = None
        self._streaming_buffer = ""
        self._set_status("Error", "error")
        self.add_message("System", f"Error: {msg}")
        self._capture_thread_timeline(str(self._active_thread_id or "").strip())

    @staticmethod
    def _strip_group_prefix(text: str) -> str:
        for prefix in ("┌ ", "├ ", "└ ", "↳ "):
            if text.startswith(prefix):
                return text[len(prefix) :]
        if ": " in text:
            return text.split(": ", 1)[1]
        return text

    def _update_previous_group_item(self, previous: QListWidgetItem) -> None:
        previous_state = previous.data(Qt.UserRole + 6)
        previous_text = str(previous.data(Qt.UserRole + 5) or self._strip_group_prefix(previous.text()))
        if previous_state == "single":
            previous.setText(f"┌ {previous_text}")
            previous.setData(Qt.UserRole + 6, "start")
            previous.setToolTip("Grouped message (start)")
        elif previous_state in {"start", "mid", "end"}:
            previous.setText(f"├ {previous_text}")
            previous.setData(Qt.UserRole + 6, "mid")
            previous.setToolTip("Grouped message (middle)")

    def _build_message_item(self, sender: str, text: str) -> tuple[QListWidgetItem, bool]:
        groupable = sender in {"Me", "Assistant"}
        group_state = "single"
        message_text = f"{sender}: {text}"
        compact = False

        if groupable and self.result_list.count() > 0:
            previous = self.result_list.item(self.result_list.count() - 1)
            prev_sender = previous.data(Qt.UserRole + 2)
            if prev_sender == sender:
                compact = True
                self._update_previous_group_item(previous)
                group_state = "end"
                message_text = f"└ {text}"

        item = QListWidgetItem(message_text)
        item.setData(Qt.UserRole + 2, sender)
        item.setData(Qt.UserRole + 5, text)
        item.setData(Qt.UserRole + 6, group_state)
        return item, compact

    @staticmethod
    def _message_prefix_for_state(sender: str, group_state: object) -> str:
        if group_state == "start":
            return "┌ "
        if group_state == "mid":
            return "├ "
        if group_state == "end":
            return "└ "
        return f"{sender}: "

    def _set_message_item_text(self, item: QListWidgetItem, sender: str, text: str) -> None:
        group_state = item.data(Qt.UserRole + 6)
        item.setText(f"{self._message_prefix_for_state(sender, group_state)}{text}")
        item.setData(Qt.UserRole + 5, text)

    def _style_message_item(self, item: QListWidgetItem, sender: str, *, compact: bool) -> None:
        group_state = item.data(Qt.UserRole + 6)
        if group_state == "start":
            item.setBackground(QColor("#e6eeff"))
            item.setSizeHint(QSize(0, 40))
            item.setToolTip(f"{sender} grouped message (start)")
        elif group_state == "mid":
            item.setBackground(QColor("#edf4ff"))
            item.setSizeHint(QSize(0, 34))
            item.setToolTip(f"{sender} grouped message (middle)")
        elif group_state == "end":
            item.setBackground(QColor("#e8f1ff"))
            item.setSizeHint(QSize(0, 38))
            item.setToolTip(f"{sender} grouped message (end)")
        elif compact:
            item.setSizeHint(QSize(0, 36))
        if sender == "Me":
            item.setForeground(QColor("#0f3f95"))
        elif sender == "Assistant":
            item.setForeground(QColor("#0f172a"))
        elif sender == "System":
            item.setForeground(QColor("#991b1b"))
        elif sender == "Action":
            item.setForeground(QColor("#1d4ed8"))
            item.setBackground(QColor("#dbeafe"))
            item.setToolTip("클릭해서 복구 액션 실행")
        elif sender == "File":
            file_missing = bool(item.data(Qt.UserRole + 8))
            if file_missing:
                item.setForeground(QColor("#7c2d12"))
                item.setBackground(QColor("#ffedd5"))
            else:
                item.setForeground(QColor("#14532d"))

    def _create_stream_item(self, sender: str) -> QListWidgetItem:
        item, compact = self._build_message_item(sender, "")
        self._style_message_item(item, sender, compact=compact)
        self.result_list.addItem(item)
        self._sync_chat_empty_state()
        return item

    def add_message(
        self,
        sender: str,
        text: str,
        file_path: str | None = None,
        *,
        file_missing: bool = False,
        action_code: str | None = None,
        action_target: str | None = None,
        persist_thread: bool = True,
    ):
        if not text:
            return
        item, compact = self._build_message_item(sender, text)

        if file_path:
            item.setData(Qt.UserRole, file_path)
            item.setData(Qt.UserRole + 8, bool(file_missing))
            item.setData(Qt.UserRole + 9, "missing" if file_missing else "ready")
            if file_missing:
                item.setToolTip(
                    f"파일이 현재 경로에 없습니다. 클릭 시 상위 폴더를 엽니다 ({self._file_open_shortcut_hint()}): {file_path}"
                )
            else:
                item.setToolTip(f"Click to open ({self._file_open_shortcut_hint()}): {file_path}")

        if action_code:
            item.setData(Qt.UserRole + 10, action_code)
            if action_target is not None:
                item.setData(Qt.UserRole + 11, action_target)
            item.setToolTip(f"클릭해서 실행: {text}")

        self._style_message_item(item, sender, compact=compact)

        self.result_list.addItem(item)
        self.result_list.scrollToBottom()
        self._sync_chat_empty_state()
        if persist_thread:
            self._capture_thread_timeline(str(self._active_thread_id or "").strip())

    @staticmethod
    def _normalize_local_file_path(raw_path: object) -> Path | None:
        if not isinstance(raw_path, str):
            return None

        candidate = raw_path.strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
            candidate = candidate[1:-1].strip()
        if not candidate:
            return None
        if "://" in candidate and not candidate.startswith("file://"):
            return None
        if candidate.startswith("file://"):
            parsed = urlparse(candidate)
            candidate = unquote(parsed.path).strip()
            if os.name == "nt" and candidate.startswith("/"):
                candidate = candidate.lstrip("/")
            if not candidate:
                return None

        path = Path(candidate).expanduser()
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            project_candidate = Path(__file__).resolve().parents[1] / path
            if cwd_candidate.exists():
                path = cwd_candidate
            elif project_candidate.exists():
                path = project_candidate
        try:
            return path.resolve(strict=False)
        except OSError:
            return path

    def _reveal_in_finder(self, file_path: Path) -> tuple[bool, str]:
        source = file_path
        resolved_path, resolution, candidates = self._resolve_click_target_file(file_path)
        target = resolved_path
        if target is None and file_path.parent.exists():
            target = file_path.parent
            resolution = f"{resolution}_parent" if resolution else "parent"
        if not target.exists():
            if resolution == "ambiguous" and candidates:
                detail = f"유사 경로 후보가 {len(candidates)}개라 위치를 단정할 수 없습니다."
            else:
                detail = "대상 경로와 상위 폴더를 찾을 수 없습니다."
            self._record_open_event(
                event="reveal_in_finder",
                path=source,
                success=False,
                detail=detail,
                resolution=resolution,
                source_path=source,
                category="ambiguous" if resolution == "ambiguous" else "not_found",
            )
            return False, detail

        if resolved_path is not None and resolved_path != source:
            self._remember_resolved_file_path(source, resolved_path)

        if sys.platform == "darwin":
            reveal_args = ["open", "-R", "--", str(target)] if target.is_file() else ["open", "--", str(target)]
            code, detail = self._run_open_command(reveal_args)
            if code == 0:
                message = (
                    "Finder에서 파일 위치를 열었습니다."
                    if target.is_file()
                    else "Finder에서 폴더를 열었습니다."
                )
                self._record_open_event(
                    event="reveal_in_finder",
                    path=target,
                    success=True,
                    detail="finder reveal",
                    resolution=resolution,
                    source_path=source if target != source else None,
                    category="ok",
                )
                return True, message
            formatted = self._format_open_error_message(detail or "finder reveal failed")
            self._record_open_event(
                event="reveal_in_finder",
                path=target,
                success=False,
                detail=formatted,
                resolution=resolution,
                source_path=source if target != source else None,
            )
            return False, formatted

        opened, info = self._open_local_file(target if target.is_dir() else target.parent)
        if opened:
            return True, info or "대상 위치를 열었습니다."
        return False, info

    def _open_local_file(self, path: Path) -> tuple[bool, str]:
        text_path = str(path)

        if os.name == "nt":
            try:
                os.startfile(text_path)  # type: ignore[attr-defined]
            except (OSError, RuntimeError, ValueError) as exc:
                detail = f"Windows open failed: {exc}"
                self._record_open_event(event="open_windows", path=path, success=False, detail=detail)
                return False, detail
            self._record_open_event(event="open_windows", path=path, success=True)
            return True, ""

        if sys.platform == "darwin":
            detail_candidates: list[str] = []
            default_code, default_detail = self._run_open_command(["open", "--", text_path])
            if default_code == 0:
                self._record_open_event(event="open_darwin_default", path=path, success=True)
                return True, ""
            if default_detail:
                detail_candidates.append(default_detail)

            default_category = self._classify_open_error(default_detail)
            user_canceled = self._is_user_initiated_cancel(default_detail)
            skip_direct_app_open = default_category == "permission" or user_canceled
            if skip_direct_app_open:
                self._record_open_event(
                    event="open_darwin_short_circuit",
                    path=path,
                    success=False,
                    detail=f"default_failed_category={default_category}; user_canceled={user_canceled}",
                    category=default_category,
                )

            # Fallback for document files where default app mapping can be broken.
            if not skip_direct_app_open and path.suffix.lower() in {
                ".pdf",
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".txt",
                ".md",
            }:
                preview_code, preview_detail = self._run_open_command(["open", "-a", "Preview", "--", text_path])
                if preview_code == 0:
                    self._record_open_event(
                        event="open_darwin_preview_fallback",
                        path=path,
                        success=True,
                        detail="default app failed; opened in Preview",
                    )
                    return True, "기본 앱 열기에 실패해 Preview로 문서를 열었습니다."
                if preview_detail:
                    detail_candidates.append(preview_detail)
            # Keep Qt fallback as last attempt because some environments rely on Qt URL handling.
            if not skip_direct_app_open and QDesktopServices.openUrl(QUrl.fromLocalFile(text_path)):
                self._record_open_event(event="open_darwin_qt_fallback", path=path, success=True)
                return True, ""

            is_file_target = path.is_file() or bool(path.suffix)
            reveal_args = ["open", "-R", "--", text_path] if is_file_target else ["open", "--", text_path]
            reveal_code, reveal_detail = self._run_open_command(reveal_args)
            if reveal_code == 0:
                detail = (
                    "permission/canceled default open; revealed in Finder"
                    if skip_direct_app_open
                    else "default app failed; revealed in Finder"
                )
                self._record_open_event(
                    event="open_darwin_reveal_fallback",
                    path=path,
                    success=True,
                    detail=detail,
                )
                if skip_direct_app_open:
                    return True, "열기 취소/권한 오류로 Finder에서 파일 위치를 열었습니다."
                return True, "기본 앱 열기에 실패해 Finder에서 파일 위치를 열었습니다."
            if reveal_detail:
                detail_candidates.append(reveal_detail)

            parent = path.parent
            if parent.exists():
                parent_code, parent_detail = self._run_open_command(["open", "--", str(parent)])
                if parent_code == 0:
                    detail = (
                        f"permission/canceled default open; opened parent={parent}"
                        if skip_direct_app_open
                        else f"default app failed; opened parent={parent}"
                    )
                    self._record_open_event(
                        event="open_darwin_parent_fallback",
                        path=path,
                        success=True,
                        detail=detail,
                    )
                    if skip_direct_app_open:
                        return True, "열기 취소/권한 오류로 상위 폴더를 열었습니다."
                    return True, "기본 앱 열기에 실패해 상위 폴더를 열었습니다."
                if parent_detail:
                    detail_candidates.append(parent_detail)
            detail = next((entry for entry in detail_candidates if entry), "") or "open command failed"
            formatted = self._format_open_error_message(detail)
            self._record_open_event(
                event="open_darwin_failed",
                path=path,
                success=False,
                detail=formatted,
            )
            return False, formatted

        proc = subprocess.run(["xdg-open", text_path], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            self._record_open_event(event="open_linux_xdg", path=path, success=True)
            return True, ""
        if QDesktopServices.openUrl(QUrl.fromLocalFile(text_path)):
            self._record_open_event(event="open_linux_qt_fallback", path=path, success=True)
            return True, ""
        detail = (proc.stderr or proc.stdout or "").strip() or "xdg-open failed"
        formatted = self._format_open_error_message(detail)
        self._record_open_event(event="open_linux_failed", path=path, success=False, detail=formatted)
        return False, formatted

    @staticmethod
    def _classify_open_error(raw_detail: str) -> str:
        detail = (raw_detail or "").strip()
        lower = detail.lower()
        if any(
            token in lower
            for token in (
                "operation not permitted",
                "not authorized",
                "permission denied",
                "권한",
            )
        ):
            return "permission"
        if any(
            token in lower
            for token in (
                "user canceled",
                "user cancelled",
                "cancelled",
                "canceled",
                "operation canceled",
                "operation cancelled",
                "작업이 취소",
                "취소되었습니다",
                "사용자가 취소",
                "취소",
            )
        ):
            return "canceled"
        if any(
            token in lower
            for token in (
                "no application",
                "no associated application",
                "unable to find application",
                "application not found",
                "lsopenurlswithrole",
                "기본 앱 연결",
            )
        ):
            return "association"
        if any(token in lower for token in ("no such file", "does not exist", "찾을 수 없습니다")):
            return "not_found"
        return "generic"

    @staticmethod
    def _is_user_initiated_cancel(raw_detail: str) -> bool:
        lower = (raw_detail or "").strip().lower()
        return any(token in lower for token in ("user canceled", "user cancelled", "사용자가 취소"))

    @staticmethod
    def _format_open_error_message(raw_detail: str) -> str:
        detail = (raw_detail or "").strip()
        category = LauncherWindow._classify_open_error(detail)
        if category == "permission":
            return "권한 문제로 파일을 열 수 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 파일 및 폴더 권한을 확인하세요."
        if category == "canceled":
            return "파일 열기가 취소되었습니다. Finder에서 직접 열기 또는 기본 앱 연결을 확인하세요."
        if category == "association":
            return "기본 앱 연결을 찾을 수 없습니다. Finder > 정보 가져오기 > 다음으로 열기에서 앱을 지정하세요."
        if category == "not_found":
            return "대상 경로를 찾을 수 없습니다. 파일이 이동/삭제되었는지 확인하세요."
        if detail:
            return detail
        return "알 수 없는 오류로 파일을 열 수 없습니다."

    def on_result_item_clicked(self, item: QListWidgetItem):
        if bool(item.data(Qt.UserRole + 12)) or bool(item.data(Qt.UserRole + 13)):
            # Action cards are button-driven; row click should not emit errors.
            return
        action_code = item.data(Qt.UserRole + 10)
        if isinstance(action_code, str) and action_code.strip():
            self._run_timeline_action(action_code.strip(), item.data(Qt.UserRole + 11))
            return
        file_path = self._normalize_local_file_path(item.data(Qt.UserRole))
        if file_path is None:
            self.add_message("System", "파일 경로를 해석할 수 없습니다.")
            return
        resolved_path, resolution, candidates = self._resolve_click_target_file(file_path)
        if resolved_path is None:
            if resolution == "ambiguous":
                self._record_open_event(
                    event="resolve_ambiguous_candidates",
                    path=file_path,
                    success=False,
                    resolution=resolution,
                    detail=f"candidate_count={len(candidates)}",
                    category="ambiguous",
                )
                self.add_message("System", f"유사 문서 후보가 여러 개라 자동 열기를 중단했습니다: {file_path.name}")
                self._append_similar_file_candidates(file_path, candidates=candidates)
                self._append_open_recovery_actions(file_path)
                return
            parent = file_path.parent
            if parent.exists():
                opened_parent, parent_info = self._open_local_file(parent)
                if opened_parent:
                    self._record_open_event(
                        event="open_parent_due_missing_file",
                        path=parent,
                        source_path=file_path,
                        success=True,
                        resolution=resolution,
                    )
                    self._set_status(f"Opened parent: {parent.name}", "ok")
                    self.add_message("System", f"파일을 찾을 수 없어 상위 폴더를 열었습니다: {parent}")
                    if parent_info:
                        self.add_message("System", parent_info)
                else:
                    self._record_open_event(
                        event="open_parent_due_missing_file",
                        path=parent,
                        source_path=file_path,
                        success=False,
                        resolution=resolution,
                        detail=parent_info,
                    )
                    self.add_message("System", f"파일을 찾을 수 없습니다: {file_path}")
                    self._append_open_failure_guidance(
                        file_path=file_path,
                        error_message=parent_info,
                        parent_attempt=True,
                        parent_path=parent,
                    )
            else:
                self.add_message("System", f"파일을 찾을 수 없습니다: {file_path}")
                self._append_open_recovery_actions(file_path)
            return

        if resolution == "cached":
            self.add_message("System", f"복구된 경로를 사용해 문서를 엽니다: {resolved_path.name}")
        elif resolution == "similar":
            self.add_message("System", f"원본 경로를 찾지 못해 유사 경로 후보를 엽니다: {resolved_path.name}")

        opened, info = self._open_local_file(resolved_path)
        if opened:
            status_text = f"Opened file: {resolved_path.name}"
            if info:
                if "Finder" in info:
                    status_text = "Revealed in Finder"
                elif "상위 폴더" in info:
                    status_text = f"Opened parent: {resolved_path.parent.name}"
            self._set_status(status_text, "ok")
            self._record_open_event(
                event="open_selected_item",
                path=resolved_path,
                source_path=file_path if resolved_path != file_path else None,
                success=True,
                resolution=resolution,
                detail=info,
            )
            if resolution in {"cached", "similar"} and resolved_path != file_path:
                self.add_message("System", f"원본 경로 대신 복구 경로를 열었습니다: {resolved_path}")
                self._remember_resolved_file_path(file_path, resolved_path)
            if info:
                self.add_message("System", info)
            return
        if not opened:
            self._record_open_event(
                event="open_selected_item",
                path=resolved_path,
                source_path=file_path if resolved_path != file_path else None,
                success=False,
                resolution=resolution,
                detail=info,
            )
            if resolution in {"cached", "similar"} and resolved_path != file_path:
                self.add_message("System", f"복구 경로 열기 실패: {resolved_path}")
            self._append_open_failure_guidance(file_path=file_path, error_message=info)

    def cleanup(self):
        pass

    def keyPressEvent(self, event):
        shortcut_mod = bool(event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        shift_mod = bool(event.modifiers() & Qt.ShiftModifier)
        if shortcut_mod and event.key() == Qt.Key_K:
            event.accept()
            self._focus_thread_search()
            return
        if shortcut_mod and event.key() == Qt.Key_L:
            event.accept()
            self._focus_composer_input()
            return
        if shortcut_mod and event.key() == Qt.Key_J:
            event.accept()
            self._focus_chat_timeline()
            return
        if shortcut_mod and event.key() == Qt.Key_M:
            event.accept()
            self._cycle_response_mode()
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_Down:
            event.accept()
            self._move_chat_selection(1)
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_Up:
            event.accept()
            self._move_chat_selection(-1)
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_P:
            event.accept()
            self._open_selected_timeline_parent()
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_R:
            event.accept()
            self._reveal_selected_timeline_item()
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_O:
            event.accept()
            self._copy_selected_timeline_file_path()
            return
        if shortcut_mod and event.key() == Qt.Key_O:
            event.accept()
            self._open_selected_timeline_item()
            return
        if shortcut_mod and shift_mod and event.key() == Qt.Key_C:
            event.accept()
            self._cite_selected_timeline_item()
            return
        if shortcut_mod and event.key() == Qt.Key_Down:
            event.accept()
            self._move_thread_selection(1)
            return
        if shortcut_mod and event.key() == Qt.Key_Up:
            event.accept()
            self._move_thread_selection(-1)
            return
        if event.key() == Qt.Key_Escape and self.thread_search.hasFocus():
            if self.thread_search.text().strip():
                self.thread_search.clear()
            self.input_field.setFocus()
            return
        if event.key() == Qt.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def show_and_activate(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.input_field.setFocus()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            path_str = url.toLocalFile()
            path = Path(path_str)
            if path.is_dir():
                self.policy_registry.add_folder(path)
                self._set_status(f"Active Folder: {path.name}", "ok")
                QTimer.singleShot(2500, lambda: self._set_status("Ready", "ok"))
            elif path.is_file():
                self.input_field.insertPlainText(f' "{path_str}" ')

    def _open_smart_folder_manager(self):
        dlg = SmartFolderManagerDialog(
            self.policy_registry,
            self,
            mode_callback=self._open_mode_profile_editor,
            runtime_policy_callback=self._open_runtime_policy_editor,
        )
        dlg.exec()

    def open_settings(self):
        hub = SettingsHubDialog(
            smart_folder_count=len(self.policy_registry.list_folders()),
            current_mode=self._response_mode,
            runtime_policy=self._runtime_policy,
            open_folders_callback=self._open_smart_folder_manager,
            open_mode_callback=self._open_mode_profile_editor,
            open_runtime_callback=self._open_runtime_policy_editor,
            on_runtime_policy_applied=self._on_runtime_policy_updated,
            on_mode_profile_applied=self._on_mode_profile_updated,
            parent=self,
        )
        hub.exec()

    def open_tasks(self):
        dlg = TaskManagerWindow(self)
        dlg.exec()

    def open_meeting_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "오디오 파일 선택",
            str(Path.home()),
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac *.webm);;All Files (*)",
        )
        if file_path:
            self.input_field.setPlainText(f'/meeting "{file_path}"')
            self.on_submit()

    def open_photo_dialog(self):
        dialog = PhotoGalleryDialog(parent=self)
        dialog.exec()
