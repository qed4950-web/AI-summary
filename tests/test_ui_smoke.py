import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QListWidgetItem, QPushButton

    from core.config.desktop_runtime_policy import load_desktop_runtime_policy, save_desktop_runtime_policy
    from core.config.mode_profiles import (
        DEFAULT_MODE_PROFILES,
        MODE_ORDER,
    )
    from core.config.mode_profiles import (
        load_mode_profiles as _core_load_mode_profiles,
    )
    from core.config.mode_profiles import (
        save_mode_profiles as _core_save_mode_profiles,
    )
    from desktop_app.tasks_ui import TaskManagerWindow
    from desktop_app.ui import LauncherWindow, ModeProfileDialog, RuntimePolicyDialog, SettingsHubDialog
except ImportError:
    pytest.skip("PySide6 not available or headless", allow_module_level=True)

load_mode_profiles = _core_load_mode_profiles


def _qt_ui_runtime_available() -> bool:
    if os.getenv("DESKTOP_UI_SMOKE_FORCE", "").strip() == "1":
        return True
    probe_env = dict(os.environ)
    probe_env.setdefault("QT_QPA_PLATFORM", "offscreen")
    probe_cmd = [
        sys.executable,
        "-c",
        (
            "import sys\n"
            "from PySide6.QtWidgets import QApplication\n"
            "app = QApplication.instance() or QApplication(sys.argv)\n"
            "print('qt-ok')\n"
        ),
    ]
    try:
        completed = subprocess.run(
            probe_cmd,
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and "qt-ok" in (completed.stdout or "")


_QT_UI_RUNTIME_OK = _qt_ui_runtime_available()

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _QT_UI_RUNTIME_OK,
        reason="Qt UI runtime unavailable (QApplication bootstrap failed)",
    ),
]


@pytest.fixture(autouse=True)
def _isolated_runtime_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_PATH", str(tmp_path / "desktop_runtime_policy.json"))
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_HISTORY_PATH", str(tmp_path / "desktop_runtime_policy_history.jsonl"))
    monkeypatch.setenv("DESKTOP_THREAD_HISTORY_PATH", str(tmp_path / "desktop_thread_history.json"))
    monkeypatch.setenv("DESKTOP_THREAD_TIMELINE_PATH", str(tmp_path / "desktop_thread_timelines.json"))
    monkeypatch.setenv("DESKTOP_FILE_RESOLUTION_CACHE_PATH", str(tmp_path / "desktop_file_resolution_cache.json"))
    monkeypatch.setenv("DESKTOP_SIMILAR_LOOKUP_CACHE_PATH", str(tmp_path / "desktop_similar_lookup_cache.json"))
    monkeypatch.setenv("DESKTOP_OPEN_EVENT_LOG_PATH", str(tmp_path / "desktop_file_open_events.jsonl"))
    mode_profiles_path = tmp_path / "desktop_mode_profiles.json"
    _core_save_mode_profiles(
        {mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER},
        mode_profiles_path,
    )
    monkeypatch.setattr(
        "desktop_app.ui.load_mode_profiles",
        lambda: _core_load_mode_profiles(mode_profiles_path),
    )
    monkeypatch.setattr(
        "desktop_app.ui.save_mode_profiles",
        lambda profiles: _core_save_mode_profiles(profiles, mode_profiles_path),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "load_mode_profiles",
        lambda: _core_load_mode_profiles(mode_profiles_path),
    )
    SettingsHubDialog._session_history_filter_state = {"source": "all", "period": "all_time"}
    for env_name in (
        "DESKTOP_MASK_PII",
        "DESKTOP_MAX_FILE_LINKS",
        "DESKTOP_MAX_REFERENCE_LINKS",
        "DESKTOP_MAX_RESPONSE_CHARS",
        "DESKTOP_MAX_SUGGESTION_CHARS",
    ):
        monkeypatch.delenv(env_name, raising=False)


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_ui_instantiation():
    QApplication.instance() or QApplication(sys.argv)

    window = TaskManagerWindow()
    try:
        assert window is not None
        assert window.windowTitle() == "Task Center"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_reference_layout_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        assert window.sidebar.width() >= 280
        assert window.header_title.text() == "Reply to 0f greeting"
        assert window.input_field.placeholderText() == "무엇이든 부탁하세요"
        assert window.composer_panel.objectName() == "ComposerPanel"
        assert window.header_open_btn.text().endswith("Open")
        assert window.thread_search.placeholderText().startswith("검색")
        assert window.thread_list.count() >= 5
        assert window.mode_button.text() == "Auto"
        assert window.mode_button.toolTip().startswith("Auto:")
        assert "검색" in window.shortcut_hint.text()
        assert "action=auto" in window.mode_hint.text()
        assert "privacy=mask" in window.mode_hint.text()
        assert "refs<=" in window.mode_hint.text()
        assert "file-links<=" in window.mode_hint.text()
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_submit_and_response_state_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        assert window.chat_empty_state.isHidden() is False
        window.input_field.setPlainText("hello")
        window.on_submit()

        assert window._query_in_flight is True
        assert window.result_list.count() >= 1
        assert window.chat_empty_state.isHidden() is True

        window.handle_stream_update("진행 중 ")
        window.handle_stream_update("응답")
        assert window.streaming_item is not None
        assert "진행 중 응답" in window.streaming_item.text()
        window.handle_stream_update(" test.user@example.com")
        assert "[REDACTED_EMAIL]" in window.streaming_item.text()

        window.handle_response("done")
        assert window._query_in_flight is False
        assert window.streaming_item is None
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_thread_row_and_composer_tuning_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        first_item = window.thread_list.item(0)
        first_row = window.thread_list.itemWidget(first_item)
        assert first_row is not None
        assert first_row.property("active") is True

        window.thread_list.setCurrentRow(1)
        second_item = window.thread_list.item(1)
        second_row = window.thread_list.itemWidget(second_item)
        assert second_row is not None
        assert second_row.property("active") is True

        assert window.composer_panel.maximumWidth() >= 900
        assert window.btn_send.text() == "▲"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_thread_search_and_shortcut_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        total_threads = window.thread_list.count()
        assert window.thread_empty_state.isHidden() is True

        window.thread_search.setText("프로젝트")
        visible_threads = sum(not window.thread_list.item(i).isHidden() for i in range(total_threads))
        assert 0 < visible_threads < total_threads
        assert window.thread_list.currentRow() >= 0

        window.thread_search.setText("zz-no-hit")
        assert window.thread_list.currentRow() == -1
        assert window.thread_empty_state.isHidden() is False

        window.thread_search.setText("focus-check")
        window.input_field.setFocus()
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_K, Qt.ControlModifier))
        assert window.thread_search.selectedText() == "focus-check"

        window.thread_search.clear()
        assert window.thread_empty_state.isHidden() is True
        window.thread_list.setCurrentRow(0)
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up, Qt.ControlModifier))
        assert window.thread_list.currentRow() == total_threads - 1
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Down, Qt.ControlModifier))
        assert window.thread_list.currentRow() == 0

        window.input_field.setPlainText("focus-composer")
        window.thread_search.setFocus()
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_L, Qt.ControlModifier))
        assert window.input_field.textCursor().hasSelection() is True
        assert window.input_field.textCursor().selectedText() == "focus-composer"
        assert window.input_field.toPlainText() == "focus-composer"

        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_M, Qt.ControlModifier))
        assert window.mode_button.text() == "Instant"
        assert "Instant" in window.model_chip.text()
        assert "top-k=3" in window.mode_hint.text()

        window.add_message("Assistant", "timeline check")
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_J, Qt.ControlModifier))
        assert window.result_list.currentRow() == window.result_list.count() - 1
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_thread_sidebar_persistence_and_show_more_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    history_path = tmp_path / "desktop_thread_history.json"
    seeded_threads = []
    now = datetime.now(timezone.utc)
    for idx in range(8):
        seeded_threads.append(
            {
                "id": f"thread-seed-{idx}",
                "title": f"프로젝트 스레드 {idx}",
                "updated_at": (now - timedelta(hours=idx)).isoformat(),
            }
        )
    history_path.write_text(json.dumps(seeded_threads, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    window = LauncherWindow()
    try:
        assert window.thread_list.count() == window._THREAD_PAGE_SIZE
        assert window.btn_show_more_threads.isEnabled() is True
        window._show_more_threads()
        assert window.thread_list.count() == len(seeded_threads)
        assert window.btn_show_more_threads.isEnabled() is False

        window._start_new_thread()
        window.input_field.setPlainText("스레드 저장 검증 질문")
        window.on_submit()
    finally:
        window.close()

    reloaded = LauncherWindow()
    try:
        row_titles: list[str] = []
        for idx in range(reloaded.thread_list.count()):
            item = reloaded.thread_list.item(idx)
            row = reloaded.thread_list.itemWidget(item)
            if row is None:
                continue
            title_widget = row.findChild(QLabel, "ThreadTitle")
            if title_widget and title_widget.text():
                row_titles.append(title_widget.text())
        assert any("스레드 저장 검증 질문" in title for title in row_titles)
    finally:
        reloaded.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_thread_timeline_restore_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        window._start_new_thread()
        window.input_field.setPlainText("alpha-query")
        window.on_submit()
        window.handle_response("alpha-answer")
        alpha_thread = window._active_thread_id

        window._start_new_thread()
        window.input_field.setPlainText("beta-query")
        window.on_submit()
        window.handle_response("beta-answer")
        beta_thread = window._active_thread_id

        assert alpha_thread
        assert beta_thread
        assert alpha_thread != beta_thread

        alpha_row = -1
        for idx in range(window.thread_list.count()):
            item = window.thread_list.item(idx)
            if str(item.data(Qt.UserRole + 20) or "") == alpha_thread:
                alpha_row = idx
                break
        assert alpha_row >= 0

        window.thread_list.setCurrentRow(alpha_row)
        rendered = [window.result_list.item(i).text() for i in range(window.result_list.count())]
        assert any("alpha-query" in line for line in rendered)
        assert any("alpha-answer" in line for line in rendered)
        assert all("beta-query" not in line for line in rendered)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_timeline_keyboard_navigation_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        window.add_message("Assistant", "one")
        window.add_message("Assistant", "two")
        window.add_message("File", "artifact", file_path=str(Path(__file__).resolve()))
        assert window.result_list.item(0).text().startswith("┌ ")
        assert window.result_list.item(1).text().startswith("└ ")

        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_J, Qt.ControlModifier))
        assert window.result_list.currentRow() == window.result_list.count() - 1

        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up, Qt.ControlModifier | Qt.ShiftModifier))
        assert window.result_list.currentRow() == window.result_list.count() - 2

        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Down, Qt.ControlModifier | Qt.ShiftModifier))
        assert window.result_list.currentRow() == window.result_list.count() - 1

        with patch.object(window, "on_result_item_clicked") as mock_open:
            window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_O, Qt.ControlModifier))
            assert mock_open.call_count == 1

        window.input_field.setPlainText("draft")
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_C, Qt.ControlModifier | Qt.ShiftModifier))
        quoted = window.input_field.toPlainText()
        assert "draft" in quoted
        assert "> File: artifact" in quoted
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        # quoted and relative path should be normalized
        normalized = window._normalize_local_file_path('"tests/test_ui_smoke.py"')
        assert normalized is not None
        assert normalized.name == "test_ui_smoke.py"
        assert normalized.exists() is True

        file_item = QListWidgetItem("File: smoke")
        file_item.setData(Qt.UserRole, '"tests/test_ui_smoke.py"')
        with patch.object(window, "_open_local_file", return_value=(True, "")) as mock_open:
            before = window.result_list.count()
            window.on_result_item_clicked(file_item)
            assert mock_open.call_count == 1
            opened_path = mock_open.call_args.args[0]
            assert isinstance(opened_path, Path)
            assert opened_path.name == "test_ui_smoke.py"
            assert window.result_list.count() == before

        with patch.object(
            window,
            "_open_local_file",
            return_value=(True, "열기 취소/권한 오류로 Finder에서 파일 위치를 열었습니다."),
        ):
            window.on_result_item_clicked(file_item)
            assert window.status_label.text() == "Revealed in Finder"

        missing_item = QListWidgetItem("File: missing")
        missing_item.setData(Qt.UserRole, "/tmp/ai-summary-missing-file.pdf")
        with (
            patch.object(window, "_resolve_click_target_file", return_value=(None, "missing", [])),
            patch.object(window, "_open_local_file", return_value=(True, "")) as mock_open_parent,
        ):
            before_missing = window.result_list.count()
            window.on_result_item_clicked(missing_item)
            assert mock_open_parent.call_count == 1
            opened_parent = mock_open_parent.call_args.args[0]
            assert isinstance(opened_parent, Path)
            assert str(opened_parent).endswith("/tmp")
            assert window.result_list.count() == before_missing + 1
            assert "상위 폴더를 열었습니다" in window.result_list.item(window.result_list.count() - 1).text()
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_auto_resolves_similar_path_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        candidate = docs_dir / "portfolio.pdf"
        candidate.write_text("stub", encoding="utf-8")

        class _Registry:
            def list_folders(self):
                return [{"path": str(docs_dir)}]

        window.policy_registry = _Registry()  # type: ignore[assignment]

        missing_item = QListWidgetItem("File: stale")
        missing_item.setData(Qt.UserRole, str(tmp_path / "stale" / "portfolio.pdf"))
        with patch.object(window, "_open_local_file", return_value=(True, "")) as mock_open:
            before = window.result_list.count()
            window.on_result_item_clicked(missing_item)
            assert mock_open.call_count == 1
            opened_target = mock_open.call_args.args[0]
            assert isinstance(opened_target, Path)
            assert opened_target.name == "portfolio.pdf"
            tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
            assert any("유사 경로 후보" in text for text in tail_texts)
            assert any("복구 경로" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_ambiguous_candidates_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "portfolio.pdf").write_text("pdf", encoding="utf-8")
        (docs_dir / "portfolio.docx").write_text("docx", encoding="utf-8")

        class _Registry:
            def list_folders(self):
                return [{"path": str(docs_dir)}]

        window.policy_registry = _Registry()  # type: ignore[assignment]

        missing_item = QListWidgetItem("File: stale-ambiguous")
        missing_item.setData(Qt.UserRole, str(tmp_path / "stale" / "portfolio.pdf"))
        with patch.object(window, "_open_local_file", return_value=(True, "")) as mock_open:
            before = window.result_list.count()
            window.on_result_item_clicked(missing_item)
            assert mock_open.call_count == 0
            tail_items = [window.result_list.item(i) for i in range(before, window.result_list.count())]
            tail_texts = [item.text() for item in tail_items]
            assert any("자동 열기를 중단했습니다" in text for text in tail_texts)
            action_items = [
                item
                for item in tail_items
                if str(item.data(Qt.UserRole + 10) or "").strip() == "open_candidate_file"
            ]
            assert len(action_items) >= 2
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_resolution_cache_persistence_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    source = tmp_path / "stale" / "portfolio.pdf"
    resolved = tmp_path / "docs" / "portfolio.pdf"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("stub", encoding="utf-8")

    window = LauncherWindow()
    try:
        window._remember_resolved_file_path(source, resolved)
        cache_file = Path(str(tmp_path / "desktop_file_resolution_cache.json"))
        assert cache_file.exists() is True
    finally:
        window.close()

    reloaded = LauncherWindow()
    try:
        cached = reloaded._cached_resolved_file_path(source)
        assert cached is not None
        assert cached.name == "portfolio.pdf"
        assert cached.exists() is True
    finally:
        reloaded.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_similar_file_lookup_cache_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "portfolio.pdf").write_text("pdf", encoding="utf-8")
    (docs_dir / "portfolio.docx").write_text("docx", encoding="utf-8")

    window = LauncherWindow()
    try:
        with patch.object(window.policy_registry, "list_folders", return_value=[{"path": str(docs_dir)}]):
            real_rglob = Path.rglob
            stats = {"folder_rglob_calls": 0}

            def _counted_rglob(path_obj: Path, pattern: str):
                if path_obj == docs_dir:
                    stats["folder_rglob_calls"] += 1
                return real_rglob(path_obj, pattern)

            with patch("pathlib.Path.rglob", autospec=True, side_effect=_counted_rglob):
                first = window._find_similar_file_candidates(Path("/missing/root/portfolio.pdf"), limit=5)
                second = window._find_similar_file_candidates(Path("/missing/root/portfolio.pdf"), limit=5)
                assert first
                assert second
                # Candidate patterns are 2개이며, 두 번째 조회는 캐시 hit라 폴더 rglob 추가 호출이 없어야 한다.
                assert stats["folder_rglob_calls"] == 2
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_similar_lookup_cache_persistence_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    candidate = docs_dir / "portfolio.pdf"
    candidate.write_text("pdf", encoding="utf-8")

    first = LauncherWindow()
    try:
        with patch.object(first.policy_registry, "list_folders", return_value=[{"path": str(docs_dir)}]):
            ranked = first._find_similar_file_candidates(Path("/missing/root/portfolio.pdf"), limit=5)
            assert ranked
        cache_path = tmp_path / "desktop_similar_lookup_cache.json"
        assert cache_path.exists() is True
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload
    finally:
        first.close()

    second = LauncherWindow()
    try:
        assert second._similar_lookup_cache
        with patch.object(second.policy_registry, "list_folders", return_value=[{"path": str(docs_dir)}]):
            ranked = second._find_similar_file_candidates(Path("/missing/root/portfolio.pdf"), limit=5)
            assert ranked
            assert ranked[0].name == "portfolio.pdf"
    finally:
        second.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_open_event_log_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.QDesktopServices.openUrl", return_value=False),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
            ]
            opened, _info = window._open_local_file(Path("/tmp/ai-summary-open-event.pdf"))
            assert opened is False

        log_path = tmp_path / "desktop_file_open_events.jsonl"
        assert log_path.exists() is True
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines
        payload = json.loads(lines[-1])
        assert payload["event"] == "open_darwin_failed"
        assert payload["success"] is False
        assert payload["category"] == "canceled"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_similar_file_scan_limit_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QApplication.instance() or QApplication(sys.argv)

    monkeypatch.setenv("DESKTOP_SIMILAR_SCAN_MAX", "1")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "portfolio.pdf").write_text("pdf", encoding="utf-8")
    (docs_dir / "portfolio.docx").write_text("docx", encoding="utf-8")

    window = LauncherWindow()
    try:
        with patch.object(window.policy_registry, "list_folders", return_value=[{"path": str(docs_dir)}]):
            ranked = window._find_similar_file_candidates(Path("/missing/root/portfolio.pdf"), limit=5)
            assert len(ranked) == 1
            assert ranked[0].name == "portfolio.pdf"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_handle_response_file_link_parse_and_privacy_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        valid_uri = Path(__file__).resolve().as_uri()
        before = window.result_list.count()
        window.handle_response(
            "연락처 test.user@example.com "
            f"[FILE_LINK:{valid_uri}] "
            f"[FILE_LINK:\"{valid_uri}\"] "
            "[FILE_LINK:http://example.com/a.pdf]"
        )

        assistant_text = window.result_list.item(before).text()
        assert "[REDACTED_EMAIL]" in assistant_text
        file_items = [
            window.result_list.item(i)
            for i in range(before, window.result_list.count())
            if window.result_list.item(i).data(Qt.UserRole)
        ]
        assert len(file_items) == 1
        file_path = str(file_items[0].data(Qt.UserRole))
        assert file_path.endswith("tests/test_ui_smoke.py")
        assert "http://example.com/a.pdf" not in file_path
        tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
        assert any("참조 문서 요약:" in text and "유효하지 않은 링크 1개는 제외했습니다" in text for text in tail_texts)
        assert any("참조 문서 요약:" in text and "중복 링크 1개는 병합했습니다" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_handle_response_legacy_path_link_conversion_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        legacy_path = str(Path(__file__).resolve())
        before = window.result_list.count()
        window.handle_response(f"legacy [FILE_LINK:{legacy_path}] [FILE_LINK:http://example.com/legacy.pdf]")

        file_items = [
            window.result_list.item(i)
            for i in range(before, window.result_list.count())
            if window.result_list.item(i).data(Qt.UserRole)
        ]
        assert len(file_items) == 1
        assert str(file_items[0].data(Qt.UserRole)).endswith("tests/test_ui_smoke.py")
        tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
        assert any(
            "참조 문서 요약:" in text and "레거시 경로 링크 1개를 표준 경로로 변환했습니다" in text
            for text in tail_texts
        )
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_handle_response_link_only_placeholder_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        existing_uri = Path(__file__).resolve().as_uri()
        missing_uri = Path("/tmp/ai-summary-missing-link.pdf").resolve().as_uri()
        before = window.result_list.count()
        window.handle_response(f"[FILE_LINK:{existing_uri}] [FILE_LINK:{missing_uri}]")

        assistant_item = window.result_list.item(before)
        assert "참조 문서만 반환되었습니다" in assistant_item.text()

        file_items = [
            window.result_list.item(i)
            for i in range(before, window.result_list.count())
            if window.result_list.item(i).data(Qt.UserRole)
        ]
        assert len(file_items) == 2
        labels = [item.text() for item in file_items]
        assert any("[missing]" in label for label in labels)
        missing_tooltips = [item.toolTip() for item in file_items if "[missing]" in item.text()]
        assert any("상위 폴더" in tip for tip in missing_tooltips)

        tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
        assert any("참조 문서 요약:" in text and "현재 경로에 없습니다" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_handle_response_file_link_overflow_notice_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        before = window.result_list.count()
        links = " ".join(
            f"[FILE_LINK:{Path(f'/tmp/ai-summary-overflow-{idx}.pdf').resolve().as_uri()}]" for idx in range(12)
        )
        window.handle_response(f"links only {links}")

        file_items = [
            window.result_list.item(i)
            for i in range(before, window.result_list.count())
            if window.result_list.item(i).data(Qt.UserRole)
        ]
        assert len(file_items) == 8

        tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
        assert any("참조 문서 요약:" in text and "총 12개 중 8개 표시" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_error_message_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        permission_msg = window._format_open_error_message("Operation not permitted")
        cancel_msg = window._format_open_error_message("User canceled.")
        local_cancel_msg = window._format_open_error_message("작업이 취소되었습니다.")
        association_msg = window._format_open_error_message("no associated application")
        not_found_msg = window._format_open_error_message("No such file")
        generic_msg = window._format_open_error_message("open command failed")
        assert "권한" in permission_msg
        assert "취소" in cancel_msg
        assert "취소" in local_cancel_msg
        assert "기본 앱" in association_msg
        assert "경로" in not_found_msg
        assert generic_msg == "open command failed"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_open_local_file_macos_preview_fallback_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.QDesktopServices.openUrl", return_value=False),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                type("Proc", (), {"returncode": 1, "stderr": "no associated application", "stdout": ""})(),
                type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
            ]
            opened, info = window._open_local_file(Path("/tmp/ai-summary-test.pdf"))
            assert opened is True
            assert "Preview" in info
            assert mock_run.call_count == 2
            first_args = mock_run.call_args_list[0].args[0]
            second_args = mock_run.call_args_list[1].args[0]
            assert first_args[:2] == ["open", "--"]
            assert second_args[:3] == ["open", "-a", "Preview"]
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_open_local_file_macos_parent_fallback_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.QDesktopServices.openUrl", return_value=False),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
            ]
            opened, info = window._open_local_file(Path("/tmp/ai-summary-test.pdf"))
            assert opened is True
            assert "상위 폴더" in info
            assert mock_run.call_count == 4
            preview_open_args = mock_run.call_args_list[1].args[0]
            assert preview_open_args[:3] == ["open", "-a", "Preview"]
            parent_open_args = mock_run.call_args_list[3].args[0]
            assert parent_open_args[:2] == ["open", "--"]
            assert parent_open_args[2] == "/tmp"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_open_local_file_macos_canceled_short_circuit_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.QDesktopServices.openUrl", return_value=False),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                type("Proc", (), {"returncode": 1, "stderr": "User canceled.", "stdout": ""})(),
                type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
            ]
            opened, info = window._open_local_file(Path("/tmp/ai-summary-test.pdf"))
            assert opened is True
            assert "Finder" in info
            assert mock_run.call_count == 2
            first_args = mock_run.call_args_list[0].args[0]
            second_args = mock_run.call_args_list[1].args[0]
            assert first_args[:2] == ["open", "--"]
            assert second_args[:3] == ["open", "-R", "--"]
            assert all("-a" not in call.args[0] for call in mock_run.call_args_list)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_open_local_file_macos_localized_cancel_preview_fallback_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.QDesktopServices.openUrl", return_value=False),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                type("Proc", (), {"returncode": 1, "stderr": "작업이 취소되었습니다.", "stdout": ""})(),
                type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
            ]
            opened, info = window._open_local_file(Path("/tmp/ai-summary-test.pdf"))
            assert opened is True
            assert "Preview" in info
            assert mock_run.call_count == 2
            first_args = mock_run.call_args_list[0].args[0]
            second_args = mock_run.call_args_list[1].args[0]
            assert first_args[:2] == ["open", "--"]
            assert second_args[:3] == ["open", "-a", "Preview"]
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_failure_guide_card_includes_finder_action_on_macos_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        with patch("desktop_app.ui.sys.platform", "darwin"):
            before = window.result_list.count()
            window._append_open_failure_cta_actions(file_path=Path(__file__).resolve(), category="canceled")
            guide_items = [
                window.result_list.item(i)
                for i in range(before, window.result_list.count())
                if bool(window.result_list.item(i).data(Qt.UserRole + 13))
            ]
            assert guide_items
            card = window.result_list.itemWidget(guide_items[-1])
            assert card is not None
            assert card.findChild(QPushButton, "GuideFinderButton") is not None
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_run_timeline_action_reveal_in_finder_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        target = Path(__file__).resolve()
        with (
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.return_value = type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})()
            before = window.result_list.count()
            window._run_timeline_action("reveal_in_finder", str(target))
            assert mock_run.call_count == 1
            args = mock_run.call_args.args[0]
            assert args[:3] == ["open", "-R", "--"]
            tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
            assert any("Finder에서 파일 위치를 열었습니다" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_reveal_in_finder_recovers_stale_path_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    candidate = docs_dir / "portfolio.pdf"
    candidate.write_text("pdf", encoding="utf-8")
    stale = tmp_path / "stale" / "portfolio.pdf"

    window = LauncherWindow()
    try:
        with (
            patch.object(window.policy_registry, "list_folders", return_value=[{"path": str(docs_dir)}]),
            patch("desktop_app.ui.sys.platform", "darwin"),
            patch("desktop_app.ui.subprocess.run") as mock_run,
        ):
            mock_run.return_value = type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})()
            revealed, info = window._reveal_in_finder(stale)
            assert revealed is True
            assert "Finder" in info
            assert mock_run.call_count == 1
            args = mock_run.call_args.args[0]
            assert args[:3] == ["open", "-R", "--"]
            assert args[3] == str(candidate)
            cached = window._cached_resolved_file_path(stale)
            assert cached is not None
            assert cached.name == "portfolio.pdf"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_mode_profile_dialog_topk_validation_contract():
    QApplication.instance() or QApplication(sys.argv)

    dialog = ModeProfileDialog(load_mode_profiles())
    try:
        desc_control = dialog._editors["Instant"]["description"]
        desc_error = dialog._desc_error_labels.get("Instant")
        status_control = dialog._editors["Instant"]["thinking_status"]
        status_error = dialog._status_error_labels.get("Instant")
        control = dialog._editors["Instant"]["topk"]
        inline_error = dialog._topk_error_labels.get("Instant")
        assert isinstance(desc_control, QLineEdit)
        assert isinstance(desc_error, QLabel)
        assert isinstance(status_control, QLineEdit)
        assert isinstance(status_error, QLabel)
        assert isinstance(control, QLineEdit)
        assert isinstance(inline_error, QLabel)

        control.setText("invalid-topk")
        assert dialog.btn_save.isEnabled() is False
        assert inline_error.isHidden() is False
        assert "양의 정수" in inline_error.text()

        control.setText("4")
        assert dialog.btn_save.isEnabled() is True
        assert inline_error.isHidden() is True

        desc_control.setText("")
        assert dialog.btn_save.isEnabled() is False
        assert desc_error.isHidden() is False
        assert "1-48자" in desc_error.text()
        desc_control.setText("즉시 답변")
        assert desc_error.isHidden() is True

        status_control.setText("a" * 40)
        assert dialog.btn_save.isEnabled() is False
        assert status_error.isHidden() is False
        assert "1-24자" in status_error.text()
        status_control.setText("Thinking fast")
        assert status_error.isHidden() is True
        assert dialog.btn_save.isEnabled() is True
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_runtime_policy_dialog_save_and_launcher_hint_contract():
    QApplication.instance() or QApplication(sys.argv)

    dialog = RuntimePolicyDialog(load_desktop_runtime_policy())
    try:
        dialog.privacy_mask.setChecked(False)
        dialog.max_file_links.setValue(5)
        dialog.max_reference_links.setValue(4)
        dialog.max_response_chars.setValue(18000)
        dialog.max_suggestion_chars.setValue(88)
        dialog._save_and_close()
    finally:
        dialog.close()

    policy = load_desktop_runtime_policy()
    assert policy["privacy_mask_enabled"] is False
    assert policy["max_file_links"] == 5
    assert policy["max_reference_links"] == 4
    assert policy["max_response_chars"] == 18000
    assert policy["max_suggestion_chars"] == 88

    window = LauncherWindow()
    try:
        assert "privacy=raw" in window.mode_hint.text()
        assert "refs<=4" in window.mode_hint.text()
        assert "file-links<=5" in window.mode_hint.text()
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_dialog_navigation_contract():
    QApplication.instance() or QApplication(sys.argv)

    calls = {"folders": 0, "mode": 0, "runtime": 0}
    applied = {"policy": 0}
    applied_mode = {"profile": 0}

    dialog = SettingsHubDialog(
        smart_folder_count=2,
        current_mode="Thinking",
        runtime_policy={
            "privacy_mask_enabled": False,
            "max_reference_links": 4,
            "max_file_links": 6,
            "max_response_chars": 20000,
            "max_suggestion_chars": 120,
        },
        open_folders_callback=lambda: calls.__setitem__("folders", calls["folders"] + 1),
        open_mode_callback=lambda: calls.__setitem__("mode", calls["mode"] + 1),
        open_runtime_callback=lambda: calls.__setitem__("runtime", calls["runtime"] + 1),
        on_runtime_policy_applied=lambda: applied.__setitem__("policy", applied["policy"] + 1),
        on_mode_profile_applied=lambda: applied_mode.__setitem__("profile", applied_mode["profile"] + 1),
    )
    try:
        assert "Smart folders: 2" in dialog.summary_folders.text()
        assert "Current mode: Thinking" in dialog.summary_mode.text()
        assert "privacy=raw" in dialog.summary_policy.text()
        assert "refs<=4" in dialog.summary_policy.text()

        dialog.inline_mode_selector.setCurrentText("Pro")
        dialog.inline_mode_action.setCurrentText("search")
        dialog.inline_mode_topk.setText("auto")
        assert dialog.btn_apply_inline_mode.isEnabled() is False
        assert "search" in dialog.inline_mode_error.text()
        dialog.inline_mode_topk.setText("invalid")
        assert dialog.btn_apply_inline_mode.isEnabled() is False
        assert "Top-k" in dialog.inline_mode_error.text()
        dialog.inline_mode_topk.setText("9")
        assert dialog.btn_apply_inline_mode.isEnabled() is True
        dialog.inline_mode_tokens.setValue(1600)
        dialog.inline_mode_temp.setValue(0.31)
        dialog.btn_apply_inline_mode.click()
        mode_profiles = load_mode_profiles()
        assert mode_profiles["Pro"]["force_action"] == "search"
        assert mode_profiles["Pro"]["topk"] == 9
        assert mode_profiles["Pro"]["llm_max_new_tokens"] == 1600
        assert mode_profiles["Pro"]["llm_temperature"] == pytest.approx(0.31)
        assert "Mode preset saved (Pro)" in dialog.inline_mode_status.text()
        assert "Mode preset saved (Pro)" in dialog.hub_status.text()
        assert applied_mode == {"profile": 1}

        dialog.inline_privacy_mask.setChecked(True)
        dialog.inline_max_refs.setValue(9)
        dialog.inline_max_links.setValue(10)
        dialog.btn_apply_inline_policy.click()
        policy = load_desktop_runtime_policy()
        assert policy["privacy_mask_enabled"] is True
        assert policy["max_reference_links"] == 9
        assert policy["max_file_links"] == 10
        assert "Inline policy saved" in dialog.inline_status.text()
        assert "Inline policy saved" in dialog.hub_status.text()
        assert "settings_inline_policy_apply" in dialog.history_box.toPlainText()
        assert applied == {"policy": 1}
        dialog.btn_folders.click()
        dialog.btn_mode.click()
        dialog.btn_runtime.click()
        assert calls == {"folders": 1, "mode": 1, "runtime": 1}
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_history_restore_contract():
    QApplication.instance() or QApplication(sys.argv)

    applied = {"policy": 0}
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": False,
            "max_reference_links": 3,
            "max_file_links": 5,
            "max_response_chars": 16000,
            "max_suggestion_chars": 90,
        }
    )
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_reference_links": 11,
            "max_file_links": 12,
            "max_response_chars": 36000,
            "max_suggestion_chars": 180,
        }
    )

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
        on_runtime_policy_applied=lambda: applied.__setitem__("policy", applied["policy"] + 1),
    )
    try:
        assert dialog.history_selector.count() >= 2
        dialog.history_selector.setCurrentIndex(1)
        assert "Preview diff:" in dialog.history_preview.toPlainText()
        assert dialog.btn_restore_history.isEnabled() is False
        assert "confirm preview" in dialog.history_status.text().lower()
        dialog._restore_selected_history_policy()
        not_restored = load_desktop_runtime_policy()
        assert not_restored["privacy_mask_enabled"] is True
        assert "확인 체크" in dialog.history_status.text()
        dialog.history_confirm.setChecked(True)
        assert dialog.btn_restore_history.isEnabled() is True
        dialog.btn_restore_history.click()
        restored = load_desktop_runtime_policy()
        assert restored["privacy_mask_enabled"] is False
        assert restored["max_reference_links"] == 3
        assert restored["max_file_links"] == 5
        assert "Selected policy restored" in dialog.history_status.text()
        assert "Selected policy restored" in dialog.hub_status.text()
        assert "settings_history_restore" in dialog.history_box.toPlainText()
        assert dialog.history_confirm.isChecked() is False
        assert applied == {"policy": 1}
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_history_filter_contract():
    QApplication.instance() or QApplication(sys.argv)

    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_reference_links": 6,
            "max_file_links": 7,
            "max_response_chars": 25000,
            "max_suggestion_chars": 120,
        },
        source="settings_inline_policy_apply",
    )
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": False,
            "max_reference_links": 4,
            "max_file_links": 6,
            "max_response_chars": 18000,
            "max_suggestion_chars": 90,
        },
        source="runtime_policy_dialog_save",
    )

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        absolute_indexes = [
            idx
            for idx in range(dialog.history_period_filter.count())
            if str(dialog.history_period_filter.itemData(idx)).startswith("absolute:")
        ]
        assert absolute_indexes
        dialog.history_period_filter.setCurrentIndex(absolute_indexes[0])
        assert "Since " in dialog.history_period_filter.currentText()

        source_index = dialog.history_source_filter.findData("settings_inline_policy_apply")
        assert source_index >= 0
        dialog.history_source_filter.setCurrentIndex(source_index)
        source_filtered = dialog.history_box.toPlainText()
        assert "settings_inline_policy_apply" in source_filtered
        assert "runtime_policy_dialog_save" not in source_filtered

        no_result_index = dialog.history_source_filter.findData("settings_history_restore")
        assert no_result_index >= 0
        dialog.history_source_filter.setCurrentIndex(no_result_index)
        assert "No runtime policy history for current filters." in dialog.history_box.toPlainText()
        assert "filter result" in dialog.history_status.text()

        all_index = dialog.history_source_filter.findData("all")
        assert all_index >= 0
        dialog.history_source_filter.setCurrentIndex(all_index)
        assert dialog.history_selector.count() >= 1
        assert "confirm preview" in dialog.history_status.text().lower()
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_history_filter_custom_datetime_contract():
    QApplication.instance() or QApplication(sys.argv)

    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_reference_links": 6,
            "max_file_links": 7,
            "max_response_chars": 25000,
            "max_suggestion_chars": 120,
        },
        source="settings_inline_policy_apply",
    )

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        dialog.history_custom_period_input.setText("bad-input")
        dialog.btn_apply_history_custom_period.click()
        assert bool(dialog.history_custom_period_input.property("invalid")) is True
        assert "YYYY-MM-DD HH:MM" in dialog.history_status.text()

        dialog.history_custom_period_input.setText("2025-01-01 00:00")
        dialog.btn_apply_history_custom_period.click()
        assert bool(dialog.history_custom_period_input.property("invalid")) is False
        assert str(dialog.history_period_filter.currentData()).startswith("absolute:")
        assert "Since custom" in dialog.history_period_filter.currentText()
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_history_filter_session_restore_contract():
    QApplication.instance() or QApplication(sys.argv)

    first = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        source_index = first.history_source_filter.findData("runtime_policy_dialog_save")
        assert source_index >= 0
        first.history_source_filter.setCurrentIndex(source_index)
        absolute_indexes = [
            idx
            for idx in range(first.history_period_filter.count())
            if str(first.history_period_filter.itemData(idx)).startswith("absolute:")
        ]
        assert absolute_indexes
        first.history_period_filter.setCurrentIndex(absolute_indexes[-1])
        saved_period_token = str(first.history_period_filter.currentData())
    finally:
        first.close()

    second = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        assert second.history_source_filter.currentData() == "runtime_policy_dialog_save"
        assert str(second.history_period_filter.currentData()) == saved_period_token
    finally:
        second.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_history_filter_stale_absolute_restore_contract():
    QApplication.instance() or QApplication(sys.argv)

    stale_token = f"absolute:{datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat()}"
    SettingsHubDialog._session_history_filter_state = {"source": "all", "period": stale_token}

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        assert str(dialog.history_period_filter.currentData()) == stale_token
        assert "Since custom" in dialog.history_period_filter.currentText()
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_missing_file_item_accessibility_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        missing_path = "/tmp/ai-summary-missing-accessibility.pdf"
        window.add_message("File", "[missing] ai-summary-missing-accessibility.pdf", file_path=missing_path, file_missing=True)
        item = window.result_list.item(window.result_list.count() - 1)
        assert item.data(Qt.UserRole + 8) is True
        assert item.data(Qt.UserRole + 9) == "missing"
        assert window._file_open_shortcut_hint() in item.toolTip()
        assert "[missing]" in item.text()
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_failure_recovery_actions_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        existing_path = str(Path(__file__).resolve())
        file_item = QListWidgetItem("File: recover")
        file_item.setData(Qt.UserRole, existing_path)
        with patch.object(window, "_open_local_file", return_value=(False, "open failed")):
            before = window.result_list.count()
            window.on_result_item_clicked(file_item)
            tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
            assert any("파일 열기 실패: open failed" in text for text in tail_texts)
            assert any(
                "다음 동작:" in text and "Shift+P" in text and "Shift+R" in text and "Shift+O" in text
                for text in tail_texts
            )
            card_items = [
                window.result_list.item(i)
                for i in range(before, window.result_list.count())
                if bool(window.result_list.item(i).data(Qt.UserRole + 12))
            ]
            assert len(card_items) == 1
            card_widget = window.result_list.itemWidget(card_items[0])
            assert card_widget is not None
            retry_button = card_widget.findChild(QPushButton, "RecoveryRetryButton")
            parent_button = card_widget.findChild(QPushButton, "RecoveryParentButton")
            reveal_button = card_widget.findChild(QPushButton, "RecoveryRevealButton")
            copy_button = card_widget.findChild(QPushButton, "RecoveryCopyButton")
            assert retry_button is not None
            assert parent_button is not None
            assert reveal_button is not None
            assert copy_button is not None
            assert card_widget.focusProxy() is retry_button
            assert retry_button.isDefault() is True
            assert retry_button.accessibleName() == "Retry open file"
            assert parent_button.accessibleName() == "Open parent folder"
            assert reveal_button.accessibleName() == "Reveal in Finder"
            assert copy_button.accessibleName() == "Copy file path"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_failure_guidance_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        existing_path = str(Path(__file__).resolve())
        file_item = QListWidgetItem("File: recover-cancel")
        file_item.setData(Qt.UserRole, existing_path)
        with patch.object(
            window,
            "_open_local_file",
            return_value=(False, "파일 열기가 취소되었습니다. Finder에서 직접 열기 또는 기본 앱 연결을 확인하세요."),
        ):
            before = window.result_list.count()
            window.on_result_item_clicked(file_item)
            tail_texts = [window.result_list.item(i).text() for i in range(before, window.result_list.count())]
            assert any("파일 열기 실패:" in text for text in tail_texts)
            assert any("취소 원인 점검:" in text for text in tail_texts)
            assert any("다음 동작:" in text for text in tail_texts)
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_failure_cta_actions_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        existing_path = str(Path(__file__).resolve())
        file_item = QListWidgetItem("File: recover-cta")
        file_item.setData(Qt.UserRole, existing_path)
        with patch.object(
            window,
            "_open_local_file",
            return_value=(False, "파일 열기가 취소되었습니다. Finder에서 직접 열기 또는 기본 앱 연결을 확인하세요."),
        ):
            before = window.result_list.count()
            window.on_result_item_clicked(file_item)
            card_items = [
                window.result_list.item(i)
                for i in range(before, window.result_list.count())
                if bool(window.result_list.item(i).data(Qt.UserRole + 13))
            ]
            assert card_items
            card_widget = window.result_list.itemWidget(card_items[0])
            assert card_widget is not None
            permission_button = card_widget.findChild(QPushButton, "GuidePermissionButton")
            association_button = card_widget.findChild(QPushButton, "GuideAssociationButton")
            similar_button = card_widget.findChild(QPushButton, "GuideSimilarButton")
            assert permission_button is not None
            assert association_button is not None
            assert similar_button is not None
            assert card_widget.focusProxy() is permission_button
            assert permission_button.isDefault() is True
            assert permission_button.accessibleName() == "권한 설정 가이드 열기"
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_similar_file_candidates_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    folder_a = tmp_path / "docs_a"
    folder_b = tmp_path / "docs_b"
    (folder_a / "old").mkdir(parents=True, exist_ok=True)
    (folder_a / "new").mkdir(parents=True, exist_ok=True)
    (folder_b / "newer").mkdir(parents=True, exist_ok=True)
    candidate_old = folder_a / "old" / "portfolio.pdf"
    candidate_new = folder_a / "new" / "portfolio.pdf"
    candidate_docx = folder_a / "new" / "portfolio.docx"
    candidate_other = folder_b / "newer" / "portfolio.pdf"
    candidate_old.write_text("old", encoding="utf-8")
    candidate_new.write_text("new", encoding="utf-8")
    candidate_docx.write_text("docx", encoding="utf-8")
    candidate_other.write_text("other", encoding="utf-8")
    old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    new_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    docx_ts = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
    other_ts = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    candidate_old.touch()
    candidate_new.touch()
    candidate_docx.touch()
    candidate_other.touch()
    # Rank: folder priority -> extension match -> path similarity -> mtime(desc)
    import os as _os

    _os.utime(candidate_old, (old_ts, old_ts))
    _os.utime(candidate_new, (new_ts, new_ts))
    _os.utime(candidate_docx, (docx_ts, docx_ts))
    _os.utime(candidate_other, (other_ts, other_ts))

    window = LauncherWindow()
    try:
        with patch.object(
            window.policy_registry,
            "list_folders",
            return_value=[{"path": str(folder_a)}, {"path": str(folder_b)}],
        ):
            ranked = window._find_similar_file_candidates(Path("/missing/new/portfolio.pdf"), limit=5)
            assert ranked
            assert ranked[0].name == "portfolio.pdf"
            assert ranked[0].parent.name == "new"
            # Same extension should beat newer .docx in same folder.
            assert ranked[1].suffix.lower() == ".pdf"
            before = window.result_list.count()
            window._append_similar_file_candidates(Path("/missing/new/portfolio.pdf"))
            tail_items = [window.result_list.item(i) for i in range(before, window.result_list.count())]
            assert any("유사 문서 후보" in item.text() for item in tail_items)
            candidate_actions = [
                item
                for item in tail_items
                if str(item.data(Qt.UserRole + 10) or "").strip() == "open_candidate_file"
            ]
            assert candidate_actions
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_open_success_status_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        existing_path = str(Path(__file__).resolve())
        file_item = QListWidgetItem("File: open-success")
        file_item.setData(Qt.UserRole, existing_path)
        with patch.object(window, "_open_local_file", return_value=(True, "")):
            window.on_result_item_clicked(file_item)
            assert window.status_label.text().startswith("Opened file:")
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_status_log_filter_and_copy_contract():
    QApplication.instance() or QApplication(sys.argv)

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        dialog._set_hub_status("sync ok", "success", allow_throttle=False)
        dialog._set_hub_status("sync fail", "error", allow_throttle=False)
        filter_index = dialog.hub_status_filter.findData("error")
        assert filter_index >= 0
        dialog.hub_status_filter.setCurrentIndex(filter_index)
        filtered = dialog.hub_status_log.toPlainText()
        assert "[error] sync fail" in filtered
        assert "[success] sync ok" not in filtered
        dialog._copy_hub_status_log()
        assert "[error] sync fail" in QApplication.clipboard().text()
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_status_log_export_json_time_filter_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        now = datetime.now(timezone.utc)
        dialog._hub_status_events = [
            {
                "stamp": "00:00:00",
                "tone": "error",
                "text": "very old error",
                "at_utc": (now - timedelta(days=2)).isoformat(),
            },
            {
                "stamp": "00:00:01",
                "tone": "error",
                "text": "recent error",
                "at_utc": (now - timedelta(minutes=20)).isoformat(),
            },
        ]
        tone_index = dialog.hub_status_filter.findData("error")
        assert tone_index >= 0
        dialog.hub_status_filter.setCurrentIndex(tone_index)
        range_index = dialog.hub_status_time_filter.findData("24h")
        assert range_index >= 0
        dialog.hub_status_time_filter.setCurrentIndex(range_index)
        export_path = tmp_path / "status-log.json"
        with patch(
            "desktop_app.ui.QFileDialog.getSaveFileName",
            return_value=(str(export_path), "JSON Files (*.json)"),
        ):
            dialog._export_hub_status_log()
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert len(payload) == 1
        assert payload[0]["tone"] == "error"
        assert "recent error" in payload[0]["text"]
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_status_log_export_csv_and_default_path_contract(tmp_path: Path):
    QApplication.instance() or QApplication(sys.argv)

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        default_path = dialog._default_status_log_export_path()
        assert default_path.name.startswith("ai-summary-status-log-")
        assert default_path.suffix == ".txt"

        now = datetime.now(timezone.utc)
        dialog._hub_status_events = [
            {
                "stamp": "00:00:01",
                "tone": "warning",
                "text": "recent warning",
                "at_utc": now.isoformat(),
            }
        ]
        dialog._render_hub_status_log()
        csv_path = tmp_path / "status-log.csv"
        with patch(
            "desktop_app.ui.QFileDialog.getSaveFileName",
            return_value=(str(csv_path), "CSV Files (*.csv)"),
        ):
            dialog._export_hub_status_log()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines
        assert lines[0] == "timestamp_utc,timestamp_local,tone,text"
        assert any("recent warning" in line for line in lines[1:])
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_settings_hub_status_banner_auto_reset_contract():
    QApplication.instance() or QApplication(sys.argv)

    dialog = SettingsHubDialog(
        smart_folder_count=1,
        current_mode="Auto",
        runtime_policy=load_desktop_runtime_policy(),
        open_folders_callback=lambda: None,
        open_mode_callback=lambda: None,
        open_runtime_callback=lambda: None,
    )
    try:
        initial_log = dialog.hub_status_log.toPlainText()
        assert "No status events yet." in initial_log
        dialog._set_hub_status("saved", "success")
        assert dialog._hub_status_reset_timer.isActive() is True
        first_log = dialog.hub_status_log.toPlainText()
        assert "[success] saved" in first_log
        first_timestamp = dialog._hub_status_last_at
        dialog._set_hub_status("saved", "success")
        assert dialog._hub_status_last_at == first_timestamp
        assert dialog.hub_status_log.toPlainText() == first_log
        dialog._set_hub_status("saved", "success", allow_throttle=False)
        assert dialog._hub_status_last_at is not None
        assert first_timestamp is not None
        assert dialog._hub_status_last_at >= first_timestamp
        assert dialog.hub_status_log.toPlainText().count("[success] saved") >= 2
        dialog._set_hub_status("error", "error")
        assert dialog._hub_status_reset_timer.isActive() is False
        dialog._reset_hub_status()
        assert dialog.hub_status.text() == "Settings ready"
    finally:
        dialog.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_file_recovery_shortcut_mapping_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        window.add_message("File", "artifact", file_path=str(Path(__file__).resolve()))
        window.result_list.setCurrentRow(window.result_list.count() - 1)
        with patch.object(window, "_open_selected_timeline_parent") as mock_parent:
            window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_P, Qt.ControlModifier | Qt.ShiftModifier))
            assert mock_parent.call_count == 1
        with patch.object(window, "_copy_selected_timeline_file_path") as mock_copy:
            window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_O, Qt.ControlModifier | Qt.ShiftModifier))
            assert mock_copy.call_count == 1
        with patch.object(window, "_reveal_selected_timeline_item") as mock_reveal:
            window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_R, Qt.ControlModifier | Qt.ShiftModifier))
            assert mock_reveal.call_count == 1
    finally:
        window.close()


@pytest.mark.skipif(sys.platform == "linux", reason="Requires Wayland/X11 on Linux")
def test_launcher_recovery_action_click_dispatch_contract():
    QApplication.instance() or QApplication(sys.argv)

    window = LauncherWindow()
    try:
        target = Path(__file__).resolve()
        before = window.result_list.count()
        window._append_open_recovery_actions(target)
        card_items = [
            window.result_list.item(i)
            for i in range(before, window.result_list.count())
            if bool(window.result_list.item(i).data(Qt.UserRole + 12))
        ]
        assert len(card_items) == 1
        card_widget = window.result_list.itemWidget(card_items[0])
        assert card_widget is not None
        retry_button = card_widget.findChild(QPushButton, "RecoveryRetryButton")
        reveal_button = card_widget.findChild(QPushButton, "RecoveryRevealButton")
        assert retry_button is not None
        assert reveal_button is not None
        assert reveal_button.accessibleName() == "Reveal in Finder"
        with patch.object(window, "_run_timeline_action") as mock_action:
            retry_button.click()
            mock_action.assert_called_once()
            args = mock_action.call_args[0]
            assert args[0] == "retry_open"
            assert str(target) in str(args[1])
        with patch.object(window, "_run_timeline_action") as mock_action:
            reveal_button.click()
            mock_action.assert_called_once()
            args = mock_action.call_args[0]
            assert args[0] == "reveal_in_finder"
            assert str(target) in str(args[1])
    finally:
        window.close()
