from __future__ import annotations

import queue

import pytest

from scripts.pipeline.infopilot_cli.watchers import WatchEventHandler

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


class _TogglePolicyEngine:
    def __init__(self, allowed: bool) -> None:
        self._allowed = allowed

    def check(self, *_args, **_kwargs):
        return self._allowed, "policy"


class _BoolPolicyEngine:
    def __init__(self, allowed: bool) -> None:
        self._allowed = allowed

    def check(self, *_args, **_kwargs):
        return self._allowed


class _DictPolicyEngine:
    def __init__(self, allowed: bool) -> None:
        self._allowed = allowed

    def check(self, *_args, **_kwargs):
        return {"allowed": self._allowed}


class _DictStringPolicyEngine:
    def __init__(self, allowed: str) -> None:
        self._allowed = allowed

    def check(self, *_args, **_kwargs):
        return {"allowed": self._allowed}


class _ListPolicyEngine:
    def __init__(self, value: list[str]) -> None:
        self._value = value

    def check(self, *_args, **_kwargs):
        return self._value


class _Event:
    def __init__(self, src_path: str, dest_path: str = "", *, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = is_directory


def test_watch_event_handler_ignore_path_exact_match_only(tmp_path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    sibling_path = tmp_path / "policy.json.backup.md"
    sibling_path.write_text("# note", encoding="utf-8")

    handler = WatchEventHandler(
        queue.Queue(),
        {".md", ".json"},
        ignore_paths={str(policy_path)},
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(str(policy_path.resolve())) is False
    assert handler._should_process(str(sibling_path.resolve())) is True


def test_watch_event_handler_skips_ignore_path_remove_events(tmp_path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("hello", encoding="utf-8")
    events: "queue.Queue[tuple[str, str]]" = queue.Queue()

    handler = WatchEventHandler(
        events,
        {".md", ".json"},
        ignore_paths={str(policy_path)},
        agent="knowledge_search",
        base_handler_cls=object,
    )

    handler.on_deleted(_Event(str(policy_path)))
    assert events.empty() is True

    handler.on_moved(_Event(str(policy_path), str(doc_path)))
    assert events.get_nowait() == ("add", str(doc_path.resolve()))
    assert events.empty() is True


def test_watch_event_handler_uses_live_policy_provider(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    state: dict[str, _TogglePolicyEngine | None] = {"engine": None}

    def policy_provider():
        return state["engine"]

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine_provider=policy_provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is True
    state["engine"] = _TogglePolicyEngine(False)
    assert handler._should_process(target_path) is False
    state["engine"] = _TogglePolicyEngine(True)
    assert handler._should_process(target_path) is True


def test_watch_event_handler_provider_precedence_over_local_engine(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def provider():
        return _TogglePolicyEngine(True)

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_TogglePolicyEngine(False),
        policy_engine_provider=provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is True


def test_watch_event_handler_policy_provider_none_falls_back_to_local_engine(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def provider():
        return None

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_TogglePolicyEngine(False),
        policy_engine_provider=provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is False


def test_watch_event_handler_policy_provider_none_without_fallback_uses_extension_gate(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def provider():
        return None

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine_provider=provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is True


def test_watch_event_handler_policy_provider_failure_falls_back(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def failing_provider():
        raise RuntimeError("provider-failure")

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_TogglePolicyEngine(False),
        policy_engine_provider=failing_provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is False


def test_watch_event_handler_policy_provider_failure_warning_throttled(tmp_path, caplog) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def failing_provider():
        raise RuntimeError("provider-failure")

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_TogglePolicyEngine(False),
        policy_engine_provider=failing_provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    with caplog.at_level("WARNING"):
        assert handler._should_process(target_path) is False
        assert handler._should_process(target_path) is False

    warning_records = [record for record in caplog.records if "policy engine provider failed" in record.message]
    assert len(warning_records) == 1


def test_watch_event_handler_policy_provider_failure_warning_resets_after_recovery(tmp_path, caplog) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    state = {"mode": "fail"}

    def provider():
        if state["mode"] == "fail":
            raise RuntimeError("provider-failure")
        return _TogglePolicyEngine(True)

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine_provider=provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    with caplog.at_level("WARNING"):
        assert handler._should_process(target_path) is False
        state["mode"] = "ok"
        assert handler._should_process(target_path) is True
        state["mode"] = "fail"
        assert handler._should_process(target_path) is False

    warning_records = [record for record in caplog.records if "policy engine provider failed" in record.message]
    assert len(warning_records) == 2


def test_watch_event_handler_policy_provider_failure_without_fallback_is_fail_closed(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    def failing_provider():
        raise RuntimeError("provider-failure")

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine_provider=failing_provider,
        agent="knowledge_search",
        base_handler_cls=object,
    )

    assert handler._should_process(target_path) is False


def test_watch_event_handler_supports_bool_policy_check_result(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_BoolPolicyEngine(True),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler._should_process(target_path) is True

    handler_false = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_BoolPolicyEngine(False),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler_false._should_process(target_path) is False


def test_watch_event_handler_supports_dict_policy_check_result(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_DictPolicyEngine(True),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler._should_process(target_path) is True

    handler_false = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_DictPolicyEngine(False),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler_false._should_process(target_path) is False


def test_watch_event_handler_dict_string_allowed_is_coerced(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    handler_true = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_DictStringPolicyEngine("true"),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler_true._should_process(target_path) is True

    handler_false = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_DictStringPolicyEngine("false"),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler_false._should_process(target_path) is False


def test_watch_event_handler_dict_unknown_string_is_fail_closed(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_DictStringPolicyEngine("maybe"),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler._should_process(target_path) is False


def test_watch_event_handler_unexpected_policy_payload_is_fail_closed(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    target_path = str(target.resolve())

    handler = WatchEventHandler(
        queue.Queue(),
        {".md"},
        policy_engine=_ListPolicyEngine(["true"]),
        agent="knowledge_search",
        base_handler_cls=object,
    )
    assert handler._should_process(target_path) is False


def test_watch_event_handler_ignores_invalid_event_paths() -> None:
    events: "queue.Queue[tuple[str, str]]" = queue.Queue()
    handler = WatchEventHandler(
        events,
        {".md"},
        agent="knowledge_search",
        base_handler_cls=object,
    )

    handler.on_created(_Event(None))  # type: ignore[arg-type]
    handler.on_modified(_Event(None))  # type: ignore[arg-type]
    handler.on_deleted(_Event(None))  # type: ignore[arg-type]
    handler.on_moved(_Event(None, None))  # type: ignore[arg-type]

    assert events.empty() is True


def test_watch_event_handler_invalid_path_warning_throttled(caplog) -> None:
    events: "queue.Queue[tuple[str, str]]" = queue.Queue()
    handler = WatchEventHandler(
        events,
        {".md"},
        agent="knowledge_search",
        base_handler_cls=object,
    )

    with caplog.at_level("WARNING"):
        handler.on_created(_Event(None))  # type: ignore[arg-type]
        handler.on_created(_Event(""))  # type: ignore[arg-type]

    warning_records = [record for record in caplog.records if "watch event path normalize failed" in record.message]
    assert len(warning_records) == 1
    assert events.empty() is True


def test_watch_event_handler_invalid_path_warning_resets_after_valid_path(tmp_path, caplog) -> None:
    events: "queue.Queue[tuple[str, str]]" = queue.Queue()
    valid_path = tmp_path / "note.md"
    valid_path.write_text("ok", encoding="utf-8")
    handler = WatchEventHandler(
        events,
        {".md"},
        agent="knowledge_search",
        base_handler_cls=object,
    )

    with caplog.at_level("WARNING"):
        handler.on_created(_Event(None))  # type: ignore[arg-type]
        handler.on_created(_Event(str(valid_path)))
        handler.on_created(_Event(None))  # type: ignore[arg-type]

    warning_records = [record for record in caplog.records if "watch event path normalize failed" in record.message]
    assert len(warning_records) == 2
