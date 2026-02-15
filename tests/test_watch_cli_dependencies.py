from __future__ import annotations

import importlib
import types
from types import SimpleNamespace

import click
import pytest

from scripts.pipeline.infopilot_cli import watch as watch_module
from scripts.pipeline.infopilot_cli.watch import _load_watch_dependencies, _normalize_watch_targets

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_load_watch_dependencies_reports_missing(monkeypatch) -> None:
    original_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in {"sentence_transformers", "watchdog.events", "watchdog.observers"}:
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr("scripts.pipeline.infopilot_cli.watch.importlib.import_module", fake_import)

    with pytest.raises(click.ClickException) as exc:
        _load_watch_dependencies()
    message = str(exc.value)
    assert "sentence-transformers" in message
    assert "watchdog" in message


def test_load_watch_dependencies_accepts_valid_symbols(monkeypatch) -> None:
    fake_sentence_transformers = types.SimpleNamespace(SentenceTransformer=object)
    fake_watchdog_events = types.SimpleNamespace(FileSystemEventHandler=object)
    fake_watchdog_observers = types.SimpleNamespace(Observer=object)

    def fake_import(name: str, package: str | None = None):
        if name == "sentence_transformers":
            return fake_sentence_transformers
        if name == "watchdog.events":
            return fake_watchdog_events
        if name == "watchdog.observers":
            return fake_watchdog_observers
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("scripts.pipeline.infopilot_cli.watch.importlib.import_module", fake_import)

    sentence_transformer_cls, observer_cls, fs_event_handler_cls = _load_watch_dependencies()
    assert sentence_transformer_cls is object
    assert observer_cls is object
    assert fs_event_handler_cls is object


def test_normalize_watch_targets_raises_when_all_missing(tmp_path) -> None:
    with pytest.raises(click.ClickException) as exc:
        _normalize_watch_targets([str(tmp_path / "missing-a"), str(tmp_path / "missing-b")])
    assert "감시할 유효한 경로가 없습니다" in str(exc.value)


def test_normalize_watch_targets_file_path_maps_to_parent_and_dedups(tmp_path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    file_path = folder / "doc.md"
    file_path.write_text("x", encoding="utf-8")

    targets = _normalize_watch_targets([str(file_path), str(folder)])
    assert targets == [folder.resolve()]


def test_cmd_watch_wires_normalized_roots_and_dynamic_policy_provider(monkeypatch, tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    file_path = root / "doc.md"
    file_path.write_text("x", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, _model_name: str):
            return

    class FakeObserver:
        def __init__(self):
            return

        def schedule(self, _handler, _path: str, recursive: bool = True) -> None:
            assert recursive is True

        def start(self) -> None:
            return

        def stop(self) -> None:
            return

        def join(self) -> None:
            return

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["roots"] = kwargs["roots"]
            captured["policy_engine_provider"] = kwargs["policy_engine_provider"]
            captured["policy_reload_callback"] = kwargs["policy_reload_callback"]

    class FakeWatchHandler:
        def __init__(self, *_args, **kwargs):
            captured["handler_policy_provider"] = kwargs["policy_engine_provider"]

    class FakePolicyHandler:
        def __init__(self, *_args, **_kwargs):
            return

    def fake_watch_loop(*_args, **_kwargs) -> None:
        return

    monkeypatch.setattr(
        watch_module,
        "_load_watch_dependencies",
        lambda: (FakeSentenceTransformer, FakeObserver, object),
    )
    monkeypatch.setattr(watch_module, "IncrementalPipeline", FakePipeline)
    monkeypatch.setattr(watch_module, "WatchEventHandler", FakeWatchHandler)
    monkeypatch.setattr(watch_module, "PolicyEventHandler", FakePolicyHandler)
    monkeypatch.setattr(watch_module, "watch_loop", fake_watch_loop)

    args = SimpleNamespace(
        output_root=str(tmp_path / "out"),
        target=[str(file_path), str(root)],
        model_name="stub-model",
        batch_size=8,
        translate=False,
        debounce=0.05,
        policy=None,
    )

    watch_module.cmd_watch(args, "knowledge_search")

    normalized_roots = captured["roots"]
    assert isinstance(normalized_roots, list)
    assert normalized_roots == [root.resolve()]

    pipeline_provider = captured["policy_engine_provider"]
    handler_provider = captured["handler_policy_provider"]
    assert callable(pipeline_provider)
    assert handler_provider is pipeline_provider

    replacement_engine = object()
    reload_callback = captured["policy_reload_callback"]
    assert callable(reload_callback)
    reload_callback(replacement_engine)
    assert pipeline_provider() is replacement_engine
