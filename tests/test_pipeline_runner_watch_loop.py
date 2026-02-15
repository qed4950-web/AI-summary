from __future__ import annotations

import queue
import threading

import pytest

from scripts.pipeline.infopilot_cli.pipeline_runner import watch_loop

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


class _FakePipeline:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.process_calls: list[tuple[set[str], set[str]]] = []
        self.policy_reload_count = 0

    def process(self, add_paths: set[str], remove_paths: set[str]) -> None:
        self.process_calls.append((set(add_paths), set(remove_paths)))
        self.stop_event.set()

    def handle_policy_change(self) -> None:
        self.policy_reload_count += 1
        self.stop_event.set()


def _run_loop(
    event_queue: "queue.Queue[tuple[str, str]]",
    pipeline: _FakePipeline,
    stop_event: threading.Event,
    debounce_sec: float,
) -> threading.Thread:
    worker = threading.Thread(
        target=watch_loop,
        args=(event_queue, pipeline, stop_event, debounce_sec),
        daemon=True,
    )
    worker.start()
    return worker


def test_watch_loop_flushes_add_event_after_idle_timeout() -> None:
    event_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
    stop_event = threading.Event()
    pipeline = _FakePipeline(stop_event)
    event_queue.put(("add", "/tmp/document.md"))

    worker = _run_loop(event_queue, pipeline, stop_event, debounce_sec=0.1)
    worker.join(timeout=2.0)

    if worker.is_alive():
        stop_event.set()
        worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert pipeline.process_calls == [({"/tmp/document.md"}, set())]
    assert pipeline.policy_reload_count == 0


def test_watch_loop_flushes_policy_reload_after_idle_timeout() -> None:
    event_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
    stop_event = threading.Event()
    pipeline = _FakePipeline(stop_event)
    event_queue.put(("policy_reload", ""))

    worker = _run_loop(event_queue, pipeline, stop_event, debounce_sec=0.1)
    worker.join(timeout=2.0)

    if worker.is_alive():
        stop_event.set()
        worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert pipeline.policy_reload_count == 1
    assert pipeline.process_calls == []
