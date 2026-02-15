# scripts/pipeline/infopilot_cli/watch.py
"""
Watch command module.
Refactored to delegate logic to `watchers.py` and `pipeline_runner.py`.
"""
from __future__ import annotations

import importlib
import queue
import threading
from pathlib import Path

import click

from core.config.paths import CACHE_DIR, CORPUS_PATH
from core.data_pipeline.scanner import DEFAULT_EXTS
from core.policy.engine import PolicyEngine

from .pipeline_runner import IncrementalPipeline, load_vector_index, sync_scan_csv, watch_loop
from .watchers import PolicyEventHandler, WatchEventHandler

__all__ = [
    "IncrementalPipeline",
    "PolicyEventHandler",
    "WatchEventHandler",
    "load_vector_index",
    "sync_scan_csv",
    "watch_loop",
    "cmd_watch",
]


def cmd_watch(args, knowledge_agent: str):
    """
    Run the incremental indexing watcher.
    """
    sentence_transformer_cls, observer_cls, fs_event_handler_cls = _load_watch_dependencies()

    output_root = Path(args.output_root).expanduser()
    scan_csv = output_root / "scan_results.csv"
    if hasattr(args, "output_csv") and args.output_csv:
        scan_csv = Path(args.output_csv)

    policy_path = getattr(args, "policy", None)
    if policy_path:
        policy_path = Path(policy_path)

    policy_engine = None
    if policy_path and policy_path.exists():
        try:
            policy_engine = PolicyEngine.from_file(policy_path)
            print(f"📜 정책 로드: {policy_path}")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            print(f"⚠️ 정책 로드 실패: {exc}")

    model_name = getattr(args, "model_name", "all-MiniLM-L6-v2")
    print(f"🔌 Encoder 로딩: {model_name}...")
    try:
        encoder = sentence_transformer_cls(model_name)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(f"임베딩 모델 로딩 실패({model_name}): {exc}") from exc

    watch_targets = _normalize_watch_targets(getattr(args, "target", []))

    policy_state: dict[str, PolicyEngine | None] = {"engine": policy_engine}

    def _current_policy_engine() -> PolicyEngine | None:
        return policy_state["engine"]

    def _handle_policy_reload(updated: PolicyEngine | None) -> None:
        policy_state["engine"] = updated

    pipeline = IncrementalPipeline(
        encoder=encoder,
        batch_size=getattr(args, "batch_size", 32),
        scan_csv=scan_csv,
        corpus_path=CORPUS_PATH,
        cache_dir=CACHE_DIR,
        translate=getattr(args, "translate", False),
        policy_engine=policy_engine,
        policy_engine_provider=_current_policy_engine,
        policy_reload_callback=_handle_policy_reload,
        policy_path=policy_path,
        roots=watch_targets,
        agent=knowledge_agent,
        require_policy_engine=bool(policy_path),
    )
    allowed_exts = {ext.lower() for ext in DEFAULT_EXTS}
    ignore_paths = {str(policy_path.resolve())} if policy_path and policy_path.exists() else set()

    event_queue = queue.Queue()
    stop_event = threading.Event()
    debounce = float(getattr(args, "debounce", 1.0))

    observer = observer_cls()

    for path in watch_targets:
        print(f"👀 감시 시작: {path}")
        handler = WatchEventHandler(
            event_queue,
            allowed_exts,
            policy_engine_provider=_current_policy_engine,
            ignore_paths=ignore_paths,
            agent=knowledge_agent,
            base_handler_cls=fs_event_handler_cls,
        )
        observer.schedule(handler, str(path), recursive=True)

    if policy_path and policy_path.exists():
        policy_dir = policy_path.parent
        policy_handler = PolicyEventHandler(
            event_queue,
            policy_path.resolve(),
            base_handler_cls=fs_event_handler_cls,
        )
        observer.schedule(policy_handler, str(policy_dir), recursive=False)

    observer_started = False
    try:
        observer.start()
        observer_started = True
    except (OSError, PermissionError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"watch observer 시작 실패: {exc}") from exc
    print("🚀 감시 루프 시작 (Ctrl+C로 종료)...")

    try:
        watch_loop(event_queue, pipeline, stop_event, debounce)
    except KeyboardInterrupt:
        print("\n🛑 중지 요청...")
        stop_event.set()
    finally:
        if observer_started:
            observer.stop()
            observer.join()
        print("👋 종료.")


def _load_watch_dependencies():
    missing: list[str] = []
    try:
        sentence_transformers_module = importlib.import_module("sentence_transformers")
    except ImportError:
        sentence_transformers_module = None
        missing.append("sentence-transformers")

    try:
        watchdog_events_module = importlib.import_module("watchdog.events")
        watchdog_observers_module = importlib.import_module("watchdog.observers")
    except ImportError:
        watchdog_events_module = None
        watchdog_observers_module = None
        missing.append("watchdog")

    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise click.ClickException(
            f"watch 명령에 필요한 의존성이 없습니다: {missing_list}. 설치 후 다시 시도하세요."
        )

    sentence_transformer_cls = getattr(sentence_transformers_module, "SentenceTransformer", None)
    observer_cls = getattr(watchdog_observers_module, "Observer", None)
    fs_event_handler_cls = getattr(watchdog_events_module, "FileSystemEventHandler", None)
    if sentence_transformer_cls is None or observer_cls is None or fs_event_handler_cls is None:
        raise click.ClickException("watch 의존성 로딩에 실패했습니다. 패키지 버전을 확인하세요.")

    return sentence_transformer_cls, observer_cls, fs_event_handler_cls


def _normalize_watch_targets(raw_targets: list[str]) -> list[Path]:
    targets = list(raw_targets) if raw_targets else ["."]
    resolved: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        try:
            path = Path(target).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            print(f"⚠️ 경로 해석 실패: {target}")
            continue
        if not path.exists():
            print(f"⚠️ 경로 없음: {path}")
            continue
        watch_root = path.parent if path.is_file() else path
        if path.is_file():
            print(f"ℹ️ 파일 경로는 상위 폴더를 감시합니다: {path} -> {watch_root}")
        if watch_root in seen:
            continue
        seen.add(watch_root)
        resolved.append(watch_root)
    if not resolved:
        raise click.ClickException("감시할 유효한 경로가 없습니다. --target 경로를 확인하세요.")
    return resolved
