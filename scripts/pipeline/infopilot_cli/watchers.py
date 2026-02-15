# scripts/pipeline/infopilot_cli/watchers.py
from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

# Third-party
try:
    from watchdog.events import FileSystemEventHandler
except ImportError:
    FileSystemEventHandler = object  # Fallback

from core.policy.engine import PolicyEngine

logger = logging.getLogger(__name__)


class WatchEventHandler(FileSystemEventHandler):
    """Handles file system events for the document corpus."""

    def __init__(
        self,
        event_queue: "queue.Queue[Tuple[str, str]]",
        allowed_exts: Set[str],
        *,
        policy_engine: Optional[PolicyEngine] = None,
        policy_engine_provider: Optional[Callable[[], Optional[PolicyEngine]]] = None,
        ignore_paths: Optional[Set[str]] = None,
        agent: str,
        base_handler_cls: type,
    ):
        base_handler_cls.__init__(self)
        self._queue = event_queue
        self._allowed_exts = allowed_exts
        self._policy_engine = policy_engine
        self._policy_engine_provider = policy_engine_provider
        self._ignore_paths = self._normalize_ignore_paths(ignore_paths)
        self._policy_agent = agent
        self._provider_failure_notified = False
        self._provider_failed_last_call = False
        self._invalid_path_warning_notified = False

    def _normalize_ignore_paths(self, ignore_paths: Optional[Set[str]]) -> Set[str]:
        normalized: Set[str] = set()
        if not ignore_paths:
            return normalized
        for raw in ignore_paths:
            if not raw:
                continue
            try:
                normalized.add(self._normalize_path(str(raw)))
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
        return normalized

    def _normalize_path(self, raw: str) -> str:
        # unicodedata normalization could go here if needed
        return str(Path(raw).resolve())

    def _normalize_event_path(self, raw: object) -> str | None:
        try:
            if raw is None:
                raise ValueError("event path is None")
            raw_path = str(raw).strip()
            if not raw_path:
                raise ValueError("event path is empty")
            path = self._normalize_path(raw_path)
            self._invalid_path_warning_notified = False
            return path
        except (OSError, RuntimeError, TypeError, ValueError):
            if not self._invalid_path_warning_notified:
                logger.warning("watch event path normalize failed: %r", raw)
                self._invalid_path_warning_notified = True
            return None

    def _current_policy_engine(self) -> Optional[PolicyEngine]:
        if self._policy_engine_provider:
            try:
                engine = self._policy_engine_provider()
                if engine is None and self._policy_engine is not None:
                    self._provider_failure_notified = False
                    self._provider_failed_last_call = False
                    return self._policy_engine
                self._provider_failure_notified = False
                self._provider_failed_last_call = False
                return engine
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                if not self._provider_failure_notified:
                    logger.warning("policy engine provider failed; using local fallback: %s", exc)
                    self._provider_failure_notified = True
                self._provider_failed_last_call = True
                return self._policy_engine
        if self._policy_engine:
            return self._policy_engine
        return None

    def _is_ignored_path(self, path: str) -> bool:
        return path in self._ignore_paths

    def _resolve_allowed(self, check_result: object) -> bool:
        raw_allowed: object
        if isinstance(check_result, tuple):
            raw_allowed = check_result[0] if check_result else False
        elif isinstance(check_result, dict):
            raw_allowed = check_result.get("allowed", False)
        else:
            raw_allowed = check_result

        if isinstance(raw_allowed, bool):
            return raw_allowed
        if isinstance(raw_allowed, (int, float)):
            return raw_allowed != 0
        if isinstance(raw_allowed, str):
            normalized = raw_allowed.strip().lower()
            if normalized in {"1", "true", "yes", "y", "allow", "allowed"}:
                return True
            if normalized in {"", "0", "false", "no", "n", "deny", "denied"}:
                return False
            return False
        return False

    def _should_process(self, path: str) -> bool:
        if self._is_ignored_path(path):
            return False

        # Policy check
        engine = self._current_policy_engine()
        if self._policy_engine_provider and self._provider_failed_last_call and engine is None:
            return False
        if engine:
            try:
                check_result = engine.check(Path(path), agent=self._policy_agent, include_manual=False)
            except (OSError, RuntimeError, TypeError, ValueError):
                return False
            allowed = self._resolve_allowed(check_result)
            if not allowed:
                return False

        # Extension check
        ext = Path(path).suffix.lower()
        return ext in self._allowed_exts

    def on_created(self, event):
        if event.is_directory:
            return
        path = self._normalize_event_path(event.src_path)
        if not path:
            return
        if self._should_process(path):
            self._queue.put(("add", path))

    def on_modified(self, event):
        if event.is_directory:
            return
        path = self._normalize_event_path(event.src_path)
        if not path:
            return
        if self._should_process(path):
            self._queue.put(("add", path))

    def on_moved(self, event):
        if event.is_directory:
            return
        src = self._normalize_event_path(event.src_path)
        dest = self._normalize_event_path(event.dest_path)
        if not src and not dest:
            return

        # Treat move as delete + add
        # (Though we might not strictly need to delete if src wasn't tracked,
        #  but safe to enqueue remove just in case)
        if src and not self._is_ignored_path(src):
            self._queue.put(("remove", src))

        if dest and self._should_process(dest):
            self._queue.put(("add", dest))

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = self._normalize_event_path(event.src_path)
        if not path:
            return
        if self._is_ignored_path(path):
            return
        self._queue.put(("remove", path))


class PolicyEventHandler(FileSystemEventHandler):
    """Monitors the policy file (smart folders) for changes."""

    def __init__(
        self,
        event_queue: "queue.Queue[Tuple[str, str]]",
        policy_path: Path,
        *,
        base_handler_cls: type,
    ) -> None:
        base_handler_cls.__init__(self)
        self._queue = event_queue
        self._policy_path = self._normalize_path(policy_path)

    def _normalize_path(self, path: Path) -> Path:
        return path.resolve()

    def _is_target(self, raw: str) -> bool:
        try:
            p = Path(raw).resolve()
            return p == self._policy_path
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    def on_created(self, event):
        if not event.is_directory and self._is_target(event.src_path):
            logger.info("📜 Policy file created.")
            self._queue.put(("policy_reload", ""))

    def on_modified(self, event):
        if not event.is_directory and self._is_target(event.src_path):
            logger.info("📜 Policy file modified.")
            self._queue.put(("policy_reload", ""))

    def on_moved(self, event):
        if not event.is_directory and self._is_target(event.dest_path):
            logger.info("📜 Policy file moved/renamed.")
            self._queue.put(("policy_reload", ""))

    def on_deleted(self, event):
        if not event.is_directory and self._is_target(event.src_path):
            logger.warning("📜 Policy file deleted.")
            # Depending on logic, might want to clear policy or just warn
            self._queue.put(("policy_reload", ""))
