from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_incremental_index as incremental_index

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


class _FakeDriftState:
    def __init__(self, *, added: list[Path], modified: list[Path], deleted: list[Path]) -> None:
        self.added = added
        self.modified = modified
        self.deleted = deleted

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return f"added={len(self.added)}, modified={len(self.modified)}, deleted={len(self.deleted)}"


class _FakeIndex:
    def __init__(self) -> None:
        self.removed: set[str] = set()
        self.saved_cache_dirs: list[Path] = []

    def remove_paths(self, paths: set[str]) -> None:
        self.removed = set(paths)

    def save(self, cache_dir: Path) -> None:
        self.saved_cache_dirs.append(Path(cache_dir))


class _FakeChunkCache:
    def __init__(self) -> None:
        self.dropped: set[str] = set()
        self.saved = 0
        self._known = {"/docs/deleted.pdf", "/docs/removed.pdf", "/docs/current.pdf"}

    def known_paths(self) -> set[str]:
        return set(self._known)

    def drop_paths(self, paths: set[str]) -> None:
        self.dropped |= set(paths)
        self._known -= set(paths)

    def save(self) -> None:
        self.saved += 1


def test_run_incremental_index_reconciles_deletions_without_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    all_files = [{"path": "/docs/current.pdf", "size": 1, "mtime": 1.0}]
    drift = _FakeDriftState(added=[], modified=[], deleted=[Path("/docs/deleted.pdf")])
    fake_index = _FakeIndex()
    removed_calls: list[list[str]] = []
    run_step2_calls: list[list[dict[str, object]]] = []
    saved_states: list[tuple[Path, dict[str, object]]] = []
    fake_chunk_cache = _FakeChunkCache()
    reports: list[dict[str, object]] = []

    monkeypatch.setattr(incremental_index, "scan_directory", lambda _root: list(all_files))
    monkeypatch.setattr(incremental_index, "load_scan_state", lambda _path: {})
    monkeypatch.setattr(incremental_index, "filter_incremental_rows", lambda rows, _state: (list(rows), {}))
    monkeypatch.setattr(incremental_index, "update_scan_state", lambda _state, _rows: {"version": 1})
    monkeypatch.setattr(
        incremental_index,
        "save_scan_state",
        lambda path, state: saved_states.append((Path(path), dict(state))),
    )
    monkeypatch.setattr(incremental_index, "default_train_config", lambda: object())
    monkeypatch.setattr(incremental_index, "run_step2", lambda *args, **kwargs: run_step2_calls.append(list(args[0])))
    monkeypatch.setattr(incremental_index, "remove_from_corpus", lambda paths, _corpus: removed_calls.append(list(paths)))
    monkeypatch.setattr(incremental_index, "load_vector_index", lambda _cache_dir: fake_index)
    monkeypatch.setattr(incremental_index, "ChunkCache", lambda _cache_path: fake_chunk_cache)
    monkeypatch.setattr(
        incremental_index,
        "_write_incremental_report",
        lambda report, path=None: reports.append(dict(report)),
    )
    monkeypatch.setattr(
        incremental_index,
        "DriftDetector",
        lambda _cache: type(
            "Detector",
            (),
            {"detect_with_incremental": lambda self, _all, _to_process: drift},
        )(),
    )

    incremental_index.main()

    assert run_step2_calls == []
    assert removed_calls
    assert set(removed_calls[0]) == {"/docs/deleted.pdf"}
    assert fake_index.removed == {"/docs/deleted.pdf"}
    assert fake_index.saved_cache_dirs
    assert fake_chunk_cache.dropped == {"/docs/deleted.pdf"}
    assert fake_chunk_cache.saved == 1
    assert saved_states
    assert reports
    assert reports[-1]["status"] == "completed"
    assert str(reports[-1].get("started_at_utc") or "")
    assert str(reports[-1].get("finished_at_utc") or "")
    assert int(reports[-1].get("duration_ms") or 0) >= 0
    assert reports[-1]["run_step2_triggered"] is False
    assert reports[-1]["processed_count"] == 0
    assert reports[-1]["deleted_reconciled_count"] == 1


def test_run_incremental_index_processes_only_added_modified_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    all_files = [
        {"path": "/docs/current.pdf", "size": 1, "mtime": 1.0},
        {"path": "/docs/new.pdf", "size": 1, "mtime": 1.0},
        {"path": "/docs/untouched.pdf", "size": 1, "mtime": 1.0},
    ]
    drift = _FakeDriftState(
        added=[Path("/docs/new.pdf")],
        modified=[Path("/docs/current.pdf")],
        deleted=[Path("/docs/removed.pdf")],
    )
    fake_index = _FakeIndex()
    removed_calls: list[list[str]] = []
    run_step2_calls: list[list[dict[str, object]]] = []
    fake_chunk_cache = _FakeChunkCache()
    reports: list[dict[str, object]] = []

    monkeypatch.setattr(incremental_index, "scan_directory", lambda _root: list(all_files))
    monkeypatch.setattr(incremental_index, "load_scan_state", lambda _path: {})
    monkeypatch.setattr(incremental_index, "filter_incremental_rows", lambda rows, _state: (list(rows), {}))
    monkeypatch.setattr(incremental_index, "update_scan_state", lambda _state, _rows: {"version": 1})
    monkeypatch.setattr(incremental_index, "save_scan_state", lambda _path, _state: None)
    monkeypatch.setattr(incremental_index, "default_train_config", lambda: object())
    monkeypatch.setattr(incremental_index, "run_step2", lambda *args, **kwargs: run_step2_calls.append(list(args[0])))
    monkeypatch.setattr(incremental_index, "remove_from_corpus", lambda paths, _corpus: removed_calls.append(list(paths)))
    monkeypatch.setattr(incremental_index, "load_vector_index", lambda _cache_dir: fake_index)
    monkeypatch.setattr(incremental_index, "ChunkCache", lambda _cache_path: fake_chunk_cache)
    monkeypatch.setattr(
        incremental_index,
        "_write_incremental_report",
        lambda report, path=None: reports.append(dict(report)),
    )
    monkeypatch.setattr(
        incremental_index,
        "DriftDetector",
        lambda _cache: type(
            "Detector",
            (),
            {"detect_with_incremental": lambda self, _all, _to_process: drift},
        )(),
    )

    incremental_index.main()

    assert removed_calls
    assert set(removed_calls[0]) == {"/docs/removed.pdf"}
    assert fake_chunk_cache.dropped == {"/docs/removed.pdf"}
    assert fake_chunk_cache.saved == 1
    assert len(run_step2_calls) == 1
    processed_paths = {str(row.get("path")) for row in run_step2_calls[0]}
    assert processed_paths == {"/docs/current.pdf", "/docs/new.pdf"}
    assert reports
    assert reports[-1]["status"] == "completed"
    assert str(reports[-1].get("started_at_utc") or "")
    assert str(reports[-1].get("finished_at_utc") or "")
    assert int(reports[-1].get("duration_ms") or 0) >= 0
    assert reports[-1]["run_step2_triggered"] is True
    assert reports[-1]["processed_count"] == 2
    assert reports[-1]["missing_target_count"] == 0


def test_run_incremental_index_large_mixed_drift_scope_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    all_files = [
        {"path": f"/docs/file-{idx:03d}.pdf", "size": 1, "mtime": 1.0}
        for idx in range(200)
    ]
    added = [Path(f"/docs/file-{idx:03d}.pdf") for idx in range(20, 40)] + [Path("/docs/file-999.pdf")]
    modified = [Path(f"/docs/file-{idx:03d}.pdf") for idx in range(80, 100)]
    deleted = [Path(f"/docs/deleted-{idx:03d}.pdf") for idx in range(30)]
    drift = _FakeDriftState(added=added, modified=modified, deleted=deleted)

    fake_index = _FakeIndex()
    fake_chunk_cache = _FakeChunkCache()
    removed_calls: list[list[str]] = []
    run_step2_calls: list[list[dict[str, object]]] = []
    reports: list[dict[str, object]] = []

    monkeypatch.setattr(incremental_index, "scan_directory", lambda _root: list(all_files))
    monkeypatch.setattr(incremental_index, "load_scan_state", lambda _path: {})
    monkeypatch.setattr(incremental_index, "filter_incremental_rows", lambda rows, _state: (list(rows), {}))
    monkeypatch.setattr(incremental_index, "update_scan_state", lambda _state, _rows: {"version": 2})
    monkeypatch.setattr(incremental_index, "save_scan_state", lambda _path, _state: None)
    monkeypatch.setattr(incremental_index, "default_train_config", lambda: object())
    monkeypatch.setattr(incremental_index, "run_step2", lambda *args, **kwargs: run_step2_calls.append(list(args[0])))
    monkeypatch.setattr(incremental_index, "remove_from_corpus", lambda paths, _corpus: removed_calls.append(list(paths)))
    monkeypatch.setattr(incremental_index, "load_vector_index", lambda _cache_dir: fake_index)
    monkeypatch.setattr(incremental_index, "ChunkCache", lambda _cache_path: fake_chunk_cache)
    monkeypatch.setattr(
        incremental_index,
        "_write_incremental_report",
        lambda report, path=None: reports.append(dict(report)),
    )
    monkeypatch.setattr(
        incremental_index,
        "DriftDetector",
        lambda _cache: type(
            "Detector",
            (),
            {"detect_with_incremental": lambda self, _all, _to_process: drift},
        )(),
    )

    incremental_index.main()

    assert removed_calls
    assert set(removed_calls[0]) == {str(path) for path in deleted}
    assert fake_index.removed == {str(path) for path in deleted}
    assert fake_chunk_cache.dropped == {str(path) for path in deleted}
    assert len(run_step2_calls) == 1
    processed_paths = {str(row.get("path")) for row in run_step2_calls[0]}
    all_path_set = {str(row.get("path")) for row in all_files}
    expected_paths = {str(path) for path in (added + modified) if str(path) in all_path_set}
    assert processed_paths == expected_paths
    assert reports
    assert reports[-1]["missing_target_count"] == 1
    assert reports[-1]["run_step2_triggered"] is True


def test_run_incremental_index_writes_failed_report_on_exception_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    all_files = [{"path": "/docs/current.pdf", "size": 1, "mtime": 1.0}]
    drift = _FakeDriftState(added=[Path("/docs/current.pdf")], modified=[], deleted=[])
    reports: list[dict[str, object]] = []

    monkeypatch.setattr(incremental_index, "scan_directory", lambda _root: list(all_files))
    monkeypatch.setattr(incremental_index, "load_scan_state", lambda _path: {})
    monkeypatch.setattr(incremental_index, "filter_incremental_rows", lambda rows, _state: (list(rows), {}))
    monkeypatch.setattr(incremental_index, "update_scan_state", lambda _state, _rows: {"version": 1})
    monkeypatch.setattr(incremental_index, "save_scan_state", lambda _path, _state: None)
    monkeypatch.setattr(incremental_index, "default_train_config", lambda: object())
    monkeypatch.setattr(incremental_index, "remove_from_corpus", lambda _paths, _corpus: None)
    monkeypatch.setattr(
        incremental_index,
        "run_step2",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("run_step2 boom")),
    )
    monkeypatch.setattr(
        incremental_index,
        "_write_incremental_report",
        lambda report, path=None: reports.append(dict(report)),
    )
    monkeypatch.setattr(
        incremental_index,
        "DriftDetector",
        lambda _cache: type(
            "Detector",
            (),
            {"detect_with_incremental": lambda self, _all, _to_process: drift},
        )(),
    )

    with pytest.raises(RuntimeError, match="run_step2 boom"):
        incremental_index.main()

    assert reports
    final = reports[-1]
    assert final["status"] == "failed"
    assert str(final.get("started_at_utc") or "")
    assert str(final.get("finished_at_utc") or "")
    assert int(final.get("duration_ms") or 0) >= 0
    assert final["failed_phase"] == "run_step2"
    assert "run_step2 boom" in str(final.get("error") or "")
    assert final["scanned_total"] == 1
