"""
Incremental Indexing Script
Uses DriftDetector to update the index efficiently.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config.paths import DATA_DIR, DOCS_DIR
from core.data_pipeline.cache_manager import ChunkCache
from core.data_pipeline.drift import DriftDetector
from core.data_pipeline.incremental import (
    filter_incremental_rows,
    load_scan_state,
    save_scan_state,
    update_scan_state,
)
from core.data_pipeline.pipeline import default_train_config, remove_from_corpus, run_step2
from core.data_pipeline.scanner import scan_directory
from scripts.pipeline.infopilot_cli.pipeline_runner import load_vector_index

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IncrementalIndex")


def _report_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    custom = os.getenv("INCREMENTAL_INDEX_REPORT_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return DATA_DIR / "cache" / "incremental_index_report.json"


def _write_incremental_report(report: dict[str, object], path: Path | None = None) -> None:
    target = _report_path(path)
    payload = dict(report)
    payload.setdefault("reported_at_utc", datetime.now(timezone.utc).isoformat())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write incremental report: %s", exc)


def _drop_deleted_from_chunk_cache(*, cache_path: Path, deleted_paths: set[str]) -> None:
    if not deleted_paths:
        return
    try:
        cache = ChunkCache(cache_path)
        before = len(cache.known_paths())
        cache.drop_paths(deleted_paths)
        cache.save()
        after = len(cache.known_paths())
        dropped = max(0, before - after)
        if dropped > 0:
            logger.info("🧽 Removed %d deleted docs from chunk cache.", dropped)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.warning("Chunk cache cleanup failed for deleted docs: %s", exc)


def _reconcile_deleted_paths(*, deleted_paths: set[str], corpus_path: Path, cache_dir: Path, cache_path: Path) -> None:
    if not deleted_paths:
        return
    logger.info("🧹 Reconciling deleted docs in corpus/index: %d", len(deleted_paths))
    remove_from_corpus(sorted(deleted_paths), corpus_path)
    index = load_vector_index(cache_dir)
    index.remove_paths(set(deleted_paths))
    index.save(cache_dir)
    _drop_deleted_from_chunk_cache(cache_path=cache_path, deleted_paths=deleted_paths)


def main() -> None:
    logger.info("🚀 Starting Incremental Indexing...")

    # 0. Setup
    cache_path = DATA_DIR / "cache" / "chunk_cache.json"
    scan_state_path = DATA_DIR / "cache" / "scan_state.json"
    run_started_at = datetime.now(timezone.utc)
    report: dict[str, object] = {
        "status": "running",
        "started_at_utc": run_started_at.isoformat(),
        "finished_at_utc": "",
        "duration_ms": 0,
        "scanned_total": 0,
        "changed_candidates": 0,
        "added_count": 0,
        "modified_count": 0,
        "deleted_count": 0,
        "deleted_reconciled_count": 0,
        "run_step2_triggered": False,
        "processed_count": 0,
        "missing_target_count": 0,
    }
    phase = "scan"

    def _finalize_report() -> None:
        finished_at = datetime.now(timezone.utc)
        report["finished_at_utc"] = finished_at.isoformat()
        report["duration_ms"] = max(0, int((finished_at - run_started_at).total_seconds() * 1000))

    try:
        # 1. Scan
        all_files = scan_directory(DOCS_DIR)
        report["scanned_total"] = len(all_files)
        logger.info("📁 Scanned %d files in %s", len(all_files), DOCS_DIR)

        # 2. Incremental Filter (Mtime/Size)
        phase = "incremental_filter"
        scan_state = load_scan_state(scan_state_path)
        to_process, _cached_meta = filter_incremental_rows(all_files, scan_state)
        report["changed_candidates"] = len(to_process)
        logger.info("🔍 Changed/New candidates based on mtime: %d", len(to_process))

        # 3. Drift Detection (Add/Del vs Modified)
        phase = "drift_detection"
        detector = DriftDetector(cache_path)
        drift = detector.detect_with_incremental(all_files, to_process)
        report["added_count"] = len(drift.added)
        report["modified_count"] = len(drift.modified)
        report["deleted_count"] = len(drift.deleted)

        logger.info("📊 Drift Report: %s", drift.summary())
        if not drift.has_changes:
            logger.info("✅ No changes detected. Index is up to date.")
            report["status"] = "no_changes"
            _finalize_report()
            _write_incremental_report(report)
            return

        phase = "reconcile_deleted"
        deleted_paths = {str(path) for path in drift.deleted if str(path).strip()}
        _reconcile_deleted_paths(
            deleted_paths=deleted_paths,
            corpus_path=DATA_DIR / "corpus.parquet",
            cache_dir=DATA_DIR / "cache",
            cache_path=cache_path,
        )
        report["deleted_reconciled_count"] = len(deleted_paths)

        # 4. Process new/modified rows only.
        phase = "run_step2"
        targets = drift.added + drift.modified
        logger.info("🔄 Processing %d new/modified files...", len(targets))
        target_set = {str(path) for path in targets}
        run_rows = [row for row in all_files if str(row.get("path") or "") in target_set]
        run_path_set = {str(row.get("path") or "") for row in run_rows if str(row.get("path") or "").strip()}
        missing_targets = {path for path in target_set if path not in run_path_set}
        if missing_targets:
            logger.warning("⚠️ Drift target mismatch: %d path(s) were not found in scan rows.", len(missing_targets))
        report["missing_target_count"] = len(missing_targets)
        report["processed_count"] = len(run_rows)

        if run_rows:
            logger.info("⚙️ Triggering Pipeline (run_step2) with %d rows...", len(run_rows))
            report["run_step2_triggered"] = True
            cfg = default_train_config()
            run_step2(
                run_rows,
                out_corpus=DATA_DIR / "corpus.parquet",
                out_model=DATA_DIR / "topic_model.joblib",
                cfg=cfg,
                use_tqdm=True,
                translate=False,
                scan_state_path=scan_state_path,
                chunk_cache_path=cache_path,
            )
        else:
            logger.info("ℹ️ No effective rows to process after drift reconciliation.")

        # 5. Update state snapshot
        phase = "save_scan_state"
        new_state = update_scan_state(scan_state, all_files)
        save_scan_state(scan_state_path, new_state)
        report["status"] = "completed"
        _finalize_report()
        _write_incremental_report(report)
        logger.info("✅ Incremental Indexing Complete.")
    except Exception as exc:
        report["status"] = "failed"
        report["failed_phase"] = phase
        report["error"] = str(exc)[:600]
        _finalize_report()
        _write_incremental_report(report)
        logger.exception("Incremental indexing failed during phase `%s`: %s", phase, exc)
        raise


if __name__ == "__main__":
    main()
