from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev.verify.verify_incremental_index_report import (
    evaluate_incremental_report,
    load_incremental_report,
    main,
)

pytestmark = [pytest.mark.smoke, pytest.mark.integration]
_STARTED_AT = "2026-02-15T00:00:00+00:00"
_FINISHED_AT = "2026-02-15T00:00:01+00:00"
_DURATION_MS = 1000


def test_incremental_report_eval_pass_contract() -> None:
    issues = evaluate_incremental_report(
        {
            "status": "completed",
            "started_at_utc": _STARTED_AT,
            "finished_at_utc": _FINISHED_AT,
            "duration_ms": _DURATION_MS,
            "scanned_total": 12,
            "changed_candidates": 4,
            "added_count": 2,
            "modified_count": 1,
            "deleted_count": 1,
            "deleted_reconciled_count": 1,
            "run_step2_triggered": True,
            "processed_count": 3,
            "missing_target_count": 0,
        },
        max_missing_targets=0,
    )
    assert issues == []


def test_incremental_report_eval_fail_contract() -> None:
    issues = evaluate_incremental_report(
        {
            "status": "completed",
            "started_at_utc": _STARTED_AT,
            "finished_at_utc": _FINISHED_AT,
            "duration_ms": _DURATION_MS,
            "scanned_total": 12,
            "changed_candidates": 4,
            "added_count": 2,
            "modified_count": 1,
            "deleted_count": 1,
            "deleted_reconciled_count": 1,
            "run_step2_triggered": True,
            "processed_count": 3,
            "missing_target_count": 3,
        },
        max_missing_targets=1,
    )
    assert issues
    assert "missing_target_count too high" in issues[0]


def test_incremental_report_eval_fail_consistency_contract() -> None:
    issues = evaluate_incremental_report(
        {
            "status": "completed",
            "started_at_utc": _STARTED_AT,
            "finished_at_utc": _FINISHED_AT,
            "duration_ms": _DURATION_MS,
            "scanned_total": 1,
            "changed_candidates": 0,
            "added_count": 1,
            "modified_count": 0,
            "deleted_count": 0,
            "deleted_reconciled_count": 0,
            "run_step2_triggered": False,
            "processed_count": 1,
            "missing_target_count": 0,
        }
    )
    assert issues
    assert any("processed_count must be 0 when run_step2_triggered is false" in item for item in issues)


def test_incremental_report_eval_failed_status_contract() -> None:
    issues = evaluate_incremental_report(
        {
            "status": "failed",
            "started_at_utc": _STARTED_AT,
            "finished_at_utc": _FINISHED_AT,
            "duration_ms": _DURATION_MS,
            "scanned_total": 1,
            "changed_candidates": 1,
            "added_count": 1,
            "modified_count": 0,
            "deleted_count": 0,
            "deleted_reconciled_count": 0,
            "run_step2_triggered": True,
            "processed_count": 1,
            "missing_target_count": 0,
            "failed_phase": "run_step2",
            "error": "boom",
        },
        allowed_statuses=("completed", "no_changes", "failed"),
    )
    assert issues == []


def test_incremental_report_eval_invalid_timing_contract() -> None:
    issues = evaluate_incremental_report(
        {
            "status": "completed",
            "started_at_utc": "2026-02-15T00:00:02+00:00",
            "finished_at_utc": "2026-02-15T00:00:01+00:00",
            "duration_ms": 1000,
            "scanned_total": 1,
            "changed_candidates": 1,
            "added_count": 1,
            "modified_count": 0,
            "deleted_count": 0,
            "deleted_reconciled_count": 0,
            "run_step2_triggered": True,
            "processed_count": 1,
            "missing_target_count": 0,
        }
    )
    assert issues
    assert any("finished_at_utc must be >= started_at_utc" in item for item in issues)


def test_incremental_report_cli_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "incremental_index_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "started_at_utc": _STARTED_AT,
                "finished_at_utc": _FINISHED_AT,
                "duration_ms": _DURATION_MS,
                "scanned_total": 1,
                "changed_candidates": 1,
                "added_count": 1,
                "modified_count": 0,
                "deleted_count": 0,
                "deleted_reconciled_count": 0,
                "run_step2_triggered": True,
                "processed_count": 1,
                "missing_target_count": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_incremental_report(report_path)
    assert loaded["status"] == "completed"

    ok_rc = main(["--report-path", str(report_path), "--max-missing-targets", "0"])
    assert ok_rc == 0

    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "started_at_utc": _STARTED_AT,
                "finished_at_utc": _FINISHED_AT,
                "duration_ms": _DURATION_MS,
                "scanned_total": 1,
                "changed_candidates": 1,
                "added_count": 1,
                "modified_count": 0,
                "deleted_count": 0,
                "deleted_reconciled_count": 0,
                "run_step2_triggered": True,
                "processed_count": 1,
                "missing_target_count": 2,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    fail_rc = main(["--report-path", str(report_path), "--max-missing-targets", "0"])
    assert fail_rc == 1
