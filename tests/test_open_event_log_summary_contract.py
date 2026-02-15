from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev.verify.summarize_open_event_log import (
    evaluate_open_event_alerts,
    load_open_events,
    main,
    render_markdown_summary,
    summarize_open_events,
)

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_open_event_summary_counts_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "open-events.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "at_utc": "2026-02-15T01:00:00+00:00",
                "event": "open_darwin_default",
                "success": True,
                "category": "ok",
                "path": "/tmp/a.pdf",
            },
            {
                "at_utc": "2026-02-15T01:01:00+00:00",
                "event": "open_darwin_failed",
                "success": False,
                "category": "canceled",
                "path": "/tmp/a.pdf",
            },
            {
                "at_utc": "2026-02-15T01:02:00+00:00",
                "event": "open_darwin_failed",
                "success": False,
                "category": "association",
                "path": "/tmp/b.pdf",
            },
            {
                "at_utc": "2026-02-15T01:03:00+00:00",
                "event": "reveal_in_finder",
                "success": True,
                "category": "ok",
                "path": "/tmp/b.pdf",
            },
            {
                "at_utc": "2026-02-15T01:04:00+00:00",
                "event": "open_darwin_short_circuit",
                "success": False,
                "category": "canceled",
                "path": "/tmp/b.pdf",
            },
        ],
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("{bad json}\n")

    events = load_open_events(log_path)
    assert len(events) == 5

    summary = summarize_open_events(events, top_n=2)
    assert summary["total_events"] == 5
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 3
    assert summary["event_counts"]["open_darwin_failed"] == 2
    assert summary["event_counts"]["reveal_in_finder"] == 1
    assert summary["category_counts"]["canceled"] == 2
    assert summary["recovery_attempt_count"] == 1
    assert summary["recovery_success_count"] == 1
    assert summary["recovery_success_rate"] == 1.0
    assert summary["short_circuit_count"] == 1
    assert summary["short_circuit_rate"] == 0.2
    assert summary["top_failure_paths"][0]["path"] == "/tmp/b.pdf"
    assert summary["top_failure_paths"][0]["failures"] == 2

    alert = evaluate_open_event_alerts(summary, min_events=2, failure_rate_threshold=0.5, canceled_rate_threshold=0.2)
    assert alert["status"] == "alert"
    assert alert["alerts"]
    summary["alert"] = alert

    markdown = render_markdown_summary(summary)
    assert "# Open Event Summary" in markdown
    assert "## Category Counts" in markdown
    assert "## Alert Status" in markdown
    assert "## Recovery Effectiveness" in markdown
    assert "## Top Failure Paths" in markdown


def test_open_event_summary_cli_writes_outputs_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "open-events.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "at_utc": "2026-02-15T01:00:00+00:00",
                "event": "open_linux_xdg",
                "success": True,
                "category": "ok",
                "path": "/tmp/a.pdf",
            }
        ],
    )
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"

    rc = main(
        [
            "--log-path",
            str(log_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--top-n",
            "3",
        ]
    )
    assert rc == 0
    assert out_json.exists() is True
    assert out_md.exists() is True

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_events"] == 1
    assert payload["success_count"] == 1
    assert payload["alert"]["status"] == "ok"
    assert "Open Event Summary" in out_md.read_text(encoding="utf-8")


def test_open_event_summary_cli_fail_on_alert_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "open-events.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "at_utc": "2026-02-15T01:00:00+00:00",
                "event": "open_darwin_failed",
                "success": False,
                "category": "canceled",
                "path": "/tmp/a.pdf",
            },
            {
                "at_utc": "2026-02-15T01:01:00+00:00",
                "event": "open_darwin_failed",
                "success": False,
                "category": "canceled",
                "path": "/tmp/b.pdf",
            },
        ],
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "--min-events",
            "2",
            "--failure-rate-threshold",
            "0.5",
            "--canceled-rate-threshold",
            "0.5",
            "--fail-on-alert",
        ]
    )
    assert rc == 2


def test_open_event_summary_recovery_alert_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "open-events.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "at_utc": "2026-02-15T01:00:00+00:00",
                "event": "reveal_in_finder",
                "success": False,
                "category": "canceled",
                "path": "/tmp/a.pdf",
            },
            {
                "at_utc": "2026-02-15T01:01:00+00:00",
                "event": "reveal_in_finder",
                "success": False,
                "category": "canceled",
                "path": "/tmp/b.pdf",
            },
            {
                "at_utc": "2026-02-15T01:02:00+00:00",
                "event": "open_darwin_parent_fallback",
                "success": False,
                "category": "canceled",
                "path": "/tmp/c.pdf",
            },
        ],
    )

    summary = summarize_open_events(load_open_events(log_path), top_n=2)
    alert = evaluate_open_event_alerts(
        summary,
        min_events=100,
        failure_rate_threshold=1.0,
        canceled_rate_threshold=1.0,
        min_recovery_attempts=2,
        recovery_success_threshold=0.6,
    )
    assert alert["status"] == "alert"
    assert any("recovery_success_rate_low" in item for item in alert["alerts"])


def test_open_event_summary_short_circuit_alert_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "open-events.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "at_utc": "2026-02-15T01:00:00+00:00",
                "event": "open_darwin_short_circuit",
                "success": False,
                "category": "canceled",
                "path": "/tmp/a.pdf",
            },
            {
                "at_utc": "2026-02-15T01:01:00+00:00",
                "event": "open_darwin_short_circuit",
                "success": False,
                "category": "permission",
                "path": "/tmp/b.pdf",
            },
            {
                "at_utc": "2026-02-15T01:02:00+00:00",
                "event": "open_darwin_default",
                "success": True,
                "category": "ok",
                "path": "/tmp/c.pdf",
            },
        ],
    )
    summary = summarize_open_events(load_open_events(log_path), top_n=2)
    alert = evaluate_open_event_alerts(
        summary,
        min_events=3,
        failure_rate_threshold=1.0,
        canceled_rate_threshold=1.0,
        short_circuit_rate_threshold=0.5,
    )
    assert alert["status"] == "alert"
    assert any("short_circuit_rate_high" in item for item in alert["alerts"])
