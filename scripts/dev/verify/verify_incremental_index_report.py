"""Static verifier for incremental index run report JSON."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def default_report_path() -> Path:
    custom = os.getenv("INCREMENTAL_INDEX_REPORT_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path("data/cache/incremental_index_report.json")


def load_incremental_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso8601(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def evaluate_incremental_report(
    report: dict[str, Any],
    *,
    max_missing_targets: int = 0,
    allowed_statuses: tuple[str, ...] = ("completed", "no_changes"),
) -> list[str]:
    issues: list[str] = []
    if not report:
        return ["incremental report is missing or invalid JSON"]
    status = str(report.get("status") or "").strip()
    if status not in allowed_statuses:
        issues.append(f"unexpected status: {status!r}")

    started_at = _parse_iso8601(report.get("started_at_utc"))
    finished_at = _parse_iso8601(report.get("finished_at_utc"))
    if started_at is None:
        issues.append("started_at_utc is missing or invalid")
    if finished_at is None:
        issues.append("finished_at_utc is missing or invalid")
    try:
        duration_ms = int(report.get("duration_ms"))
    except (TypeError, ValueError):
        duration_ms = -1
        issues.append("duration_ms is missing or invalid")
    if duration_ms < 0:
        issues.append("duration_ms must be >= 0")
    if started_at is not None and finished_at is not None:
        elapsed_ms = int((finished_at - started_at).total_seconds() * 1000)
        if elapsed_ms < 0:
            issues.append("finished_at_utc must be >= started_at_utc")
        elif duration_ms >= 0 and abs(elapsed_ms - duration_ms) > 2000:
            issues.append("duration_ms is inconsistent with started_at_utc/finished_at_utc")

    required_int_fields = (
        "scanned_total",
        "changed_candidates",
        "added_count",
        "modified_count",
        "deleted_count",
        "deleted_reconciled_count",
        "processed_count",
        "missing_target_count",
    )
    int_values: dict[str, int] = {}
    for key in required_int_fields:
        raw = report.get(key)
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            issues.append(f"{key} is missing or invalid")
            continue
        if parsed < 0:
            issues.append(f"{key} must be >= 0")
            continue
        int_values[key] = parsed

    run_step2_triggered_raw = report.get("run_step2_triggered")
    if not isinstance(run_step2_triggered_raw, bool):
        issues.append("run_step2_triggered is missing or invalid")
    else:
        run_step2_triggered = bool(run_step2_triggered_raw)
        processed_count = int_values.get("processed_count", 0)
        if run_step2_triggered and processed_count <= 0:
            issues.append("processed_count must be > 0 when run_step2_triggered is true")
        if not run_step2_triggered and processed_count > 0:
            issues.append("processed_count must be 0 when run_step2_triggered is false")

    deleted_count = int_values.get("deleted_count", 0)
    deleted_reconciled_count = int_values.get("deleted_reconciled_count", 0)
    if deleted_reconciled_count > deleted_count:
        issues.append("deleted_reconciled_count cannot exceed deleted_count")

    target_count = int_values.get("added_count", 0) + int_values.get("modified_count", 0)
    processed_count = int_values.get("processed_count", 0)
    if processed_count > target_count:
        issues.append("processed_count cannot exceed added_count + modified_count")

    if status == "failed":
        failed_phase = str(report.get("failed_phase") or "").strip()
        error = str(report.get("error") or "").strip()
        if not failed_phase:
            issues.append("failed report missing `failed_phase`")
        if not error:
            issues.append("failed report missing `error`")

    try:
        missing_target_count = int(report.get("missing_target_count") or 0)
    except (TypeError, ValueError):
        missing_target_count = -1
    if missing_target_count < 0:
        issues.append("missing_target_count is invalid")
    elif missing_target_count > max(0, int(max_missing_targets)):
        issues.append(
            f"missing_target_count too high: {missing_target_count} > {max(0, int(max_missing_targets))}"
        )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify incremental_index_report.json contract")
    parser.add_argument("--report-path", type=Path, default=default_report_path(), help="Path to report JSON")
    parser.add_argument("--max-missing-targets", type=int, default=0, help="Allowed upper bound for missing targets")
    parser.add_argument(
        "--allow-status",
        action="append",
        dest="allow_statuses",
        default=[],
        help="Additional allowed status values",
    )
    args = parser.parse_args(argv)

    allow_statuses = ("completed", "no_changes")
    if args.allow_statuses:
        extras = tuple(str(item).strip() for item in args.allow_statuses if str(item).strip())
        if extras:
            allow_statuses = tuple(dict.fromkeys((*allow_statuses, *extras)))

    report = load_incremental_report(args.report_path)
    issues = evaluate_incremental_report(
        report,
        max_missing_targets=max(0, int(args.max_missing_targets)),
        allowed_statuses=allow_statuses,
    )
    if issues:
        print("[FAIL] incremental index report contract")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("[OK] incremental index report contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
