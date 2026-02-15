"""Summarize desktop file-open telemetry events (JSONL)."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def default_log_path() -> Path:
    custom = os.getenv("DESKTOP_OPEN_EVENT_LOG_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".ai-summary" / "desktop_file_open_events.jsonl"


def load_open_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                events.append(payload)
    except OSError:
        return []
    return events


def summarize_open_events(events: list[dict[str, Any]], *, top_n: int = 5) -> dict[str, Any]:
    total = len(events)
    successes = 0
    failures = 0
    category_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    failure_paths: defaultdict[str, int] = defaultdict(int)
    recovery_attempt_count = 0
    recovery_success_count = 0
    short_circuit_count = 0

    recovery_events = {
        "open_darwin_reveal_fallback",
        "open_darwin_parent_fallback",
        "open_parent_due_missing_file",
        "reveal_in_finder",
    }

    min_ts = ""
    max_ts = ""
    for row in events:
        event_name = str(row.get("event") or "").strip() or "unknown"
        category = str(row.get("category") or "").strip() or "generic"
        success = bool(row.get("success"))
        path = str(row.get("path") or "").strip()
        at_utc = str(row.get("at_utc") or "").strip()

        event_counts[event_name] += 1
        category_counts[category] += 1
        if event_name in recovery_events:
            recovery_attempt_count += 1
            if success:
                recovery_success_count += 1
        if event_name == "open_darwin_short_circuit":
            short_circuit_count += 1
        if success:
            successes += 1
        else:
            failures += 1
            if path:
                failure_paths[path] += 1

        if at_utc:
            if not min_ts or at_utc < min_ts:
                min_ts = at_utc
            if not max_ts or at_utc > max_ts:
                max_ts = at_utc

    top_fail_paths = [
        {"path": path, "failures": count}
        for path, count in sorted(failure_paths.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, top_n)]
    ]
    return {
        "total_events": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": round((successes / total), 4) if total else 0.0,
        "recovery_attempt_count": recovery_attempt_count,
        "recovery_success_count": recovery_success_count,
        "recovery_success_rate": round((recovery_success_count / recovery_attempt_count), 4)
        if recovery_attempt_count
        else 0.0,
        "short_circuit_count": short_circuit_count,
        "short_circuit_rate": round((short_circuit_count / total), 4) if total else 0.0,
        "window_start_utc": min_ts,
        "window_end_utc": max_ts,
        "event_counts": dict(event_counts),
        "category_counts": dict(category_counts),
        "top_failure_paths": top_fail_paths,
    }


def evaluate_open_event_alerts(
    summary: dict[str, Any],
    *,
    min_events: int = 20,
    failure_rate_threshold: float = 0.35,
    canceled_rate_threshold: float = 0.2,
    min_recovery_attempts: int = 5,
    recovery_success_threshold: float = 0.7,
    short_circuit_rate_threshold: float = 0.15,
) -> dict[str, Any]:
    total = int(summary.get("total_events") or 0)
    failures = int(summary.get("failure_count") or 0)
    recovery_attempt_count = int(summary.get("recovery_attempt_count") or 0)
    recovery_success_count = int(summary.get("recovery_success_count") or 0)
    short_circuit_count = int(summary.get("short_circuit_count") or 0)
    categories = summary.get("category_counts", {})
    if isinstance(categories, dict):
        canceled = int(categories.get("canceled") or 0)
    else:
        canceled = 0
    failure_rate = (failures / total) if total else 0.0
    canceled_rate = (canceled / total) if total else 0.0
    recovery_success_rate = (recovery_success_count / recovery_attempt_count) if recovery_attempt_count else 0.0
    short_circuit_rate = (short_circuit_count / total) if total else 0.0
    alerts: list[str] = []

    if total >= max(1, int(min_events)):
        if failure_rate >= float(failure_rate_threshold):
            alerts.append(
                f"failure_rate_high: {failure_rate:.2%} >= {float(failure_rate_threshold):.2%} (events={total})"
            )
        if canceled_rate >= float(canceled_rate_threshold):
            alerts.append(
                f"canceled_rate_high: {canceled_rate:.2%} >= {float(canceled_rate_threshold):.2%} (events={total})"
            )
        if short_circuit_rate >= float(short_circuit_rate_threshold):
            alerts.append(
                f"short_circuit_rate_high: {short_circuit_rate:.2%} >= {float(short_circuit_rate_threshold):.2%} "
                f"(events={total})"
            )
    if recovery_attempt_count >= max(1, int(min_recovery_attempts)):
        if recovery_success_rate < float(recovery_success_threshold):
            alerts.append(
                "recovery_success_rate_low: "
                f"{recovery_success_rate:.2%} < {float(recovery_success_threshold):.2%} "
                f"(attempts={recovery_attempt_count})"
            )

    return {
        "status": "alert" if alerts else "ok",
        "alerts": alerts,
        "failure_rate": round(failure_rate, 4),
        "canceled_rate": round(canceled_rate, 4),
        "recovery_success_rate": round(recovery_success_rate, 4),
        "short_circuit_rate": round(short_circuit_rate, 4),
        "min_events": max(1, int(min_events)),
        "failure_rate_threshold": float(failure_rate_threshold),
        "canceled_rate_threshold": float(canceled_rate_threshold),
        "min_recovery_attempts": max(1, int(min_recovery_attempts)),
        "recovery_success_threshold": float(recovery_success_threshold),
        "short_circuit_rate_threshold": float(short_circuit_rate_threshold),
    }


def render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Open Event Summary",
        "",
        f"- total_events: {int(summary.get('total_events') or 0)}",
        f"- success_count: {int(summary.get('success_count') or 0)}",
        f"- failure_count: {int(summary.get('failure_count') or 0)}",
        f"- success_rate: {float(summary.get('success_rate') or 0.0):.2%}",
        f"- recovery_attempt_count: {int(summary.get('recovery_attempt_count') or 0)}",
        f"- recovery_success_count: {int(summary.get('recovery_success_count') or 0)}",
        f"- recovery_success_rate: {float(summary.get('recovery_success_rate') or 0.0):.2%}",
        f"- short_circuit_count: {int(summary.get('short_circuit_count') or 0)}",
        f"- short_circuit_rate: {float(summary.get('short_circuit_rate') or 0.0):.2%}",
        f"- window_start_utc: {summary.get('window_start_utc') or '-'}",
        f"- window_end_utc: {summary.get('window_end_utc') or '-'}",
        "",
        "## Category Counts",
    ]
    category_counts = summary.get("category_counts", {})
    if isinstance(category_counts, dict) and category_counts:
        for key, value in sorted(category_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Alert Status")
    alert = summary.get("alert", {})
    alert_status = "ok"
    if isinstance(alert, dict):
        alert_status = str(alert.get("status") or "ok")
    lines.append(f"- status: {alert_status}")
    if isinstance(alert, dict):
        lines.append(f"- failure_rate: {float(alert.get('failure_rate') or 0.0):.2%}")
        lines.append(f"- canceled_rate: {float(alert.get('canceled_rate') or 0.0):.2%}")
        lines.append(f"- recovery_success_rate: {float(alert.get('recovery_success_rate') or 0.0):.2%}")
        lines.append(f"- short_circuit_rate: {float(alert.get('short_circuit_rate') or 0.0):.2%}")
        raw_alerts = alert.get("alerts", [])
        alerts = raw_alerts if isinstance(raw_alerts, list) else []
        if alerts:
            for item in alerts:
                lines.append(f"- alert: {item}")

    lines.append("")
    lines.append("## Recovery Effectiveness")
    lines.append(f"- attempts: {int(summary.get('recovery_attempt_count') or 0)}")
    lines.append(f"- successes: {int(summary.get('recovery_success_count') or 0)}")
    lines.append(f"- success_rate: {float(summary.get('recovery_success_rate') or 0.0):.2%}")
    lines.append(f"- short_circuit_count: {int(summary.get('short_circuit_count') or 0)}")
    lines.append("")
    lines.append("## Top Failure Paths")
    top_failure_paths = summary.get("top_failure_paths", [])
    if isinstance(top_failure_paths, list) and top_failure_paths:
        for row in top_failure_paths:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('path')}: {int(row.get('failures') or 0)}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize desktop file-open telemetry events.")
    parser.add_argument("--log-path", type=Path, default=default_log_path(), help="Input JSONL log path")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON summary path")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional output Markdown summary path")
    parser.add_argument("--top-n", type=int, default=5, help="Top failure paths count")
    parser.add_argument("--min-events", type=int, default=20, help="Minimum events before alert thresholds apply")
    parser.add_argument("--failure-rate-threshold", type=float, default=0.35, help="Alert threshold for failure rate")
    parser.add_argument("--canceled-rate-threshold", type=float, default=0.2, help="Alert threshold for canceled rate")
    parser.add_argument("--min-recovery-attempts", type=int, default=5, help="Minimum recovery attempts before threshold applies")
    parser.add_argument(
        "--recovery-success-threshold",
        type=float,
        default=0.7,
        help="Alert threshold for recovery success rate (alert when below threshold)",
    )
    parser.add_argument(
        "--short-circuit-rate-threshold",
        type=float,
        default=0.15,
        help="Alert threshold for open short-circuit rate",
    )
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit non-zero when alert status is raised")
    args = parser.parse_args(argv)

    events = load_open_events(args.log_path)
    summary = summarize_open_events(events, top_n=max(1, int(args.top_n)))
    summary["alert"] = evaluate_open_event_alerts(
        summary,
        min_events=max(1, int(args.min_events)),
        failure_rate_threshold=max(0.0, float(args.failure_rate_threshold)),
        canceled_rate_threshold=max(0.0, float(args.canceled_rate_threshold)),
        min_recovery_attempts=max(1, int(args.min_recovery_attempts)),
        recovery_success_threshold=max(0.0, min(1.0, float(args.recovery_success_threshold))),
        short_circuit_rate_threshold=max(0.0, min(1.0, float(args.short_circuit_rate_threshold))),
    )
    markdown = render_markdown_summary(summary)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")

    if args.out_json is None and args.out_md is None:
        print(markdown)
    if args.fail_on_alert:
        alert = summary.get("alert", {})
        if isinstance(alert, dict) and str(alert.get("status") or "ok") == "alert":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
