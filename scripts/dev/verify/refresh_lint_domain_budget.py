"""Refresh lint debt domain budget totals from current Ruff statistics.

This utility updates `budget_total` per domain in the domain budget file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOMAIN_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_domain_budget.json"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_domain_refresh_summary.md"
STAT_LINE_RE = re.compile(r"^\s*(\d+)\s+([A-Z]\d{3})\b")
DOMAIN_KEYS: tuple[str, ...] = ("engine", "ui_ux", "tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh lint debt domain budget totals")
    parser.add_argument(
        "--domain-budget-file",
        default=str(DEFAULT_DOMAIN_BUDGET_FILE),
        help="Path to lint domain budget JSON file",
    )
    parser.add_argument(
        "--summary-file",
        default=str(DEFAULT_SUMMARY_FILE),
        help="Path to markdown summary file",
    )
    return parser.parse_args()


def _parse_statistics(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Found "):
            continue
        match = STAT_LINE_RE.match(line)
        if not match:
            continue
        counts[match.group(2)] = int(match.group(1))
    return counts


def _run_ruff_statistics(paths: list[str]) -> tuple[dict[str, int], bool]:
    effective = [path for path in paths if (PROJECT_ROOT / path).exists()]
    if not effective:
        return {}, False
    proc = subprocess.run(
        ["ruff", "check", *effective, "--statistics"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return {}, False
    return _parse_statistics(proc.stdout), True


def _load_budget(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be object")
    domains = payload.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("key `domains` must be object")
    return payload


def _refresh_budget(payload: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    notes: list[str] = []
    domains = payload.get("domains", {})
    assert isinstance(domains, dict)

    for key in DOMAIN_KEYS:
        raw = domains.get(key, {})
        if not isinstance(raw, dict):
            notes.append(f"- {key}: skipped (invalid domain payload)")
            continue
        raw_paths = raw.get("paths", [])
        if not isinstance(raw_paths, list):
            notes.append(f"- {key}: skipped (invalid paths payload)")
            continue
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        counts, ok = _run_ruff_statistics(paths)
        if not ok:
            notes.append(f"- {key}: skipped (ruff statistics unavailable)")
            continue
        total = sum(counts.values())
        raw["budget_total"] = int(total)
        domains[key] = raw
        notes.append(f"- {key}: budget_total -> {total}")

    payload["domains"] = domains
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return payload, notes


def _write_summary(path: Path, payload: dict[str, object], notes: list[str]) -> None:
    domains = payload.get("domains", {})
    if not isinstance(domains, dict):
        domains = {}
    lines = [
        "# Lint Domain Refresh Summary",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc', 'n/a')}`",
        "",
        "## Domain Totals",
        "",
    ]
    for key in DOMAIN_KEYS:
        raw = domains.get(key, {})
        if not isinstance(raw, dict):
            lines.append(f"- {key}: unavailable")
            continue
        label = str(raw.get("label", key))
        budget_total = int(raw.get("budget_total", 0))
        raw_paths = raw.get("paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        path_list = ", ".join(str(path) for path in raw_paths if str(path).strip())
        lines.append(f"- {label}: `{budget_total}` ({path_list})")

    lines.extend(["", "## Refresh Notes", ""])
    if notes:
        lines.extend(notes)
    else:
        lines.append("- no refresh notes")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    budget_file = Path(args.domain_budget_file)
    summary_file = Path(args.summary_file)
    try:
        payload = _load_budget(budget_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] invalid lint domain budget file: {budget_file} ({exc})")
        return 1

    refreshed, notes = _refresh_budget(payload)
    budget_file.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary(summary_file, refreshed, notes)
    print(f"[OK] refreshed lint domain budget: {budget_file}")
    print(f"[OK] wrote lint domain summary: {summary_file}")
    for line in notes:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
