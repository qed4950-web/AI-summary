"""Verify Ruff lint debt stays within the configured budget.

This checker is intended for CI and local static governance checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_budget.json"
STAT_LINE_PREFIX = "[*]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Ruff lint debt budget")
    parser.add_argument(
        "--budget-file",
        default=str(DEFAULT_BUDGET_FILE),
        help="Path to lint debt budget JSON file",
    )
    parser.add_argument(
        "--report-file",
        default="",
        help="Optional path to ruff statistics report file",
    )
    parser.add_argument(
        "--slack",
        type=int,
        default=0,
        help="Allowed increment per lint code above budget",
    )
    parser.add_argument(
        "--write-current",
        action="store_true",
        help="Write current statistics as new budget baseline",
    )
    return parser.parse_args()


def _parse_statistics(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Found ") or line.startswith(STAT_LINE_PREFIX):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if not parts[0].isdigit():
            continue
        code = parts[1]
        if len(code) != 4 or not code[0].isalpha() or not code[1:].isdigit():
            continue
        counts[code] = int(parts[0])
    return counts


def _run_ruff_statistics() -> str:
    proc = subprocess.run(
        ["ruff", "check", ".", "--statistics"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _load_statistics(report_file: Path | None) -> dict[str, int]:
    text = ""
    if report_file and report_file.exists():
        text = report_file.read_text(encoding="utf-8")
    else:
        text = _run_ruff_statistics()
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(text, encoding="utf-8")
    return _parse_statistics(text)


def _load_budget(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_codes = payload.get("codes", {})
    if not isinstance(raw_codes, dict):
        return {}
    result: dict[str, int] = {}
    for code, count in raw_codes.items():
        if isinstance(code, str) and isinstance(count, int):
            result[code] = count
    return result


def _write_budget(path: Path, counts: dict[str, int]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "codes": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_top_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for code, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]:
        print(f"  - {code}: {count}")


def main() -> int:
    args = parse_args()
    budget_file = Path(args.budget_file)
    report_file = Path(args.report_file) if args.report_file else None

    current_counts = _load_statistics(report_file)
    if not current_counts:
        print("[FAIL] no lint statistics parsed from Ruff output")
        return 1

    if args.write_current:
        _write_budget(budget_file, current_counts)
        print(f"[OK] wrote lint debt budget baseline: {budget_file}")
        _print_top_counts("Current lint debt snapshot:", current_counts)
        return 0

    budget_counts = _load_budget(budget_file)
    if not budget_counts:
        print(f"[FAIL] missing or invalid lint debt budget: {budget_file}")
        print("Run with --write-current once to initialize baseline.")
        return 1

    failures: list[str] = []
    all_codes = sorted(set(budget_counts) | set(current_counts))
    for code in all_codes:
        budget = budget_counts.get(code, 0)
        current = current_counts.get(code, 0)
        if current > budget + args.slack:
            failures.append(f"{code}: current {current} > budget {budget} (+slack {args.slack})")

    _print_top_counts("Current lint debt snapshot:", current_counts)
    if failures:
        print("[FAIL] lint debt budget exceeded")
        for issue in failures:
            print(f"  - {issue}")
        return 1

    print(f"[OK] lint debt within budget ({budget_file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
