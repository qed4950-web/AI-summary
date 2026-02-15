"""Generate release metadata and notes artifacts.

This script is designed for CI release workflows and does not execute tests.
It compiles static project signals into JSON and Markdown artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
MAKEFILE = PROJECT_ROOT / "Makefile"
TESTS_DIR = PROJECT_ROOT / "tests"
LINT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "lint" / "ruff_statistics.txt"
IMPACT_SCORE_POLICY_PATH = PROJECT_ROOT / "docs" / "plan" / "impact_score_policy.json"
LINT_DOMAIN_BUDGET_PATH = PROJECT_ROOT / "docs" / "plan" / "lint_debt_domain_budget.json"
MAX_CHANGE_FILES_PER_DOMAIN = 8

TARGET_VARS: tuple[str, ...] = ("SMOKE_TESTS", "INTEGRATION_TESTS", "STATIC_RUFF_TARGETS")
MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    "smoke": re.compile(r"pytest\.mark\.smoke"),
    "full": re.compile(r"pytest\.mark\.full"),
    "integration": re.compile(r"pytest\.mark\.integration"),
}
LINT_STAT_LINE_RE = re.compile(r"^\s*(\d+)\s+([A-Z]\d{3})\b")
CHANGE_DOMAINS: tuple[str, ...] = ("engine", "ui_ux", "platform")
LINT_DOMAIN_ORDER: tuple[str, ...] = ("engine", "ui_ux", "tests")
DEFAULT_IMPACT_SCORE_POLICY: dict[str, object] = {
    "version": "default",
    "base_score": 55,
    "weights": {
        "engine_per_file": 2,
        "ui_ux_per_file": 3,
        "platform_per_file": 1,
    },
    "caps": {
        "engine": 16,
        "ui_ux": 12,
        "platform": 8,
        "lint_penalty": 20,
    },
    "integration_bonus": {
        "threshold": 10,
        "met": 7,
        "not_met": 3,
    },
    "lint_penalty_divisor": 80,
    "tiers": {
        "medium_at": 60,
        "high_at": 75,
    },
}
DEFAULT_LINT_DOMAIN_BUDGET: dict[str, object] = {
    "version": "default",
    "domains": {
        "engine": {
            "label": "engine",
            "paths": ["core", "scripts/pipeline", "scripts/dev/verify"],
            "budget_total": 900,
        },
        "ui_ux": {
            "label": "ui/ux",
            "paths": ["desktop_app"],
            "budget_total": 500,
        },
        "tests": {
            "label": "tests",
            "paths": ["tests", "scripts/dev/tests"],
            "budget_total": 900,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate release metadata artifacts")
    parser.add_argument("--release-tag", default="", help="Release tag (vX.Y.Z)")
    parser.add_argument("--output-dir", default="artifacts/release", help="Artifact output directory")
    return parser.parse_args()


def _parse_makefile_targets() -> dict[str, list[str]]:
    if not MAKEFILE.exists():
        return {name: [] for name in TARGET_VARS}
    text = MAKEFILE.read_text(encoding="utf-8")
    targets: dict[str, list[str]] = {}
    for name in TARGET_VARS:
        match = re.search(rf"^{name}\s*=\s*(.+)$", text, flags=re.MULTILINE)
        targets[name] = match.group(1).split() if match else []
    return targets


def _count_markers() -> dict[str, int]:
    counts = Counter({name: 0 for name in MARKER_PATTERNS})
    for file_path in TESTS_DIR.rglob("*.py"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for marker, pattern in MARKER_PATTERNS.items():
            counts[marker] += len(pattern.findall(text))
    return dict(counts)


def _collect_workflows() -> list[str]:
    if not WORKFLOWS_DIR.exists():
        return []
    return sorted(path.name for path in WORKFLOWS_DIR.glob("*.yml"))


def _detect_release_tag(explicit_tag: str) -> str:
    if explicit_tag.strip():
        return explicit_tag.strip()
    for env_key in ("RELEASE_TAG", "GITHUB_REF_NAME"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return "unresolved"


def _run_git_command(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "-c", "color.ui=false", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]


def _collect_status_changed_files() -> list[str]:
    lines = _run_git_command(["status", "--porcelain"])
    paths: list[str] = []
    for line in lines:
        match = re.match(r"^..\s+(.+)$", line)
        payload = match.group(1).strip() if match else ""
        if not payload:
            continue
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1].strip()
        paths.append(payload)
    return sorted(set(paths))


def _collect_changed_files() -> list[str]:
    local_changes = _collect_status_changed_files()
    if local_changes and not os.getenv("GITHUB_SHA", "").strip():
        return local_changes

    before = os.getenv("GITHUB_EVENT_BEFORE", "").strip()
    sha = os.getenv("GITHUB_SHA", "").strip()
    if before and sha and before != "0000000000000000000000000000000000000000":
        changed = _run_git_command(["diff", "--name-only", f"{before}..{sha}"])
        if changed:
            return sorted(set(changed))

    changed = _run_git_command(["show", "--name-only", "--pretty=", "HEAD"])
    return sorted(set(changed))


def _classify_change_domain(path: str) -> str:
    if path.startswith(("desktop_app/", "tests/test_ui")):
        return "ui_ux"
    if path.startswith(("core/", "scripts/pipeline/", "scripts/dev/verify/")):
        return "engine"
    return "platform"


def _summarize_changes(paths: list[str]) -> dict[str, object]:
    grouped: dict[str, list[str]] = {domain: [] for domain in CHANGE_DOMAINS}
    for path in paths:
        grouped[_classify_change_domain(path)].append(path)

    compact: dict[str, object] = {
        domain: {
            "count": len(items),
            "files": items[:MAX_CHANGE_FILES_PER_DOMAIN],
            "truncated": len(items) > MAX_CHANGE_FILES_PER_DOMAIN,
        }
        for domain, items in grouped.items()
    }
    compact["total"] = len(paths)
    return compact


def _parse_lint_snapshot() -> list[dict[str, object]]:
    if not LINT_REPORT_PATH.exists():
        return []
    text = LINT_REPORT_PATH.read_text(encoding="utf-8")
    counts = _parse_lint_statistics(text)
    rows: list[dict[str, object]] = []
    for code, count in counts.items():
        rows.append({"code": code, "count": count})
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:8]


def _parse_lint_statistics(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = LINT_STAT_LINE_RE.match(line)
        if not match:
            continue
        counts[match.group(2)] = int(match.group(1))
    return counts


def _run_ruff_statistics_for_paths(paths: list[str]) -> tuple[dict[str, int], bool]:
    effective_paths = [path for path in paths if (PROJECT_ROOT / path).exists()]
    if not effective_paths:
        return {}, False
    try:
        proc = subprocess.run(
            ["ruff", "check", *effective_paths, "--statistics"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}, False
    if proc.returncode not in (0, 1):
        return {}, False
    return _parse_lint_statistics(proc.stdout), True


def _to_int(value: object, default: int, *, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _load_impact_score_policy() -> dict[str, object]:
    policy = json.loads(json.dumps(DEFAULT_IMPACT_SCORE_POLICY))
    source = "default"
    if IMPACT_SCORE_POLICY_PATH.exists():
        try:
            payload = json.loads(IMPACT_SCORE_POLICY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                source = str(IMPACT_SCORE_POLICY_PATH.relative_to(PROJECT_ROOT))
                policy["version"] = str(payload.get("version", policy["version"]))
                policy["base_score"] = _to_int(payload.get("base_score"), int(policy["base_score"]))

                raw_weights = payload.get("weights", {})
                if isinstance(raw_weights, dict):
                    weights = policy.get("weights", {})
                    if not isinstance(weights, dict):
                        weights = {}
                        policy["weights"] = weights
                    weights["engine_per_file"] = _to_int(
                        raw_weights.get("engine_per_file"),
                        _to_int(weights.get("engine_per_file"), 2),
                    )
                    weights["ui_ux_per_file"] = _to_int(
                        raw_weights.get("ui_ux_per_file"),
                        _to_int(weights.get("ui_ux_per_file"), 3),
                    )
                    weights["platform_per_file"] = _to_int(
                        raw_weights.get("platform_per_file"),
                        _to_int(weights.get("platform_per_file"), 1),
                    )

                raw_caps = payload.get("caps", {})
                if isinstance(raw_caps, dict):
                    caps = policy.get("caps", {})
                    if not isinstance(caps, dict):
                        caps = {}
                        policy["caps"] = caps
                    caps["engine"] = _to_int(raw_caps.get("engine"), _to_int(caps.get("engine"), 16))
                    caps["ui_ux"] = _to_int(raw_caps.get("ui_ux"), _to_int(caps.get("ui_ux"), 12))
                    caps["platform"] = _to_int(raw_caps.get("platform"), _to_int(caps.get("platform"), 8))
                    caps["lint_penalty"] = _to_int(raw_caps.get("lint_penalty"), _to_int(caps.get("lint_penalty"), 20))

                raw_integration_bonus = payload.get("integration_bonus", {})
                if isinstance(raw_integration_bonus, dict):
                    integration_bonus = policy.get("integration_bonus", {})
                    if not isinstance(integration_bonus, dict):
                        integration_bonus = {}
                        policy["integration_bonus"] = integration_bonus
                    integration_bonus["threshold"] = _to_int(
                        raw_integration_bonus.get("threshold"),
                        _to_int(integration_bonus.get("threshold"), 10),
                    )
                    integration_bonus["met"] = _to_int(
                        raw_integration_bonus.get("met"),
                        _to_int(integration_bonus.get("met"), 7),
                    )
                    integration_bonus["not_met"] = _to_int(
                        raw_integration_bonus.get("not_met"),
                        _to_int(integration_bonus.get("not_met"), 3),
                    )

                policy["lint_penalty_divisor"] = _to_int(
                    payload.get("lint_penalty_divisor"),
                    int(policy["lint_penalty_divisor"]),
                    min_value=1,
                )

                raw_tiers = payload.get("tiers", {})
                if isinstance(raw_tiers, dict):
                    tiers = policy.get("tiers", {})
                    if not isinstance(tiers, dict):
                        tiers = {}
                        policy["tiers"] = tiers
                    tiers["medium_at"] = _to_int(raw_tiers.get("medium_at"), _to_int(tiers.get("medium_at"), 60))
                    tiers["high_at"] = _to_int(raw_tiers.get("high_at"), _to_int(tiers.get("high_at"), 75))
        except (OSError, ValueError, json.JSONDecodeError):
            source = "default (invalid policy file)"
    policy["source"] = source
    return policy


def _load_lint_domain_budget() -> dict[str, object]:
    budget = json.loads(json.dumps(DEFAULT_LINT_DOMAIN_BUDGET))
    source = "default"
    if LINT_DOMAIN_BUDGET_PATH.exists():
        try:
            payload = json.loads(LINT_DOMAIN_BUDGET_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                source = str(LINT_DOMAIN_BUDGET_PATH.relative_to(PROJECT_ROOT))
                budget["version"] = str(payload.get("version", budget["version"]))
                raw_domains = payload.get("domains", {})
                if isinstance(raw_domains, dict):
                    current_domains = budget.get("domains", {})
                    if not isinstance(current_domains, dict):
                        current_domains = {}
                        budget["domains"] = current_domains
                    for domain in LINT_DOMAIN_ORDER:
                        raw_domain = raw_domains.get(domain)
                        if not isinstance(raw_domain, dict):
                            continue
                        current = current_domains.get(domain, {})
                        if not isinstance(current, dict):
                            current = {}
                            current_domains[domain] = current
                        current["label"] = str(raw_domain.get("label", current.get("label", domain))).strip() or domain
                        raw_paths = raw_domain.get("paths", current.get("paths", []))
                        if not isinstance(raw_paths, list):
                            raw_paths = []
                        current["paths"] = [str(path).strip() for path in raw_paths if str(path).strip()]
                        current["budget_total"] = _to_int(
                            raw_domain.get("budget_total"),
                            _to_int(current.get("budget_total"), 0),
                        )
        except (OSError, ValueError, json.JSONDecodeError):
            source = "default (invalid lint domain budget file)"
    budget["source"] = source
    return budget


def _build_lint_domain_summary(lint_domain_budget: dict[str, object]) -> dict[str, object]:
    raw_domains = lint_domain_budget.get("domains", {})
    if not isinstance(raw_domains, dict):
        return {"available": False, "domains": {}}

    domains: dict[str, dict[str, object]] = {}
    overall_available = True
    for domain in LINT_DOMAIN_ORDER:
        raw = raw_domains.get(domain, {})
        if not isinstance(raw, dict):
            raw = {}

        label = str(raw.get("label", domain)).strip() or domain
        raw_paths = raw.get("paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        budget_total = _to_int(raw.get("budget_total"), 0)

        counts, available = _run_ruff_statistics_for_paths(paths)
        overall_available = overall_available and available
        top_codes = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
        current_total = sum(counts.values())
        delta = current_total - budget_total
        trend = "flat"
        if delta > 0:
            trend = "increase"
        elif delta < 0:
            trend = "decrease"

        domains[domain] = {
            "label": label,
            "paths": paths,
            "budget_total": budget_total,
            "current_total": current_total,
            "delta": delta,
            "trend": trend,
            "top_codes": [{"code": code, "count": count} for code, count in top_codes],
            "available": available,
        }

    return {"available": overall_available, "domains": domains}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_change_intent_lines(change_summary: dict[str, object]) -> list[str]:
    lines: list[str] = []
    if change_summary["engine"]["count"] > 0:
        lines.append("- engine: 정책/파이프라인/검증 경계 안정화에 초점을 둔 변경이 포함되었습니다.")
    if change_summary["ui_ux"]["count"] > 0:
        lines.append("- ui/ux: 탐색/입력/피드백 동선을 단축하는 인터랙션 보강이 포함되었습니다.")
    if change_summary["platform"]["count"] > 0:
        lines.append("- platform: CI 게이트/릴리즈 거버넌스/문서 계약 정합성 변경이 포함되었습니다.")
    if not lines:
        lines.append("- 변경 파일이 감지되지 않았습니다.")
    return lines


def _build_risk_watchpoint_lines(
    lint_snapshot: list[dict[str, object]],
    makefile_targets: dict[str, list[str]],
    lint_domain_summary: dict[str, object],
) -> list[str]:
    lines: list[str] = []
    if lint_snapshot:
        top = lint_snapshot[0]
        lines.append(f"- lint debt: `{top['code']}`가 `{top['count']}`건으로 최다이며 회귀 감시가 필요합니다.")
    lines.append(
        f"- smoke/integration 표면: `{len(makefile_targets['SMOKE_TESTS'])}` / `{len(makefile_targets['INTEGRATION_TESTS'])}` 테스트로 관리됩니다."
    )
    raw_domains = lint_domain_summary.get("domains", {})
    if isinstance(raw_domains, dict):
        raised: list[str] = []
        for domain in LINT_DOMAIN_ORDER:
            payload = raw_domains.get(domain, {})
            if not isinstance(payload, dict):
                continue
            delta = _to_int(payload.get("delta"), 0, min_value=-10_000_000)
            if delta > 0:
                label = str(payload.get("label", domain))
                raised.append(f"{label} +{delta}")
        if raised:
            lines.append("- lint domain delta: " + ", ".join(raised) + " (budget 초과)")
    lines.append("- release 게이트 실패 시 artifact 업로드 누락 여부를 함께 확인해야 합니다.")
    return lines


def _build_rollback_lines(changed_files: list[str]) -> list[str]:
    priority_paths = (
        ".github/workflows/release.yml",
        ".github/workflows/integration.yml",
        ".github/workflows/lint.yml",
        "Makefile",
        "desktop_app/ui.py",
        "scripts/dev/release/generate_release_metadata.py",
    )
    lines: list[str] = []
    for path in priority_paths:
        if path in changed_files:
            lines.append(f"- `{path}`")
    if not lines:
        lines.append("- 핵심 롤백 포인트가 이번 변경 목록에 없습니다.")
    return lines


def _build_impact_score(
    change_summary: dict[str, object],
    lint_snapshot: list[dict[str, object]],
    makefile_targets: dict[str, list[str]],
    policy: dict[str, object],
) -> dict[str, object]:
    engine_count = int(change_summary["engine"]["count"])
    ui_count = int(change_summary["ui_ux"]["count"])
    platform_count = int(change_summary["platform"]["count"])
    integration_surface = len(makefile_targets["INTEGRATION_TESTS"])
    top_lint = int(lint_snapshot[0]["count"]) if lint_snapshot else 0

    weights = policy.get("weights", {})
    caps = policy.get("caps", {})
    integration_bonus = policy.get("integration_bonus", {})
    tiers = policy.get("tiers", {})
    if not isinstance(weights, dict):
        weights = {}
    if not isinstance(caps, dict):
        caps = {}
    if not isinstance(integration_bonus, dict):
        integration_bonus = {}
    if not isinstance(tiers, dict):
        tiers = {}

    engine_per_file = _to_int(weights.get("engine_per_file"), 2)
    ui_ux_per_file = _to_int(weights.get("ui_ux_per_file"), 3)
    platform_per_file = _to_int(weights.get("platform_per_file"), 1)
    engine_cap = _to_int(caps.get("engine"), 16)
    ui_ux_cap = _to_int(caps.get("ui_ux"), 12)
    platform_cap = _to_int(caps.get("platform"), 8)
    lint_penalty_cap = _to_int(caps.get("lint_penalty"), 20)
    integration_threshold = _to_int(integration_bonus.get("threshold"), 10)
    integration_bonus_met = _to_int(integration_bonus.get("met"), 7)
    integration_bonus_not_met = _to_int(integration_bonus.get("not_met"), 3)
    medium_at = _to_int(tiers.get("medium_at"), 60)
    high_at = _to_int(tiers.get("high_at"), 75)
    base_score = _to_int(policy.get("base_score"), 55)
    lint_penalty_divisor = _to_int(policy.get("lint_penalty_divisor"), 80, min_value=1)

    score = base_score
    score += min(engine_cap, engine_count * engine_per_file)
    score += min(ui_ux_cap, ui_count * ui_ux_per_file)
    score += min(platform_cap, platform_count * platform_per_file)
    score += (
        integration_bonus_met
        if integration_surface >= integration_threshold
        else integration_bonus_not_met
    )
    score -= min(lint_penalty_cap, top_lint // lint_penalty_divisor)
    score = max(0, min(100, score))

    tier = "low"
    if score >= high_at:
        tier = "high"
    elif score >= medium_at:
        tier = "medium"

    return {
        "score": score,
        "tier": tier,
        "top_lint_code": lint_snapshot[0]["code"] if lint_snapshot else "n/a",
        "top_lint_count": top_lint,
        "integration_surface": integration_surface,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    marker_counts = payload["marker_counts"]
    makefile_targets = payload["makefile_targets"]
    change_summary = payload["change_summary"]
    lint_snapshot = payload["lint_snapshot_top"]
    lint_domain_summary = payload.get("lint_domain_summary", {})
    changed_files = payload["changed_files"]
    impact = payload["impact_score"]
    impact_policy = payload["impact_score_policy"]
    if not isinstance(impact_policy, dict):
        impact_policy = {"source": "default", "version": "default"}

    def _fmt_domain(domain_key: str, label: str) -> list[str]:
        domain = change_summary[domain_key]
        lines = [f"- {label}: `{domain['count']}` files"]
        for file_path in domain["files"]:
            lines.append(f"  - `{file_path}`")
        if domain["truncated"]:
            lines.append("  - `...`")
        return lines

    lines = [
        "# Release Notes (Auto)",
        "",
        f"- Generated at: `{payload['generated_at_utc']}`",
        f"- Release tag: `{payload['release_tag']}`",
        f"- Commit: `{payload['commit_sha']}`",
        "",
        "## Workflow Coverage",
        "",
        f"- Workflows: `{', '.join(payload['workflows'])}`",
        "",
        "## Test Marker Snapshot",
        "",
        f"- smoke: `{marker_counts['smoke']}`",
        f"- full: `{marker_counts['full']}`",
        f"- integration: `{marker_counts['integration']}`",
        "",
        "## Gate Surface Snapshot",
        "",
        f"- smoke tests: `{len(makefile_targets['SMOKE_TESTS'])}`",
        f"- integration tests: `{len(makefile_targets['INTEGRATION_TESTS'])}`",
        f"- static lint targets: `{len(makefile_targets['STATIC_RUFF_TARGETS'])}`",
        "",
        "## Engine/UI/UX Change Summary",
        "",
        f"- total changed files: `{change_summary['total']}`",
    ]
    lines.extend(_fmt_domain("engine", "engine"))
    lines.extend(_fmt_domain("ui_ux", "ui/ux"))
    lines.extend(_fmt_domain("platform", "platform"))

    lines.extend(["", "## Lint Debt Top Codes", ""])
    if lint_snapshot:
        for row in lint_snapshot:
            lines.append(f"- `{row['code']}`: `{row['count']}`")
    else:
        lines.append("- lint snapshot not found (`artifacts/lint/ruff_statistics.txt`)")

    lines.extend(["", "## Lint Debt Domain Delta", ""])
    raw_domains = lint_domain_summary.get("domains", {}) if isinstance(lint_domain_summary, dict) else {}
    if isinstance(raw_domains, dict) and raw_domains:
        for domain in LINT_DOMAIN_ORDER:
            domain_payload = raw_domains.get(domain, {})
            if not isinstance(domain_payload, dict):
                continue
            label = str(domain_payload.get("label", domain))
            current_total = _to_int(domain_payload.get("current_total"), 0)
            budget_total = _to_int(domain_payload.get("budget_total"), 0)
            delta = _to_int(domain_payload.get("delta"), 0, min_value=-10_000_000)
            trend = str(domain_payload.get("trend", "flat"))
            lines.append(
                f"- {label}: current `{current_total}` / budget `{budget_total}` / delta `{delta:+d}` ({trend})"
            )
            top_codes = domain_payload.get("top_codes", [])
            if isinstance(top_codes, list) and top_codes:
                top = ", ".join(
                    f"{entry.get('code', 'n/a')}:{_to_int(entry.get('count'), 0)}"
                    for entry in top_codes
                    if isinstance(entry, dict)
                )
                if top:
                    lines.append(f"  - top codes: `{top}`")
    else:
        lines.append("- lint domain summary unavailable")

    lines.extend(["", "## Change Intent", ""])
    lines.extend(_build_change_intent_lines(change_summary))

    lines.extend(["", "## Risk Watchpoints", ""])
    lines.extend(_build_risk_watchpoint_lines(lint_snapshot, makefile_targets, lint_domain_summary))

    lines.extend(["", "## Rollback Points", ""])
    lines.extend(_build_rollback_lines(changed_files))

    lines.extend(["", "## Impact Score Policy", ""])
    lines.append(f"- source: `{impact_policy.get('source', 'default')}`")
    lines.append(f"- version: `{impact_policy.get('version', 'default')}`")

    lines.extend(["", "## Impact Score", ""])
    lines.append(f"- score: `{impact['score']}/100` ({impact['tier']})")
    lines.append(f"- integration surface: `{impact['integration_surface']}` tests")
    lines.append(f"- top lint pressure: `{impact['top_lint_code']}` `{impact['top_lint_count']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    release_tag = _detect_release_tag(args.release_tag)
    changed_files = _collect_changed_files()
    makefile_targets = _parse_makefile_targets()
    change_summary = _summarize_changes(changed_files)
    lint_snapshot = _parse_lint_snapshot()
    lint_domain_budget = _load_lint_domain_budget()
    lint_domain_summary = _build_lint_domain_summary(lint_domain_budget)
    impact_score_policy = _load_impact_score_policy()
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_tag": release_tag,
        "commit_sha": os.getenv("GITHUB_SHA", "local"),
        "workflows": _collect_workflows(),
        "marker_counts": _count_markers(),
        "makefile_targets": makefile_targets,
        "change_summary": change_summary,
        "lint_snapshot_top": lint_snapshot,
        "lint_domain_budget": lint_domain_budget,
        "lint_domain_summary": lint_domain_summary,
        "changed_files": changed_files,
        "impact_score_policy": impact_score_policy,
        "impact_score": _build_impact_score(change_summary, lint_snapshot, makefile_targets, impact_score_policy),
    }

    json_path = output_dir / "release_metadata.json"
    markdown_path = output_dir / "RELEASE_NOTES.auto.md"
    _write_json(json_path, payload)
    _write_markdown(markdown_path, payload)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
