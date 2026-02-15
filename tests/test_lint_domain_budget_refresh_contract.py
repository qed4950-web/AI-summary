from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"
REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "verify" / "refresh_lint_domain_budget.py"
DOMAIN_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_domain_budget.json"
SUMMARY_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_domain_refresh_summary.md"
PR_TEMPLATE_FILE = PROJECT_ROOT / ".github" / "pull_request_template.md"

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_refresh_lint_domain_budget_script_contract() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    for token in (
        "--domain-budget-file",
        "--summary-file",
        "DEFAULT_SUMMARY_FILE",
        "_write_summary",
        "DOMAIN_KEYS",
        "budget_total",
        "ruff",
        "generated_at_utc",
    ):
        assert token in text


def test_makefile_lint_domain_refresh_contract() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "LINT_DOMAIN_BUDGET_FILE ?=" in text
    assert "LINT_DOMAIN_SUMMARY_FILE ?=" in text
    assert "lint-debt-domain-refresh:" in text
    assert (
        'python scripts/dev/verify/refresh_lint_domain_budget.py --domain-budget-file "$(LINT_DOMAIN_BUDGET_FILE)" --summary-file "$(LINT_DOMAIN_SUMMARY_FILE)"'
        in text
    )
    assert "$(MAKE) lint-debt-domain-refresh" in text


def test_lint_domain_budget_core_keys_contract() -> None:
    payload = json.loads(DOMAIN_BUDGET_FILE.read_text(encoding="utf-8"))
    domains = payload.get("domains")
    assert isinstance(domains, dict)
    for key in ("engine", "ui_ux", "tests"):
        domain = domains.get(key)
        assert isinstance(domain, dict)
        assert isinstance(domain.get("budget_total"), int)


def test_lint_domain_refresh_summary_contract() -> None:
    text = SUMMARY_FILE.read_text(encoding="utf-8")
    for token in ("# Lint Domain Refresh Summary", "## Domain Totals", "## Refresh Notes"):
        assert token in text


def test_pr_template_mentions_lint_domain_refresh_contract() -> None:
    text = PR_TEMPLATE_FILE.read_text(encoding="utf-8")
    for token in ("## Lint Domain Refresh", "make lint-debt-domain-refresh", "lint_domain_refresh_summary.md"):
        assert token in text
