from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_METADATA_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "release" / "generate_release_metadata.py"
LINT_DOMAIN_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_domain_budget.json"

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_release_metadata_script_has_lint_domain_summary_contract() -> None:
    text = RELEASE_METADATA_SCRIPT.read_text(encoding="utf-8")
    for token in (
        "LINT_DOMAIN_BUDGET_PATH",
        "_load_lint_domain_budget",
        "_build_lint_domain_summary",
        "\"lint_domain_summary\"",
        "## Lint Debt Domain Delta",
    ):
        assert token in text


def test_lint_domain_budget_file_contract() -> None:
    payload = json.loads(LINT_DOMAIN_BUDGET_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "version" in payload
    domains = payload.get("domains")
    assert isinstance(domains, dict)
    for key in ("engine", "ui_ux", "tests"):
        domain = domains.get(key)
        assert isinstance(domain, dict)
        paths = domain.get("paths")
        assert isinstance(paths, list) and paths
        assert all(isinstance(path, str) and path.strip() for path in paths)
        assert isinstance(domain.get("budget_total"), int)
