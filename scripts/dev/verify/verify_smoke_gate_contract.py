"""Static guard for smoke workflow gate and test-surface contracts.

Ensures smoke CI and local smoke-check command stay aligned for critical
engine resilience coverage.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMOKE_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "smoke.yml"
MAKEFILE = PROJECT_ROOT / "Makefile"

REQUIRED_WORKFLOW_TOKENS: tuple[str, ...] = (
    "name: Baseline smoke tests",
    "id: baseline_smoke",
    "name: Engine resilience contract tests",
    "id: engine_resilience",
    "name: Publish smoke summary",
    "steps.baseline_smoke.outcome",
    "steps.engine_resilience.outcome",
)

REQUIRED_SMOKE_TEST_PATHS: tuple[str, ...] = (
    "tests/test_infopilot_cli_contract.py",
    "tests/test_policy_engine.py",
    "tests/test_pipeline_sensitive_paths.py",
    "tests/test_meeting_policy_scope.py",
    "tests/test_photo_policy_scope.py",
    "tests/test_watch_cli_dependencies.py",
    "tests/test_watch_event_handler_contract.py",
    "tests/test_pipeline_policy_provider_contract.py",
    "tests/test_pipeline_runner_watch_loop.py",
    "tests/test_desktop_backend_mode_contract.py",
    "tests/test_llm_client_option_contract.py",
    "tests/test_mode_profiles_config_contract.py",
    "tests/test_release_metadata_lint_domain_contract.py",
    "tests/test_lint_domain_budget_refresh_contract.py",
)


def _check_workflow() -> list[str]:
    if not SMOKE_WORKFLOW.exists():
        return [f"missing file: {SMOKE_WORKFLOW}"]
    text = SMOKE_WORKFLOW.read_text(encoding="utf-8")
    failures: list[str] = []

    for token in REQUIRED_WORKFLOW_TOKENS:
        if token not in text:
            failures.append(f"smoke.yml: missing token `{token}`")

    for test_path in REQUIRED_SMOKE_TEST_PATHS:
        if test_path not in text:
            failures.append(f"smoke.yml: missing smoke test `{test_path}`")

    return failures


def _check_makefile() -> list[str]:
    if not MAKEFILE.exists():
        return [f"missing file: {MAKEFILE}"]
    text = MAKEFILE.read_text(encoding="utf-8")
    failures: list[str] = []

    smoke_tests_match = re.search(r"^SMOKE_TESTS\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not smoke_tests_match:
        failures.append("Makefile: missing `SMOKE_TESTS = ...` declaration")
        return failures
    smoke_tests = smoke_tests_match.group(1).split()
    if not re.search(r"^SMOKE_PYTEST_ARGS\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `SMOKE_PYTEST_ARGS ?=` declaration")

    if "smoke-check:" not in text:
        failures.append("Makefile: missing target `smoke-check`")
        return failures
    if "pytest -q $(SMOKE_TESTS) $(SMOKE_PYTEST_ARGS)" not in text:
        failures.append("Makefile: smoke-check must run `pytest -q $(SMOKE_TESTS) $(SMOKE_PYTEST_ARGS)`")

    for test_path in REQUIRED_SMOKE_TEST_PATHS:
        if test_path not in smoke_tests:
            failures.append(f"Makefile: SMOKE_TESTS missing `{test_path}`")

    return failures


def main() -> int:
    print("Checking smoke gate contracts...")
    failures = _check_workflow() + _check_makefile()
    if failures:
        print("[FAIL] smoke gate contracts")
        for issue in failures:
            print(f"  - {issue}")
        return 1
    print("[OK] smoke workflow and local smoke-check are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
