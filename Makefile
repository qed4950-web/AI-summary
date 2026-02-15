.PHONY: static-check smoke-check integration-check release-check lint-debt-snapshot lint-debt-check lint-debt-domain-refresh lint-debt-refresh test-full test-integration

SMOKE_TESTS = tests/test_infopilot_cli_contract.py tests/test_policy_engine.py tests/test_pipeline_sensitive_paths.py tests/test_meeting_policy_scope.py tests/test_photo_policy_scope.py tests/test_watch_cli_dependencies.py tests/test_watch_event_handler_contract.py tests/test_pipeline_policy_provider_contract.py tests/test_pipeline_runner_watch_loop.py tests/test_desktop_backend_mode_contract.py tests/test_desktop_runtime_policy_config_contract.py tests/test_llm_client_option_contract.py tests/test_mode_profiles_config_contract.py tests/test_release_metadata_lint_domain_contract.py tests/test_lint_domain_budget_refresh_contract.py
SMOKE_PYTEST_ARGS ?=
INTEGRATION_TESTS = tests/test_infopilot_cli_contract.py tests/test_watch_cli_dependencies.py tests/test_watch_event_handler_contract.py tests/test_pipeline_policy_provider_contract.py tests/test_pipeline_runner_watch_loop.py tests/test_desktop_backend_mode_contract.py tests/test_desktop_runtime_policy_config_contract.py tests/test_llm_client_option_contract.py tests/test_mode_profiles_config_contract.py tests/test_release_metadata_lint_domain_contract.py tests/test_lint_domain_budget_refresh_contract.py tests/test_incremental_index_report_contract.py tests/test_open_event_log_summary_contract.py tests/test_run_incremental_index_contract.py tests/test_ui_smoke.py tests/test_chat_pipeline.py tests/integration/test_policy_integration.py tests/integration/test_rag_retrieval.py tests/integration/test_meeting_e2e.py tests/integration/test_llama_build.py
INTEGRATION_PYTEST_ARGS ?=
RELEASE_TAG ?=
STATIC_RUFF_TARGETS = scripts/pipeline/infopilot.py scripts/pipeline/infopilot_cli/chat.py scripts/pipeline/infopilot_cli/watch.py scripts/pipeline/infopilot_cli/watchers.py scripts/pipeline/infopilot_cli/pipeline_runner.py desktop_app/ui.py desktop_app/backend.py core/config/mode_profiles.py core/config/desktop_runtime_policy.py core/conversation/llm_client.py tests/test_ui_smoke.py tests/test_chat_pipeline.py tests/test_desktop_backend_mode_contract.py tests/test_desktop_runtime_policy_config_contract.py tests/test_llm_client_option_contract.py tests/test_mode_profiles_config_contract.py tests/test_release_metadata_lint_domain_contract.py tests/test_lint_domain_budget_refresh_contract.py tests/test_incremental_index_report_contract.py tests/test_open_event_log_summary_contract.py tests/test_infopilot_cli_contract.py tests/test_watch_cli_dependencies.py tests/test_watch_event_handler_contract.py tests/test_pipeline_policy_provider_contract.py tests/test_pipeline_runner_watch_loop.py tests/integration/test_policy_integration.py tests/integration/test_rag_retrieval.py tests/integration/test_meeting_e2e.py scripts/dev/verify/verify_smoke_gate_contract.py scripts/dev/verify/verify_release_integration_contract.py scripts/dev/verify/verify_incremental_index_report.py scripts/dev/verify/summarize_open_event_log.py scripts/dev/verify/verify_lint_debt_budget.py scripts/dev/verify/refresh_lint_domain_budget.py scripts/dev/release/generate_release_metadata.py
LINT_DEBT_REPORT_PATH ?= artifacts/lint/ruff_statistics.txt
LINT_DEBT_BUDGET_FILE ?= docs/plan/lint_debt_budget.json
LINT_DOMAIN_BUDGET_FILE ?= docs/plan/lint_debt_domain_budget.json
LINT_DOMAIN_SUMMARY_FILE ?= docs/plan/lint_domain_refresh_summary.md
LINT_DEBT_SLACK ?= 0

static-check:
	ruff check $(STATIC_RUFF_TARGETS)
	python scripts/dev/verify/verify_smoke_gate_contract.py
	python scripts/dev/verify/verify_release_integration_contract.py

smoke-check:
	pytest -q $(SMOKE_TESTS) $(SMOKE_PYTEST_ARGS)

integration-check:
	pytest -q -m "integration" $(INTEGRATION_TESTS) $(INTEGRATION_PYTEST_ARGS)

release-check:
	python scripts/dev/verify/verify_release_integration_contract.py --release-tag "$(RELEASE_TAG)"

lint-debt-snapshot:
	mkdir -p $$(dirname "$(LINT_DEBT_REPORT_PATH)")
	-ruff check . --statistics > "$(LINT_DEBT_REPORT_PATH)"

lint-debt-check:
	python scripts/dev/verify/verify_lint_debt_budget.py --budget-file "$(LINT_DEBT_BUDGET_FILE)" --report-file "$(LINT_DEBT_REPORT_PATH)" --slack "$(LINT_DEBT_SLACK)"

lint-debt-domain-refresh:
	python scripts/dev/verify/refresh_lint_domain_budget.py --domain-budget-file "$(LINT_DOMAIN_BUDGET_FILE)" --summary-file "$(LINT_DOMAIN_SUMMARY_FILE)"

lint-debt-refresh:
	python scripts/dev/verify/verify_lint_debt_budget.py --budget-file "$(LINT_DEBT_BUDGET_FILE)" --report-file "$(LINT_DEBT_REPORT_PATH)" --write-current
	$(MAKE) lint-debt-domain-refresh

test-full:
	pytest -q -m "full"

test-integration:
	$(MAKE) integration-check
