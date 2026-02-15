"""Static guard for release/integration CI gate contracts.

This verifier keeps workflow and Makefile contracts aligned for release and
integration governance checks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RELEASE_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
INTEGRATION_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "integration.yml"
MAKEFILE = PROJECT_ROOT / "Makefile"
RELEASE_CHECKLIST_GLOB = "release_readiness_checklist_*.md"
RELEASE_CHECKLIST_FILE = "release_readiness_checklist.md"
RELEASE_METADATA_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "release" / "generate_release_metadata.py"
IMPACT_SCORE_POLICY_FILE = PROJECT_ROOT / "docs" / "plan" / "impact_score_policy.json"
LINT_DEBT_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "verify" / "verify_lint_debt_budget.py"
LINT_DOMAIN_REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "verify" / "refresh_lint_domain_budget.py"
OPEN_EVENT_SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "verify" / "summarize_open_event_log.py"
INCREMENTAL_REPORT_VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "verify" / "verify_incremental_index_report.py"
LINT_DEBT_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_budget.json"
LINT_DOMAIN_BUDGET_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_debt_domain_budget.json"
LINT_DOMAIN_SUMMARY_FILE = PROJECT_ROOT / "docs" / "plan" / "lint_domain_refresh_summary.md"
PR_TEMPLATE_FILE = PROJECT_ROOT / ".github" / "pull_request_template.md"
DESKTOP_BACKEND_FILE = PROJECT_ROOT / "desktop_app" / "backend.py"
DESKTOP_UI_FILE = PROJECT_ROOT / "desktop_app" / "ui.py"
DESKTOP_RUNTIME_POLICY_FILE = PROJECT_ROOT / "core" / "config" / "desktop_runtime_policy.py"
DESKTOP_MODE_CONTRACT_TEST_FILE = PROJECT_ROOT / "tests" / "test_desktop_backend_mode_contract.py"
MODE_PROFILE_CONFIG_TEST_FILE = PROJECT_ROOT / "tests" / "test_mode_profiles_config_contract.py"
RUNTIME_POLICY_CONFIG_TEST_FILE = PROJECT_ROOT / "tests" / "test_desktop_runtime_policy_config_contract.py"
UI_SMOKE_TEST_FILE = PROJECT_ROOT / "tests" / "test_ui_smoke.py"
RETRIEVAL_STRATEGY_TEST_FILE = PROJECT_ROOT / "tests" / "test_retrieval_strategy_contract.py"

RELEASE_WORKFLOW_TOKENS: tuple[str, ...] = (
    "name: Release",
    "workflow_dispatch:",
    "tags:",
    "- \"v*\"",
    "name: Resolve release tag",
    "name: Release static checks",
    "name: Release smoke checks",
    "name: Release lint debt budget check",
    "name: Release governance summary",
    "name: Build release metadata artifact",
    "name: Upload release artifact bundle",
    "make static-check",
    "make lint-debt-snapshot",
    "make lint-debt-check",
    "python scripts/dev/verify/verify_release_integration_contract.py --release-tag",
    "python scripts/dev/release/generate_release_metadata.py",
    "uses: actions/upload-artifact@v4",
    "artifacts/lint/ruff_statistics.txt",
    "steps.release_smoke.outcome",
    "### Lint Domain Delta",
    "artifacts/release/release_metadata.json",
    "top_codes",
    "top codes:",
)

INTEGRATION_WORKFLOW_TOKENS: tuple[str, ...] = (
    "name: Integration",
    "name: Integration static guard",
    "name: Integration contract tests",
    "id: integration_contract",
    "make test-integration",
    "name: Publish integration summary",
    "steps.integration_contract.outcome",
)

REQUIRED_INTEGRATION_TEST_PATHS: tuple[str, ...] = (
    "tests/test_infopilot_cli_contract.py",
    "tests/test_watch_cli_dependencies.py",
    "tests/test_watch_event_handler_contract.py",
    "tests/test_pipeline_policy_provider_contract.py",
    "tests/test_pipeline_runner_watch_loop.py",
    "tests/test_desktop_backend_mode_contract.py",
    "tests/test_desktop_runtime_policy_config_contract.py",
    "tests/test_llm_client_option_contract.py",
    "tests/test_mode_profiles_config_contract.py",
    "tests/test_release_metadata_lint_domain_contract.py",
    "tests/test_lint_domain_budget_refresh_contract.py",
    "tests/test_incremental_index_report_contract.py",
    "tests/test_open_event_log_summary_contract.py",
    "tests/test_run_incremental_index_contract.py",
    "tests/test_ui_smoke.py",
    "tests/test_chat_pipeline.py",
    "tests/integration/test_policy_integration.py",
    "tests/integration/test_rag_retrieval.py",
    "tests/integration/test_meeting_e2e.py",
    "tests/integration/test_llama_build.py",
)

REQUIRED_STATIC_RUFF_PATHS: tuple[str, ...] = (
    "scripts/pipeline/infopilot.py",
    "scripts/pipeline/infopilot_cli/chat.py",
    "scripts/pipeline/infopilot_cli/watch.py",
    "scripts/pipeline/infopilot_cli/watchers.py",
    "scripts/pipeline/infopilot_cli/pipeline_runner.py",
    "core/config/mode_profiles.py",
    "core/config/desktop_runtime_policy.py",
    "core/conversation/llm_client.py",
    "desktop_app/backend.py",
    "desktop_app/ui.py",
    "tests/test_desktop_backend_mode_contract.py",
    "tests/test_desktop_runtime_policy_config_contract.py",
    "tests/test_llm_client_option_contract.py",
    "tests/test_mode_profiles_config_contract.py",
    "tests/test_release_metadata_lint_domain_contract.py",
    "tests/test_lint_domain_budget_refresh_contract.py",
    "tests/test_incremental_index_report_contract.py",
    "tests/test_open_event_log_summary_contract.py",
    "tests/test_ui_smoke.py",
    "tests/test_chat_pipeline.py",
    "tests/test_infopilot_cli_contract.py",
    "tests/test_watch_cli_dependencies.py",
    "tests/test_watch_event_handler_contract.py",
    "tests/test_pipeline_policy_provider_contract.py",
    "tests/test_pipeline_runner_watch_loop.py",
    "tests/integration/test_policy_integration.py",
    "tests/integration/test_rag_retrieval.py",
    "tests/integration/test_meeting_e2e.py",
    "scripts/dev/verify/verify_smoke_gate_contract.py",
    "scripts/dev/verify/verify_release_integration_contract.py",
    "scripts/dev/verify/verify_incremental_index_report.py",
    "scripts/dev/verify/summarize_open_event_log.py",
    "scripts/dev/verify/verify_lint_debt_budget.py",
    "scripts/dev/verify/refresh_lint_domain_budget.py",
    "scripts/dev/release/generate_release_metadata.py",
)

SEMVER_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
BACKEND_MODE_TOKENS: tuple[str, ...] = (
    "_mode_profiles",
    "_reload_mode_profiles",
    "_reload_runtime_policy",
    "refresh_runtime_policy",
    "_apply_mode_runtime_profile",
    "_format_runtime_status",
    "load_desktop_runtime_policy",
    "_collect_document_links",
    "_normalize_answer_value",
    "_sanitize_reserved_link_token",
    "_mask_answer_text",
    "_normalize_suggestions",
    "_truncate_response_text",
    "_to_file_link_token",
    "_build_reference_section",
    "_normalize_response_payload",
    "응답 결과가 비어 있습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
    "참조 문서 안내: 총",
    "건 중 상위",
    "건만 포함되었습니다.",
    "지원되지 않거나 유효하지 않은 링크",
    "중복 링크 {merged_count}건은 병합되었습니다.",
    "[FILE_LINK_BLOCKED:",
    "privacy={'mask' if self._mask_answer_pii else 'raw'}",
    "refs<={self._max_reference_links}",
    "Runtime policy synced",
    "self._max_response_chars = max(1200",
    "self._max_suggestion_chars = max(24",
    "load_mode_profiles",
    "@Slot(str, str)",
    "llm_max_new_tokens",
    "llm_temperature",
)
UI_MODE_TOKENS: tuple[str, ...] = (
    "query_requested = Signal(str, str)",
    "runtime_policy_refresh_requested = Signal()",
    "SettingsHubDialog",
    "RuntimePolicyDialog",
    "Inline Runtime Policy",
    "Inline Mode Preset",
    "_runtime_policy",
    "_reload_runtime_policy",
    "_open_runtime_policy_editor",
    "_open_smart_folder_manager",
    "_on_runtime_policy_updated",
    "_on_mode_profile_updated",
    "_sync_inline_mode_controls",
    "_apply_inline_runtime_policy",
    "_apply_inline_mode_profile",
    "_restore_selected_history_policy",
    "_sync_history_restore_state",
    "_set_hub_status",
    "_reset_hub_status",
    "_append_hub_status_event",
    "_selected_hub_status_filter",
    "_selected_hub_status_time_filter",
    "_filtered_hub_status_events",
    "_render_hub_status_log",
    "_copy_hub_status_log",
    "_default_status_log_export_path",
    "_export_hub_status_log",
    "_apply_history_filters",
    "_is_history_entry_visible",
    "_selected_history_source_filter",
    "_selected_history_period_token",
    "_selected_history_period_days",
    "_selected_history_period_threshold",
    "_parse_history_timestamp",
    "_parse_custom_history_period_input",
    "_apply_custom_history_period",
    "_capture_history_filter_state",
    "_ensure_history_period_option",
    "_restore_history_filter_state",
    "_on_history_filters_changed",
    "_format_history_source",
    "on_mode_profile_applied",
    "_response_mode_runtime",
    "_response_mode_profiles",
    "_open_mode_profile_editor",
    "ModeProfileDialog",
    "_desc_fields",
    "_status_fields",
    "_topk_fields",
    "_desc_error_labels",
    "_status_error_labels",
    "_topk_error_labels",
    "_is_valid_description",
    "_is_valid_status",
    "_validate_form",
    "_on_text_fields_changed",
    "_classify_open_error",
    "_format_open_error_message",
    "_append_open_failure_guidance",
    "_append_open_failure_cta_actions",
    "_path_similarity_score",
    "_find_similar_file_candidates",
    "_append_similar_file_candidates",
    "_add_failure_guide_card",
    "_open_local_file",
    "_reveal_in_finder",
    "_open_command_timeout_seconds",
    "_run_open_command",
    "open_darwin_short_circuit",
    "_similar_lookup_scan_limit",
    "_similar_lookup_cache_path",
    "_load_similar_lookup_cache",
    "_save_similar_lookup_cache",
    "_extract_file_links",
    "_file_open_shortcut_hint",
    "_format_file_item_label",
    "_thread_history_path",
    "_thread_timeline_path",
    "_load_thread_entries",
    "_load_thread_timelines",
    "_save_thread_entries",
    "_save_thread_timelines",
    "_capture_thread_timeline",
    "_restore_thread_timeline",
    "_sync_active_thread_for_query",
    "_open_selected_timeline_parent",
    "_reveal_selected_timeline_item",
    "_copy_selected_timeline_file_path",
    "_append_open_recovery_actions",
    "_run_timeline_action",
    "_mask_display_text",
    "load_desktop_runtime_policy_history",
    "_privacy_mask_enabled",
    "Runtime policy updated",
    "SettingsHubStatusBanner",
    "SettingsHubStatusLog",
    "Status timeline (recent)",
    "Filter",
    "Range",
    "Last 10m",
    "Last 1h",
    "Last 24h",
    "Copy",
    "Export",
    "CSV Files (*.csv)",
    "timestamp_utc",
    "timestamp_local",
    "_hub_status_reset_timer",
    "_hub_status_last_key",
    "_hub_status_last_at",
    "_hub_status_throttle_ms",
    "_hub_status_events",
    "_hub_status_event_limit",
    "_session_history_filter_state",
    "ModeInlineError",
    "Description은 1-48자여야 합니다.",
    "Status는 1-24자여야 합니다.",
    "privacy={'mask' if self._privacy_mask_enabled else 'raw'}",
    "refs<={self._max_reference_links}",
    "file-links<={self._max_file_links}",
    "상위 폴더를 열었습니다",
    "UI 표시 단계에서 민감정보 일부 마스킹됨",
    "참조 문서만 반환되었습니다. 아래 파일을 확인하세요.",
    "[missing]",
    "참조 문서 요약:",
    "총 {len(file_links) + overflow_count}개 중 {len(file_links)}개 표시",
    "현재 경로에 없습니다",
    "유효하지 않은 링크 {invalid_count}개는 제외했습니다",
    "중복 링크 {merged_duplicate_count}개는 병합했습니다",
    "레거시 경로 링크 {legacy_converted_count}개를 표준 경로로 변환했습니다",
    "파일이 현재 경로에 없습니다. 클릭 시 상위 폴더를 엽니다",
    "(open parent folder)",
    "다음 동작:",
    "Shift+P",
    "Shift+O",
    "Qt.UserRole + 9",
    "Qt.UserRole + 10",
    "Qt.UserRole + 11",
    "Restore Selected Policy",
    "Select a version and confirm preview to restore",
    "History source",
    "History period",
    "Custom from",
    "Apply custom",
    "Since today 00:00",
    "Since custom (",
    "absolute:",
    "Custom datetime format: YYYY-MM-DD HH:MM",
    "Custom period applied",
    "No status events for current filter.",
    "No runtime policy history for current filters.",
    "No restorable history (filter result)",
    "Tab 순서:",
    "DESKTOP_THREAD_HISTORY_PATH",
    "DESKTOP_THREAD_TIMELINE_PATH",
    "DESKTOP_FILE_RESOLUTION_CACHE_PATH",
    "DESKTOP_SIMILAR_SCAN_MAX",
    "DESKTOP_SIMILAR_LOOKUP_CACHE_PATH",
    "DESKTOP_OPEN_EVENT_LOG_PATH",
    "desktop_file_resolution_cache.json",
    "desktop_file_open_events.jsonl",
    "모든 스레드를 표시 중입니다.",
    "No more",
    "취소 원인 점검:",
    "기본 앱 점검:",
    "권한 점검:",
    "권한 설정 가이드 열기",
    "기본 앱 연결 가이드 열기",
    "이름 유사 문서 찾기",
    "유사 문서 후보",
    "FailureGuideCard",
    "GuidePermissionButton",
    "GuideAssociationButton",
    "GuideSimilarButton",
    "GuideFinderButton",
    "RecoveryRevealButton",
    "권장 조치:",
    "open_permission_guide",
    "open_app_association_guide",
    "search_similar_files",
    "reveal_in_finder",
    "open_candidate_file",
    "유사 문서 후보가 여러 개라 자동 열기를 중단했습니다",
    "복구된 경로를 사용해 문서를 엽니다",
    "_save_file_resolution_cache",
    "_load_file_resolution_cache",
    "_record_open_event",
    "_open_event_log_path",
    "_load_similar_candidates_for_folder",
    "similar_scan_capped",
    "기본 앱 열기에 실패해 Preview로 문서를 열었습니다.",
    "기본 앱 열기에 실패해 상위 폴더를 열었습니다.",
    "Opened file:",
    "Opened parent:",
    "settings_inline_policy_apply",
    "settings_history_restore",
    "Mode preset saved (",
    "QLineEdit[invalid=\"true\"]",
    "self.mode_hint",
    "grouped message (start)",
    "grouped message (middle)",
    "grouped message (end)",
    "query_requested.emit(query, self._response_mode)",
    "Shift+R",
)
MODE_CONTRACT_BACKEND_TEST_TOKENS: tuple[str, ...] = (
    "test_backend_apply_mode_runtime_profile_updates_llm_client",
    "test_backend_handle_query_passes_mode_profile_to_chat",
    "test_backend_status_message_exposes_runtime_contract",
    "test_backend_masks_answer_pii_by_default",
    "test_backend_dedupes_and_normalizes_file_links",
    "test_backend_resolves_relative_hit_path_against_docs_dir_contract",
    "test_backend_drops_unresolvable_relative_hit_path_contract",
    "test_backend_truncates_very_long_answer",
    "test_backend_truncation_keeps_file_links_clickable",
    "test_backend_normalizes_non_dict_response_payload",
    "test_backend_structured_answer_payload_is_json_normalized",
    "test_backend_truncates_and_normalizes_suggestions",
    "test_backend_normalizes_tuple_hits_and_suggestions",
    "test_backend_empty_payload_has_user_fallback_message",
    "test_backend_reference_overflow_notice_contract",
    "test_backend_reference_limit_from_env_contract",
    "test_backend_reference_limit_from_policy_file_contract",
    "test_backend_refresh_runtime_policy_slot_contract",
    "test_backend_file_link_token_uri_encoding_contract",
    "test_backend_masks_fallback_file_name_when_title_missing",
    "test_backend_masks_reference_title_and_suggestion_pii_contract",
    "test_backend_reports_invalid_reference_links_excluded",
    "test_backend_sanitizes_reserved_file_link_tokens_in_answer_and_suggestions",
)
MODE_CONTRACT_PROFILE_TEST_TOKENS: tuple[str, ...] = (
    "test_mode_profiles_save_and_reload_roundtrip",
)
MODE_CONTRACT_RUNTIME_POLICY_TEST_TOKENS: tuple[str, ...] = (
    "test_runtime_policy_loads_defaults_when_file_missing",
    "test_runtime_policy_uses_env_fallback_when_file_missing",
    "test_runtime_policy_save_and_reload_roundtrip",
    "test_runtime_policy_history_written_on_save",
    "test_runtime_policy_history_source_contract",
)
MODE_CONTRACT_UI_TEST_TOKENS: tuple[str, ...] = (
    "test_launcher_file_open_contract",
    "test_launcher_file_open_auto_resolves_similar_path_contract",
    "test_launcher_file_open_ambiguous_candidates_contract",
    "test_launcher_file_resolution_cache_persistence_contract",
    "test_launcher_similar_file_lookup_cache_contract",
    "test_launcher_similar_lookup_cache_persistence_contract",
    "test_launcher_open_event_log_contract",
    "test_launcher_similar_file_scan_limit_contract",
    "test_launcher_file_open_error_message_contract",
    "test_launcher_open_local_file_macos_preview_fallback_contract",
    "test_launcher_open_local_file_macos_parent_fallback_contract",
    "test_launcher_open_local_file_macos_canceled_short_circuit_contract",
    "test_launcher_failure_guide_card_includes_finder_action_on_macos_contract",
    "test_launcher_run_timeline_action_reveal_in_finder_contract",
    "test_launcher_reveal_in_finder_recovers_stale_path_contract",
    "test_launcher_handle_response_file_link_parse_and_privacy_contract",
    "test_launcher_handle_response_legacy_path_link_conversion_contract",
    "test_launcher_handle_response_link_only_placeholder_contract",
    "test_launcher_handle_response_file_link_overflow_notice_contract",
    "test_runtime_policy_dialog_save_and_launcher_hint_contract",
    "test_settings_hub_dialog_navigation_contract",
    "test_settings_hub_history_restore_contract",
    "test_settings_hub_history_filter_contract",
    "test_settings_hub_history_filter_custom_datetime_contract",
    "test_settings_hub_history_filter_session_restore_contract",
    "test_settings_hub_history_filter_stale_absolute_restore_contract",
    "test_launcher_missing_file_item_accessibility_contract",
    "test_launcher_file_open_failure_recovery_actions_contract",
    "test_launcher_file_open_failure_guidance_contract",
    "test_launcher_file_open_failure_cta_actions_contract",
    "test_launcher_similar_file_candidates_contract",
    "test_launcher_file_open_success_status_contract",
    "test_settings_hub_status_log_filter_and_copy_contract",
    "test_settings_hub_status_log_export_json_time_filter_contract",
    "test_settings_hub_status_log_export_csv_and_default_path_contract",
    "test_settings_hub_status_banner_auto_reset_contract",
    "test_launcher_thread_sidebar_persistence_and_show_more_contract",
    "test_launcher_thread_timeline_restore_contract",
    "test_launcher_file_recovery_shortcut_mapping_contract",
    "test_launcher_recovery_action_click_dispatch_contract",
)
RETRIEVAL_STRATEGY_TEST_TOKENS: tuple[str, ...] = (
    "test_init_retriever_rebuild_invokes_ready_non_blocking",
    "test_init_retriever_rebuild_handles_legacy_ready_signature",
    "test_init_retriever_rebuild_falls_back_to_index_manager_schedule",
)


def _check_release_workflow() -> list[str]:
    if not RELEASE_WORKFLOW.exists():
        return [f"missing file: {RELEASE_WORKFLOW}"]
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    failures: list[str] = []

    for token in RELEASE_WORKFLOW_TOKENS:
        if token not in text:
            failures.append(f"release.yml: missing token `{token}`")
    return failures


def _check_integration_workflow() -> list[str]:
    if not INTEGRATION_WORKFLOW.exists():
        return [f"missing file: {INTEGRATION_WORKFLOW}"]
    text = INTEGRATION_WORKFLOW.read_text(encoding="utf-8")
    failures: list[str] = []

    for token in INTEGRATION_WORKFLOW_TOKENS:
        if token not in text:
            failures.append(f"integration.yml: missing token `{token}`")

    return failures


def _check_makefile() -> list[str]:
    if not MAKEFILE.exists():
        return [f"missing file: {MAKEFILE}"]
    text = MAKEFILE.read_text(encoding="utf-8")
    failures: list[str] = []

    integration_tests_match = re.search(r"^INTEGRATION_TESTS\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not integration_tests_match:
        failures.append("Makefile: missing `INTEGRATION_TESTS = ...` declaration")
        return failures
    integration_tests = integration_tests_match.group(1).split()

    static_ruff_targets_match = re.search(r"^STATIC_RUFF_TARGETS\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not static_ruff_targets_match:
        failures.append("Makefile: missing `STATIC_RUFF_TARGETS = ...` declaration")
        return failures
    static_ruff_targets = static_ruff_targets_match.group(1).split()

    if not re.search(r"^INTEGRATION_PYTEST_ARGS\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `INTEGRATION_PYTEST_ARGS ?=` declaration")
    if not re.search(r"^RELEASE_TAG\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `RELEASE_TAG ?=` declaration")
    if not re.search(r"^LINT_DEBT_REPORT_PATH\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `LINT_DEBT_REPORT_PATH ?=` declaration")
    if not re.search(r"^LINT_DEBT_BUDGET_FILE\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `LINT_DEBT_BUDGET_FILE ?=` declaration")
    if not re.search(r"^LINT_DOMAIN_BUDGET_FILE\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `LINT_DOMAIN_BUDGET_FILE ?=` declaration")
    if not re.search(r"^LINT_DOMAIN_SUMMARY_FILE\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `LINT_DOMAIN_SUMMARY_FILE ?=` declaration")
    if not re.search(r"^LINT_DEBT_SLACK\s*\?=", text, flags=re.MULTILINE):
        failures.append("Makefile: missing `LINT_DEBT_SLACK ?=` declaration")

    if "integration-check:" not in text:
        failures.append("Makefile: missing target `integration-check`")
    if 'pytest -q -m "integration" $(INTEGRATION_TESTS) $(INTEGRATION_PYTEST_ARGS)' not in text:
        failures.append(
            'Makefile: integration-check must run `pytest -q -m "integration" $(INTEGRATION_TESTS) $(INTEGRATION_PYTEST_ARGS)`'
        )
    if "static-check:" not in text:
        failures.append("Makefile: missing target `static-check`")
    if "ruff check $(STATIC_RUFF_TARGETS)" not in text:
        failures.append("Makefile: static-check must run `ruff check $(STATIC_RUFF_TARGETS)`")
    if "release-check:" not in text:
        failures.append("Makefile: missing target `release-check`")
    if 'python scripts/dev/verify/verify_release_integration_contract.py --release-tag "$(RELEASE_TAG)"' not in text:
        failures.append("Makefile: release-check must pass RELEASE_TAG to contract verifier")
    if "lint-debt-snapshot:" not in text:
        failures.append("Makefile: missing target `lint-debt-snapshot`")
    if '-ruff check . --statistics > "$(LINT_DEBT_REPORT_PATH)"' not in text:
        failures.append("Makefile: lint-debt-snapshot must persist `ruff check . --statistics` output")
    if "lint-debt-check:" not in text:
        failures.append("Makefile: missing target `lint-debt-check`")
    if 'python scripts/dev/verify/verify_lint_debt_budget.py --budget-file "$(LINT_DEBT_BUDGET_FILE)" --report-file "$(LINT_DEBT_REPORT_PATH)" --slack "$(LINT_DEBT_SLACK)"' not in text:
        failures.append("Makefile: lint-debt-check must validate lint debt budget with report and slack")
    if "lint-debt-refresh:" not in text:
        failures.append("Makefile: missing target `lint-debt-refresh`")
    if 'python scripts/dev/verify/verify_lint_debt_budget.py --budget-file "$(LINT_DEBT_BUDGET_FILE)" --report-file "$(LINT_DEBT_REPORT_PATH)" --write-current' not in text:
        failures.append("Makefile: lint-debt-refresh must write budget baseline from current report")
    if "lint-debt-domain-refresh:" not in text:
        failures.append("Makefile: missing target `lint-debt-domain-refresh`")
    if 'python scripts/dev/verify/refresh_lint_domain_budget.py --domain-budget-file "$(LINT_DOMAIN_BUDGET_FILE)" --summary-file "$(LINT_DOMAIN_SUMMARY_FILE)"' not in text:
        failures.append("Makefile: lint-debt-domain-refresh must refresh domain lint budget file")
    if "$(MAKE) lint-debt-domain-refresh" not in text:
        failures.append("Makefile: lint-debt-refresh must chain lint-debt-domain-refresh")

    for test_path in REQUIRED_INTEGRATION_TEST_PATHS:
        if test_path not in integration_tests:
            failures.append(f"Makefile: INTEGRATION_TESTS missing `{test_path}`")
    for source_path in REQUIRED_STATIC_RUFF_PATHS:
        if source_path not in static_ruff_targets:
            failures.append(f"Makefile: STATIC_RUFF_TARGETS missing `{source_path}`")
    return failures


def _check_release_checklist() -> list[str]:
    release_docs_dir = PROJECT_ROOT / "docs" / "plan"
    dated_candidates = sorted(release_docs_dir.glob(RELEASE_CHECKLIST_GLOB))
    canonical = release_docs_dir / RELEASE_CHECKLIST_FILE
    if not dated_candidates and not canonical.exists():
        return [f"docs/plan: missing `{RELEASE_CHECKLIST_GLOB}` or `{RELEASE_CHECKLIST_FILE}`"]

    latest = canonical if canonical.exists() else dated_candidates[-1]
    text = latest.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in ("# AI-summary", "## 요약", "## 남은 차단/권장 이슈", "lint_domain_refresh_summary.md"):
        if token not in text:
            failures.append(f"{latest}: missing section token `{token}`")
    return failures


def _check_release_tag(tag: str) -> list[str]:
    if not tag:
        return []
    if SEMVER_TAG_RE.fullmatch(tag):
        return []
    return [f"release tag must match semver pattern `vX.Y.Z` (received `{tag}`)"]


def _check_release_metadata_script() -> list[str]:
    if not RELEASE_METADATA_SCRIPT.exists():
        return [f"missing file: {RELEASE_METADATA_SCRIPT}"]
    text = RELEASE_METADATA_SCRIPT.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in (
        "IMPACT_SCORE_POLICY_PATH",
        "_load_impact_score_policy",
        "\"impact_score_policy\"",
        "LINT_DOMAIN_BUDGET_PATH",
        "_load_lint_domain_budget",
        "\"lint_domain_summary\"",
        "## Lint Debt Domain Delta",
    ):
        if token not in text:
            failures.append(f"generate_release_metadata.py: missing token `{token}`")
    return failures


def _check_impact_score_policy_file() -> list[str]:
    if not IMPACT_SCORE_POLICY_FILE.exists():
        return [f"missing file: {IMPACT_SCORE_POLICY_FILE}"]
    try:
        payload = json.loads(IMPACT_SCORE_POLICY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid json: {IMPACT_SCORE_POLICY_FILE} ({exc})"]
    if not isinstance(payload, dict):
        return [f"{IMPACT_SCORE_POLICY_FILE}: top-level JSON must be object"]
    failures: list[str] = []
    for key in ("version", "base_score", "weights", "caps", "integration_bonus", "tiers"):
        if key not in payload:
            failures.append(f"{IMPACT_SCORE_POLICY_FILE}: missing key `{key}`")
    return failures


def _check_desktop_mode_contract() -> list[str]:
    failures: list[str] = []
    if not DESKTOP_BACKEND_FILE.exists():
        failures.append(f"missing file: {DESKTOP_BACKEND_FILE}")
        return failures
    if not DESKTOP_UI_FILE.exists():
        failures.append(f"missing file: {DESKTOP_UI_FILE}")
        return failures
    if not DESKTOP_RUNTIME_POLICY_FILE.exists():
        failures.append(f"missing file: {DESKTOP_RUNTIME_POLICY_FILE}")
        return failures
    if not DESKTOP_MODE_CONTRACT_TEST_FILE.exists():
        failures.append(f"missing file: {DESKTOP_MODE_CONTRACT_TEST_FILE}")
        return failures
    if not MODE_PROFILE_CONFIG_TEST_FILE.exists():
        failures.append(f"missing file: {MODE_PROFILE_CONFIG_TEST_FILE}")
        return failures
    if not RUNTIME_POLICY_CONFIG_TEST_FILE.exists():
        failures.append(f"missing file: {RUNTIME_POLICY_CONFIG_TEST_FILE}")
        return failures
    if not UI_SMOKE_TEST_FILE.exists():
        failures.append(f"missing file: {UI_SMOKE_TEST_FILE}")
        return failures
    if not RETRIEVAL_STRATEGY_TEST_FILE.exists():
        failures.append(f"missing file: {RETRIEVAL_STRATEGY_TEST_FILE}")
        return failures

    backend_text = DESKTOP_BACKEND_FILE.read_text(encoding="utf-8")
    ui_text = DESKTOP_UI_FILE.read_text(encoding="utf-8")
    runtime_policy_text = DESKTOP_RUNTIME_POLICY_FILE.read_text(encoding="utf-8")
    test_text = DESKTOP_MODE_CONTRACT_TEST_FILE.read_text(encoding="utf-8")
    mode_profile_test_text = MODE_PROFILE_CONFIG_TEST_FILE.read_text(encoding="utf-8")
    runtime_policy_test_text = RUNTIME_POLICY_CONFIG_TEST_FILE.read_text(encoding="utf-8")
    ui_smoke_text = UI_SMOKE_TEST_FILE.read_text(encoding="utf-8")
    retrieval_test_text = RETRIEVAL_STRATEGY_TEST_FILE.read_text(encoding="utf-8")
    for token in BACKEND_MODE_TOKENS:
        if token not in backend_text:
            failures.append(f"desktop_app/backend.py: missing token `{token}`")
    for token in UI_MODE_TOKENS:
        if token not in ui_text:
            failures.append(f"desktop_app/ui.py: missing token `{token}`")
    for token in (
        "DEFAULT_DESKTOP_RUNTIME_POLICY",
        "DESKTOP_RUNTIME_POLICY_HISTORY_PATH",
        "load_desktop_runtime_policy",
        "save_desktop_runtime_policy",
        "append_desktop_runtime_policy_history",
        "load_desktop_runtime_policy_history",
    ):
        if token not in runtime_policy_text:
            failures.append(f"core/config/desktop_runtime_policy.py: missing token `{token}`")
    for token in MODE_CONTRACT_BACKEND_TEST_TOKENS:
        if token not in test_text:
            failures.append(f"tests/test_desktop_backend_mode_contract.py: missing token `{token}`")
    for token in MODE_CONTRACT_PROFILE_TEST_TOKENS:
        if token not in mode_profile_test_text:
            failures.append(f"tests/test_mode_profiles_config_contract.py: missing token `{token}`")
    for token in MODE_CONTRACT_RUNTIME_POLICY_TEST_TOKENS:
        if token not in runtime_policy_test_text:
            failures.append(f"tests/test_desktop_runtime_policy_config_contract.py: missing token `{token}`")
    for token in MODE_CONTRACT_UI_TEST_TOKENS:
        if token not in ui_smoke_text:
            failures.append(f"tests/test_ui_smoke.py: missing token `{token}`")
    for token in RETRIEVAL_STRATEGY_TEST_TOKENS:
        if token not in retrieval_test_text:
            failures.append(f"tests/test_retrieval_strategy_contract.py: missing token `{token}`")
    return failures


def _check_lint_debt_script() -> list[str]:
    if not LINT_DEBT_SCRIPT.exists():
        return [f"missing file: {LINT_DEBT_SCRIPT}"]
    return []


def _check_lint_domain_refresh_script() -> list[str]:
    if not LINT_DOMAIN_REFRESH_SCRIPT.exists():
        return [f"missing file: {LINT_DOMAIN_REFRESH_SCRIPT}"]
    text = LINT_DOMAIN_REFRESH_SCRIPT.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in ("--domain-budget-file", "--summary-file", "_write_summary", "budget_total", "DOMAIN_KEYS"):
        if token not in text:
            failures.append(f"refresh_lint_domain_budget.py: missing token `{token}`")
    return failures


def _check_open_event_summary_script() -> list[str]:
    if not OPEN_EVENT_SUMMARY_SCRIPT.exists():
        return [f"missing file: {OPEN_EVENT_SUMMARY_SCRIPT}"]
    text = OPEN_EVENT_SUMMARY_SCRIPT.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in (
        "load_open_events",
        "summarize_open_events",
        "evaluate_open_event_alerts",
        "render_markdown_summary",
        "recovery_attempt_count",
        "recovery_success_rate",
        "short_circuit_count",
        "short_circuit_rate",
        "short_circuit_rate_high",
        "--log-path",
        "--out-json",
        "--failure-rate-threshold",
        "--canceled-rate-threshold",
        "--min-recovery-attempts",
        "--recovery-success-threshold",
        "--short-circuit-rate-threshold",
        "--fail-on-alert",
    ):
        if token not in text:
            failures.append(f"{OPEN_EVENT_SUMMARY_SCRIPT}: missing token `{token}`")
    return failures


def _check_incremental_report_verify_script() -> list[str]:
    if not INCREMENTAL_REPORT_VERIFY_SCRIPT.exists():
        return [f"missing file: {INCREMENTAL_REPORT_VERIFY_SCRIPT}"]
    text = INCREMENTAL_REPORT_VERIFY_SCRIPT.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in (
        "load_incremental_report",
        "evaluate_incremental_report",
        "required_int_fields",
        "run_step2_triggered",
        "failed_phase",
        "started_at_utc",
        "finished_at_utc",
        "duration_ms",
        "_parse_iso8601",
        "fromisoformat",
        "processed_count must be > 0 when run_step2_triggered is true",
        "--report-path",
        "--max-missing-targets",
        "--allow-status",
    ):
        if token not in text:
            failures.append(f"{INCREMENTAL_REPORT_VERIFY_SCRIPT}: missing token `{token}`")
    return failures


def _check_lint_debt_budget_file() -> list[str]:
    if not LINT_DEBT_BUDGET_FILE.exists():
        return [f"missing file: {LINT_DEBT_BUDGET_FILE}"]
    return []


def _check_lint_domain_budget_file() -> list[str]:
    if not LINT_DOMAIN_BUDGET_FILE.exists():
        return [f"missing file: {LINT_DOMAIN_BUDGET_FILE}"]
    try:
        payload = json.loads(LINT_DOMAIN_BUDGET_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid json: {LINT_DOMAIN_BUDGET_FILE} ({exc})"]
    if not isinstance(payload, dict):
        return [f"{LINT_DOMAIN_BUDGET_FILE}: top-level JSON must be object"]
    failures: list[str] = []
    if "version" not in payload:
        failures.append(f"{LINT_DOMAIN_BUDGET_FILE}: missing key `version`")
    raw_domains = payload.get("domains", {})
    if not isinstance(raw_domains, dict):
        failures.append(f"{LINT_DOMAIN_BUDGET_FILE}: key `domains` must be object")
        return failures
    for domain in ("engine", "ui_ux", "tests"):
        raw = raw_domains.get(domain)
        if not isinstance(raw, dict):
            failures.append(f"{LINT_DOMAIN_BUDGET_FILE}: missing object `domains.{domain}`")
            continue
        if "paths" not in raw:
            failures.append(f"{LINT_DOMAIN_BUDGET_FILE}: missing key `domains.{domain}.paths`")
        if "budget_total" not in raw:
            failures.append(f"{LINT_DOMAIN_BUDGET_FILE}: missing key `domains.{domain}.budget_total`")
    return failures


def _check_lint_domain_summary_file() -> list[str]:
    if not LINT_DOMAIN_SUMMARY_FILE.exists():
        return [f"missing file: {LINT_DOMAIN_SUMMARY_FILE}"]
    text = LINT_DOMAIN_SUMMARY_FILE.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in ("# Lint Domain Refresh Summary", "## Domain Totals", "## Refresh Notes"):
        if token not in text:
            failures.append(f"{LINT_DOMAIN_SUMMARY_FILE}: missing section token `{token}`")
    return failures


def _check_pr_template() -> list[str]:
    if not PR_TEMPLATE_FILE.exists():
        return [f"missing file: {PR_TEMPLATE_FILE}"]
    text = PR_TEMPLATE_FILE.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in ("## Lint Domain Refresh", "make lint-debt-domain-refresh", "lint_domain_refresh_summary.md"):
        if token not in text:
            failures.append(f"{PR_TEMPLATE_FILE}: missing token `{token}`")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify release/integration CI contracts")
    parser.add_argument(
        "--release-tag",
        default="",
        help="Release tag to validate (defaults to GITHUB_REF_NAME when set).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.release_tag.strip() or os.getenv("GITHUB_REF_NAME", "").strip()

    print("Checking release/integration gate contracts...")
    failures = (
        _check_release_workflow()
        + _check_integration_workflow()
        + _check_makefile()
        + _check_release_checklist()
        + _check_release_metadata_script()
        + _check_impact_score_policy_file()
        + _check_desktop_mode_contract()
        + _check_lint_debt_script()
        + _check_lint_domain_refresh_script()
        + _check_open_event_summary_script()
        + _check_incremental_report_verify_script()
        + _check_lint_debt_budget_file()
        + _check_lint_domain_budget_file()
        + _check_lint_domain_summary_file()
        + _check_pr_template()
        + _check_release_tag(tag)
    )
    if failures:
        print("[FAIL] release/integration gate contracts")
        for issue in failures:
            print(f"  - {issue}")
        return 1
    print("[OK] release/integration workflow and local gate commands are aligned")
    if tag:
        print(f"[OK] release tag format: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
