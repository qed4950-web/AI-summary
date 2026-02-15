from __future__ import annotations

from pathlib import Path

import pytest

from core.config.desktop_runtime_policy import (
    DEFAULT_DESKTOP_RUNTIME_POLICY,
    load_desktop_runtime_policy,
    load_desktop_runtime_policy_history,
    save_desktop_runtime_policy,
)

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _isolated_policy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "desktop_runtime_policy.json"
    history_path = tmp_path / "desktop_runtime_policy_history.jsonl"
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_PATH", str(path))
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_HISTORY_PATH", str(history_path))
    for env_name in (
        "DESKTOP_MASK_PII",
        "DESKTOP_MAX_FILE_LINKS",
        "DESKTOP_MAX_REFERENCE_LINKS",
        "DESKTOP_MAX_RESPONSE_CHARS",
        "DESKTOP_MAX_SUGGESTION_CHARS",
    ):
        monkeypatch.delenv(env_name, raising=False)
    return path


def test_runtime_policy_loads_defaults_when_file_missing() -> None:
    policy = load_desktop_runtime_policy()
    assert policy == DEFAULT_DESKTOP_RUNTIME_POLICY


def test_runtime_policy_uses_env_fallback_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKTOP_MASK_PII", "0")
    monkeypatch.setenv("DESKTOP_MAX_FILE_LINKS", "11")
    monkeypatch.setenv("DESKTOP_MAX_REFERENCE_LINKS", "7")
    monkeypatch.setenv("DESKTOP_MAX_RESPONSE_CHARS", "32000")
    monkeypatch.setenv("DESKTOP_MAX_SUGGESTION_CHARS", "140")

    policy = load_desktop_runtime_policy()

    assert policy["privacy_mask_enabled"] is False
    assert policy["max_file_links"] == 11
    assert policy["max_reference_links"] == 7
    assert policy["max_response_chars"] == 32000
    assert policy["max_suggestion_chars"] == 140


def test_runtime_policy_save_and_reload_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_file_links": 4,
            "max_reference_links": 3,
            "max_response_chars": 18000,
            "max_suggestion_chars": 96,
        }
    )
    monkeypatch.setenv("DESKTOP_MASK_PII", "0")
    monkeypatch.setenv("DESKTOP_MAX_FILE_LINKS", "32")

    policy = load_desktop_runtime_policy()

    assert policy["privacy_mask_enabled"] is True
    assert policy["max_file_links"] == 4
    assert policy["max_reference_links"] == 3
    assert policy["max_response_chars"] == 18000
    assert policy["max_suggestion_chars"] == 96


def test_runtime_policy_history_written_on_save() -> None:
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": False,
            "max_file_links": 6,
            "max_reference_links": 4,
            "max_response_chars": 21000,
            "max_suggestion_chars": 110,
        }
    )
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_file_links": 7,
            "max_reference_links": 5,
            "max_response_chars": 22000,
            "max_suggestion_chars": 120,
        }
    )

    history = load_desktop_runtime_policy_history(limit=5)

    assert len(history) >= 2
    latest = history[0]
    assert latest["source"] == "save_desktop_runtime_policy"
    latest_policy = latest.get("policy", {})
    assert isinstance(latest_policy, dict)
    assert latest_policy["privacy_mask_enabled"] is True
    assert latest_policy["max_file_links"] == 7


def test_runtime_policy_history_source_contract() -> None:
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": True,
            "max_file_links": 8,
            "max_reference_links": 6,
            "max_response_chars": 24000,
            "max_suggestion_chars": 120,
        },
        source="settings_inline_policy_apply",
    )

    history = load_desktop_runtime_policy_history(limit=1)

    assert len(history) == 1
    assert history[0].get("source") == "settings_inline_policy_apply"
