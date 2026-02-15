from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.mode_profiles import DEFAULT_MODE_PROFILES, MODE_ORDER, load_mode_profiles, save_mode_profiles

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_mode_profiles_load_defaults_when_file_missing(tmp_path: Path) -> None:
    profiles = load_mode_profiles(tmp_path / "missing.json")
    assert set(profiles.keys()) == set(MODE_ORDER)
    assert profiles["Auto"]["description"] == DEFAULT_MODE_PROFILES["Auto"]["description"]


def test_mode_profiles_save_and_reload_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mode_profiles.json"
    raw = {mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER}
    raw["Thinking"]["llm_max_new_tokens"] = 896
    raw["Thinking"]["llm_temperature"] = 0.22
    raw["Instant"]["topk"] = 2
    raw["Pro"]["force_action"] = "search"

    save_mode_profiles(raw, path)
    loaded = load_mode_profiles(path)

    assert loaded["Thinking"]["llm_max_new_tokens"] == 896
    assert loaded["Thinking"]["llm_temperature"] == pytest.approx(0.22)
    assert loaded["Instant"]["topk"] == 2
    assert loaded["Pro"]["force_action"] == "search"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "Auto" in payload
    assert payload["Thinking"]["llm_max_new_tokens"] == 896
