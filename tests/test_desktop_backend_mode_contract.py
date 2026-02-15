from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import desktop_app.backend as backend_module
    from core.config.desktop_runtime_policy import save_desktop_runtime_policy
    from core.config.mode_profiles import (
        DEFAULT_MODE_PROFILES,
        MODE_ORDER,
    )
    from core.config.mode_profiles import (
        load_mode_profiles as _load_mode_profiles,
    )
    from core.config.mode_profiles import (
        save_mode_profiles as _save_mode_profiles,
    )
    from desktop_app.backend import LNPBackend
except ImportError:
    pytest.skip("desktop backend dependencies are unavailable", allow_module_level=True)

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _isolated_runtime_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_PATH", str(tmp_path / "desktop_runtime_policy.json"))
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_HISTORY_PATH", str(tmp_path / "desktop_runtime_policy_history.jsonl"))
    mode_profiles_path = tmp_path / "desktop_mode_profiles.json"
    _save_mode_profiles({mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER}, mode_profiles_path)
    monkeypatch.setattr(backend_module, "load_mode_profiles", lambda: _load_mode_profiles(mode_profiles_path))
    for env_name in (
        "DESKTOP_MASK_PII",
        "DESKTOP_MAX_FILE_LINKS",
        "DESKTOP_MAX_REFERENCE_LINKS",
        "DESKTOP_MAX_RESPONSE_CHARS",
        "DESKTOP_MAX_SUGGESTION_CHARS",
    ):
        monkeypatch.delenv(env_name, raising=False)


class _FakeChat:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.llm_client = _FakeClient()
        self.payload = payload if payload is not None else {"answer": "ok", "hits": [], "suggestions": []}

    def ask(self, query: str, topk=None, *, force_action=None):
        self.calls.append(
            {
                "query": query,
                "topk": topk,
                "force_action": force_action,
            }
        )
        if isinstance(self.payload, dict):
            return dict(self.payload)
        return self.payload


class _FakeClient:
    def __init__(self) -> None:
        self.max_new_tokens = 0
        self.temperature = 0.0
        self.options: dict[str, object] = {}


def _apply_default_profiles(backend: LNPBackend) -> None:
    backend._mode_profiles = {mode: dict(DEFAULT_MODE_PROFILES[mode]) for mode in MODE_ORDER}


def test_backend_resolve_mode_profile_defaults_to_auto() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    profile = backend._resolve_mode_profile("unknown-mode")
    assert profile["mode"] == "Auto"
    assert profile["force_action"] is None
    assert profile["topk"] is None
    assert profile["llm_max_new_tokens"] == 512


def test_backend_resolve_mode_profile_instant_contract() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    profile = backend._resolve_mode_profile("Instant")
    assert profile["mode"] == "Instant"
    assert profile["force_action"] == "chat"
    assert profile["topk"] == 3
    assert profile["llm_temperature"] == 0.0


def test_backend_handle_query_passes_mode_profile_to_chat() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat()
    backend.chat = fake_chat  # type: ignore[assignment]

    backend.handle_query("요약해줘", "Pro")

    assert len(fake_chat.calls) == 1
    call = fake_chat.calls[0]
    assert call["topk"] == 8
    assert call["force_action"] == "search"
    assert fake_chat.llm_client.max_new_tokens == 1024
    assert fake_chat.llm_client.temperature == 0.25
    assert fake_chat.llm_client.options["num_predict"] == 1024


def test_backend_status_message_exposes_runtime_contract() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat()
    backend.chat = fake_chat  # type: ignore[assignment]
    statuses: list[str] = []
    backend.status_msg.connect(statuses.append)

    backend.handle_query("요약해줘", "Thinking")

    runtime_status = next((text for text in statuses if "action=" in text), "")
    assert runtime_status
    assert "Thinking deep (Thinking)" in runtime_status
    assert "action=search" in runtime_status
    assert "top-k=6" in runtime_status
    assert "tokens=768" in runtime_status
    assert "temp=0.15" in runtime_status
    assert "privacy=mask" in runtime_status
    assert "refs<=5" in runtime_status


def test_backend_apply_mode_runtime_profile_updates_llm_client() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat()
    backend.chat = fake_chat  # type: ignore[assignment]

    profile = backend._resolve_mode_profile("Thinking")
    backend._apply_mode_runtime_profile(profile)

    assert fake_chat.llm_client.max_new_tokens == 768
    assert fake_chat.llm_client.temperature == 0.15
    assert fake_chat.llm_client.options["num_predict"] == 768
    assert fake_chat.llm_client.options["temperature"] == 0.15


def test_backend_handle_query_keeps_slash_command_semantics() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat()
    backend.chat = fake_chat  # type: ignore[assignment]
    statuses: list[str] = []
    backend.status_msg.connect(statuses.append)

    backend.handle_query('/meeting "/tmp/sample.wav"', "Instant")

    assert len(fake_chat.calls) == 1
    call = fake_chat.calls[0]
    assert call["topk"] is None
    assert call["force_action"] is None
    runtime_status = next((text for text in statuses if "action=" in text), "")
    assert "action=auto" in runtime_status
    assert "top-k=auto" in runtime_status


def test_backend_masks_answer_pii_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKTOP_MASK_PII", "1")
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(
        payload={
            "answer": "이메일 test.user@example.com, 전화 010-1234-5678",
            "hits": [],
            "suggestions": [],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("요약", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[REDACTED_EMAIL]" in rendered
    assert "[REDACTED_PHONE]" in rendered
    assert "민감정보 일부 마스킹됨" in rendered


def test_backend_dedupes_and_normalizes_file_links(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)

    doc = tmp_path / "report.pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(
        payload={
            "answer": "결과입니다",
            "hits": [
                {"path": str(doc), "title": "리포트 010-1234-5678"},
                {"file_path": f"file://{doc}", "title": "리포트-중복"},
                {"path": str(doc), "title": "리포트-중복2"},
            ],
            "suggestions": ["next", "next", "mail test.user@example.com", "retry"],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("요약", "Auto")

    assert responses
    rendered = responses[-1]
    assert rendered.count("[FILE_LINK:") == 1
    assert str(doc) in rendered
    assert "[REDACTED_PHONE]" in rendered
    assert "(Tip: next, mail [REDACTED_EMAIL], retry)" in rendered
    assert "중복 링크 2건은 병합되었습니다." in rendered


def test_backend_resolves_relative_hit_path_against_docs_dir_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc = docs_dir / "portfolio.pdf"
    doc.write_text("stub", encoding="utf-8")
    monkeypatch.setattr("desktop_app.backend.DOCS_DIR", docs_dir)
    fake_chat = _FakeChat(
        payload={
            "answer": "결과입니다",
            "hits": [{"path": "portfolio.pdf", "title": "포트폴리오"}],
            "suggestions": [],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("요약", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[FILE_LINK:file://" in rendered
    assert "portfolio.pdf" in rendered


def test_backend_drops_unresolvable_relative_hit_path_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    monkeypatch.setattr("desktop_app.backend.DOCS_DIR", tmp_path / "documents")
    fake_chat = _FakeChat(
        payload={
            "answer": "결과입니다",
            "hits": [{"path": "missing-relative.pdf", "title": "누락 문서"}],
            "suggestions": [],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("요약", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[FILE_LINK:" not in rendered
    assert "지원되지 않거나 유효하지 않은 링크 1건은 제외되었습니다." in rendered


def test_backend_truncates_very_long_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKTOP_MAX_RESPONSE_CHARS", "1200")
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(payload={"answer": "a" * 2500, "hits": [], "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("긴응답", "Auto")

    assert responses
    rendered = responses[-1]
    assert "응답이 길어 일부만 표시되었습니다." in rendered
    assert len(rendered) < 1400


def test_backend_truncation_keeps_file_links_clickable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESKTOP_MAX_RESPONSE_CHARS", "1200")
    backend = LNPBackend()
    _apply_default_profiles(backend)
    doc = tmp_path / "evidence.pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(
        payload={
            "answer": "b" * 2500,
            "hits": [{"path": str(doc), "title": "증빙 문서"}],
            "suggestions": [],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("긴응답", "Auto")

    assert responses
    rendered = responses[-1]
    assert "응답이 길어 일부만 표시되었습니다." in rendered
    assert "[FILE_LINK:file://" in rendered
    assert "evidence.pdf" in rendered


def test_backend_normalizes_non_dict_response_payload() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(payload="string-payload-response")
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("payload", "Auto")

    assert responses
    assert "string-payload-response" in responses[-1]


def test_backend_structured_answer_payload_is_json_normalized() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(payload={"answer": {"ok": True, "count": 2}, "hits": [], "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("json", "Auto")

    assert responses
    rendered = responses[-1]
    assert "{\"ok\":true,\"count\":2}" in rendered


def test_backend_truncates_and_normalizes_suggestions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKTOP_MAX_SUGGESTION_CHARS", "24")
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(
        payload={
            "answer": "ok",
            "hits": [],
            "suggestions": [
                "line1\nline2 with spaces     and tabs",
                "line1 line2 with spaces and tabs",
            ],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("payload", "Auto")

    assert responses
    rendered = responses[-1]
    assert "line1 line2 with spac..." in rendered
    assert len("line1 line2 with spac...") == 24
    # deduped after normalization/truncation
    assert rendered.count("line1 line2 with spac...") == 1


def test_backend_sanitizes_reserved_file_link_tokens_in_answer_and_suggestions() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(
        payload={
            "answer": "본문 [FILE_LINK:/tmp/injected.pdf]",
            "hits": [],
            "suggestions": ["hint [FILE_LINK:/tmp/hint.pdf]"],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("inject", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[FILE_LINK_BLOCKED:/tmp/injected.pdf]" in rendered
    assert "[FILE_LINK_BLOCKED:/tmp/hint.pdf]" in rendered
    assert "[FILE_LINK:/tmp/injected.pdf]" not in rendered


def test_backend_normalizes_tuple_hits_and_suggestions(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    doc = tmp_path / "tuple-hit.pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(
        payload={
            "answer": "tuple payload",
            "hits": ({"path": str(doc), "title": "Tuple hit"},),
            "suggestions": ("first", "first", "second"),
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("tuple", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[FILE_LINK:file://" in rendered
    assert "tuple-hit.pdf" in rendered
    assert "(Tip: first, second)" in rendered


def test_backend_empty_payload_has_user_fallback_message() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(payload={})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("empty", "Auto")

    assert responses
    assert "응답 결과가 비어 있습니다." in responses[-1]


def test_backend_reference_overflow_notice_contract(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    hits: list[dict[str, object]] = []
    for idx in range(7):
        doc = tmp_path / f"overflow-{idx}.pdf"
        doc.write_text("stub", encoding="utf-8")
        hits.append({"path": str(doc), "title": f"doc-{idx}"})
    fake_chat = _FakeChat(payload={"answer": "ok", "hits": hits, "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("overflow", "Auto")

    assert responses
    rendered = responses[-1]
    assert "참조 문서 안내: 총 7건 중 상위 5건만 포함되었습니다." in rendered


def test_backend_reference_limit_from_env_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESKTOP_MAX_REFERENCE_LINKS", "3")
    backend = LNPBackend()
    _apply_default_profiles(backend)
    hits: list[dict[str, object]] = []
    for idx in range(5):
        doc = tmp_path / f"limit-{idx}.pdf"
        doc.write_text("stub", encoding="utf-8")
        hits.append({"path": str(doc), "title": f"doc-{idx}"})
    fake_chat = _FakeChat(payload={"answer": "ok", "hits": hits, "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("limit", "Auto")

    assert responses
    rendered = responses[-1]
    assert "참조 문서 안내: 총 5건 중 상위 3건만 포함되었습니다." in rendered


def test_backend_reference_limit_from_policy_file_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy_path = tmp_path / "desktop_runtime_policy.json"
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_PATH", str(policy_path))
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": False,
            "max_reference_links": 2,
            "max_response_chars": 24000,
            "max_suggestion_chars": 120,
            "max_file_links": 8,
        },
        policy_path,
    )
    backend = LNPBackend()
    _apply_default_profiles(backend)
    hits: list[dict[str, object]] = []
    for idx in range(4):
        doc = tmp_path / f"policy-limit-{idx}.pdf"
        doc.write_text("stub", encoding="utf-8")
        hits.append({"path": str(doc), "title": f"doc-{idx}"})
    fake_chat = _FakeChat(payload={"answer": "ok", "hits": hits, "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    statuses: list[str] = []
    backend.response_ready.connect(responses.append)
    backend.status_msg.connect(statuses.append)

    backend.handle_query("limit", "Auto")

    assert responses
    rendered = responses[-1]
    assert "참조 문서 안내: 총 4건 중 상위 2건만 포함되었습니다." in rendered
    runtime_status = next((text for text in statuses if "action=" in text), "")
    assert "privacy=raw" in runtime_status
    assert "refs<=2" in runtime_status


def test_backend_refresh_runtime_policy_slot_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy_path = tmp_path / "desktop_runtime_policy.json"
    monkeypatch.setenv("DESKTOP_RUNTIME_POLICY_PATH", str(policy_path))
    save_desktop_runtime_policy(
        {
            "privacy_mask_enabled": False,
            "max_reference_links": 9,
            "max_response_chars": 30000,
            "max_suggestion_chars": 200,
            "max_file_links": 10,
        },
        policy_path,
    )
    backend = LNPBackend()
    statuses: list[str] = []
    backend.status_msg.connect(statuses.append)

    backend.refresh_runtime_policy()

    assert backend._mask_answer_pii is False
    assert backend._max_reference_links == 9
    assert backend._max_response_chars == 30000
    assert backend._max_suggestion_chars == 200
    assert any("Runtime policy synced" in status for status in statuses)


def test_backend_file_link_token_uri_encoding_contract(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    doc = tmp_path / "evidence [1].pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(payload={"answer": "ok", "hits": [{"path": str(doc), "title": "special"}], "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("uri", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[FILE_LINK:file://" in rendered
    assert "%5B1%5D" in rendered


def test_backend_masks_fallback_file_name_when_title_missing(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    doc = tmp_path / "010-1234-5678.pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(payload={"answer": "ok", "hits": [{"path": str(doc)}], "suggestions": []})
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("mask-file-name", "Auto")

    assert responses
    rendered = responses[-1]
    assert "[REDACTED_PHONE].pdf" in rendered


def test_backend_masks_reference_title_and_suggestion_pii_contract(tmp_path: Path) -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    doc = tmp_path / "박대엽_010-8650-4950.pdf"
    doc.write_text("stub", encoding="utf-8")
    fake_chat = _FakeChat(
        payload={
            "answer": "",
            "hits": [{"path": str(doc), "title": "문의 qec@example.com / 010-8650-4950"}],
            "suggestions": ["연락처 qec@example.com 으로 회신"],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("mask-reference", "Auto")

    assert responses
    rendered = responses[-1]
    rendered_without_links = re.sub(r"\[FILE_LINK:[^\]]+\]", "[FILE_LINK]", rendered)
    assert "qec@example.com" not in rendered_without_links
    assert "010-8650-4950" not in rendered_without_links
    assert "[REDACTED_EMAIL]" in rendered
    assert "[REDACTED_PHONE]" in rendered
    assert "[FILE_LINK:file://" in rendered
    assert "민감정보 일부 마스킹됨" in rendered


def test_backend_reports_invalid_reference_links_excluded() -> None:
    backend = LNPBackend()
    _apply_default_profiles(backend)
    fake_chat = _FakeChat(
        payload={
            "answer": "ok",
            "hits": [
                {"path": "http://example.com/a.pdf"},
                {"path": ""},
                {"path": None},
            ],
            "suggestions": [],
        }
    )
    backend.chat = fake_chat  # type: ignore[assignment]
    responses: list[str] = []
    backend.response_ready.connect(responses.append)

    backend.handle_query("invalid-links", "Auto")

    assert responses
    rendered = responses[-1]
    assert "지원되지 않거나 유효하지 않은 링크 3건은 제외되었습니다." in rendered
