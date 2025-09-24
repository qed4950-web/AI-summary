from __future__ import annotations

from pathlib import Path

from infopilot_core.data_pipeline.policies.engine import PolicyEngine


def _write_policy(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def test_policy_engine_resolves_roots(tmp_path: Path) -> None:
    policy_file = _write_policy(
        tmp_path / "policies.json",
        """
        [
          {
            "path": "docs",
            "agents": ["knowledge_search"],
            "security": {"processing": "local_only", "pii_filter": true},
            "indexing": {"mode": "realtime"}
          }
        ]
        """,
    )
    engine = PolicyEngine.from_file(policy_file)
    assert engine.has_policies
    roots = engine.roots_for_agent("knowledge_search")
    expected_root = (tmp_path / "docs").resolve()
    assert roots == [expected_root]
    assert engine.allows(expected_root / "report.pdf", agent="knowledge_search")
    assert not engine.allows(expected_root / "report.pdf", agent="meeting")


def test_policy_engine_manual_mode(tmp_path: Path) -> None:
    policy_file = _write_policy(
        tmp_path / "manual.json",
        """
        {
          "path": "manual",
          "agents": ["knowledge_search"],
          "indexing": {"mode": "manual"}
        }
        """,
    )
    engine = PolicyEngine.from_file(policy_file)
    target = (tmp_path / "manual" / "doc.pdf").resolve()
    assert engine.allows(target, agent="knowledge_search", include_manual=True)
    assert not engine.allows(target, agent="knowledge_search", include_manual=False)


def test_policy_engine_filters_records(tmp_path: Path) -> None:
    policy_file = _write_policy(
        tmp_path / "filter.json",
        """
        [
          {"path": "docs", "agents": ["knowledge_search"]},
          {"path": "private", "agents": ["meeting"], "indexing": {"mode": "manual"}}
        ]
        """,
    )
    engine = PolicyEngine.from_file(policy_file)
    docs_root = (tmp_path / "docs").resolve()
    records = [
        {"path": str(docs_root / "keep.pdf")},
        {"path": str((tmp_path / "private" / "secret.pdf").resolve())},
        {"path": str((tmp_path / "other" / "ignore.pdf").resolve())},
    ]
    filtered = engine.filter_records(records, agent="knowledge_search", include_manual=False)
    assert len(filtered) == 1
    assert filtered[0]["path"].endswith("keep.pdf")
