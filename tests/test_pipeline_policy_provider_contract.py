from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pipeline.infopilot_cli.pipeline_runner import IncrementalPipeline

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


class _Engine:
    pass


def _build_pipeline(
    tmp_path: Path,
    *,
    policy_engine: _Engine | None = None,
    policy_engine_provider=None,
    require_policy_engine: bool = False,
) -> IncrementalPipeline:
    return IncrementalPipeline(
        encoder=object(),
        batch_size=1,
        scan_csv=tmp_path / "scan.csv",
        corpus_path=tmp_path / "corpus.parquet",
        cache_dir=tmp_path / "cache",
        translate=False,
        policy_engine=policy_engine,  # type: ignore[arg-type]
        policy_engine_provider=policy_engine_provider,
        agent="knowledge_search",
        require_policy_engine=require_policy_engine,
    )


def test_incremental_pipeline_provider_none_falls_back_to_local_engine(tmp_path: Path) -> None:
    local_engine = _Engine()
    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=local_engine,
        policy_engine_provider=lambda: None,
    )
    assert pipeline._current_policy_engine() is local_engine


def test_incremental_pipeline_provider_failure_falls_back_to_local_engine(tmp_path: Path) -> None:
    local_engine = _Engine()

    def failing_provider():
        raise RuntimeError("provider-failure")

    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=local_engine,
        policy_engine_provider=failing_provider,
    )
    assert pipeline._current_policy_engine() is local_engine


def test_incremental_pipeline_provider_precedence_over_local_engine(tmp_path: Path) -> None:
    local_engine = _Engine()
    provider_engine = _Engine()
    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=local_engine,
        policy_engine_provider=lambda: provider_engine,
    )
    assert pipeline._current_policy_engine() is provider_engine


def test_incremental_pipeline_provider_none_without_local_engine_returns_none(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=None,
        policy_engine_provider=lambda: None,
    )
    assert pipeline._current_policy_engine() is None


def test_incremental_pipeline_required_policy_blocks_add_paths_when_engine_missing(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=None,
        policy_engine_provider=lambda: None,
        require_policy_engine=True,
    )
    filtered = pipeline._filter_add_paths_for_policy_gate({"a.md", "b.txt"}, pipeline._current_policy_engine())
    assert filtered == set()


def test_incremental_pipeline_required_policy_allows_add_paths_with_fallback_engine(tmp_path: Path) -> None:
    local_engine = _Engine()
    pipeline = _build_pipeline(
        tmp_path,
        policy_engine=local_engine,
        policy_engine_provider=lambda: None,
        require_policy_engine=True,
    )
    filtered = pipeline._filter_add_paths_for_policy_gate({"a.md"}, pipeline._current_policy_engine())
    assert filtered == {"a.md"}
