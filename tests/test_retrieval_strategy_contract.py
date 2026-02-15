from __future__ import annotations

from pathlib import Path

from core.conversation import retrieval_strategy


class _FakeRetrieverWithReady:
    last_instance: "_FakeRetrieverWithReady | None" = None

    def __init__(self, **_kwargs) -> None:
        type(self).last_instance = self
        self.ready_calls: list[dict[str, object]] = []

    def ready(self, rebuild: bool = False, *, wait: bool = True) -> bool:
        self.ready_calls.append({"rebuild": rebuild, "wait": wait})
        return True


class _FakeRetrieverLegacyReady:
    last_instance: "_FakeRetrieverLegacyReady | None" = None

    def __init__(self, **_kwargs) -> None:
        type(self).last_instance = self
        self.ready_calls: list[dict[str, object]] = []

    def ready(self, rebuild: bool = False) -> bool:  # legacy signature without wait
        self.ready_calls.append({"rebuild": rebuild})
        return True


class _FakeScheduleIndexManager:
    def __init__(self) -> None:
        self.schedule_calls: list[bool] = []

    def schedule_rebuild(self, *, priority: bool = False):
        self.schedule_calls.append(bool(priority))
        return object()


class _FakeRetrieverNoReady:
    last_instance: "_FakeRetrieverNoReady | None" = None

    def __init__(self, **_kwargs) -> None:
        type(self).last_instance = self
        self.index_manager = _FakeScheduleIndexManager()


def _build_retriever(*, rebuild: bool):
    return retrieval_strategy.init_retriever(
        model_path=Path("/tmp/missing-model"),
        corpus_path=Path("/tmp/missing-corpus"),
        cache_dir=Path("/tmp/cache-dir"),
        topk=5,
        use_rerank=False,
        rerank_model_name="",
        rerank_depth=0,
        rerank_batch_size=1,
        rerank_device=None,
        rerank_min_score=None,
        min_similarity=0.3,
        lexical_weight=0.0,
        rebuild=rebuild,
    )


def test_init_retriever_rebuild_invokes_ready_non_blocking(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_strategy, "HybridRetriever", _FakeRetrieverWithReady)

    retr = _build_retriever(rebuild=True)

    assert retr is not None
    instance = _FakeRetrieverWithReady.last_instance
    assert instance is not None
    assert instance.ready_calls
    call = instance.ready_calls[0]
    assert call["rebuild"] is True
    assert call["wait"] is False


def test_init_retriever_rebuild_handles_legacy_ready_signature(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_strategy, "HybridRetriever", _FakeRetrieverLegacyReady)

    retr = _build_retriever(rebuild=True)

    assert retr is not None
    instance = _FakeRetrieverLegacyReady.last_instance
    assert instance is not None
    assert instance.ready_calls == [{"rebuild": True}]


def test_init_retriever_rebuild_falls_back_to_index_manager_schedule(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_strategy, "HybridRetriever", _FakeRetrieverNoReady)

    retr = _build_retriever(rebuild=True)

    assert retr is not None
    instance = _FakeRetrieverNoReady.last_instance
    assert instance is not None
    assert instance.index_manager.schedule_calls == [True]
