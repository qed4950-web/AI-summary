from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.search.retriever import Retriever


@pytest.fixture
def rag_test_env(tmp_path):
    corpus_dir = tmp_path / "corpus"
    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    corpus_dir.mkdir()
    model_dir.mkdir()
    cache_dir.mkdir()

    (corpus_dir / "project_alpha.txt").write_text(
        "Project Alpha is a secret initiative to build a flying car.",
        encoding="utf-8",
    )
    (corpus_dir / "project_beta.txt").write_text(
        "Project Beta focuses on underwater exploration and deep sea mining.",
        encoding="utf-8",
    )
    (corpus_dir / "gundam_specs.md").write_text(
        "RX-78-2 Gundam has a beam rifle and beam saber.",
        encoding="utf-8",
    )

    return {"corpus": corpus_dir, "model": model_dir, "cache": cache_dir}


@pytest.mark.smoke
@pytest.mark.integration
def test_retriever_hybrid_search(rag_test_env) -> None:
    with (
        patch("core.search.retriever.IndexManager") as mock_index_manager,
        patch("core.search.retriever.QueryEncoder"),
        patch("core.search.retriever.CrossEncoderReranker"),
    ):
        mock_index = MagicMock()
        mock_index_manager.return_value.get_index.return_value = mock_index
        mock_index.doc_ids = [0, 1, 2]
        mock_index.exts = [".txt", ".txt", ".md"]
        mock_index.search.return_value = [
            {
                "doc_id": 0,
                "path": str(rag_test_env["corpus"] / "project_alpha.txt"),
                "preview": "build a flying car",
                "score": 0.9,
                "ext": ".txt",
            },
            {
                "doc_id": 2,
                "path": str(rag_test_env["corpus"] / "gundam_specs.md"),
                "preview": "beam rifle",
                "score": 0.1,
                "ext": ".md",
            },
        ]

        retriever = Retriever(
            model_path=rag_test_env["model"],
            corpus_path=rag_test_env["corpus"],
            cache_dir=rag_test_env["cache"],
            use_rerank=False,
        )
        hits = retriever.search("flying car", top_k=2)

    assert hits
    assert "project_alpha" in str(hits[0]["path"])
    assert hits[0]["score"] > 0.0
