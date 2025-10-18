"""Compatibility helper for rebuilding search indices."""
from __future__ import annotations

from pathlib import Path

from core.search.retriever import Retriever

from src.config import TOPIC_MODEL_PATH


def run_indexing(*, corpus_path: Path, cache_dir: Path) -> None:
    """Rebuild the ANN index using the modern retriever."""
    if not TOPIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"토픽 모델 파일을 찾을 수 없습니다: {TOPIC_MODEL_PATH}. "
            "먼저 학습을 실행해 topic_model.joblib을 생성하세요."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(
        model_path=TOPIC_MODEL_PATH,
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        auto_refresh=False,
    )
    try:
        retriever.ready(rebuild=True, wait=True)
    finally:
        retriever.shutdown()

