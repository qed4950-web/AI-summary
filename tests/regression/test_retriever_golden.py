from __future__ import annotations

from pathlib import Path

import pytest

from core.config.paths import CACHE_DIR, CORPUS_PATH, TOPIC_MODEL_PATH
from core.search.retriever import Retriever
from scripts.evaluate_rag import evaluate, load_cases

GOLDEN_PATH = Path("data/eval/golden_queries.jsonl")

pytestmark = pytest.mark.full


def test_golden_queries_recall():
    if not GOLDEN_PATH.exists():
        pytest.skip("golden query set이 아직 준비되지 않았습니다.")
    if not TOPIC_MODEL_PATH.exists() or not CORPUS_PATH.exists():
        pytest.skip("학습된 모델/코퍼스를 찾을 수 없어 golden 테스트를 건너뜁니다.")

    cases = load_cases(GOLDEN_PATH)
    if not cases:
        pytest.skip("golden query 파일이 비어 있습니다.")

    try:
        retriever = Retriever(
            model_path=TOPIC_MODEL_PATH,
            corpus_path=CORPUS_PATH,
            cache_dir=CACHE_DIR,
            auto_refresh=False,
        )
        retriever.ready(rebuild=False, wait=True)
    except FileNotFoundError as exc:  # pragma: no cover - 환경 의존
        pytest.skip(f"필수 아티팩트가 없어 golden 테스트를 건너뜁니다: {exc}")

    results = evaluate(retriever, cases, top_k=5)
    failures = [detail for detail in results["details"] if not detail["topk"]]
    if failures:
        missing_queries = ", ".join(detail["query"] for detail in failures)
        pytest.fail(f"golden 질의가 예상 문서를 찾지 못했습니다: {missing_queries}")
