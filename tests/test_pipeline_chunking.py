import pandas as pd
import pytest

from core.data_pipeline.pipeline import _apply_uniform_chunks


@pytest.mark.full
def test_adaptive_chunk_window_splits_long_documents():
    text = " ".join([f"sentence{i}" for i in range(600)])
    df = pd.DataFrame([{"text": text}])
    chunked = _apply_uniform_chunks(df, min_tokens=64, max_tokens=256)
    assert len(chunked) > 1
    assert chunked.iloc[0]["chunk_tokens"] >= 64


@pytest.mark.full
def test_adaptive_chunk_window_single_chunk_for_short_text():
    df = pd.DataFrame([{"text": "short text with just a few words."}])
    chunked = _apply_uniform_chunks(df, min_tokens=64, max_tokens=256)
    assert len(chunked) == 1
