from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.data_pipeline.pipeline import run_step2


@pytest.fixture
def mock_policy_engine() -> MagicMock:
    engine = MagicMock()

    def allows_side_effect(path, agent, include_manual=True):  # noqa: ARG001
        return "secret" not in str(path)

    engine.allows.side_effect = allows_side_effect
    return engine


@pytest.mark.smoke
@pytest.mark.integration
def test_pipeline_policy_integration(mock_policy_engine: MagicMock, tmp_path) -> None:
    safe_file = tmp_path / "safe.txt"
    secret_file = tmp_path / "secret.txt"
    safe_file.touch()
    secret_file.touch()

    rows = [
        {"path": str(safe_file), "size": 10, "mtime": 100.0, "ext": ".txt"},
        {"path": str(secret_file), "size": 10, "mtime": 100.0, "ext": ".txt"},
    ]

    with (
        patch("core.data_pipeline.pipeline.pd", new=MagicMock()),
        patch("core.data_pipeline.pipeline._create_chunk_cache", return_value=None),
        patch("core.data_pipeline.pipeline.load_scan_state", return_value=None),
        patch("core.data_pipeline.pipeline.CorpusBuilder") as mock_builder,
        patch("core.data_pipeline.pipeline.TopicModel"),
        patch("builtins.print") as mock_print,
    ):
        mock_builder.return_value.process_rows.return_value = (None, None)

        try:
            run_step2(
                rows,
                out_corpus=tmp_path / "corpus.parquet",
                out_model=tmp_path / "model.joblib",
                policy_engine=mock_policy_engine,
                use_tqdm=False,
                train_embeddings=False,
            )
        except Exception:
            # This test validates policy filtering invocation and warning output only.
            pass

    assert mock_policy_engine.allows.call_count >= 2
    print_calls = [str(call) for call in mock_print.mock_calls]
    assert any("정책에 위반되는 1개 파일" in call for call in print_calls)
