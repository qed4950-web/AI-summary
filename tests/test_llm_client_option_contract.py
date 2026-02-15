from __future__ import annotations

import pytest

from core.conversation.llm_client import LocalLlamaCliClient, create_llm_client

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_create_local_llama_cli_client_resolves_temperature_and_tokens() -> None:
    client = create_llm_client(
        "local_llama_cli",
        model="models/gguf/dummy.gguf",
        options={
            "n_ctx": "1024",
            "n_threads": "2",
            "n_gpu_layers": "0",
            "max_new_tokens": "321",
            "temperature": "0.42",
        },
    )
    assert isinstance(client, LocalLlamaCliClient)
    assert client.max_new_tokens == 321
    assert client.temperature == pytest.approx(0.42)


def test_create_local_llamacpp_fallback_keeps_runtime_options() -> None:
    client = create_llm_client(
        "local_llamacpp",
        model="models/gguf/missing-model.gguf",
        options={
            "n_gpu_layers": "0",
            "max_new_tokens": "111",
            "temperature": "0.33",
        },
    )
    assert client is not None
    assert getattr(client, "max_new_tokens", None) == 111
    assert float(getattr(client, "temperature", 0.0)) == pytest.approx(0.33)
