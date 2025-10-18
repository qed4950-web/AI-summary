from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional


class LLMClientError(RuntimeError):
    """Raised when the local LLM backend fails."""


@dataclass
class LLMClient:
    """Abstract base for lightweight local LLM clients."""

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, *, system: Optional[str] = None, timeout: float = 30.0) -> str:
        raise NotImplementedError


@dataclass
class OllamaClient(LLMClient):
    """Executes prompts against a locally running Ollama CLI."""

    model: str = "llama3"
    host: str = ""
    options: Dict[str, str] = field(default_factory=dict)

    def is_available(self) -> bool:
        return shutil.which("ollama") is not None

    def generate(self, prompt: str, *, system: Optional[str] = None, timeout: float = 30.0) -> str:
        if not self.is_available():
            raise LLMClientError("ollama executable not found in PATH")

        env = os.environ.copy()
        if self.host:
            env["OLLAMA_HOST"] = self.host

        full_prompt = prompt if not system else f"{system.strip()}\n\n{prompt.strip()}"
        cmd = ["ollama", "run", self.model]
        try:
            result = subprocess.run(
                cmd,
                input=full_prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMClientError(f"ollama request timed out after {timeout}s") from exc
        except OSError as exc:
            raise LLMClientError(f"failed to invoke ollama: {exc}") from exc

        if result.returncode != 0:
            raise LLMClientError(
                f"ollama run failed (exit {result.returncode}): {result.stderr.decode('utf-8', 'ignore').strip()}"
            )
        return result.stdout.decode("utf-8", "ignore").strip()


def create_llm_client(backend: Optional[str], *, model: str, host: str = "", options: Optional[Dict[str, str]] = None) -> Optional[LLMClient]:
    backend = (backend or "").strip().lower()
    if not backend:
        return None
    if backend == "ollama":
        client = OllamaClient(model=model or "llama3", host=host or "", options=options or {})
        if not client.is_available():
            raise LLMClientError("ollama backend requested but executable not found")
        return client
    raise LLMClientError(f"unsupported LLM backend: {backend}")
