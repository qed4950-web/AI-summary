from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, request

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None

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
    """Executes prompts against an Ollama daemon via its HTTP API."""

    model: str = "llama3"
    host: str = ""
    options: Dict[str, str] = field(default_factory=dict)
    _context: Optional[List[int]] = field(init=False, default=None)

    def _resolve_base_url(self) -> str:
        host = (self.host or os.getenv("OLLAMA_HOST") or "127.0.0.1:11434").strip()
        if not host:
            host = "127.0.0.1:11434"
        if host.startswith(("http://", "https://")):
            base = host
        else:
            base = f"http://{host}"
        return base.rstrip("/")

    def is_available(self) -> bool:
        base = self._resolve_base_url()
        try:
            with request.urlopen(f"{base}/api/tags", timeout=2.0) as resp:
                code = getattr(resp, "status", None)
                if code is None:
                    code = resp.getcode()
                return 200 <= int(code) < 400
        except Exception:
            return False

    def generate(self, prompt: str, *, system: Optional[str] = None, timeout: float = 30.0) -> str:
        base = self._resolve_base_url()
        full_prompt = prompt if not system else f"{system.strip()}\n\n{prompt.strip()}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }
        if self._context:
            payload["context"] = self._context

        options = self._prepare_options(system)
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{base}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "ignore")
        except socket.timeout as exc:
            raise LLMClientError(f"ollama request timed out after {timeout}s") from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore") if hasattr(exc, "read") else ""
            message = detail.strip() or str(exc)
            raise LLMClientError(f"ollama request failed ({exc.code}): {message}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"ollama connection failed: {exc.reason}") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMClientError("ollama response decoding failed") from exc

        if not isinstance(payload, dict):
            raise LLMClientError("ollama response was not a JSON object")
        if "error" in payload:
            raise LLMClientError(f"ollama returned an error: {payload['error']}")

        context = payload.get("context")
        if isinstance(context, list):
            self._context = context
        text = payload.get("response", "")
        if not isinstance(text, str):
            text = ""
        return text.strip()

    def _prepare_options(self, system: Optional[str]) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        for key, value in (self.options or {}).items():
            if key.lower() == "api_key":
                continue
            options[key] = self._coerce_option_value(value)

        env_default = os.getenv("LNPCHAT_OLLAMA_NUM_PREDICT")
        if env_default:
            options.setdefault("num_predict", self._coerce_option_value(env_default))
        else:
            options.setdefault("num_predict", 512)

        if system and "health check" in system.lower():
            try:
                max_tokens = int(options.get("num_predict", 64))
            except (TypeError, ValueError):
                max_tokens = 64
            options["num_predict"] = max(1, min(64, max_tokens))
            options.setdefault("temperature", 0.0)

        return options

    @staticmethod
    def _coerce_option_value(value: Any) -> Any:
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return raw
            lowered = raw.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                if "." in raw:
                    return float(raw)
                return int(raw)
            except ValueError:
                return raw
        return value


@dataclass
class LocalGemmaClient(LLMClient):
    """Loads a local Gemma (or any HF causal LM) without running a server."""

    model: str
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 512
    _pipe: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if pipeline is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise LLMClientError("transformers가 필요합니다. `pip install transformers` 후 다시 시도하세요.")
        dtype = self._resolve_dtype(self.torch_dtype)
        model_kwargs: Dict[str, Any] = {}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        try:
            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.model,
                model_kwargs=model_kwargs,
                device_map=self.device or "auto",
                trust_remote_code=False,
            )
        except Exception as exc:
            raise LLMClientError(f"로컬 모델 로드에 실패했습니다: {exc}") from exc

    def is_available(self) -> bool:
        return self._pipe is not None

    def generate(self, prompt: str, *, system: Optional[str] = None, timeout: float = 30.0) -> str:
        if self._pipe is None:
            raise LLMClientError("로컬 모델이 초기화되지 않았습니다.")
        full_prompt = prompt if not system else f"{system.strip()}\n\n{prompt.strip()}"
        try:
            outputs = self._pipe(
                full_prompt,
                max_new_tokens=max(1, int(self.max_new_tokens)),
                do_sample=False,
                num_return_sequences=1,
            )
        except Exception as exc:
            raise LLMClientError(f"로컬 모델 호출에 실패했습니다: {exc}") from exc

        if not outputs:
            return ""
        first = outputs[0]
        if isinstance(first, dict):
            text = first.get("generated_text", "")
        else:
            text = str(first)
        text = text or ""
        # Best-effort: 제거할 프롬프트가 있으면 잘라낸다.
        if text.startswith(full_prompt):
            text = text[len(full_prompt) :].lstrip()
        return text

    @staticmethod
    def _resolve_dtype(raw: str) -> Any:
        if torch is None:
            return "auto"
        lowered = (raw or "").strip().lower()
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        return "auto"


@dataclass
class LocalLlamaCppClient(LLMClient):
    """Loads a local GGUF (e.g., Gemma) via llama.cpp without any server."""

    model: str
    n_ctx: int = 4096
    n_threads: int = 0
    max_new_tokens: int = 512
    _llm: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if Llama is None:
            raise LLMClientError("llama-cpp-python이 필요합니다. `pip install llama-cpp-python` 후 다시 시도하세요.")
        try:
            self._llm = Llama(
                model_path=self.model,
                n_ctx=max(256, int(self.n_ctx)),
                n_threads=int(self.n_threads) if self.n_threads else None,
                logits_all=False,
            )
        except Exception as exc:
            raise LLMClientError(f"GGUF 로드에 실패했습니다: {exc}") from exc

    def is_available(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, *, system: Optional[str] = None, timeout: float = 30.0) -> str:
        if self._llm is None:
            raise LLMClientError("llama.cpp 모델이 초기화되지 않았습니다.")
        full_prompt = prompt if not system else f"{system.strip()}\n\n{prompt.strip()}"
        try:
            out = self._llm(
                prompt=full_prompt,
                max_tokens=max(1, int(self.max_new_tokens)),
                temperature=0.0,
                stop=["</s>"],
            )
        except Exception as exc:
            raise LLMClientError(f"llama.cpp 호출에 실패했습니다: {exc}") from exc
        text = ""
        if isinstance(out, dict):
            choices = out.get("choices") or []
            if choices and isinstance(choices[0], dict):
                text = choices[0].get("text", "") or ""
        if not text:
            text = str(out)
        return text.strip()


def create_llm_client(backend: Optional[str], *, model: str, host: str = "", options: Optional[Dict[str, str]] = None) -> Optional[LLMClient]:
    backend = (backend or "").strip().lower()
    if not backend:
        return None
    if backend == "ollama":
        client = OllamaClient(model=model or "llama3", host=host or "", options=options or {})
        if not client.is_available():
            raise LLMClientError("ollama backend requested but server is not reachable")
        return client
    if backend == "local_gemma":
        opts = options or {}
        device = opts.get("device") or host or "auto"
        torch_dtype = opts.get("torch_dtype") or "auto"
        max_new_tokens = opts.get("max_new_tokens") or opts.get("num_predict") or 512
        return LocalGemmaClient(
            model=model,
            device=str(device),
            torch_dtype=str(torch_dtype),
            max_new_tokens=int(max_new_tokens),
        )
    if backend == "local_llamacpp":
        opts = options or {}
        return LocalLlamaCppClient(
            model=model,
            n_ctx=int(opts.get("n_ctx", 4096)),
            n_threads=int(opts.get("n_threads", 0)),
            max_new_tokens=int(opts.get("max_new_tokens", opts.get("num_predict", 512))),
        )
    raise LLMClientError(f"unsupported LLM backend: {backend}")
