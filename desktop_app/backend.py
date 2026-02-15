
# desktop_app/backend.py
"""
Backend logic for the Desktop App, running in a separate QThread.
Directly imports core modules instead of using subprocess IPC.
"""
# ruff: noqa: E402

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from PySide6.QtCore import QObject, Signal, Slot

# Core Imports
# Ensure core is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config.desktop_runtime_policy import load_desktop_runtime_policy
from core.config.mode_profiles import MODE_ORDER, load_mode_profiles
from core.config.paths import DOCS_DIR
from core.conversation.lnp_chat import LNPChat

try:
    from core.agents.meeting.pii import mask_text as _mask_pii_text
except Exception:  # pragma: no cover - optional privacy helper import
    def _mask_pii_text(text: str) -> str:
        return text


class LNPBackend(QObject):
    """
    Worker that runs in a background thread.
    Handles Model Loading and Chat Interactions.
    """
    # Signals
    ready = Signal()                # Backend initialized
    response_ready = Signal(str)    # Final response text
    stream_update = Signal(str)     # Streaming chunk (if supported later)
    error_occurred = Signal(str)    # Error message
    status_msg = Signal(str)        # "Loading model...", "Thinking..."

    def __init__(self):
        super().__init__()
        self.chat: Optional[LNPChat] = None
        self._is_loading = False
        self._mode_profiles = load_mode_profiles()
        self._runtime_policy: Dict[str, Any] = {}
        self._mask_answer_pii = True
        self._max_response_chars = 24000
        self._max_suggestion_chars = 120
        self._max_reference_links = 5
        self._reload_runtime_policy()

    @staticmethod
    def _normalize_mode(raw_mode: str) -> str:
        mode = (raw_mode or "").strip().title()
        return mode if mode in MODE_ORDER else "Auto"

    def _resolve_mode_profile(self, raw_mode: str) -> Dict[str, Any]:
        mode = self._normalize_mode(raw_mode)
        profile = dict(self._mode_profiles.get(mode, self._mode_profiles.get("Auto", {})))
        profile["mode"] = mode
        return profile

    def _reload_mode_profiles(self) -> None:
        self._mode_profiles = load_mode_profiles()

    def _reload_runtime_policy(self) -> None:
        policy = load_desktop_runtime_policy()
        self._runtime_policy = policy
        self._mask_answer_pii = bool(policy.get("privacy_mask_enabled", True))
        self._max_response_chars = max(1200, self._as_int(policy.get("max_response_chars"), 24000))
        self._max_suggestion_chars = max(24, self._as_int(policy.get("max_suggestion_chars"), 120))
        self._max_reference_links = max(1, self._as_int(policy.get("max_reference_links"), 5))

    @Slot()
    def refresh_runtime_policy(self) -> None:
        self._reload_runtime_policy()
        self.status_msg.emit(
            f"Runtime policy synced • privacy={'mask' if self._mask_answer_pii else 'raw'} • refs<={self._max_reference_links}"
        )

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _apply_mode_runtime_profile(self, profile: Dict[str, Any]) -> None:
        if not self.chat:
            return
        client = getattr(self.chat, "llm_client", None)
        if client is None:
            return

        max_new_tokens = max(1, self._as_int(profile.get("llm_max_new_tokens"), 512))
        temperature = max(0.0, self._as_float(profile.get("llm_temperature"), 0.0))

        if hasattr(client, "max_new_tokens"):
            try:
                setattr(client, "max_new_tokens", max_new_tokens)
            except (AttributeError, TypeError, ValueError):
                pass
        if hasattr(client, "temperature"):
            try:
                setattr(client, "temperature", temperature)
            except (AttributeError, TypeError, ValueError):
                pass

        options = getattr(client, "options", None)
        if isinstance(options, dict):
            options["num_predict"] = max_new_tokens
            options["temperature"] = temperature

    def _format_runtime_status(
        self,
        profile: Dict[str, Any],
        mode_label: str,
        *,
        topk: object,
        force_action: object,
    ) -> str:
        thinking = str(profile.get("thinking_status", "Thinking")).strip() or "Thinking"
        action = "auto" if force_action in (None, "", "auto") else str(force_action)
        topk_text = "auto" if topk in (None, "", "auto") else str(topk)
        tokens = max(1, self._as_int(profile.get("llm_max_new_tokens"), 512))
        temperature = max(0.0, self._as_float(profile.get("llm_temperature"), 0.0))
        return (
            f"{thinking} ({mode_label}) • "
            f"action={action} • top-k={topk_text} • "
            f"tokens={tokens} • temp={temperature:.2f} • "
            f"privacy={'mask' if self._mask_answer_pii else 'raw'} • "
            f"refs<={self._max_reference_links}"
        )

    @staticmethod
    def _normalize_hit_path(raw_path: object) -> str:
        if not isinstance(raw_path, str):
            return ""
        candidate = raw_path.strip()
        if not candidate:
            return ""
        if "://" in candidate and not candidate.startswith("file://"):
            return ""

        if candidate.startswith("file://"):
            parsed = urlparse(candidate)
            candidate = unquote(parsed.path)
            if os.name == "nt" and candidate.startswith("/"):
                candidate = candidate.lstrip("/")

        path = Path(candidate).expanduser()
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            project_candidate = Path(project_root) / path
            docs_candidate = DOCS_DIR / path
            if cwd_candidate.exists():
                path = cwd_candidate
            elif project_candidate.exists():
                path = project_candidate
            elif docs_candidate.exists():
                path = docs_candidate
            else:
                # Avoid emitting unresolved relative paths as file links.
                return ""
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        return str(resolved)

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return unique

    def _collect_document_links(
        self, raw_hits: object, *, limit: int = 5
    ) -> tuple[list[dict[str, str]], bool, int, int, int]:
        if not isinstance(raw_hits, list):
            return [], False, 0, 0, 0
        links: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        masked_any = False
        overflow_count = 0
        skipped_invalid_count = 0
        merged_duplicate_count = 0
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, dict):
                skipped_invalid_count += 1
                continue
            path = self._normalize_hit_path(raw_hit.get("path", raw_hit.get("file_path", "")))
            if not path:
                skipped_invalid_count += 1
                continue
            dedupe_key = path.casefold() if os.name == "nt" else path
            if dedupe_key in seen_paths:
                merged_duplicate_count += 1
                continue
            seen_paths.add(dedupe_key)
            if len(links) >= limit:
                overflow_count += 1
                continue
            title_raw = str(raw_hit.get("title", raw_hit.get("filename", ""))).strip()
            if self._mask_answer_pii and title_raw:
                masked_title = _mask_pii_text(title_raw)
                masked_any = masked_any or (masked_title != title_raw)
                title_raw = masked_title
            fallback_name = Path(path).name or "Unknown"
            if self._mask_answer_pii and fallback_name:
                fallback_masked = _mask_pii_text(fallback_name)
                masked_any = masked_any or (fallback_masked != fallback_name)
                fallback_name = fallback_masked
            title = title_raw or fallback_name
            links.append({"title": title, "path": path})
        return links, masked_any, overflow_count, skipped_invalid_count, merged_duplicate_count

    def _mask_answer_text(self, answer: object) -> tuple[str, bool]:
        text = self._normalize_answer_value(answer)
        if not text:
            return "", False
        text = self._sanitize_reserved_link_token(text)
        if not self._mask_answer_pii:
            return text, False
        masked = _mask_pii_text(text)
        return masked, masked != text

    @staticmethod
    def _sanitize_reserved_link_token(text: str) -> str:
        if "[FILE_LINK:" not in text:
            return text
        return text.replace("[FILE_LINK:", "[FILE_LINK_BLOCKED:")

    @staticmethod
    def _normalize_answer_value(answer: object) -> str:
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer
        if isinstance(answer, tuple):
            answer = list(answer)
        if isinstance(answer, (dict, list, int, float, bool)):
            try:
                return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                return str(answer)
        return str(answer)

    def _normalize_suggestions(self, raw_suggestions: object) -> tuple[list[str], bool]:
        suggestions: list[str] = []
        masked_any = False
        if isinstance(raw_suggestions, list):
            for item in raw_suggestions:
                text = str(item).strip()
                if not text:
                    continue
                masked = _mask_pii_text(text) if self._mask_answer_pii else text
                masked_any = masked_any or (masked != text)
                masked = self._sanitize_reserved_link_token(masked)
                normalized = " ".join(masked.split())
                if len(normalized) > self._max_suggestion_chars:
                    normalized = normalized[: self._max_suggestion_chars - 3].rstrip() + "..."
                suggestions.append(normalized)
        return self._dedupe_strings(suggestions), masked_any

    def _truncate_response_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._max_response_chars:
            return text, False
        clipped = text[: self._max_response_chars].rstrip()
        return clipped, True

    @staticmethod
    def _to_file_link_token(path_str: str) -> str:
        path = Path(path_str)
        try:
            if path.is_absolute():
                return path.as_uri()
        except (ValueError, OSError):
            return path_str
        return path_str

    @staticmethod
    def _build_reference_section(links: list[dict[str, str]]) -> str:
        if not links:
            return ""
        lines = ["📎 참조 문서:"]
        for link in links:
            token = LNPBackend._to_file_link_token(str(link["path"]))
            lines.append(f"• {link['title']} [FILE_LINK:{token}]")
        return "\n".join(lines)

    @staticmethod
    def _build_reference_note(*, shown_count: int, overflow_count: int, invalid_count: int, merged_count: int) -> str:
        details: list[str] = []
        if overflow_count > 0:
            # Contract token: "참조 문서 안내: 총"
            details.append(f"총 {shown_count + overflow_count}건 중 상위 {shown_count}건만 포함되었습니다.")
        if invalid_count > 0:
            details.append(f"지원되지 않거나 유효하지 않은 링크 {invalid_count}건은 제외되었습니다.")
        if merged_count > 0:
            details.append(f"중복 링크 {merged_count}건은 병합되었습니다.")
        if not details:
            return ""
        return "(참조 문서 안내: " + " / ".join(details) + ")"

    @staticmethod
    def _normalize_response_payload(raw_payload: object) -> dict[str, object]:
        if isinstance(raw_payload, dict):
            payload = dict(raw_payload)
        elif raw_payload is None:
            payload = {}
        else:
            payload = {"answer": str(raw_payload)}

        answer = payload.get("answer", "")
        hits = payload.get("hits", [])
        suggestions = payload.get("suggestions", [])
        normalized_hits = list(hits) if isinstance(hits, (list, tuple)) else []
        normalized_suggestions = list(suggestions) if isinstance(suggestions, (list, tuple)) else []
        return {
            "answer": answer,
            "hits": normalized_hits,
            "suggestions": normalized_suggestions,
        }

    @Slot()
    def initialize(self):
        """Initializes the heavy LNPChat components."""
        if self.chat:
            self.ready.emit()
            return

        self._is_loading = True
        self.status_msg.emit("Initializing AI Core...")

        try:
            # 1. Initialize LLM (Ensure GPU is used via env var or default)
            # Fix: Explicitly ensure GPU layers setting here if needed,
            # though lnp_chat.py now defaults to -1.

            # 2. Create LNPChat Instance
            # Use absolute paths based on project_root
            project_path = Path(project_root)
            self.chat = LNPChat(
                model_path=project_path / "data/topic_model.joblib",
                corpus_path=project_path / "data/corpus.parquet",
                cache_dir=project_path / "data/cache",
                llm_model=str(project_path / "models/gguf/gemma-3-4b-it-Q4_K_M.gguf")
            )

            # 3. Build Retriever & LLM
            # LNPChat.__post_init__ calls _reset_llm_client but we might want explicit control.
            # Using .ready() triggers retriever build
            self.status_msg.emit("Loading Search Index...")
            self.chat.ready(rebuild=False)

            self.status_msg.emit("Ready")
            self.ready.emit()

        except Exception as e:
            self.error_occurred.emit(f"Initialization Failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_loading = False

    @Slot(str)
    @Slot(str, str)
    def handle_query(self, query: str, response_mode: str = "Auto"):
        """Process a user query."""
        if not self.chat:
            self.error_occurred.emit("Backend not initialized.")
            return

        try:
            self._reload_runtime_policy()
            self._reload_mode_profiles()
            profile = self._resolve_mode_profile(response_mode)
            mode_label = str(profile["mode"])
            topk = profile["topk"]
            force_action = profile["force_action"]
            if query.strip().startswith("/"):
                # Slash commands keep explicit command semantics.
                topk = None
                force_action = None

            self._apply_mode_runtime_profile(profile)
            self.status_msg.emit(
                self._format_runtime_status(
                    profile,
                    mode_label,
                    topk=topk,
                    force_action=force_action,
                )
            )

            # Using chat.ask() which is the public API we verified
            raw_payload = self.chat.ask(query, topk=topk, force_action=force_action)
            response_dict = self._normalize_response_payload(raw_payload)

            # Extract text answer with optional PII masking.
            answer_body, pii_masked = self._mask_answer_text(response_dict.get("answer", ""))

            # Truncate body first so file-link tokens remain parseable and clickable.
            answer_body, truncated = self._truncate_response_text(answer_body)

            # Extract document hits for clickable links (deduped/normalized).
            links, links_masked, links_overflow, links_skipped, links_merged = self._collect_document_links(
                response_dict.get("hits", []),
                limit=self._max_reference_links,
            )
            reference_section = self._build_reference_section(links)
            reference_note = self._build_reference_note(
                shown_count=len(links),
                overflow_count=links_overflow,
                invalid_count=links_skipped,
                merged_count=links_merged,
            )

            # Handle suggestions if present.
            suggestions, suggestions_masked = self._normalize_suggestions(response_dict.get("suggestions", []))
            tip_text = "(Tip: " + ", ".join(suggestions[:5]) + ")" if suggestions else ""

            sections: list[str] = []
            if answer_body.strip():
                sections.append(answer_body.strip())
            if reference_section:
                sections.append(reference_section)
            if reference_note:
                sections.append(reference_note)
            if tip_text:
                sections.append(tip_text)
            if truncated:
                sections.append("(안내: 응답이 길어 일부만 표시되었습니다.)")
            if pii_masked or suggestions_masked or links_masked:
                sections.append("(보안: 민감정보 일부 마스킹됨)")
            if not sections:
                sections.append("응답 결과가 비어 있습니다. 질문을 조금 더 구체적으로 입력해 주세요.")
            answer = "\n\n".join(sections).strip()

            self.response_ready.emit(answer)
            self.status_msg.emit("Ready")

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
            self.status_msg.emit("Error")
            import traceback
            traceback.print_exc()
