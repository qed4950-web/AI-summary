"""PII masking helpers for meeting artifacts."""
from __future__ import annotations

from typing import List, Optional

from .constants import PII_EMAIL_RE, PII_PHONE_RE


def mask_text(text: Optional[str]) -> str:
    """Redact email addresses and phone numbers from free text."""
    if not text:
        return ""
    masked = PII_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    masked = PII_PHONE_RE.sub("[REDACTED_PHONE]", masked)
    return masked


def mask_segments(segments: Optional[List[dict]]) -> List[dict]:
    """Return a new list with text fields masked."""
    if not segments:
        return []
    masked_segments: List[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        masked_segments.append(
            {
                **segment,
                "text": mask_text(segment.get("text")),
            }
        )
    return masked_segments


__all__ = ["mask_text", "mask_segments"]
