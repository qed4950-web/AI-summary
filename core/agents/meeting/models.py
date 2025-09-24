"""Dataclasses describing meeting agent inputs/outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class MeetingJobConfig:
    audio_path: Path
    output_dir: Path
    language: str = "ko"
    diarize: bool = False
    speaker_count: Optional[int] = None
    policy_tag: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MeetingTranscriptionResult:
    text: str
    segments: List[dict]
    duration_seconds: float
    language: str


@dataclass
class MeetingSummary:
    highlights: List[str]
    action_items: List[str]
    decisions: List[str]
    raw_summary: str
    transcript_path: Path
    structured_summary: dict
