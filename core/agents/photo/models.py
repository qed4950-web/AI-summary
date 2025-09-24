"""Dataclasses for photo agent inputs and outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PhotoJobConfig:
    roots: List[Path]
    output_dir: Path
    policy_tag: Optional[str] = None
    prefer_gpu: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PhotoAsset:
    path: Path
    tags: List[str]
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PhotoRecommendation:
    best_shots: List[PhotoAsset]
    duplicates: List[List[PhotoAsset]]
    similar_groups: List[List[PhotoAsset]]
    report_path: Path
