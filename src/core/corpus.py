"""Compatibility wrapper around the new corpus builder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.data_pipeline.pipeline import CorpusBuilder as _CorpusBuilderImpl


@dataclass
class CorpusBuilder:
    """Thin wrapper that preserves the legacy interface."""

    progress: bool = True
    translate: bool = False
    max_workers: int | None = None

    def __post_init__(self) -> None:
        self._builder = _CorpusBuilderImpl(
            progress=self.progress,
            translate=self.translate,
            max_workers=self.max_workers,
        )

    def build(self, file_rows: List[Dict[str, Any]]):
        return self._builder.build(file_rows)

    @staticmethod
    def save(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

