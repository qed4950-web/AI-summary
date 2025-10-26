"""Chunk/document cache helpers for incremental training."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore[assignment]


@dataclass
class CacheEntry:
    path: str
    doc_hash: str
    chunk_count: int
    updated_at: float


class ChunkCache:
    """Persist a lightweight mapping of document hashes for reuse/dedup."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._entries: Dict[str, CacheEntry] = {}
        self._hash_index: Dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for path, meta in payload.items():
                    if not isinstance(meta, dict):
                        continue
                    entry = CacheEntry(
                        path=path,
                        doc_hash=str(meta.get("doc_hash") or ""),
                        chunk_count=int(meta.get("chunk_count") or 0),
                        updated_at=float(meta.get("updated_at") or 0.0),
                    )
                    self._entries[path] = entry
                    if entry.doc_hash:
                        self._hash_index[entry.doc_hash] = path
        except Exception:
            # best-effort; corrupted cache will be rebuilt
            self._entries = {}
            self._hash_index = {}

    def mark_dirty(self) -> None:
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {path: asdict(entry) for path, entry in self._entries.items()}
        self.cache_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
        self._dirty = False

    def unchanged_paths(self, df: "pd.DataFrame") -> Set[str]:
        """Return paths whose doc_hash matches cached value."""
        if pd is None or df is None or df.empty or "path" not in df.columns:
            return set()
        if "doc_hash" not in df.columns:
            return set()
        unchanged: Set[str] = set()
        for path, doc_hash in zip(df["path"], df["doc_hash"]):
            key = str(path or "")
            cached = self._entries.get(key)
            if not key or not doc_hash or cached is None:
                continue
            if cached.doc_hash == str(doc_hash):
                unchanged.add(key)
        return unchanged

    def update_from_frame(self, df: "pd.DataFrame") -> None:
        """Refresh cache entries using the final corpus frame."""
        if pd is None or df is None or df.empty or "path" not in df.columns:
            return
        now = time.time()
        grouped = df.groupby("path", dropna=True)
        for path, group in grouped:
            doc_hash = ""
            if "doc_hash" in group.columns:
                doc_hash = str(group["doc_hash"].fillna("").iloc[0])
            if "chunk_count" in group.columns and not group["chunk_count"].isnull().all():
                chunk_count = int(group["chunk_count"].fillna(0).iloc[0])
            else:
                chunk_count = int(len(group))
            entry = CacheEntry(
                path=str(path),
                doc_hash=doc_hash,
                chunk_count=chunk_count,
                updated_at=now,
            )
            self._entries[str(path)] = entry
            if entry.doc_hash:
                self._hash_index[entry.doc_hash] = str(path)
            self._dirty = True

    def drop_paths(self, missing: Iterable[str]) -> None:
        removed = False
        for path in missing:
            entry = self._entries.pop(path, None)
            if entry and entry.doc_hash in self._hash_index:
                self._hash_index.pop(entry.doc_hash, None)
                removed = True
        if removed:
            self._dirty = True

    def known_paths(self) -> Set[str]:
        return set(self._entries.keys())
