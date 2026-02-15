# scripts/pipeline/infopilot_cli/pipeline_runner.py
from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

# Core Imports
from core.data_pipeline.pipeline import CorpusBuilder, remove_from_corpus, update_corpus_file
from core.data_pipeline.scanner import DEFAULT_EXTS, collect_file_metadata, scan_directory
from core.policy.engine import PolicyEngine
from core.search.retriever import MODEL_TEXT_COLUMN, VectorIndex, _split_tokens

logger = logging.getLogger(__name__)


def load_vector_index(cache_dir: Path) -> VectorIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = cache_dir / "doc_meta.json"
    emb = cache_dir / "doc_embeddings.npy"
    faiss_path = cache_dir / "doc_index.faiss"

    index = VectorIndex()
    if meta.exists():
        try:
            index.load(
                emb if emb.exists() else None,
                meta,
                faiss_path=faiss_path if faiss_path.exists() else None,
                use_mmap=False,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"⚠️ 인덱스 로드 실패 → 새 인덱스를 생성합니다: {exc}")
            index = VectorIndex()
    return index


def sync_scan_csv(
    scan_csv: Path,
    rows_to_add: List[Dict[str, Any]],
    paths_to_remove: Set[str],
) -> None:
    if not rows_to_add and not paths_to_remove:
        return

    scan_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, Any]] = []
    if scan_csv.exists():
        try:
            with scan_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    existing_rows.append(dict(row))
        except (OSError, csv.Error):
            existing_rows = []

    additions_by_path: Dict[str, Dict[str, Any]] = {}
    for row in rows_to_add:
        path = str(row.get("path", "") or "").strip()
        if not path:
            continue
        additions_by_path[path] = row

    keep: List[Dict[str, Any]] = []
    removed = {str(p) for p in paths_to_remove}
    seen_paths: Set[str] = set()
    for row in existing_rows:
        path = str(row.get("path", "") or "").strip()
        if not path:
            continue
        if path in removed:
            continue
        if path in additions_by_path:
            continue
        if path in seen_paths:
            continue
        keep.append(row)
        seen_paths.add(path)

    for path, row in sorted(additions_by_path.items(), key=lambda kv: kv[0]):
        if path in removed or path in seen_paths:
            continue
        keep.append(row)
        seen_paths.add(path)

    fieldnames: List[str] = []
    for row in keep:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["path"]

    tmp = scan_csv.with_suffix(scan_csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in keep:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(scan_csv)


class IncrementalPipeline:
    def __init__(
        self,
        *,
        encoder: Any,
        batch_size: int,
        scan_csv: Path,
        corpus_path: Path,
        cache_dir: Path,
        translate: bool,
        policy_engine: Optional[PolicyEngine] = None,
        policy_engine_provider: Optional[Callable[[], Optional[PolicyEngine]]] = None,
        policy_reload_callback: Optional[Callable[[Optional[PolicyEngine]], None]] = None,
        policy_path: Optional[Path] = None,
        roots: Optional[List[Path]] = None,
        agent: str = "knowledge_search",
        require_policy_engine: bool = False,
    ) -> None:
        self.encoder = encoder
        self.batch_size = max(1, int(batch_size))
        self.scan_csv = scan_csv
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.translate = translate
        self.allowed_exts = {ext.lower() for ext in DEFAULT_EXTS}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy_engine = policy_engine
        self._policy_engine_provider = policy_engine_provider
        self._policy_reload_callback = policy_reload_callback
        self.policy_path = policy_path
        self.roots = list(roots) if roots else []
        self.policy_agent = agent
        self.require_policy_engine = require_policy_engine
        self._policy_gate_block_notified = False

    def _current_policy_engine(self) -> Optional[PolicyEngine]:
        if self._policy_engine_provider is None:
            return self.policy_engine
        try:
            provided = self._policy_engine_provider()
            if provided is None:
                return self.policy_engine
            return provided
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return self.policy_engine

    def process(self, add_paths: Set[str], remove_paths: Set[str]) -> None:
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")
        if np is None:
            raise RuntimeError("numpy 필요. pip install numpy")
        ignore_policy = str(self.policy_path) if self.policy_path else ""
        add_paths = {p for p in add_paths if Path(p).suffix.lower() in self.allowed_exts and p != ignore_policy}
        remove_paths = {p for p in remove_paths if Path(p).suffix.lower() in self.allowed_exts}
        policy_engine = self._current_policy_engine()
        add_paths = self._filter_add_paths_for_policy_gate(add_paths, policy_engine)
        if policy_engine and policy_engine.has_policies and add_paths:
            add_paths = {
                p
                for p in add_paths
                if policy_engine.allows(Path(p), agent=self.policy_agent, include_manual=False)
            }

        rows_to_add: List[Dict[str, Any]] = []
        for raw_path in sorted(add_paths):
            path = Path(raw_path)
            if policy_engine and policy_engine.has_policies and not policy_engine.allows(
                path, agent=self.policy_agent, include_manual=False
            ):
                continue
            meta = collect_file_metadata(path, allowed_exts=self.allowed_exts)
            if meta:
                if policy_engine and policy_engine.has_policies:
                    meta["policy_mask_pii"] = policy_engine.pii_mask_enabled_for_path(path, agent=self.policy_agent)
                rows_to_add.append(meta)

        sync_scan_csv(self.scan_csv, rows_to_add, {str(p) for p in remove_paths})

        if remove_paths:
            remove_from_corpus(list(remove_paths), self.corpus_path)

        if rows_to_add:
            cb = CorpusBuilder(progress=False, translate=self.translate)
            new_records = cb.build(rows_to_add)
        else:
            new_records = None

        if new_records is not None and not new_records.empty:
            update_corpus_file(new_records, self.corpus_path)

        index = load_vector_index(self.cache_dir)

        paths_to_remove_idx = set(remove_paths)
        paths_to_remove_idx.update(row["path"] for row in rows_to_add if "path" in row)
        if paths_to_remove_idx:
            index.remove_paths(paths_to_remove_idx)

        if new_records is None or new_records.empty:
            index.save(self.cache_dir)
            if rows_to_add or remove_paths:
                print(
                    f"⚡ watcher: removed {len(paths_to_remove_idx)} 문서, 새 문서 없음.",
                    flush=True,
                )
            return

        valid_mask = new_records.get("ok", True)
        if isinstance(valid_mask, pd.Series):
            valid_df = new_records[valid_mask & (new_records[MODEL_TEXT_COLUMN].astype(str).str.len() > 0)].copy()
        else:
            valid_df = new_records.copy()

        if valid_df.empty:
            index.save(self.cache_dir)
            print(
                f"⚡ watcher: 갱신 {len(rows_to_add)}건 중 유효 텍스트가 없습니다.",
                flush=True,
            )
            return

        texts = valid_df[MODEL_TEXT_COLUMN].astype(str).tolist()
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        token_lists = [[tok for tok in _split_tokens(text.lower()) if tok] for text in texts]
        previews_series = valid_df["text_original"] if "text_original" in valid_df.columns else valid_df["text"]
        previews = previews_series.fillna("").astype(str).tolist()

        for idx, (_, row) in enumerate(valid_df.iterrows()):
            index.upsert(
                path=str(row.get("path", "")),
                ext=str(row.get("ext", "")),
                embedding=embeddings[idx],
                preview=previews[idx],
                size=int(row.get("size", 0) or 0),
                mtime=float(row.get("mtime", 0.0) or 0.0),
                ctime=float(row.get("ctime", 0.0) or 0.0),
                owner=str(row.get("owner", "") or ""),
                tokens=token_lists[idx],
            )

        index.save(self.cache_dir)
        print(
            f"⚡ watcher: 문서 {len(valid_df)}건 업데이트 (제거 {len(paths_to_remove_idx)})",
            flush=True,
        )

    def _filter_add_paths_for_policy_gate(
        self,
        add_paths: Set[str],
        policy_engine: Optional[PolicyEngine],
    ) -> Set[str]:
        if not add_paths:
            return add_paths
        if not self.require_policy_engine or policy_engine is not None:
            self._policy_gate_block_notified = False
            return add_paths
        if not self._policy_gate_block_notified:
            print("🔒 정책 엔진이 준비되지 않아 add 이벤트를 fail-closed로 차단합니다.", flush=True)
            self._policy_gate_block_notified = True
        return set()

    def handle_policy_change(self) -> None:
        if not self.policy_path:
            return
        try:
            updated = PolicyEngine.from_file(self.policy_path)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"⚠️ 정책 리로드 실패({self.policy_path}): {exc}")
            return

        self.policy_engine = updated
        if self._policy_reload_callback is not None:
            try:
                self._policy_reload_callback(updated)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Policy reload callback failed: %s", exc)

        if not self.roots:
            return

        candidates = []
        for root in self.roots:
            if root.exists():
                candidates.extend(scan_directory(root, exts=self.allowed_exts))

        allowed_paths: Set[str] = set()
        updated_rows: List[Dict[str, Any]] = []
        policy_path_str = str(self.policy_path) if self.policy_path else ""
        for row in candidates:
            raw_path = str(row.get("path", "") or "").strip()
            if not raw_path:
                continue
            if policy_path_str and raw_path == policy_path_str:
                continue
            path = Path(raw_path)
            if updated and updated.has_policies and not updated.allows(path, agent=self.policy_agent, include_manual=False):
                continue
            record = dict(row)
            record["policy_mask_pii"] = (
                bool(updated.pii_mask_enabled_for_path(path, agent=self.policy_agent)) if updated and updated.has_policies else False
            )
            allowed_paths.add(raw_path)
            updated_rows.append(record)

        existing_paths: Set[str] = set()
        if self.scan_csv.exists():
            try:
                with self.scan_csv.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for record in reader:
                        raw = str(record.get("path", "") or "").strip()
                        if raw:
                            existing_paths.add(raw)
            except (OSError, csv.Error):
                existing_paths = set()

        removed_paths = existing_paths - allowed_paths
        added_paths = allowed_paths - existing_paths
        if removed_paths or added_paths:
            sync_scan_csv(self.scan_csv, updated_rows, removed_paths)
            self.process(added_paths, removed_paths)
        print(
            f"🔄 정책 변경 반영: +{len(added_paths)} / -{len(removed_paths)} (roots={len(self.roots)})",
            flush=True,
        )


def watch_loop(
    event_queue: "queue.Queue[Tuple[str, str]]",
    pipeline_ctx: IncrementalPipeline,
    stop_event: threading.Event,
    debounce_sec: float,
) -> None:
    pending_add: Set[str] = set()
    pending_remove: Set[str] = set()
    policy_dirty = False
    last_event = 0.0

    def _log_throughput(add_count: int, remove_count: int, elapsed: float) -> None:
        total = add_count + remove_count
        if total <= 0:
            return
        rate = total / elapsed if elapsed > 0 else 0.0
        print(
            (
                "⚙️ watcher: processed add={add} remove={rem} in {secs:.2f}s "
                "(~{rate:.1f}/s)"
            ).format(add=add_count, rem=remove_count, secs=elapsed, rate=rate),
            flush=True,
        )

    while not stop_event.is_set():
        try:
            event_type, path = event_queue.get(timeout=0.5)
            path = str(path)
            if event_type == "policy_changed":
                policy_dirty = True
                last_event = time.time()
            elif event_type == "deleted":
                pending_remove.add(path)
                pending_add.discard(path)
                last_event = time.time()
            # Normalized event types from watchers.py.
            elif event_type == "add":
                pending_add.add(path)
                pending_remove.discard(path)
                last_event = time.time()
            elif event_type == "remove":
                pending_remove.add(path)
                pending_add.discard(path)
                last_event = time.time()
            elif event_type == "policy_reload":
                policy_dirty = True
                last_event = time.time()
            # Legacy fallback
            elif event_type in ("created", "modified"):
                pending_add.add(path)
                pending_remove.discard(path)
                last_event = time.time()

        except queue.Empty:
            pass

        now = time.time()
        if (policy_dirty or pending_add or pending_remove) and (now - last_event) >= debounce_sec:
            if policy_dirty:
                policy_dirty = False
                pending_add.clear()
                pending_remove.clear()
                try:
                    pipeline_ctx.handle_policy_change()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    print(f"⚠️ 정책 변경 처리 중 오류: {exc}")
                continue
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            pending_add.clear()
            pending_remove.clear()
            try:
                t0 = time.time()
                pipeline_ctx.process(to_add, to_remove)
                _log_throughput(len(to_add), len(to_remove), time.time() - t0)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"⚠️ 증분 파이프라인 처리 중 오류: {exc}")

    if pending_add or pending_remove:
        try:
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            t0 = time.time()
            pipeline_ctx.process(to_add, to_remove)
            _log_throughput(len(to_add), len(to_remove), time.time() - t0)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"⚠️ 증분 파이프라인 종료 처리 중 오류: {exc}")
