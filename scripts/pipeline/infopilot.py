# infopilot.py
from __future__ import annotations

import contextlib
import json
import hashlib
import itertools
import math
import os
import shutil
import queue
import sys
import threading
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import click

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import csv

csv.field_size_limit(10**7)  # 10MB까지 허용

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    FileSystemEventHandler = object  # type: ignore
    Observer = None


HISTORY_PATH = Path.home() / ".infopilot" / "agent_history.json"
MAX_AGENT_HISTORY = 5


# 모듈 임포트
from core.config.paths import (
    CACHE_DIR,
    CORPUS_PATH,
    DATA_DIR,
    TOPIC_MODEL_PATH,
    MODELS_DIR,
)
from core.data_pipeline.filefinder import FileFinder
from core.data_pipeline.policies.engine import PolicyEngine, SmartFolderPolicy
from core.data_pipeline.pipeline import (
    run_step2,
    TrainConfig,
    DEFAULT_N_COMPONENTS,
    DEFAULT_EMBED_MODEL,
    PARQUET_ENGINE,
    update_corpus_file,
    remove_from_corpus,
    CorpusBuilder,
    SentenceBertModel,
)
from core.infra.scheduler import JobScheduler, ScheduleSpec, ScheduledJob
from core.infra.models import ModelManager
from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.agents.meeting import MeetingAgent
from core.agents.photo import PhotoAgent
from core.conversation.orchestrator import AssistantOrchestrator
from core.search.retriever import (
    VectorIndex,
    MODEL_TEXT_COLUMN,
    _split_tokens,
)
from core.monitor import check_drift, ResourceLogger
from scripts.utils.mlflow_logger import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    mlflow_session,
)
from scripts.utils.quantizer import export_to_onnx


KNOWLEDGE_AGENT = "knowledge_search"
DEFAULT_POLICY_PATH = Path("./core/config/smart_folders.json")
DEFAULT_FOUND_FILES = DATA_DIR / "found_files.csv"
DEFAULT_SCHEDULED_ROOT = DATA_DIR / "scheduled"
DEFAULT_SCAN_STATE = DATA_DIR / "scan_state.json"
DEFAULT_CHUNK_CACHE = CACHE_DIR / "chunk_cache.json"
DEFAULT_RESOURCE_LOG = Path("logs/resource_log.jsonl")
DEFAULT_DRIFT_LOG = Path("logs/drift_log.jsonl")
DEFAULT_SEMANTIC_BASELINE = Path("logs/semantic_baseline.json")

_POLICY_CACHE: Dict[Path, PolicyEngine] = {}
_SENTENCE_ENCODER_MANAGER: Optional[ModelManager] = None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _require_pandas() -> None:
    if pd is None:
        raise click.ClickException(
            "pandas 라이브러리가 필요합니다. `pip install pandas` 또는 `bash scripts/setup_env.sh` 후 다시 시도하세요."
        )


def _load_agent_history() -> Dict[str, List[str]]:
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"meeting_audio": [], "photo_roots": []}
        meeting = [str(item) for item in payload.get("meeting_audio", []) if isinstance(item, str)]
        photo = [str(item) for item in payload.get("photo_roots", []) if isinstance(item, str)]
        return {
            "meeting_audio": meeting[:MAX_AGENT_HISTORY],
            "photo_roots": photo[:MAX_AGENT_HISTORY],
        }
    except Exception:
        return {"meeting_audio": [], "photo_roots": []}


def _save_agent_history(history: Dict[str, List[str]]) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _remember_agent_history(kind: str, values: Iterable[str]) -> None:
    if kind not in {"meeting_audio", "photo_roots"}:
        return
    history = _load_agent_history()
    original = history.get(kind, [])
    merged: List[str] = []
    for value in values:
        normalised = str(Path(value).expanduser())
        if normalised and normalised not in merged:
            merged.append(normalised)
    for existing in original:
        if existing not in merged:
            merged.append(existing)
    history[kind] = merged[:MAX_AGENT_HISTORY]
    _save_agent_history(history)


@contextlib.contextmanager
def _command_session(ctx: click.Context, run_name: str):
    """Attach MLflow + resource logger lifecycle to each CLI command."""

    settings = ctx.ensure_object(dict)
    use_mlflow: bool = settings.get("use_mlflow", True)
    tracking_uri: str = settings.get("mlflow_uri", DEFAULT_TRACKING_URI)
    experiment: str = settings.get("mlflow_experiment", DEFAULT_EXPERIMENT)
    resource_path: Optional[Path] = settings.get("resource_log_path")
    resource_interval: float = settings.get("resource_interval", 30.0)

    if use_mlflow:
        mlflow_cm = mlflow_session(
            run_name,
            experiment=experiment,
            tracking_uri=tracking_uri,
            tags={"command": run_name},
        )
    else:
        mlflow_cm = contextlib.nullcontext(None)

    resource_logger = None
    if resource_path:
        resource_logger = ResourceLogger(Path(resource_path), interval=resource_interval)
        resource_logger.start(context=run_name)

    try:
        with mlflow_cm as session:
            yield session
    finally:
        if resource_logger:
            resource_logger.stop()


def _configure_offline_transformers() -> None:
    """Ensure HuggingFace-dependent components run offline when weights exist locally."""
    base_dir = MODELS_DIR / "sentence_transformers"
    if not base_dir.exists():
        return
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(base_dir))
    os.environ.setdefault("HF_HOME", str(base_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Mac(MPS) 환경에서 OOM이 잦을 때만 수동으로 CPU 강제
    force_cpu = (os.getenv("INFOPILOT_FORCE_CPU", "") or "").strip().lower()
    if force_cpu in {"1", "true", "yes", "on"}:
        os.environ.setdefault("PYTORCH_MPS_ENABLE", "0")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _disable_mps_for_inference(reason: str = "") -> None:
    """Force CPU fallback for sentence-transformers when MPS OOM/errors occur."""
    if os.getenv("PYTORCH_MPS_ENABLE", "1") == "0":
        return
    os.environ["PYTORCH_MPS_ENABLE"] = "0"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if reason:
        print(f"⚠️ MPS 비활성화 후 CPU로 재시도합니다: {reason}", flush=True)


_configure_offline_transformers()


NORMALIZED_ALIASES = {
    "path": ("path", "filepath", "file_path", "fullpath", "full_path", "absolute_path"),
    "size": ("size", "filesize", "file_size", "bytes"),
    "mtime": ("mtime", "modified", "modified_time", "lastmodified", "timestamp"),
    "ctime": ("ctime", "created", "created_time", "creation", "creation_time"),
    "ext": ("ext", "extension", "suffix"),
    "drive": ("drive", "volume", "root"),
    "owner": ("owner", "user", "username", "author", "created_by"),
}


def _get_sentence_encoder_manager() -> ModelManager:
    global _SENTENCE_ENCODER_MANAGER
    if _SENTENCE_ENCODER_MANAGER is None:
        def _load(model_name: str):
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers 패키지가 필요합니다. pip install sentence-transformers")
            local_dir = MODELS_DIR / "sentence_transformers" / model_name
            if local_dir.exists():
                return SentenceTransformer(str(local_dir))
            return SentenceTransformer(model_name)

        _SENTENCE_ENCODER_MANAGER = ModelManager(loader=_load)
    return _SENTENCE_ENCODER_MANAGER


def _normalize_key(name: str) -> str:
    """Normalize header names by stripping non-alphanumerics and lowering case."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _pick_value(row: Dict[str, str], aliases) -> str:
    normalized = {_normalize_key(k): (k, v) for k, v in row.items() if k}
    for alias in aliases:
        alias_norm = _normalize_key(alias)
        data = normalized.get(alias_norm)
        if data:
            value = (data[1] or "").strip()
            if value:
                return value
    return ""


def _normalize_scan_row(raw: Dict[str, str], *, context: str = "") -> Dict[str, Any] | None:
    path = _pick_value(raw, NORMALIZED_ALIASES["path"])
    if not path:
        columns = ", ".join(k for k in raw.keys() if k)
        location = f" ({context})" if context else ""
        print(f"⚠️ 경고: 'path' 값을 찾지 못해 행을 건너뜁니다{location}. (감지한 열: {columns or '없음'})")
        return None

    size_raw = _pick_value(raw, NORMALIZED_ALIASES["size"])
    mtime_raw = _pick_value(raw, NORMALIZED_ALIASES["mtime"])
    ext = _pick_value(raw, NORMALIZED_ALIASES["ext"])
    drive = _pick_value(raw, NORMALIZED_ALIASES["drive"])
    ctime_raw = _pick_value(raw, NORMALIZED_ALIASES["ctime"])
    owner = _pick_value(raw, NORMALIZED_ALIASES["owner"])

    def to_int(value: str) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def to_float(value: str) -> float:
        try:
            out = float(value)
            if math.isnan(out) or math.isinf(out):
                return 0.0
            return out
        except (TypeError, ValueError):
            return 0.0

    normalized = dict(raw)
    normalized["path"] = path
    normalized["size"] = to_int(size_raw)
    normalized["mtime"] = to_float(mtime_raw)
    normalized["ctime"] = to_float(ctime_raw)
    if ext:
        normalized["ext"] = ext
    if drive:
        normalized["drive"] = drive
    if owner:
        normalized["owner"] = owner
    return normalized


def _parse_roots(raw_roots: List[str] | None) -> List[Path] | None:
    if not raw_roots:
        return None
    roots: List[Path] = []
    for raw in raw_roots:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"⚠️ 경고: 지정한 루트 '{p}'이(가) 존재하지 않아 건너뜁니다.")
            continue
        roots.append(p)
    if not roots:
        print("⚠️ 경고: 사용할 수 있는 루트가 없습니다.")
        return None
    return roots


def _load_policy_engine(
    policy_arg: Optional[str],
    *,
    fail_if_missing: bool = False,
    stage: str = "pipeline",
) -> PolicyEngine:
    """Load a policy engine with optional fail-closed semantics."""

    raw = (policy_arg or str(DEFAULT_POLICY_PATH)).strip()
    normalized = raw.lower()
    if normalized in {"none", ""}:
        if fail_if_missing:
            raise click.ClickException(
                f"[{stage}] 스마트 폴더 정책이 없어 파이프라인을 중단합니다. "
                "정책 파일을 지정하거나 --policy none 과 함께 --root 옵션을 명시하세요."
            )
        return PolicyEngine.empty()

    path = Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    if not resolved.exists():
        message = f"[{stage}] 스마트 폴더 정책 파일을 찾을 수 없습니다: {resolved}"
        if fail_if_missing:
            raise click.ClickException(message)
        print(f"⚠️ {message} (정책 미적용 상태로 진행)", flush=True)
        return PolicyEngine.empty()

    cache_key = resolved
    engine = _POLICY_CACHE.get(cache_key)
    if engine is None:
        try:
            engine = PolicyEngine.from_file(resolved)
        except Exception as exc:
            message = f"[{stage}] 정책 파일을 불러오지 못했습니다 ({resolved}): {exc}"
            if fail_if_missing:
                raise click.ClickException(message) from exc
            print(f"⚠️ {message}", flush=True)
            return PolicyEngine.empty()
        _POLICY_CACHE[cache_key] = engine
    return engine


def _run_scan(
    out: Path,
    roots: List[Path] | None = None,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    exts: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    scan_roots = roots
    if policy_engine and policy_engine.has_policies and not roots:
        candidate_roots = policy_engine.roots_for_agent(KNOWLEDGE_AGENT, include_manual=True)
        if candidate_roots:
            scan_roots = candidate_roots
            print("📁 정책 기반 스캔 루트:")
            for root in candidate_roots:
                print(f"   - {root}")

    normalized_exts: Optional[Set[str]] = None
    if exts:
        normalized_exts = set()
        for ext in exts:
            value = (ext or "").strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            normalized_exts.add(value)
        if not normalized_exts:
            normalized_exts = None

    finder = FileFinder(
        exts=normalized_exts or FileFinder.DEFAULT_EXTS,
        scan_all_drives=True,
        start_from_current_drive_only=False,
        follow_symlinks=False,
        max_depth=None,
        show_progress=True,
        progress_update_secs=0.5,
        estimate_total_dirs=False,
        startup_banner=True,
    )
    files = finder.find(roots=scan_roots, run_async=False)
    if policy_engine and policy_engine.has_policies:
        files = policy_engine.filter_records(files, agent=KNOWLEDGE_AGENT, include_manual=True)
    FileFinder.to_csv(files, out)
    print(f"📦 스캔 결과 저장: {out}")
    return files


def cmd_scan(args) -> int:
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="scan")
    roots = _parse_roots(args.roots)
    if not roots and policy_engine and policy_engine.has_policies:
        roots = policy_engine.roots_for_agent(KNOWLEDGE_AGENT, include_manual=True)
    if not roots:
        raise click.ClickException(
            "스마트 폴더 정책이나 스캔 루트가 없어 scan을 중단합니다. "
            "Park David Foundation 스펙에 따라 정책 기반 경계가 필수입니다."
        )
    rows = _run_scan(
        Path(args.out),
        roots,
        policy_engine=policy_engine,
        exts=getattr(args, "exts", None),
    )
    return len(rows)


def _resolve_scan_csv(path: Path) -> Path:
    if path.exists():
        return path

    search_root = path.parent if path.parent else Path(".")
    candidates = []
    for candidate in sorted(search_root.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with candidate.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
        except OSError:
            continue
        header_norm = {_normalize_key(h) for h in headers}
        if any(_normalize_key(alias) in header_norm for alias in NORMALIZED_ALIASES["path"]):
            candidates.append(candidate)

    if candidates:
        picked = candidates[0]
        print(f"⚠️ '{path}' 파일이 없어 '{picked}'을(를) 사용합니다.")
        return picked

    raise FileNotFoundError(f"스캔 CSV를 찾을 수 없습니다: {path}")


def _iter_scan_rows(scan_csv: Path) -> Iterator[Dict[str, Any]]:
    with scan_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=2):
            normalized = _normalize_scan_row(raw, context=f"{scan_csv}:{idx}")
            if normalized:
                yield normalized


def _load_scan_rows(
    scan_csv: Path,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    include_manual: bool = True,
) -> Iterator[Dict[str, Any]]:
    for row in _iter_scan_rows(scan_csv):
        if policy_engine and policy_engine.has_policies:
            raw_path = row.get("path")
            if not raw_path:
                continue
            if not policy_engine.allows(
                Path(str(raw_path)),
                agent=KNOWLEDGE_AGENT,
                include_manual=include_manual,
            ):
                continue
        yield row


@dataclass
class _PolicyArtifacts:
    base_dir: Path
    scan_csv: Path
    corpus: Path
    model: Path
    cache_dir: Path

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def _policy_slug(policy: SmartFolderPolicy) -> str:
    digest = hashlib.sha1(str(policy.path).encode("utf-8")).hexdigest()[:8]
    candidate = policy.path.name or policy.path.anchor.strip("\\/") or "policy"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in candidate).strip("_") or "policy"
    return f"{safe}-{digest}"


def _policy_artifacts(root: Path, policy: SmartFolderPolicy) -> _PolicyArtifacts:
    slug = _policy_slug(policy)
    base_dir = root / slug
    return _PolicyArtifacts(
        base_dir=base_dir,
        scan_csv=base_dir / "found_files.csv",
        corpus=base_dir / "corpus.parquet",
        model=base_dir / "topic_model.joblib",
        cache_dir=base_dir / "cache",
    )


def _sync_scan_csv(
    scan_csv: Path,
    rows_to_add: List[Dict[str, Any]],
    paths_to_remove: Set[str],
) -> None:
    if not rows_to_add and not paths_to_remove:
        return

    def _normalize_path(raw: Any) -> str:
        return str(raw or "").strip()

    fieldnames = ["path", "size", "mtime", "ctime", "ext", "drive", "owner"]
    additions: Dict[str, Dict[str, Any]] = {}
    for row in rows_to_add:
        path_key = _normalize_path(row.get("path"))
        if not path_key:
            continue
        additions[path_key] = {name: row.get(name) for name in fieldnames}

    removals = {_normalize_path(path) for path in paths_to_remove if _normalize_path(path)}
    removals.difference_update(additions.keys())

    scan_csv.parent.mkdir(parents=True, exist_ok=True)

    if not scan_csv.exists():
        with scan_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for record in additions.values():
                writer.writerow(record)
        return

    temp_path = scan_csv.with_suffix(scan_csv.suffix + ".tmp")
    with scan_csv.open("r", encoding="utf-8", newline="") as src, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            path_key = _normalize_path(row.get("path"))
            if not path_key or path_key in removals or path_key in additions:
                continue
            writer.writerow({name: row.get(name) for name in fieldnames})

        for record in additions.values():
            writer.writerow(record)

    temp_path.replace(scan_csv)


def _load_sentence_encoder(model_path: Path) -> Tuple[Optional[SentenceTransformer], int, str]:
    model_name = DEFAULT_EMBED_MODEL
    batch_size = 32

    if joblib is not None and model_path.exists():
        try:
            payload = joblib.load(model_path)
            model_name = payload.get("model_name", model_name)
            cfg = payload.get("train_config")
            if cfg and hasattr(cfg, "embedding_batch_size"):
                batch_size = int(getattr(cfg, "embedding_batch_size", batch_size) or batch_size)
        except Exception as exc:
            print(f"⚠️ 임베딩 모델 메타 로드 실패 → 기본값 사용({model_name}): {exc}")

    try:
        manager = _get_sentence_encoder_manager()
    except RuntimeError as exc:
        print(f"⚠️ SentenceTransformer 로더 초기화 실패: {exc}")
        return None, batch_size, model_name

    try:
        encoder = manager.get(model_name)
    except Exception as exc:
        print(f"⚠️ SentenceTransformer 모델 로드 실패({model_name}): {exc}")
        # MPS OOM/호환 문제일 수 있으니 CPU 강제 후 한 번만 재시도
        try:
            _disable_mps_for_inference(str(exc))
            # manager는 캐시를 갖고 있어 재생성이 필요하다
            global _SENTENCE_ENCODER_MANAGER
            _SENTENCE_ENCODER_MANAGER = None
            manager = _get_sentence_encoder_manager()
            encoder = manager.get(model_name)
        except Exception as exc_retry:
            print(f"⚠️ CPU 강제 재시도도 실패({model_name}): {exc_retry}")
            return None, batch_size, model_name
    return encoder, batch_size, model_name


def _load_vector_index(cache_dir: Path) -> VectorIndex:
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
        except Exception as exc:
            print(f"⚠️ 인덱스 로드 실패 → 새 인덱스를 생성합니다: {exc}")
            index = VectorIndex()
    return index


class WatchEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        event_queue: "queue.Queue[Tuple[str, str]]",
        allowed_exts: Set[str],
        *,
        policy_engine: Optional[PolicyEngine] = None,
        agent: str = KNOWLEDGE_AGENT,
    ) -> None:
        super().__init__()
        self._queue = event_queue
        self._allowed_exts = {ext.lower() for ext in allowed_exts}
        self._policy_engine = policy_engine
        self._policy_agent = agent

    def _should_process(self, path: str) -> bool:
        if not path:
            return False
        ext = Path(path).suffix.lower()
        if ext not in self._allowed_exts:
            return False
        if self._policy_engine and self._policy_engine.has_policies and not self._policy_engine.allows(
            Path(path), agent=self._policy_agent, include_manual=False
        ):
            return False
        return True

    def on_created(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("created", event.src_path))

    def on_modified(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("modified", event.src_path))

    def on_moved(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("deleted", event.src_path))
        if self._should_process(event.dest_path):
            self._queue.put(("created", event.dest_path))

    def on_deleted(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("deleted", event.src_path))


class IncrementalPipeline:
    def __init__(
        self,
        *,
        encoder: SentenceTransformer,
        batch_size: int,
        scan_csv: Path,
        corpus_path: Path,
        cache_dir: Path,
        translate: bool,
        policy_engine: Optional[PolicyEngine] = None,
    ) -> None:
        self.encoder = encoder
        self.batch_size = max(1, int(batch_size))
        self.scan_csv = scan_csv
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.translate = translate
        self.allowed_exts = {ext.lower() for ext in FileFinder.DEFAULT_EXTS}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy_engine = policy_engine
        self.policy_agent = KNOWLEDGE_AGENT

    def process(self, add_paths: Set[str], remove_paths: Set[str]) -> None:
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")
        add_paths = {p for p in add_paths if Path(p).suffix.lower() in self.allowed_exts}
        remove_paths = {p for p in remove_paths if Path(p).suffix.lower() in self.allowed_exts}
        if self.policy_engine and self.policy_engine.has_policies and add_paths:
            add_paths = {
                p
                for p in add_paths
                if self.policy_engine.allows(Path(p), agent=self.policy_agent, include_manual=False)
            }

        rows_to_add: List[Dict[str, Any]] = []
        for raw_path in sorted(add_paths):
            if self.policy_engine and self.policy_engine.has_policies and not self.policy_engine.allows(
                Path(raw_path), agent=self.policy_agent, include_manual=False
            ):
                continue
            meta = FileFinder.collect_file_metadata(Path(raw_path), allowed_exts=self.allowed_exts)
            if meta:
                rows_to_add.append(meta)

        _sync_scan_csv(self.scan_csv, rows_to_add, {str(p) for p in remove_paths})

        if remove_paths:
            remove_from_corpus(list(remove_paths), self.corpus_path)

        new_records = None
        if rows_to_add:
            cb = CorpusBuilder(progress=False, translate=self.translate)
            new_records = cb.build(rows_to_add)
        else:
            new_records = None

        if new_records is not None and not new_records.empty:
            update_corpus_file(new_records, self.corpus_path)

        index = _load_vector_index(self.cache_dir)

        paths_to_remove = set(remove_paths)
        paths_to_remove.update(row["path"] for row in rows_to_add if "path" in row)
        if paths_to_remove:
            index.remove_paths(paths_to_remove)

        if new_records is None or new_records.empty:
            index.save(self.cache_dir)
            if rows_to_add or remove_paths:
                print(
                    f"⚡ watcher: removed {len(paths_to_remove)} 문서, 새 문서 없음.",
                    flush=True,
                )
            return

        valid_mask = new_records.get("ok", True)
        if pd is not None and isinstance(valid_mask, pd.Series):
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
            f"⚡ watcher: 문서 {len(valid_df)}건 업데이트 (제거 {len(paths_to_remove)})",
            flush=True,
        )


def _watch_loop(
    event_queue: "queue.Queue[Tuple[str, str]]",
    pipeline_ctx: IncrementalPipeline,
    stop_event: threading.Event,
    debounce_sec: float,
) -> None:
    pending_add: Set[str] = set()
    pending_remove: Set[str] = set()
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
            if event_type == "deleted":
                pending_remove.add(path)
                pending_add.discard(path)
            else:
                pending_add.add(path)
                pending_remove.discard(path)
            last_event = time.time()
        except queue.Empty:
            pass

        now = time.time()
        if (pending_add or pending_remove) and (now - last_event) >= debounce_sec:
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            pending_add.clear()
            pending_remove.clear()
            try:
                t0 = time.time()
                pipeline_ctx.process(to_add, to_remove)
                _log_throughput(len(to_add), len(to_remove), time.time() - t0)
            except Exception as exc:
                print(f"⚠️ 증분 파이프라인 처리 중 오류: {exc}")

    # Flush remaining events
    if pending_add or pending_remove:
        try:
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            t0 = time.time()
            pipeline_ctx.process(to_add, to_remove)
            _log_throughput(len(to_add), len(to_remove), time.time() - t0)
        except Exception as exc:
            print(f"⚠️ 증분 파이프라인 종료 처리 중 오류: {exc}")


def _register_policy_jobs(
    scheduler: JobScheduler,
    *,
    policy_engine: PolicyEngine,
    agent: str,
    output_root: Path,
    translate: bool,
) -> List[ScheduledJob]:
    if not policy_engine or not policy_engine.has_policies:
        return []

    registered: List[ScheduledJob] = []
    output_root = output_root.expanduser()

    for policy in policy_engine.iter_policies():
        if not policy.allows_agent(agent):
            continue
        schedule = ScheduleSpec.from_policy(policy)
        if schedule.mode != "scheduled":
            continue

        artifacts = _policy_artifacts(output_root, policy)

        def _job(policy=policy, artifacts=artifacts) -> None:
            artifacts.ensure_dirs()
            rows = _run_scan(artifacts.scan_csv, [policy.path], policy_engine=policy_engine)
            filtered = policy_engine.filter_records(rows, agent=agent, include_manual=True)
            if not filtered and rows:
                filtered = rows
            if not filtered:
                print(f"⚠️ 스케줄러: {policy.path}에 처리할 문서가 없어 건너뜁니다.")
                return
            cfg = _default_train_config()
            run_step2(
                filtered,
                out_corpus=artifacts.corpus,
                out_model=artifacts.model,
                cfg=cfg,
                use_tqdm=False,
                translate=translate,
            )
            print(f"✅ 스케줄러: {policy.path} 학습 완료 → {artifacts.base_dir}")

        job_name = f"{agent}:{_policy_slug(policy)}"
        metadata = {
            "path": str(policy.path),
            "artifact_dir": str(artifacts.base_dir),
            "mode": schedule.mode,
        }
        job = scheduler.register_callable(
            job_name,
            _job,
            schedule,
            metadata=metadata,
            overwrite=True,
        )
        registered.append(job)

    return registered


def _build_train_config(args) -> TrainConfig:
    return TrainConfig(
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        use_sentence_transformer=getattr(args, "use_embedding", True),
        embedding_model=getattr(args, "embedding_model", DEFAULT_EMBED_MODEL),
        embedding_batch_size=getattr(args, "embedding_batch_size", 32),
        async_embeddings=getattr(args, "async_embed", True),
        embedding_concurrency=max(1, int(getattr(args, "embedding_concurrency", 1))),
        embedding_dtype=getattr(args, "embedding_dtype", "auto"),
        embedding_chunk_size=max(0, int(getattr(args, "embedding_chunk_size", 0) or 0)),
        embedding_chunk_start=max(0, int(getattr(args, "embedding_chunk_start", 0) or 0)),
        embedding_chunk_end=int(getattr(args, "embedding_chunk_end", -1) or -1),
        embedding_subprocess_fallback=bool(getattr(args, "embedding_subprocess_fallback", True)),
    )


def _maybe_limit_rows(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    iterator = iter(rows)
    if limit and limit > 0:
        limited = list(itertools.islice(iterator, limit))
        if next(iterator, None) is not None:
            print(f"⚡ 테스트 모드: 상위 {limit}개 파일만 사용합니다.")
        return limited
    return list(iterator)


def _default_train_config() -> TrainConfig:
    return TrainConfig(
        max_features=50000,
        n_components=DEFAULT_N_COMPONENTS,
        n_clusters=25,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        use_sentence_transformer=True,
        embedding_model=DEFAULT_EMBED_MODEL,
        embedding_batch_size=32,
        embedding_chunk_size=0,
        embedding_chunk_start=0,
        embedding_chunk_end=-1,
        embedding_subprocess_fallback=True,
    )


def _ensure_chat_artifacts(
    scan_csv: Path,
    corpus: Path,
    model: Path,
    *,
    translate: bool,
    auto_train: bool,
    policy_engine: Optional[PolicyEngine],
) -> bool:
    """Ensure chat artifacts exist and are up to date. Returns True if training ran."""

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    resolved_scan: Optional[Path] = None
    if scan_csv:
        try:
            resolved_scan = _resolve_scan_csv(scan_csv)
        except FileNotFoundError:
            resolved_scan = None

    artifacts_exist = corpus.exists() and model.exists()
    needs_train = not artifacts_exist

    if not needs_train and resolved_scan:
        scan_mtime = mtime(resolved_scan)
        artifacts_mtime = min(mtime(corpus), mtime(model))
        if scan_mtime > artifacts_mtime:
            needs_train = True

    if not needs_train:
        print("🔄 인덱스 최신성 확인 완료.")
        return False

    if resolved_scan is None:
        msg = (
            "⚠️ 학습 산출물이 없거나 오래되었지만 사용할 스캔 CSV를 찾지 못했습니다."
            " `--scan_csv` 경로를 확인하거나 'scan' 명령을 다시 실행해주세요."
        )
        raise FileNotFoundError(msg)

    if not auto_train:
        raise RuntimeError(
            "학습 산출물이 최신이 아닙니다. 'python infopilot.py train --scan_csv "
            f"{resolved_scan}'를 실행한 뒤 다시 시도해주세요."
        )

    print("⚠️ 스캔 결과가 모델보다 최신입니다. 자동으로 train 단계를 실행합니다.")
    rows = list(
        _load_scan_rows(
            resolved_scan,
            policy_engine=policy_engine,
            include_manual=False,
        )
    )
    if not rows:
        raise ValueError("자동 학습을 위한 유효한 행이 없습니다. 스캔 결과를 확인해주세요.")

    cfg = _default_train_config()
    run_step2(
        rows,
        out_corpus=corpus,
        out_model=model,
        cfg=cfg,
        use_tqdm=True,
        translate=translate,
    )
    print("✅ 자동 학습 완료")
    return True


def cmd_train(args):
    scan_csv = _resolve_scan_csv(Path(args.scan_csv))
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="train")
    row_iter = _load_scan_rows(scan_csv, policy_engine=policy_engine, include_manual=True)
    rows = _maybe_limit_rows(row_iter, args.limit_files)

    if not rows:
        raise ValueError("유효한 학습 대상 행이 없습니다. 스캔 CSV를 확인해주세요.")

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    chunk_cache_path = Path(getattr(args, "chunk_cache", DEFAULT_CHUNK_CACHE))
    state_path = Path(getattr(args, "state_file", DEFAULT_SCAN_STATE))
    df, tm = run_step2(
        rows,
        out_corpus=out_corpus,
        out_model=out_model,
        cfg=cfg,
        use_tqdm=True,
        translate=args.translate,
        scan_state_path=state_path,
        chunk_cache_path=chunk_cache_path,
        skip_extract=bool(getattr(args, "skip_extract", False)),
    )
    metrics = df.attrs.get("metrics", {}) if hasattr(df, "attrs") else {}
    incremental = df.attrs.get("incremental", {}) if hasattr(df, "attrs") else {}
    if metrics:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"📊 임베딩 품질 지표: {metric_str}")
    print("✅ 학습 완료")
    return {
        "rows": len(rows),
        "corpus": str(out_corpus),
        "model": str(out_model),
        "metrics": metrics,
        "incremental": incremental,
    }


def cmd_extract(args):
    scan_csv = _resolve_scan_csv(Path(args.scan_csv))
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="extract")
    row_iter = _load_scan_rows(scan_csv, policy_engine=policy_engine, include_manual=True)
    rows = _maybe_limit_rows(row_iter, args.limit_files)

    if not rows:
        raise ValueError("유효한 추출 대상 행이 없습니다. 스캔 CSV를 확인해주세요.")

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    chunk_cache_path = Path(getattr(args, "chunk_cache", DEFAULT_CHUNK_CACHE))
    state_path = Path(getattr(args, "state_file", DEFAULT_SCAN_STATE))
    df, _ = run_step2(
        rows,
        out_corpus=out_corpus,
        out_model=out_model,
        cfg=cfg,
        use_tqdm=True,
        translate=args.translate,
        scan_state_path=state_path,
        chunk_cache_path=chunk_cache_path,
        skip_extract=False,
        train_embeddings=False,
    )
    incremental = df.attrs.get("incremental", {}) if hasattr(df, "attrs") else {}
    print("✅ 추출 완료 (임베딩/모델 생성 없음)")
    return {
        "rows": len(rows),
        "corpus": str(out_corpus),
        "incremental": incremental,
    }


def cmd_embed(args):
    scan_csv = _resolve_scan_csv(Path(args.scan_csv))
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"기존 corpus가 없어 임베딩을 진행할 수 없습니다: {corpus_path}. 먼저 extract/train을 실행하세요."
        )

    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="embed")
    row_iter = _load_scan_rows(scan_csv, policy_engine=policy_engine, include_manual=True)
    rows = _maybe_limit_rows(row_iter, args.limit_files)

    if not rows:
        raise ValueError("유효한 임베딩 대상 행이 없습니다. 스캔 CSV를 확인해주세요.")

    cfg = _build_train_config(args)
    out_model = Path(args.model)
    chunk_cache_path = Path(getattr(args, "chunk_cache", DEFAULT_CHUNK_CACHE))
    state_path = Path(getattr(args, "state_file", DEFAULT_SCAN_STATE))
    df, tm = run_step2(
        rows,
        out_corpus=corpus_path,
        out_model=out_model,
        cfg=cfg,
        use_tqdm=True,
        translate=args.translate,
        scan_state_path=state_path,
        chunk_cache_path=chunk_cache_path,
        skip_extract=True,
        train_embeddings=True,
    )
    metrics = df.attrs.get("metrics", {}) if hasattr(df, "attrs") else {}
    incremental = df.attrs.get("incremental", {}) if hasattr(df, "attrs") else {}
    if metrics:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"📊 임베딩 품질 지표: {metric_str}")
    print("✅ 임베딩/모델 생성 완료 (기존 corpus 사용)")
    return {
        "rows": len(rows),
        "corpus": str(corpus_path),
        "model": str(out_model),
        "metrics": metrics,
        "incremental": incremental,
    }


def cmd_pipeline(args):
    out = Path(args.out)
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="pipeline")
    roots = _parse_roots(args.roots)
    if not roots and policy_engine and policy_engine.has_policies:
        roots = policy_engine.roots_for_agent(KNOWLEDGE_AGENT, include_manual=True)
    if not roots:
        raise click.ClickException(
            "스마트 폴더 정책이나 스캔 루트가 없어 파이프라인을 중단합니다. "
            "정책 파일을 지정하거나 --policy none 과 함께 --root를 명시하세요."
        )
    scan_rows = _run_scan(
        out,
        roots,
        policy_engine=policy_engine,
        exts=getattr(args, "exts", None),
    )
    filtered_rows = (
        scan_rows
        if not policy_engine or not policy_engine.has_policies
        else policy_engine.filter_records(scan_rows, agent=KNOWLEDGE_AGENT, include_manual=True)
    )
    rows = _maybe_limit_rows(filtered_rows, args.limit_files)

    if not rows:
        print("⚠️ 스캔 결과가 없어 파이프라인을 종료합니다. (목록에 포함될 문서가 없습니다.)")
        print("   → 다른 루트를 지정하거나 스마트 폴더 정책을 확인하세요.")
        return {}

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    chunk_cache_path = Path(args.chunk_cache) if getattr(args, "chunk_cache", None) else Path(args.cache) / "chunk_cache.json"
    state_path = Path(getattr(args, "state_file", DEFAULT_SCAN_STATE))
    df, tm = run_step2(
        rows,
        out_corpus=out_corpus,
        out_model=out_model,
        cfg=cfg,
        use_tqdm=True,
        translate=args.translate,
        scan_state_path=state_path,
        chunk_cache_path=chunk_cache_path,
    )
    print("✅ 파이프라인 완료")

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "ℹ️ 파이프라인은 scan/train 단계까지만 자동 실행되며 chat 모드는 별도 실행이 필요합니다.\n"
        f"   → python infopilot.py run chat --model {out_model} --corpus {out_corpus} --cache {cache_dir}"
    )

    if getattr(args, "launch_chat", False):
        print("\n💬 바로 chat 모드를 실행합니다. (종료하려면 'exit')")
        chat_args = SimpleNamespace(
            model=str(out_model),
            corpus=str(out_corpus),
            cache=str(cache_dir),
            scan_csv=str(out),
            topk=5,
            translate=args.translate,
            auto_train=True,
            rerank=True,
            rerank_model="BAAI/bge-reranker-large",
            rerank_depth=80,
            rerank_batch_size=16,
            rerank_device=None,
            rerank_min_score=0.35,
            lexical_weight=getattr(args, "lexical_weight", 0.35),
            show_translation=False,
            translation_lang="en",
            min_similarity=0.35,
            policy=str(getattr(args, "policy", str(DEFAULT_POLICY_PATH))),
        )
        cmd_chat(chat_args)

    metrics = df.attrs.get("metrics", {}) if hasattr(df, "attrs") else {}
    incremental = df.attrs.get("incremental", {}) if hasattr(df, "attrs") else {}
    if metrics:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"📊 임베딩 품질 지표: {metric_str}")

    return {
        "rows": len(rows),
        "corpus": str(out_corpus),
        "model": str(out_model),
        "cache": str(cache_dir),
        "metrics": metrics,
        "incremental": incremental,
    }


def cmd_index(args):
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="index")
    scope = getattr(args, "scope", "auto")

    limit = max(0, int(getattr(args, "limit_files", 0) or 0))
    corpus_path = Path(args.corpus)
    tmp_corpus: Optional[Path] = None

    if limit:
        _require_pandas()
        try:
            df = pd.read_parquet(corpus_path)
        except Exception as exc:
            raise click.ClickException(f"코퍼스를 불러오지 못했습니다: {exc}") from exc
        if len(df) > limit:
            df = df.iloc[:limit].copy()
        tmp_dir = Path(args.cache) / "tmp_index"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            try:
                tmp_dir = Path(
                    tempfile.mkdtemp(prefix="tmp_index_", dir=str(Path(args.cache).parent))
                )
            except Exception:
                tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_index_"))
        tmp_corpus = tmp_dir / f"corpus_limit_{limit}.parquet"
        engine = PARQUET_ENGINE or "pyarrow"
        df.to_parquet(tmp_corpus, engine=engine, index=False)
        corpus_path = tmp_corpus
        click.echo(f"⚡ 상위 {limit:,}개 문서로 제한하여 인덱싱합니다. ({corpus_path})")

    cfg = DocumentAgentConfig(
        model_path=Path(args.model),
        corpus_path=corpus_path,
        cache_dir=Path(args.cache),
        translate=getattr(args, "translate", False),
        rerank=False,
        policy_engine=policy_engine,
        policy_scope=scope,
        policy_agent=KNOWLEDGE_AGENT,
        rebuild_index=True,
    )
    agent = DocumentAgent(cfg)
    agent.prepare()
    cache_usage = _dir_size_bytes(cfg.cache_dir)
    print(f"✅ 인덱스/캐시 갱신 완료 (cache ~{cache_usage:,} bytes)")
    return {
        "cache": str(cfg.cache_dir),
        "corpus": str(cfg.corpus_path),
        "cache_usage_bytes": cache_usage,
    }


def cmd_chat(args):
    """대화형 검색 모드 (LNPChat 사용)"""
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="chat")
    def _env_or_arg(name: str, default: Optional[str] = None) -> Optional[str]:
        value = getattr(args, name, None)
        if value:
            value = str(value).strip()
            if value:
                return value
        env_name = f"LNPCHAT_{name.upper()}"
        env_value = os.getenv(env_name)
        if env_value is None:
            return default
        env_value = env_value.strip()
        return env_value or default

    llm_backend = _env_or_arg("llm_backend")
    llm_model = _env_or_arg("llm_model", default="llama3")
    llm_host = _env_or_arg("llm_host", default="")
    auto_trained = _ensure_chat_artifacts(
        scan_csv=Path(args.scan_csv),
        corpus=Path(args.corpus),
        model=Path(args.model),
        translate=args.translate,
        auto_train=args.auto_train,
        policy_engine=policy_engine,
    )

    document_agent = DocumentAgent(
        DocumentAgentConfig(
            model_path=Path(args.model),
            corpus_path=Path(args.corpus),
            cache_dir=Path(args.cache),
            topk=args.topk,
            translate=args.translate,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_depth=args.rerank_depth,
            rerank_batch_size=args.rerank_batch_size,
            rerank_device=args.rerank_device or None,
            rerank_min_score=args.rerank_min_score,
            lexical_weight=args.lexical_weight,
            show_translation=args.show_translation,
            translation_lang=args.translation_lang,
            min_similarity=args.min_similarity,
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_host=llm_host,
            llm_options={},
            policy_engine=policy_engine if policy_engine and policy_engine.has_policies else policy_engine,
            policy_scope=(getattr(args, "scope", "auto") or "auto").lower(),
            policy_agent=KNOWLEDGE_AGENT,
            rebuild_index=auto_trained,
        )
    )
    meeting_agent = MeetingAgent()
    photo_agent = PhotoAgent()

    orchestrator = AssistantOrchestrator(
        [document_agent, meeting_agent, photo_agent],
        llm_client=document_agent.llm_client,
    )

    def _print_response(resp: "OrchestratorResponse") -> None:
        prefix = f"[{resp.agent}] " if resp.agent else ""
        print(prefix + resp.message)
        if resp.suggestions:
            print("\n💡 이런 질문은 어떠세요?")
            for suggestion in resp.suggestions:
                print(f"   - {suggestion}")

    def _cli_progress_handler(agent_label: str) -> Callable[[Dict[str, Any]], None]:
        def _handler(event: Dict[str, Any]) -> None:
            stage = event.get("stage")
            status = event.get("status")
            prefix = f"[{agent_label}]"
            if status == "running":
                print(f"{prefix} ▶ {stage} 시작")
            elif status == "completed":
                print(f"{prefix} ✅ {stage} 완료")
            elif status == "failed":
                error = event.get("error")
                print(f"{prefix} ❌ {stage} 실패: {error}")
            elif status == "cancelled":
                print(f"{prefix} ⛔ {stage} 취소")
        return _handler

    def _prompt_follow_up(reason: Optional[str], message: str) -> Optional[Dict[str, object]]:
        print(message)
        history = _load_agent_history()
        if reason == "needs_audio":
            recent = history.get("meeting_audio", [])
            if recent:
                print("\n📁 최근 사용한 오디오 파일:")
                for idx, item in enumerate(recent, start=1):
                    print(f"  {idx}. {item}")
            prompt = "회의 요약을 실행하려면 오디오 파일 전체 경로를 입력하거나 번호를 선택하세요> "
            raw = input(prompt).strip()
            if not raw:
                print("⚠️ 경로를 입력하지 않아 요청을 취소했습니다.")
                return None
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(recent):
                    audio_path = recent[index]
                else:
                    print("⚠️ 번호가 유효하지 않아 요청을 취소했습니다.")
                    return None
            else:
                audio_path = raw
            _remember_agent_history("meeting_audio", [audio_path])
            return {"audio_path": audio_path, "enable_resume": True}
        if reason == "needs_roots":
            recent = history.get("photo_roots", [])
            if recent:
                print("\n📸 최근 사용한 사진 폴더:")
                for idx, item in enumerate(recent, start=1):
                    print(f"  {idx}. {item}")
            prompt = "사진 폴더 경로를 입력하거나 번호(여러 개는 콤마)로 선택하세요> "
            raw = input(prompt).strip()
            if not raw:
                print("⚠️ 경로를 입력하지 않아 요청을 취소했습니다.")
                return None
            roots: List[str] = []
            for token in [part.strip() for part in raw.split(",") if part.strip()]:
                if token.isdigit():
                    index = int(token) - 1
                    if 0 <= index < len(recent):
                        roots.append(recent[index])
                    else:
                        print(f"⚠️ 번호 {token}가 유효하지 않아 무시합니다.")
                else:
                    roots.append(token)
            if not roots:
                print("⚠️ 유효한 경로가 없어 요청을 취소했습니다.")
                return None
            _remember_agent_history("photo_roots", roots)
            return {"roots": roots}
        extra = input("추가 정보를 입력하세요> ").strip()
        if not extra:
            print("⚠️ 추가 정보를 입력하지 않아 요청을 취소했습니다.")
            return None
        return {"details": extra}

    def _resolve_follow_up(original_query: str, initial_response: "OrchestratorResponse") -> "OrchestratorResponse":
        response = initial_response
        while response.agent == "follow_up":
            follow_context = _prompt_follow_up(response.reason, response.message)
            if not follow_context:
                break
            if response.reason == "needs_audio":
                follow_context.setdefault("__progress_callback", _cli_progress_handler("회의 비서"))
            elif response.reason == "needs_roots":
                follow_context.setdefault("__progress_callback", _cli_progress_handler("사진 비서"))
            response = orchestrator.handle(original_query, follow_context)
        return response

    single_query = getattr(args, "query", None)
    json_mode = bool(getattr(args, "json", False))
    if json_mode and not single_query:
        raise SystemExit("--json 옵션은 --query와 함께 사용해야 합니다.")

    if single_query:
        response = orchestrator.handle(single_query)
        if response.agent == "follow_up" and not json_mode:
            response = _resolve_follow_up(single_query, response)
        if json_mode:
            metadata = response.metadata if isinstance(response.metadata, dict) else {}
            payload = {
                "query": single_query,
                "answer": response.message,
                "agent": response.agent,
                "reason": response.reason,
                "metadata": metadata,
                "suggestions": response.suggestions or [],
                "results": [],
            }
            if response.agent == "follow_up":
                metadata["follow_up"] = response.reason
            hits = response.metadata.get("hits", []) if isinstance(response.metadata, dict) else []
            for hit in hits[: args.topk]:
                payload["results"].append(
                    {
                        "title": Path(str(hit.get("path") or "")).name,
                        "path": hit.get("path"),
                        "ext": hit.get("ext"),
                        "score": hit.get("similarity", hit.get("vector_similarity")),
                        "vector_score": hit.get("vector_similarity"),
                        "lexical_score": hit.get("lexical_score"),
                        "match_reasons": hit.get("match_reasons") or [],
                        "preview": hit.get("preview"),
                        "translation": hit.get("translation"),
                    }
                )
            print(json.dumps(payload, ensure_ascii=False))
        else:
            _print_response(response)
        return

    print("\n💬 InfoPilot Chat 모드입니다. 자유롭게 대화하고, 문서 검색이 필요하면 '/search 질문'처럼 입력해 보세요. (종료하려면 'exit' 또는 '종료' 입력)")
    print("   명령어: /search <질문>, /meeting, /photo")
    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "종료"}:
            print("👋 종료합니다.")
            break

        response = orchestrator.handle(query)
        response = _resolve_follow_up(query, response)
        _print_response(response)
        print("-" * 80)


def cmd_watch(args):
    if Observer is None:
        raise click.ClickException("watchdog 라이브러리가 필요합니다. pip install watchdog")

    encoder, batch_size, model_name = _load_sentence_encoder(Path(args.model))
    if encoder is None:
        raise RuntimeError("sentence-transformers 모델을 로드할 수 없어 watcher를 실행할 수 없습니다.")

    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="watch")
    roots = _parse_roots(args.roots)
    if not roots and policy_engine and policy_engine.has_policies:
        roots = policy_engine.roots_for_agent(KNOWLEDGE_AGENT, include_manual=False)
    if not roots:
        raise click.ClickException(
            "스마트 폴더 정책이나 감시 루트가 지정되지 않아 watcher를 시작할 수 없습니다. "
            "정책 파일을 지정하거나 --policy none 과 함께 --root를 명시하세요."
        )

    deduped_roots: List[Path] = []
    seen_roots: Set[str] = set()
    for root in roots:
        resolved = Path(root).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        key = str(resolved)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(resolved)

    roots = deduped_roots
    existing_roots: List[Path] = []
    for root in roots:
        if root.exists():
            existing_roots.append(root)
        else:
            print(f"⚠️ 감시 루트가 존재하지 않아 제외합니다: {root}")
    if not existing_roots:
        raise click.ClickException("유효한 감시 루트가 없습니다. 경로를 다시 확인하세요.")
    roots = existing_roots

    event_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    allowed_exts = {ext.lower() for ext in FileFinder.DEFAULT_EXTS}
    handler = WatchEventHandler(event_queue, allowed_exts, policy_engine=policy_engine, agent=KNOWLEDGE_AGENT)
    observer = Observer()
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)

    pipeline_ctx = IncrementalPipeline(
        encoder=encoder,
        batch_size=batch_size,
        scan_csv=Path(args.scan_csv),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        translate=args.translate,
        policy_engine=policy_engine,
    )

    debounce_sec = max(0.5, args.debounce_ms / 1000.0)
    stop_event = threading.Event()

    policy_info = " (정책 기반)" if policy_engine and policy_engine.has_policies else ""
    print(
        "👀 파일 변경 감시를 시작합니다. (Ctrl+C로 종료)"
        f"\n   roots{policy_info}: {', '.join(str(r) for r in roots)}"
        f"\n   embedding model: {model_name} (batch={batch_size})"
        f"\n   debounce: {debounce_sec:.2f}s"
    )

    observer.start()
    try:
        _watch_loop(event_queue, pipeline_ctx, stop_event, debounce_sec)
    except KeyboardInterrupt:
        print("\n👋 watcher를 종료합니다.")
    finally:
        stop_event.set()
        observer.stop()
        observer.join()
        try:
            _get_sentence_encoder_manager().release(model_name)
        except Exception:
            pass


def cmd_schedule(args):
    policy_engine = _load_policy_engine(getattr(args, "policy", None), fail_if_missing=True, stage="schedule")
    if not policy_engine or not policy_engine.has_policies:
        print("⚠️ 스케줄러: 정책이 없어 종료합니다.")
        return

    if args.agent != KNOWLEDGE_AGENT:
        print("⚠️ 스케줄러: 현재는 knowledge_search 에이전트 예약만 지원합니다.")
        return

    scheduler = JobScheduler()
    jobs = _register_policy_jobs(
        scheduler,
        policy_engine=policy_engine,
        agent=args.agent,
        output_root=Path(args.output_root),
        translate=args.translate,
    )

    if not jobs:
        print("⚠️ 스케줄러: 예약 작업이 없습니다. 정책의 indexing.mode를 확인해주세요.")
        return

    for job in jobs:
        next_run = job.next_run.isoformat() if job.next_run else "manual"
        print(f"⏱️ {job.metadata.get('path', job.name)} → 다음 실행: {next_run}")

    poll = max(5.0, float(getattr(args, "poll_seconds", 60.0)))
    if getattr(args, "once", False):
        scheduler.run_pending()
        return

    print("🚀 정책 스케줄러를 시작합니다. (Ctrl+C로 종료)")
    try:
        while True:
            scheduler.run_pending()
            time.sleep(poll)
    except KeyboardInterrupt:
        print("👋 스케줄러를 종료합니다.")

# ---------------------------------------------------------------------------
# Click 기반 CLI
# ---------------------------------------------------------------------------


def _scan_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--ext",
        "exts",
        multiple=True,
        help="스캔할 확장자를 지정합니다 (예: --ext pdf --ext docx). 지정하지 않으면 기본 확장자를 사용합니다.",
    )(func)
    func = click.option(
        "--root",
        "--roots",
        "roots",
        multiple=True,
        type=click.Path(file_okay=False, dir_okay=True, path_type=str),
        help="스캔/감시할 루트 경로 (여러 번 지정 가능).",
    )(func)
    func = click.option(
        "--out",
        default=str(DEFAULT_FOUND_FILES),
        show_default=True,
        type=click.Path(dir_okay=False, path_type=str),
        help="스캔 결과 CSV 경로.",
    )(func)
    return func


def _train_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--state-file",
        default=str(DEFAULT_SCAN_STATE),
        show_default=True,
        type=click.Path(dir_okay=False, path_type=str),
        help="증분 학습을 위한 스캔 상태 파일",
    )(func)
    func = click.option(
        "--chunk-cache",
        default=str(DEFAULT_CHUNK_CACHE),
        show_default=True,
        type=click.Path(dir_okay=False, path_type=str),
        help="문서 해시 캐시 경로",
    )(func)
    func = click.option(
        "--use-embedding/--no-embedding",
        default=True,
        show_default=True,
        help="Sentence-BERT 임베딩 사용 여부.",
    )(func)
    func = click.option(
        "--translate/--no-translate",
        default=False,
        show_default=True,
        help="문서 학습 시 번역 파이프라인 사용 여부.",
    )(func)
    func = click.option(
        "--limit-files",
        "--limit",
        "limit_files",
        type=int,
        default=0,
        show_default=True,
        help="테스트용 상위 N개 파일만 사용 (0=전체).",
    )(func)
    func = click.option("--embedding-batch-size", type=int, default=32, show_default=True)(func)
    func = click.option("--embedding-concurrency", type=int, default=1, show_default=True, help="Sentence-BERT 임베딩 비동기 작업자 수")(func)
    func = click.option("--async-embed/--no-async-embed", default=True, show_default=True, help="임베딩 비동기 큐 사용 여부")(func)
    func = click.option("--embedding-chunk-size", type=int, default=0, show_default=True, help="Sentence-BERT 임베딩 청크 크기(0이면 전체 한 번에)")(func)
    func = click.option("--embedding-chunk-start", type=int, default=0, show_default=True, help="청크 임베딩 시작 인덱스")(func)
    func = click.option("--embedding-chunk-end", type=int, default=-1, show_default=True, help="청크 임베딩 끝 인덱스(-1이면 끝까지)")(func)
    func = click.option(
        "--embedding-subprocess-fallback/--no-embedding-subprocess-fallback",
        default=True,
        show_default=True,
        help="청크 임베딩 실패 시 CPU로 재시도(subprocess).",
    )(func)
    func = click.option(
        "--skip-extract",
        is_flag=True,
        help="기존 corpus/parquet을 그대로 사용하고 추출 단계를 건너뜁니다. (증분/스캔 상태를 업데이트하지 않음)",
    )(func)
    func = click.option(
        "--embedding-dtype",
        type=click.Choice(["auto", "fp32", "fp16"], case_sensitive=False),
        default="auto",
        show_default=True,
        help="Sentence-BERT 임베딩 dtype (auto=GPU면 FP16 사용).",
    )(func)
    func = click.option("--embedding-model", default=DEFAULT_EMBED_MODEL, show_default=True)(func)
    func = click.option("--max-df", type=float, default=0.85, show_default=True)(func)
    func = click.option("--min-df", type=int, default=2, show_default=True)(func)
    func = click.option("--n-clusters", type=int, default=25, show_default=True)(func)
    func = click.option("--n-components", type=int, default=DEFAULT_N_COMPONENTS, show_default=True)(func)
    func = click.option("--max-features", type=int, default=50000, show_default=True)(func)
    func = click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    func = click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    func = click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    return func


def _pipeline_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--state-file",
        default=str(DEFAULT_SCAN_STATE),
        show_default=True,
        type=click.Path(dir_okay=False, path_type=str),
        help="증분 학습 상태 파일",
    )(func)
    func = click.option(
        "--chunk-cache",
        default="",
        type=click.Path(dir_okay=False, path_type=str),
        help="문서 해시 캐시 경로 (기본: cache 디렉터리 내 chunk_cache.json)",
    )(func)
    func = click.option(
        "--launch-chat/--no-launch-chat",
        default=False,
        show_default=True,
        help="파이프라인 완료 직후 chat 모드 실행 여부.",
    )(func)
    func = click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--use-embedding/--no-embedding", default=True, show_default=True)(func)
    func = click.option("--translate/--no-translate", default=False, show_default=True)(func)
    func = click.option(
        "--limit-files",
        "--limit",
        "limit_files",
        type=int,
        default=0,
        show_default=True,
        help="테스트용 상위 N개만 사용.",
    )(func)
    func = click.option("--embedding-batch-size", type=int, default=32, show_default=True)(func)
    func = click.option("--embedding-concurrency", type=int, default=1, show_default=True)(func)
    func = click.option("--async-embed/--no-async-embed", default=True, show_default=True)(func)
    func = click.option("--embedding-chunk-size", type=int, default=0, show_default=True, help="Sentence-BERT 임베딩 청크 크기(0이면 전체 한 번에)")(func)
    func = click.option("--embedding-chunk-start", type=int, default=0, show_default=True, help="청크 임베딩 시작 인덱스")(func)
    func = click.option("--embedding-chunk-end", type=int, default=-1, show_default=True, help="청크 임베딩 끝 인덱스(-1이면 끝까지)")(func)
    func = click.option(
        "--embedding-subprocess-fallback/--no-embedding-subprocess-fallback",
        default=True,
        show_default=True,
        help="청크 임베딩 실패 시 CPU로 재시도(subprocess).",
    )(func)
    func = click.option(
        "--embedding-dtype",
        type=click.Choice(["auto", "fp32", "fp16"], case_sensitive=False),
        default="auto",
        show_default=True,
        help="Sentence-BERT 임베딩 dtype (auto=GPU면 FP16 사용).",
    )(func)
    func = click.option("--embedding-model", default=DEFAULT_EMBED_MODEL, show_default=True)(func)
    func = click.option("--max-df", type=float, default=0.85, show_default=True)(func)
    func = click.option("--min-df", type=int, default=2, show_default=True)(func)
    func = click.option("--n-clusters", type=int, default=25, show_default=True)(func)
    func = click.option("--n-components", type=int, default=DEFAULT_N_COMPONENTS, show_default=True)(func)
    func = click.option("--max-features", type=int, default=50000, show_default=True)(func)
    func = click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    func = click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    func = click.option("--out", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(dir_okay=False, path_type=str))(func)
    func = click.option(
        "--ext",
        "exts",
        multiple=True,
        help="파이프라인 스캔 단계에서 사용할 확장자 (예: --ext pdf --ext docx).",
    )(func)
    func = click.option(
        "--root",
        "--roots",
        "roots",
        multiple=True,
        type=click.Path(file_okay=False, dir_okay=True, path_type=str),
        help="스캔할 루트 (여러 번 지정 가능).",
    )(func)
    return func


def _index_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--limit-files",
        type=int,
        default=0,
        show_default=True,
        help="테스트용 상위 N개 문서만 인덱싱 (0=전체).",
    )(func)
    func = click.option(
        "--scope",
        type=click.Choice(["auto", "policy", "global"]),
        default="auto",
        show_default=True,
        help="검색 범위 (auto=정책 자동, policy=정책 고정, global=전체).",
    )(func)
    func = click.option("--translate/--no-translate", default=False, show_default=True)(func)
    func = click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(path_type=str))(func)
    return func


def _chat_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--scope",
        type=click.Choice(["auto", "policy", "global"]),
        default="auto",
        show_default=True,
    )(func)
    func = click.option("--json/--no-json", "json_mode", default=False, show_default=True, help="결과를 JSON으로 출력 후 종료")(func)
    func = click.option("--query", help="비대화형 단일 질의")(func)
    func = click.option("--auto-train/--no-auto-train", default=True, show_default=True, help="scan CSV 최신 시 자동 학습")(func)
    func = click.option("--translate/--no-translate", default=False, show_default=True)(func)
    func = click.option("--topk", type=int, default=5, show_default=True)(func)
    func = click.option("--min-similarity", type=float, default=0.35, show_default=True)(func)
    func = click.option("--lexical-weight", type=float, default=0.35, show_default=True)(func)
    func = click.option("--rerank/--no-rerank", default=True, show_default=True)(func)
    func = click.option("--rerank-model", default="BAAI/bge-reranker-large", show_default=True)(func)
    func = click.option("--rerank-depth", type=int, default=80, show_default=True)(func)
    func = click.option("--rerank-batch-size", type=int, default=16, show_default=True)(func)
    func = click.option("--rerank-device", default=None)(func)
    func = click.option("--rerank-min-score", type=float, default=0.35, show_default=True)(func)
    func = click.option("--show-translation/--hide-translation", default=False, show_default=True)(func)
    func = click.option("--translation-lang", default="en", show_default=True)(func)
    func = click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))(func)
    return func


def _watch_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option("--translate/--no-translate", default=False, show_default=True)(func)
    func = click.option("--debounce-ms", type=int, default=2000, show_default=True)(func)
    func = click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option(
        "--root",
        "--roots",
        "roots",
        multiple=True,
        type=click.Path(file_okay=False, dir_okay=True, path_type=str),
        help="감시할 루트 경로 (여러 번 지정 가능).",
    )(func)
    return func


def _schedule_options(func):
    func = click.option(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        show_default=True,
        help="스마트 폴더 정책 경로 (비활성화하려면 'none').",
    )(func)
    func = click.option(
        "--agent",
        type=click.Choice(["knowledge_search", "meeting", "photo"]),
        default=KNOWLEDGE_AGENT,
        show_default=True,
        help="예약 실행할 에이전트.",
    )(func)
    func = click.option("--output-root", default=str(DEFAULT_SCHEDULED_ROOT), show_default=True, type=click.Path(path_type=str))(func)
    func = click.option("--translate/--no-translate", default=False, show_default=True)(func)
    func = click.option("--once", is_flag=True, help="예약 없이 즉시 실행 후 종료")(func)
    func = click.option("--poll-seconds", type=float, default=60.0, show_default=True)(func)
    return func


@click.group(
    help="🧠 AI-summary CLI — 로컬 문서 수집·임베딩·검색 파이프라인",
    invoke_without_command=False,
)
@click.option("--mlflow/--no-mlflow", default=True, show_default=True, help="MLflow 로깅 사용 여부.")
@click.option("--mlflow-uri", default=DEFAULT_TRACKING_URI, show_default=True, help="MLflow Tracking URI.")
@click.option("--mlflow-experiment", default=DEFAULT_EXPERIMENT, show_default=True)
@click.option(
    "--resource-log-path",
    default=str(DEFAULT_RESOURCE_LOG),
    show_default=True,
    type=click.Path(path_type=str),
    help="psutil 리소스 로그 경로.",
)
@click.option("--resource-interval", default=30.0, show_default=True, help="리소스 로그 주기(초).")
@click.option("--no-resource-log", is_flag=True, help="리소스 로깅 비활성화")
@click.pass_context
def cli(
    ctx: click.Context,
    mlflow: bool,
    mlflow_uri: str,
    mlflow_experiment: str,
    resource_log_path: str,
    resource_interval: float,
    no_resource_log: bool,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "use_mlflow": mlflow,
            "mlflow_uri": mlflow_uri,
            "mlflow_experiment": mlflow_experiment,
            "resource_log_path": None if no_resource_log else Path(resource_log_path),
            "resource_interval": resource_interval,
        }
    )


@cli.group("run", help="핵심 파이프라인 단계(run scan/train/index/chat/watch).")
@click.pass_context
def run(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


@cli.group("logs", help="MLflow/psutil 로그 확인 및 정리.")
def logs() -> None:
    pass


@cli.group("model", help="모델 목록 조회 및 ONNX 양자화.")
def model_group() -> None:
    pass


@cli.group("drift", help="드리프트 점검 및 재임베딩.")
def drift_group() -> None:
    pass


def _perform_drift_check(
    ctx: click.Context,
    *,
    run_name: str,
    scan_csv: str,
    corpus: str,
    cache_dir: str,
    semantic_baseline: str,
    semantic_threshold: float,
    log_path: str,
    alert_threshold: float,
):
    _require_pandas()
    cache_path = Path(cache_dir)
    baseline_path = Path(semantic_baseline) if semantic_baseline else None
    with _command_session(ctx, run_name) as session:
        report = check_drift(
            Path(scan_csv),
            Path(corpus),
            cache_dir=cache_path,
            log_path=Path(log_path),
            alert_threshold=alert_threshold,
            semantic_baseline=baseline_path,
            semantic_threshold=semantic_threshold,
        )
        if session:
            session.log_metrics(
                {
                    "hash_drift_ratio": report.hash_drift_ratio,
                    "semantic_shift": report.semantic_shift,
                    "new_files": float(len(report.new_files)),
                    "changed_files": float(len(report.changed_files)),
                    "missing_files": float(len(report.missing_files)),
                }
            )
    return report


def _print_drift_report(report, semantic_threshold: float) -> None:
    click.echo(f"📈 hash drift ratio={report.hash_drift_ratio:.3f} (scan={report.scan_rows}, corpus={report.corpus_rows})")
    if report.new_files:
        click.echo(f"➕ 신규 문서 {len(report.new_files)}건 (상위 5개):")
        for path in report.new_files[:5]:
            click.echo(f"   + {path}")
    if report.changed_files:
        click.echo(f"🌀 변경 감지 {len(report.changed_files)}건 (상위 5개):")
        for path in report.changed_files[:5]:
            click.echo(f"   * {path}")
    if report.missing_files:
        click.echo(f"➖ 누락 문서 {len(report.missing_files)}건 (상위 5개):")
        for path in report.missing_files[:5]:
            click.echo(f"   - {path}")
    click.echo(
        f"🎯 semantic shift={report.semantic_shift:.3f} (threshold={semantic_threshold:.2f}, sample={report.semantic_sample_size})"
    )
    if report.reembed_candidates:
        click.echo(f"🔁 재임베딩 후보 {len(report.reembed_candidates)}건 (로그에 기록)")
    if report.recommendations:
        click.echo(f"✅ 권장 조치: {', '.join(report.recommendations)}")


def _auto_reembed_targets(report, *, max_candidates: int, include_changed: bool, include_new: bool) -> Set[str]:
    ordered: List[str] = []
    ordered.extend(report.reembed_candidates or [])
    if include_changed:
        ordered.extend(report.changed_files or [])
    if include_new:
        ordered.extend(report.new_files or [])
    deduped: List[str] = []
    seen: Set[str] = set()
    for path in ordered:
        path = str(path or "")
        if not path or path in seen:
            continue
        deduped.append(path)
        seen.add(path)
        if max_candidates and len(deduped) >= max_candidates:
            break
    return set(deduped)


def _run_reembed_pipeline(
    ctx: click.Context,
    paths: Set[str],
    *,
    scan_csv: str,
    corpus: str,
    cache: str,
    model: str,
    translate: bool,
    policy: str,
    run_name: str,
) -> None:
    encoder, batch_size, model_name = _load_sentence_encoder(Path(model))
    if encoder is None:
        raise click.ClickException("SentenceTransformer 모델 로드 실패로 재임베딩을 진행할 수 없습니다.")
    policy_normalized = (policy or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy, fail_if_missing=policy_required, stage="reembed")
    pipeline_ctx = IncrementalPipeline(
        encoder=encoder,
        batch_size=batch_size,
        scan_csv=Path(scan_csv),
        corpus_path=Path(corpus),
        cache_dir=Path(cache),
        translate=translate,
        policy_engine=policy_engine,
    )
    with _command_session(ctx, run_name) as session:
        pipeline_ctx.process(set(paths), set())
        if session:
            session.log_metrics({"reembedded": float(len(paths))})
    click.echo(f"🔁 재임베딩 완료 ({len(paths)}건)")
    try:
        _get_sentence_encoder_manager().release(model_name)
    except Exception:
        pass


@click.command("embed-chunk", help="내부용: 텍스트 리스트(JSON)를 임베딩해 npy로 저장합니다.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=str))
@click.option("--model", default=DEFAULT_EMBED_MODEL, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--concurrency", type=int, default=1, show_default=True)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "fp32", "fp16"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.option("--async/--no-async", "async_embed", default=True, show_default=True, help="비동기 임베딩 사용 여부")
def embed_chunk_command(
    input_path: str,
    output_path: str,
    model: str,
    batch_size: int,
    concurrency: int,
    dtype: str,
    async_embed: bool,
) -> None:
    if SentenceTransformer is None:
        raise click.ClickException("sentence-transformers 패키지가 필요합니다. pip install sentence-transformers")
    try:
        import numpy as np
    except Exception:
        raise click.ClickException("numpy 패키지가 필요합니다. pip install numpy")

    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"입력 JSON 로드 실패: {exc}")
    if not isinstance(payload, list):
        raise click.ClickException("입력 파일은 텍스트 리스트(JSON 배열)여야 합니다.")
    texts = [str(item or "") for item in payload]

    cfg = TrainConfig(
        embedding_model=model,
        embedding_batch_size=batch_size,
        embedding_concurrency=max(1, int(concurrency)),
        embedding_dtype=dtype or "auto",
        async_embeddings=async_embed,
        use_sentence_transformer=True,
        n_clusters=0,
    )
    semantic_model = SentenceBertModel(cfg)
    embeddings = semantic_model.encode(texts, show_progress=False)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings.astype(np.float32, copy=False))
    click.echo(f"✅ chunk 임베딩 완료 (docs={len(texts):,}) → {out_path}")


@click.command("scan")
@_scan_options
@click.pass_context
def scan_command(ctx: click.Context, out: str, roots: Tuple[str, ...], exts: Tuple[str, ...], policy: str) -> None:
    _require_pandas()
    args = SimpleNamespace(out=out, roots=list(roots) if roots else None, policy=policy, exts=list(exts) if exts else None)
    with _command_session(ctx, "scan") as session:
        count = cmd_scan(args) or 0
        if session:
            session.log_params({"policy": policy})
            session.log_metrics({"files": float(count)})
    click.echo(f"📦 스캔 완료: {count}건 기록 ({out})")


@click.command("extract")
@_train_options
@click.pass_context
def extract_command(
    ctx: click.Context,
    scan_csv: str,
    corpus: str,
    model: str,
    max_features: int,
    n_components: int,
    n_clusters: int,
    min_df: int,
    max_df: float,
    embedding_model: str,
    embedding_batch_size: int,
    limit_files: int,
    translate: bool,
    use_embedding: bool,
    policy: str,
    state_file: str,
    chunk_cache: str,
    embedding_concurrency: int,
    async_embed: bool,
    embedding_dtype: str,
    embedding_chunk_size: int,
    embedding_chunk_start: int,
    embedding_chunk_end: int,
    embedding_subprocess_fallback: bool,
    skip_extract: bool,
) -> None:
    _require_pandas()
    args = SimpleNamespace(
        scan_csv=scan_csv,
        corpus=corpus,
        model=model,
        max_features=max_features,
        n_components=n_components,
        n_clusters=n_clusters,
        min_df=min_df,
        max_df=max_df,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        limit_files=limit_files,
        translate=translate,
        use_embedding=use_embedding,
        policy=policy,
        state_file=state_file,
        chunk_cache=chunk_cache,
        embedding_concurrency=embedding_concurrency,
        async_embed=async_embed,
        embedding_dtype=embedding_dtype,
        embedding_chunk_size=embedding_chunk_size,
        embedding_chunk_start=embedding_chunk_start,
        embedding_chunk_end=embedding_chunk_end,
        embedding_subprocess_fallback=embedding_subprocess_fallback,
        skip_extract=False,
    )
    with _command_session(ctx, "extract") as session:
        stats = cmd_extract(args) or {}
        if session and stats:
            session.log_params({"corpus": stats.get("corpus"), "policy": policy})
            session.log_metrics({"rows": float(stats.get("rows", 0))})
    click.echo(f"📦 추출 완료 → corpus={corpus}")


@click.command("train")
@_train_options
@click.pass_context
def train_command(
    ctx: click.Context,
    scan_csv: str,
    corpus: str,
    model: str,
    max_features: int,
    n_components: int,
    n_clusters: int,
    min_df: int,
    max_df: float,
    embedding_model: str,
    embedding_batch_size: int,
    limit_files: int,
    translate: bool,
    use_embedding: bool,
    policy: str,
    state_file: str,
    chunk_cache: str,
    embedding_concurrency: int,
    async_embed: bool,
    embedding_dtype: str,
    embedding_chunk_size: int,
    embedding_chunk_start: int,
    embedding_chunk_end: int,
    embedding_subprocess_fallback: bool,
    skip_extract: bool,
) -> None:
    _require_pandas()
    args = SimpleNamespace(
        scan_csv=scan_csv,
        corpus=corpus,
        model=model,
        max_features=max_features,
        n_components=n_components,
        n_clusters=n_clusters,
        min_df=min_df,
        max_df=max_df,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        limit_files=limit_files,
        translate=translate,
        use_embedding=use_embedding,
        policy=policy,
        state_file=state_file,
        chunk_cache=chunk_cache,
        embedding_concurrency=embedding_concurrency,
        async_embed=async_embed,
        embedding_dtype=embedding_dtype,
        embedding_chunk_size=embedding_chunk_size,
        embedding_chunk_start=embedding_chunk_start,
        embedding_chunk_end=embedding_chunk_end,
        embedding_subprocess_fallback=embedding_subprocess_fallback,
        skip_extract=skip_extract,
    )
    with _command_session(ctx, "train") as session:
        stats = cmd_train(args) or {}
        if session and stats:
            session.log_params(
                {
                    "corpus": stats.get("corpus"),
                    "model": stats.get("model"),
                    "embedding_model": embedding_model,
                    "use_embedding": str(use_embedding),
                }
            )
            session.log_metrics({"rows": float(stats.get("rows", 0))})
            extra_metrics = stats.get("metrics") or {}
            if extra_metrics:
                session.log_metrics(extra_metrics)
    click.echo(f"🧠 학습 완료 → corpus={corpus}")


@click.command("embed")
@_train_options
@click.pass_context
def embed_command(
    ctx: click.Context,
    scan_csv: str,
    corpus: str,
    model: str,
    max_features: int,
    n_components: int,
    n_clusters: int,
    min_df: int,
    max_df: float,
    embedding_model: str,
    embedding_batch_size: int,
    limit_files: int,
    translate: bool,
    use_embedding: bool,
    policy: str,
    state_file: str,
    chunk_cache: str,
    embedding_concurrency: int,
    async_embed: bool,
    embedding_dtype: str,
    embedding_chunk_size: int,
    embedding_chunk_start: int,
    embedding_chunk_end: int,
    embedding_subprocess_fallback: bool,
    skip_extract: bool,
) -> None:
    _require_pandas()
    args = SimpleNamespace(
        scan_csv=scan_csv,
        corpus=corpus,
        model=model,
        max_features=max_features,
        n_components=n_components,
        n_clusters=n_clusters,
        min_df=min_df,
        max_df=max_df,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        limit_files=limit_files,
        translate=translate,
        use_embedding=use_embedding,
        policy=policy,
        state_file=state_file,
        chunk_cache=chunk_cache,
        embedding_concurrency=embedding_concurrency,
        async_embed=async_embed,
        embedding_dtype=embedding_dtype,
        embedding_chunk_size=embedding_chunk_size,
        embedding_chunk_start=embedding_chunk_start,
        embedding_chunk_end=embedding_chunk_end,
        embedding_subprocess_fallback=embedding_subprocess_fallback,
        skip_extract=True,
    )
    with _command_session(ctx, "embed") as session:
        stats = cmd_embed(args) or {}
        if session and stats:
            session.log_params(
                {
                    "corpus": stats.get("corpus"),
                    "model": stats.get("model"),
                    "embedding_model": embedding_model,
                    "policy": policy,
                }
            )
            session.log_metrics({"rows": float(stats.get("rows", 0))})
            extra_metrics = stats.get("metrics") or {}
            if extra_metrics:
                session.log_metrics(extra_metrics)
    click.echo(f"🧠 임베딩/모델 완료 → corpus={corpus}")


@click.command("pipeline")
@_pipeline_options
@click.argument("target", required=False, default="all")
@click.pass_context
def pipeline_command(
    ctx: click.Context,
    target: str,
    out: str,
    roots: Tuple[str, ...],
    exts: Tuple[str, ...],
    corpus: str,
    model: str,
    cache: str,
    max_features: int,
    n_components: int,
    n_clusters: int,
    min_df: int,
    max_df: float,
    embedding_model: str,
    embedding_batch_size: int,
    limit_files: int,
    translate: bool,
    use_embedding: bool,
    launch_chat: bool,
    policy: str,
    state_file: str,
    chunk_cache: str,
    embedding_concurrency: int,
    async_embed: bool,
    embedding_dtype: str,
    embedding_chunk_size: int,
    embedding_chunk_start: int,
    embedding_chunk_end: int,
    embedding_subprocess_fallback: bool,
) -> None:
    _require_pandas()
    normalized = (target or "all").strip().lower()
    if normalized not in {"", "all"}:
        raise click.UsageError("지원하지 않는 파이프라인 타겟입니다 (all만 지원).")
    resolved_chunk_cache = chunk_cache or str(Path(cache) / "chunk_cache.json")
    args = SimpleNamespace(
        out=out,
        roots=list(roots) if roots else None,
        exts=list(exts) if exts else None,
        corpus=corpus,
        model=model,
        cache=cache,
        max_features=max_features,
        n_components=n_components,
        n_clusters=n_clusters,
        min_df=min_df,
        max_df=max_df,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        limit_files=limit_files,
        translate=translate,
        use_embedding=use_embedding,
        launch_chat=launch_chat,
        policy=policy,
        state_file=state_file,
        chunk_cache=resolved_chunk_cache,
        embedding_concurrency=embedding_concurrency,
        async_embed=async_embed,
        embedding_dtype=embedding_dtype,
        embedding_chunk_size=embedding_chunk_size,
        embedding_chunk_start=embedding_chunk_start,
        embedding_chunk_end=embedding_chunk_end,
        embedding_subprocess_fallback=embedding_subprocess_fallback,
    )
    with _command_session(ctx, "pipeline") as session:
        stats = cmd_pipeline(args) or {}
        if session and stats:
            session.log_params(
                {
                    "corpus": stats.get("corpus"),
                    "model": stats.get("model"),
                    "cache": stats.get("cache"),
                    "launch_chat": str(launch_chat),
                }
            )
            session.log_metrics({"rows": float(stats.get("rows", 0))})
            extra_metrics = stats.get("metrics") or {}
            if extra_metrics:
                session.log_metrics(extra_metrics)
    click.echo("🚀 pipeline all 완료")


@click.command("index")
@_index_options
@click.pass_context
def index_command(
    ctx: click.Context,
    model: str,
    corpus: str,
    cache: str,
    translate: bool,
    scope: str,
    policy: str,
    limit_files: int,
) -> None:
    _require_pandas()
    args = SimpleNamespace(
        model=model,
        corpus=corpus,
        cache=cache,
        translate=translate,
        scope=scope,
        policy=policy,
        limit_files=limit_files,
    )
    with _command_session(ctx, "index") as session:
        stats = cmd_index(args) or {}
        if session and stats:
            session.log_params({"cache": stats.get("cache"), "scope": scope})
    click.echo(f"🧱 인덱스 캐시 갱신 완료 → {cache}")


@click.command("chat")
@_chat_options
@click.pass_context
def chat_command(
    ctx: click.Context,
    model: str,
    corpus: str,
    cache: str,
    scan_csv: str,
    topk: int,
    translate: bool,
    auto_train: bool,
    rerank: bool,
    rerank_model: str,
    rerank_depth: int,
    rerank_batch_size: int,
    rerank_device: Optional[str],
    rerank_min_score: float,
    lexical_weight: float,
    show_translation: bool,
    translation_lang: str,
    min_similarity: float,
    policy: str,
    scope: str,
    query: Optional[str],
    json_mode: bool,
) -> None:
    args = SimpleNamespace(
        model=model,
        corpus=corpus,
        cache=cache,
        scan_csv=scan_csv,
        topk=topk,
        translate=translate,
        auto_train=auto_train,
        rerank=rerank,
        rerank_model=rerank_model,
        rerank_depth=rerank_depth,
        rerank_batch_size=rerank_batch_size,
        rerank_device=rerank_device,
        rerank_min_score=rerank_min_score,
        lexical_weight=lexical_weight,
        show_translation=show_translation,
        translation_lang=translation_lang,
        min_similarity=min_similarity,
        policy=policy,
        scope=scope,
        query=query,
        json=json_mode,
    )
    with _command_session(ctx, "chat"):
        cmd_chat(args)


@click.command("watch")
@_watch_options
@click.pass_context
def watch_command(
    ctx: click.Context,
    roots: Tuple[str, ...],
    scan_csv: str,
    corpus: str,
    model: str,
    cache: str,
    debounce_ms: int,
    translate: bool,
    policy: str,
) -> None:
    args = SimpleNamespace(
        roots=list(roots) if roots else None,
        scan_csv=scan_csv,
        corpus=corpus,
        model=model,
        cache=cache,
        debounce_ms=debounce_ms,
        translate=translate,
        policy=policy,
    )
    with _command_session(ctx, "watch"):
        cmd_watch(args)


@click.command("schedule")
@_schedule_options
@click.pass_context
def schedule_command(
    ctx: click.Context,
    policy: str,
    agent: str,
    output_root: str,
    translate: bool,
    once: bool,
    poll_seconds: float,
) -> None:
    args = SimpleNamespace(
        policy=policy,
        agent=agent,
        output_root=output_root,
        translate=translate,
        once=once,
        poll_seconds=poll_seconds,
    )
    with _command_session(ctx, "schedule"):
        cmd_schedule(args)


cli.add_command(scan_command)
cli.add_command(train_command)
cli.add_command(pipeline_command)
cli.add_command(index_command)
cli.add_command(chat_command)
cli.add_command(watch_command)
cli.add_command(schedule_command)
cli.add_command(extract_command)
cli.add_command(embed_command)
run.add_command(scan_command)
run.add_command(extract_command)
run.add_command(train_command)
run.add_command(embed_command)
run.add_command(index_command)
run.add_command(chat_command)
run.add_command(watch_command)


@logs.command("show")
@click.option("--tail", type=int, default=20, show_default=True, help="표시할 최근 로그 라인 수.")
@click.option(
    "--resource-log-path",
    default=str(DEFAULT_RESOURCE_LOG),
    show_default=True,
    type=click.Path(path_type=str),
)
@click.option(
    "--drift-log-path",
    default=str(DEFAULT_DRIFT_LOG),
    show_default=True,
    type=click.Path(path_type=str),
)
def logs_show(tail: int, resource_log_path: str, drift_log_path: str) -> None:
    mlflow_dir = DEFAULT_TRACKING_URI.replace("file:", "")
    click.echo(f"📁 MLflow tracking: {mlflow_dir}")

    def _tail(path: Path) -> List[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-tail:]

    res_path = Path(resource_log_path)
    drift_path = Path(drift_log_path)
    res_lines = _tail(res_path)
    drift_lines = _tail(drift_path)
    click.echo(f"📊 Resource log ({res_path}):")
    if res_lines:
        for line in res_lines:
            click.echo(f"  {line.rstrip()}")
    else:
        click.echo("  (no entries)")
    click.echo(f"📉 Drift log ({drift_path}):")
    if drift_lines:
        for line in drift_lines:
            click.echo(f"  {line.rstrip()}")
    else:
        click.echo("  (no entries)")


@logs.command("clean")
@click.option("--resource", is_flag=True, help="리소스 로그 삭제")
@click.option("--drift", is_flag=True, help="드리프트 로그 삭제")
@click.option("--mlflow", "clean_mlflow", is_flag=True, help=".mlruns 디렉터리 비우기")
def logs_clean(resource: bool, drift: bool, clean_mlflow: bool) -> None:
    if resource:
        path = DEFAULT_RESOURCE_LOG
        if path.exists():
            path.unlink()
        click.echo(f"🧹 resource log 제거: {path}")
    if drift:
        path = DEFAULT_DRIFT_LOG
        if path.exists():
            path.unlink()
        click.echo(f"🧹 drift log 제거: {path}")
    if clean_mlflow:
        tracking_dir = Path(DEFAULT_TRACKING_URI.replace("file:", ""))
        if tracking_dir.exists():
            shutil.rmtree(tracking_dir)
        click.echo(f"🧹 MLflow 기록 제거: {tracking_dir}")


@model_group.command("list")
def model_list() -> None:
    if not MODELS_DIR.exists():
        click.echo("⚠️ MODELS_DIR가 존재하지 않습니다.")
        return
    click.echo("📚 모델 목록:")
    for item in sorted(MODELS_DIR.glob("*")):
        marker = "[DIR]" if item.is_dir() else "     "
        click.echo(f"  {marker} {item}")


@model_group.command("quantize")
@click.option("--model", "model_name", required=True, help="HuggingFace 모델 ID 또는 로컬 경로")
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=str),
    help="생성할 ONNX 경로",
)
@click.option("--seq-length", type=int, default=384, show_default=True, help="ONNX 내 최대 토큰 길이")
@click.option("--opset", type=int, default=17, show_default=True)
@click.option("--int8/--fp32", "int8", default=True, show_default=True, help="INT8 양자화 사용 여부")
def model_quantize(model_name: str, output: str, seq_length: int, opset: int, int8: bool) -> None:
    result = export_to_onnx(
        model_name,
        output_path=Path(output),
        sequence_length=seq_length,
        opset=opset,
        quantize_int8=int8,
    )
    mode = "int8" if result.quantized else "fp32"
    size_mb = result.file_size / (1024 * 1024) if result.file_size else 0
    click.echo(f"✅ ONNX {mode} 저장 완료 ({size_mb:.1f} MB) → {result.output}")


def _drift_log_candidates(log_path: Path, limit: int = 64) -> List[str]:
    if limit <= 0 or not log_path.exists():
        return []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []

    seen: Set[str] = set()
    picked: List[str] = []
    for raw in reversed(lines):
        entry = raw.strip()
        if not entry:
            continue
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError:
            continue
        for key in ("reembed_candidates", "changed_files", "new_files"):
            for path in payload.get(key, []) or []:
                normalized = str(path).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                picked.append(normalized)
                if len(picked) >= limit:
                    return picked
        if picked:
            # Stop after the most recent entry that yielded candidates.
            return picked
    return picked


@drift_group.command("check")
@click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))
@click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--cache-dir", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-baseline", default=str(DEFAULT_SEMANTIC_BASELINE), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-threshold", type=float, default=0.15, show_default=True, help="semantic drift 임계값 (cosine)")
@click.option("--log-path", default=str(DEFAULT_DRIFT_LOG), show_default=True, type=click.Path(path_type=str))
@click.option("--alert-threshold", type=float, default=0.1, show_default=True, help="hash drift 비율 알림 임계값")
@click.pass_context
def drift_check(
    ctx: click.Context,
    scan_csv: str,
    corpus: str,
    cache_dir: str,
    semantic_baseline: str,
    semantic_threshold: float,
    log_path: str,
    alert_threshold: float,
) -> None:
    report = _perform_drift_check(
        ctx,
        run_name="drift-check",
        scan_csv=scan_csv,
        corpus=corpus,
        cache_dir=cache_dir,
        semantic_baseline=semantic_baseline,
        semantic_threshold=semantic_threshold,
        log_path=log_path,
        alert_threshold=alert_threshold,
    )
    _print_drift_report(report, semantic_threshold)


@drift_group.command("reembed")
@click.option("--paths-file", type=click.Path(exists=True, path_type=str), help="재임베딩할 경로 리스트 파일")
@click.option(
    "--path",
    "paths",
    multiple=True,
    help="재임베딩할 단일 경로 (여러 번 지정 가능)",
)
@click.option("--from-drift-log", "use_drift_log", is_flag=True, help="최신 드리프트 로그에서 자동 대상 추출")
@click.option(
    "--drift-log-path",
    default=str(DEFAULT_DRIFT_LOG),
    show_default=True,
    type=click.Path(path_type=str),
    help="드리프트 체크 JSONL 경로",
)
@click.option("--max-candidates", type=int, default=64, show_default=True, help="로그에서 불러올 최대 문서 수")
@click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))
@click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))
@click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--translate/--no-translate", default=False, show_default=True)
@click.option("--policy", default=str(DEFAULT_POLICY_PATH), show_default=True)
@click.pass_context
def drift_reembed(
    ctx: click.Context,
    paths_file: Optional[str],
    paths: Tuple[str, ...],
    use_drift_log: bool,
    drift_log_path: str,
    max_candidates: int,
    scan_csv: str,
    corpus: str,
    cache: str,
    model: str,
    translate: bool,
    policy: str,
) -> None:
    _require_pandas()
    candidate_paths: Set[str] = set(paths or [])
    if paths_file:
        file_lines = Path(paths_file).read_text(encoding="utf-8").splitlines()
        candidate_paths.update(line.strip() for line in file_lines if line.strip())
    if use_drift_log:
        auto_paths = _drift_log_candidates(Path(drift_log_path), limit=max_candidates)
        if auto_paths:
            click.echo(f"📥 드리프트 로그에서 {len(auto_paths)}건 자동 수집")
            candidate_paths.update(auto_paths)
        else:
            click.echo("⚠️ 드리프트 로그에서 자동으로 선택할 문서를 찾지 못했습니다.")
    if not candidate_paths:
        raise click.UsageError("재임베딩할 경로를 --path 또는 --paths-file로 지정하세요.")
    _run_reembed_pipeline(
        ctx,
        candidate_paths,
        scan_csv=scan_csv,
        corpus=corpus,
        cache=cache,
        model=model,
        translate=translate,
        policy=policy,
        run_name="drift-reembed",
    )


@drift_group.command("auto")
@click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))
@click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--cache", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-baseline", default=str(DEFAULT_SEMANTIC_BASELINE), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-threshold", type=float, default=0.15, show_default=True, help="semantic drift 임계값 (cosine)")
@click.option("--log-path", default=str(DEFAULT_DRIFT_LOG), show_default=True, type=click.Path(path_type=str))
@click.option("--alert-threshold", type=float, default=0.1, show_default=True, help="hash drift 비율 알림 임계값")
@click.option("--model", default=str(TOPIC_MODEL_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--translate/--no-translate", default=False, show_default=True)
@click.option("--policy", default=str(DEFAULT_POLICY_PATH), show_default=True)
@click.option("--max-reembed", type=int, default=32, show_default=True, help="자동 재임베딩 상한")
@click.option("--include-changed/--skip-changed", default=True, show_default=True)
@click.option("--include-new/--skip-new", default=False, show_default=True)
@click.pass_context
def drift_auto(
    ctx: click.Context,
    scan_csv: str,
    corpus: str,
    cache: str,
    semantic_baseline: str,
    semantic_threshold: float,
    log_path: str,
    alert_threshold: float,
    model: str,
    translate: bool,
    policy: str,
    max_reembed: int,
    include_changed: bool,
    include_new: bool,
) -> None:
    report = _perform_drift_check(
        ctx,
        run_name="drift-auto",
        scan_csv=scan_csv,
        corpus=corpus,
        cache_dir=cache,
        semantic_baseline=semantic_baseline,
        semantic_threshold=semantic_threshold,
        log_path=log_path,
        alert_threshold=alert_threshold,
    )
    _print_drift_report(report, semantic_threshold)

    targets = _auto_reembed_targets(
        report,
        max_candidates=max_reembed,
        include_changed=include_changed,
        include_new=include_new,
    )
    if not targets:
        click.echo("✨ 자동 재임베딩 대상이 없어 종료합니다.")
        return

    click.echo(f"🔁 자동 재임베딩 대상 {len(targets)}건 처리 중…")
    _run_reembed_pipeline(
        ctx,
        targets,
        scan_csv=scan_csv,
        corpus=corpus,
        cache=cache,
        model=model,
        translate=translate,
        policy=policy,
        run_name="drift-auto-reembed",
    )


# 내부 임베딩 청크 명령 등록
cli.add_command(embed_chunk_command)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
