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
from core.errors import (
    AccessDeniedError,
    PolicyViolationError,
    ModelLoadError,
    DriftError,
    ScanError,
)
from core.logging.runtime import configure_runtime_logging
from scripts.pipeline.infopilot_cli.history import (
    HISTORY_PATH,
    MAX_AGENT_HISTORY,
    load_agent_history as _load_agent_history,
    remember_agent_history as _remember_agent_history,
)
from scripts.pipeline.infopilot_cli.drift import (
    auto_reembed_targets as _auto_reembed_targets,
    perform_drift_check as _perform_drift_check,
    print_drift_report as _print_drift_report,
)
from scripts.pipeline.infopilot_cli.policy import (
    dir_size_bytes as _dir_size_bytes,
    enforce_cache_limit as _enforce_cache_limit,
    load_policy_engine as _load_policy_engine_impl,
    normalize_exts as _normalize_exts,
    parse_roots as _parse_roots,
    warn_if_cache_limit_exceeded as _warn_if_cache_limit_exceeded,
)
from scripts.pipeline.infopilot_cli.scan import cmd_scan as _cmd_scan_impl, run_scan as _run_scan_impl
from scripts.pipeline.infopilot_cli.scan_rows import load_scan_rows as _load_scan_rows_impl, resolve_scan_csv as _resolve_scan_csv_impl
from scripts.pipeline.infopilot_cli.steps import (
    cmd_embed as _cmd_embed_impl,
    cmd_extract as _cmd_extract_impl,
    cmd_train as _cmd_train_impl,
)
from scripts.pipeline.infopilot_cli.index import cmd_index as _cmd_index_impl
from scripts.pipeline.infopilot_cli.chat import cmd_chat as _cmd_chat_impl
from scripts.pipeline.infopilot_cli.train_config import (
    build_train_config as _build_train_config_impl,
    default_train_config as _default_train_config_impl,
    maybe_limit_rows as _maybe_limit_rows_impl,
)
from scripts.pipeline.infopilot_cli.watch import (
    IncrementalPipeline,
    PolicyEventHandler,
    WatchEventHandler,
    watch_loop as _watch_loop,
)
from scripts.pipeline.infopilot_cli.session import command_session as _command_session

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


# 모듈 임포트
from core.config.paths import (
    CACHE_DIR,
    CORPUS_PATH,
    DATA_DIR,
    DRIFT_LOG_PATH,
    RESOURCE_LOG_PATH,
    SEMANTIC_BASELINE_PATH,
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
from core.agents.document import DocumentAgentConfig
from core.search.retriever import (
    VectorIndex,
    MODEL_TEXT_COLUMN,
    _split_tokens,
)
from core.monitor import check_drift
from scripts.utils.mlflow_logger import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
)
from scripts.utils.quantizer import export_to_onnx


KNOWLEDGE_AGENT = "knowledge_search"
DEFAULT_POLICY_PATH = Path("./core/config/smart_folders.json")
DEFAULT_FOUND_FILES = DATA_DIR / "found_files.csv"
DEFAULT_SCHEDULED_ROOT = DATA_DIR / "scheduled"
DEFAULT_SCAN_STATE = DATA_DIR / "scan_state.json"
DEFAULT_CHUNK_CACHE = CACHE_DIR / "chunk_cache.json"
DEFAULT_RESOURCE_LOG = RESOURCE_LOG_PATH
DEFAULT_DRIFT_LOG = DRIFT_LOG_PATH
DEFAULT_SEMANTIC_BASELINE = SEMANTIC_BASELINE_PATH

_SENTENCE_ENCODER_MANAGER: Optional[ModelManager] = None


def _require_pandas() -> None:
    if pd is None:
        raise ScanError("pandas 라이브러리가 필요합니다.", hint="pip install pandas 또는 scripts/setup_env.sh 실행")


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


def _load_policy_engine(
    policy_arg: Optional[str],
    *,
    fail_if_missing: bool = False,
    stage: str = "pipeline",
) -> PolicyEngine:
    return _load_policy_engine_impl(
        policy_arg,
        default_policy_path=DEFAULT_POLICY_PATH,
        fail_if_missing=fail_if_missing,
        stage=stage,
    )


def _run_scan(
    out: Path,
    roots: List[Path] | None = None,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    exts: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    return _run_scan_impl(out, roots, policy_engine=policy_engine, exts=exts, agent=KNOWLEDGE_AGENT)


def cmd_scan(args) -> int:
    return _cmd_scan_impl(args, default_policy_path=DEFAULT_POLICY_PATH, agent=KNOWLEDGE_AGENT)


def _resolve_scan_csv(path: Path) -> Path:
    return _resolve_scan_csv_impl(path)


def _load_scan_rows(
    scan_csv: Path,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    include_manual: bool = True,
) -> Iterator[Dict[str, Any]]:
    return _load_scan_rows_impl(
        scan_csv,
        policy_engine=policy_engine,
        include_manual=include_manual,
        agent=KNOWLEDGE_AGENT,
    )


def _build_train_config(args) -> TrainConfig:
    return _build_train_config_impl(args)


def _maybe_limit_rows(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return _maybe_limit_rows_impl(rows, limit)


def _default_train_config() -> TrainConfig:
    return _default_train_config_impl()

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


def _make_watch_handler(
    event_queue: "queue.Queue[Tuple[str, str]]",
    allowed_exts: Set[str],
    *,
    policy_engine: Optional[PolicyEngine],
    policy_engine_provider=None,
    ignore_paths: Optional[Set[str]] = None,
) -> FileSystemEventHandler:
    return WatchEventHandler(
        event_queue,
        allowed_exts,
        policy_engine=policy_engine,
        policy_engine_provider=policy_engine_provider,
        ignore_paths=ignore_paths,
        agent=KNOWLEDGE_AGENT,
        base_handler_cls=FileSystemEventHandler,
    )


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


def cmd_train(args):
    return _cmd_train_impl(
        args,
        default_policy_path=DEFAULT_POLICY_PATH,
        default_chunk_cache=DEFAULT_CHUNK_CACHE,
        default_scan_state=DEFAULT_SCAN_STATE,
        agent=KNOWLEDGE_AGENT,
    )


def cmd_extract(args):
    return _cmd_extract_impl(
        args,
        default_policy_path=DEFAULT_POLICY_PATH,
        default_chunk_cache=DEFAULT_CHUNK_CACHE,
        default_scan_state=DEFAULT_SCAN_STATE,
        agent=KNOWLEDGE_AGENT,
    )


def cmd_embed(args):
    return _cmd_embed_impl(
        args,
        default_policy_path=DEFAULT_POLICY_PATH,
        default_chunk_cache=DEFAULT_CHUNK_CACHE,
        default_scan_state=DEFAULT_SCAN_STATE,
        agent=KNOWLEDGE_AGENT,
    )


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
        raise PolicyViolationError(
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
    return _cmd_index_impl(args, default_policy_path=DEFAULT_POLICY_PATH, agent=KNOWLEDGE_AGENT)


def cmd_chat(args):
    return _cmd_chat_impl(args, default_policy_path=DEFAULT_POLICY_PATH, policy_agent=KNOWLEDGE_AGENT)


def cmd_watch(args):
    if Observer is None:
        raise PolicyViolationError("watchdog 라이브러리가 필요합니다. pip install watchdog")

    encoder, batch_size, model_name = _load_sentence_encoder(Path(args.model))
    if encoder is None:
        raise ModelLoadError("sentence-transformers 모델을 로드할 수 없어 watcher를 실행할 수 없습니다.")

    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = _load_policy_engine(policy_arg, fail_if_missing=policy_required, stage="watch")
    policy_path = getattr(policy_engine, "source", None) if policy_engine and policy_engine.has_policies else None
    policy_box = {"engine": policy_engine}

    def _get_policy_engine() -> Optional[PolicyEngine]:
        return policy_box.get("engine")

    def _set_policy_engine(updated: Optional[PolicyEngine]) -> None:
        policy_box["engine"] = updated
    roots = _parse_roots(args.roots)
    if not roots and policy_engine and policy_engine.has_policies:
        roots = policy_engine.roots_for_agent(KNOWLEDGE_AGENT, include_manual=False)
    if not roots:
        raise PolicyViolationError(
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
        raise PolicyViolationError("유효한 감시 루트가 없습니다. 경로를 다시 확인하세요.")
    roots = existing_roots

    event_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    allowed_exts = {ext.lower() for ext in FileFinder.DEFAULT_EXTS}
    ignore_paths: Set[str] = set()
    if policy_path is not None:
        ignore_paths.add(str(policy_path))
    handler = _make_watch_handler(
        event_queue,
        allowed_exts,
        policy_engine=policy_engine,
        policy_engine_provider=_get_policy_engine,
        ignore_paths=ignore_paths,
    )
    observer = Observer()
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)
    if policy_path is not None and policy_path.exists():
        policy_handler = PolicyEventHandler(event_queue, policy_path, base_handler_cls=FileSystemEventHandler)
        observer.schedule(policy_handler, str(policy_path.parent), recursive=False)

    pipeline_ctx = IncrementalPipeline(
        encoder=encoder,
        batch_size=batch_size,
        scan_csv=Path(args.scan_csv),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        translate=args.translate,
        policy_engine=policy_engine,
        policy_engine_provider=_get_policy_engine,
        policy_reload_callback=_set_policy_engine,
        policy_path=policy_path,
        roots=roots,
        agent=KNOWLEDGE_AGENT,
    )
    _enforce_cache_limit(
        Path(args.cache),
        policy_engine,
        hard_limit=getattr(args, "cache_hard_limit", False),
        clean_on_limit=getattr(args, "cache_clean_on_limit", False),
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
    func = click.option(
        "--include-denied/--allowed-only",
        default=False,
        show_default=True,
        help="정책에 의해 차단된 파일도 CSV에 기록할지 여부 (allowed, deny_reason 컬럼 추가).",
    )(func)
    func = click.option(
        "--hash/--no-hash",
        "include_hash",
        default=False,
        show_default=True,
        help="allowed 파일에 대해 SHA256 해시를 계산해 기록합니다(느릴 수 있음).",
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
    func = click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")(func)
    func = click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")(func)
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
    func = click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")(func)
    func = click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")(func)
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
    func = click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")(func)
    func = click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")(func)
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
    configure_runtime_logging()
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
    policy: str,
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
):
    return _perform_drift_check(
        ctx,
        run_name=run_name,
        scan_csv=scan_csv,
        corpus=corpus,
        cache_dir=cache_dir,
        semantic_baseline=semantic_baseline,
        semantic_threshold=semantic_threshold,
        log_path=log_path,
        alert_threshold=alert_threshold,
        policy=policy,
        policy_agent=KNOWLEDGE_AGENT,
        default_policy_path=DEFAULT_POLICY_PATH,
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
    )


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
    cache_hard_limit: bool = False,
    cache_clean_on_limit: bool = False,
) -> None:
    encoder, batch_size, model_name = _load_sentence_encoder(Path(model))
    if encoder is None:
        raise ModelLoadError("SentenceTransformer 모델 로드 실패로 재임베딩을 진행할 수 없습니다.")
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
    cache_action = _enforce_cache_limit(
        Path(cache),
        policy_engine,
        hard_limit=cache_hard_limit,
        clean_on_limit=cache_clean_on_limit,
    )
    with _command_session(ctx, run_name) as session:
        pipeline_ctx.process(set(paths), set())
        if session:
            session.log_metrics({"reembedded": float(len(paths))})
            session.set_tags(
                {
                    "policy": str(policy),
                    "cache_hard_limit": str(cache_hard_limit),
                    "cache_clean_on_limit": str(cache_clean_on_limit),
                    "cache_action": cache_action or "",
                    "policy_source": str(getattr(policy_engine, "source", "")) if policy_engine else "",
                }
            )
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
        raise PolicyViolationError("sentence-transformers 패키지가 필요합니다. pip install sentence-transformers")
    try:
        import numpy as np
    except Exception:
        raise PolicyViolationError("numpy 패키지가 필요합니다. pip install numpy")

    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise PolicyViolationError(f"입력 JSON 로드 실패: {exc}")
    if not isinstance(payload, list):
        raise PolicyViolationError("입력 파일은 텍스트 리스트(JSON 배열)여야 합니다.")
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
def scan_command(
    ctx: click.Context,
    out: str,
    roots: Tuple[str, ...],
    exts: Tuple[str, ...],
    policy: str,
    include_denied: bool,
    include_hash: bool,
) -> None:
    _require_pandas()
    args = SimpleNamespace(
        out=out,
        roots=list(roots) if roots else None,
        policy=policy,
        exts=list(exts) if exts else None,
        include_denied=include_denied,
        include_hash=include_hash,
    )
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
@click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")
@click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")
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
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
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
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
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
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
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


def _drift_log_candidates(log_path: Path, limit: int = 64) -> Tuple[List[str], Set[str], Set[str]]:
    if limit <= 0 or not log_path.exists():
        return [], set(), set()
    try:
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return [], set(), set()

    seen: Set[str] = set()
    picked: List[str] = []
    policy_ids: Set[str] = set()
    cache_actions: Set[str] = set()
    for raw in reversed(lines):
        entry = raw.strip()
        if not entry:
            continue
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError:
            continue
        policy_id = payload.get("policy_id")
        if isinstance(policy_id, str) and policy_id:
            policy_ids.add(policy_id)
        cache_action = payload.get("cache_action")
        if isinstance(cache_action, str) and cache_action:
            cache_actions.add(cache_action)
        for key in ("reembed_candidates", "changed_files", "new_files"):
            for path in payload.get(key, []) or []:
                normalized = str(path).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                picked.append(normalized)
                if len(picked) >= limit:
                    return picked, policy_ids, cache_actions
        if picked:
            # Stop after the most recent entry that yielded candidates.
            return picked, policy_ids, cache_actions
    return picked, policy_ids, cache_actions


@drift_group.command("check")
@click.option("--scan-csv", default=str(DEFAULT_FOUND_FILES), show_default=True, type=click.Path(path_type=str))
@click.option("--corpus", default=str(CORPUS_PATH), show_default=True, type=click.Path(path_type=str))
@click.option("--cache-dir", default=str(CACHE_DIR), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-baseline", default=str(DEFAULT_SEMANTIC_BASELINE), show_default=True, type=click.Path(path_type=str))
@click.option("--semantic-threshold", type=float, default=0.15, show_default=True, help="semantic drift 임계값 (cosine)")
@click.option("--log-path", default=str(DEFAULT_DRIFT_LOG), show_default=True, type=click.Path(path_type=str))
@click.option("--alert-threshold", type=float, default=0.1, show_default=True, help="hash drift 비율 알림 임계값")
@click.option("--policy", default=str(DEFAULT_POLICY_PATH), show_default=True, help="스마트 폴더 정책 파일")
@click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")
@click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")
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
    policy: str,
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
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
        policy=policy,
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
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
@click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")
@click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")
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
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
) -> None:
    _require_pandas()
    candidate_paths: Set[str] = set(paths or [])
    if paths_file:
        file_lines = Path(paths_file).read_text(encoding="utf-8").splitlines()
        candidate_paths.update(line.strip() for line in file_lines if line.strip())
    if use_drift_log:
        auto_paths, policies, cache_actions = _drift_log_candidates(Path(drift_log_path), limit=max_candidates)
        if auto_paths:
            meta = []
            if policies:
                meta.append(f"policy={','.join(sorted(policies))}")
            if cache_actions:
                meta.append(f"cache_action={','.join(sorted(cache_actions))}")
            meta_str = f" ({'; '.join(meta)})" if meta else ""
            click.echo(f"📥 드리프트 로그에서 {len(auto_paths)}건 자동 수집{meta_str}")
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
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
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
@click.option("--dry-run", is_flag=True, help="재임베딩을 실행하지 않고 대상만 출력")
@click.option("--yes", is_flag=True, help="확인 프롬프트 없이 재임베딩 실행")
@click.option("--cache-hard-limit", is_flag=True, help="캐시 한도 초과 시 중단")
@click.option("--cache-clean-on-limit", is_flag=True, help="캐시 한도 초과 시 캐시 초기화 후 진행")
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
    dry_run: bool,
    yes: bool,
    cache_hard_limit: bool,
    cache_clean_on_limit: bool,
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
        policy=policy,
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
    )
    _print_drift_report(report, semantic_threshold)

    targets = _auto_reembed_targets(
        report,
        max_candidates=max_reembed,
        include_changed=include_changed,
        include_new=include_new,
    )
    if report.reembed_candidates and cache_clean_on_limit:
        click.echo("♻️ cache clean on limit enabled; cache will reset if over limit before reembed.")
    if not targets:
        click.echo("✨ 자동 재임베딩 대상이 없어 종료합니다.")
        return

    target_list = sorted(targets)
    click.echo(f"🔁 자동 재임베딩 대상 {len(target_list)}건 (상위 10개):")
    for item in target_list[:10]:
        click.echo(f"   - {item}")
    if dry_run:
        click.echo("✅ dry-run 모드: 재임베딩을 실행하지 않습니다.")
        return
    if not yes:
        if not click.confirm(f"위 {len(target_list)}건을 재임베딩할까요?", default=False):
            click.echo("⛔ 취소했습니다.")
            return

    policy_engine = _load_policy_engine(policy, fail_if_missing=False, stage="drift")
    policy_source = str(getattr(policy_engine, "source", "")) if policy_engine else ""
    click.echo(
        f"🔁 자동 재임베딩 대상 {len(targets)}건 처리 중… "
        f"(policy={policy_source or 'n/a'}, cache_clean={cache_clean_on_limit}, cache_hard={cache_hard_limit})"
    )
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
        cache_hard_limit=cache_hard_limit,
        cache_clean_on_limit=cache_clean_on_limit,
    )


# 내부 임베딩 청크 명령 등록
cli.add_command(embed_chunk_command)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
