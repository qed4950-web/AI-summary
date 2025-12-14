from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from core.data_pipeline.policies.engine import PolicyEngine
from core.errors import PolicyViolationError

_POLICY_CACHE: Dict[Path, PolicyEngine] = {}


def dir_size_bytes(path: Path) -> int:
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


def warn_if_cache_limit_exceeded(cache_dir: Path, policy_engine: Optional[PolicyEngine]) -> Optional[tuple[int, int]]:
    if not policy_engine or not policy_engine.has_policies:
        return None
    try:
        policy = policy_engine.policy_for_path(cache_dir)
    except Exception:
        policy = None
    if not policy:
        return None
    cache_limit = policy.cache.get("max_bytes")
    if cache_limit is None:
        return None
    usage = dir_size_bytes(cache_dir)
    if usage > int(cache_limit):
        print(
            f"⚠️ 캐시 사용량이 정책 한도({int(cache_limit):,} bytes)를 초과했습니다: {usage:,} bytes",
            flush=True,
        )
    return int(cache_limit), usage


def enforce_cache_limit(
    cache_dir: Path,
    policy_engine: Optional[PolicyEngine],
    *,
    hard_limit: bool = False,
    clean_on_limit: bool = False,
) -> Optional[str]:
    if not policy_engine or not policy_engine.has_policies:
        return None
    try:
        policy = policy_engine.policy_for_path(cache_dir)
    except Exception:
        policy = None
    if not policy:
        return None
    cache_limit = policy.cache.get("max_bytes")
    purge_days = policy.cache.get("purge_days")
    cache_action = None
    if purge_days:
        try:
            threshold = time.time() - (int(purge_days) * 86400)
            for entry in cache_dir.rglob("*"):
                try:
                    if entry.is_file() and entry.stat().st_mtime < threshold:
                        entry.unlink()
                        cache_action = "purge_old"
                except OSError:
                    continue
        except Exception:
            pass
    if cache_limit is None:
        return cache_action
    usage = dir_size_bytes(cache_dir)
    if usage <= int(cache_limit):
        return cache_action
    if clean_on_limit:
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            pass
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"🧹 캐시 한도 초과로 캐시를 초기화했습니다. ({usage:,} bytes → 0)", flush=True)
        return "clean_on_limit"
    if hard_limit:
        raise PolicyViolationError(
            f"캐시 사용량이 정책 한도({int(cache_limit):,} bytes)를 초과했습니다: {usage:,} bytes"
        )
    warn_if_cache_limit_exceeded(cache_dir, policy_engine)
    return cache_action


def parse_roots(raw_roots: List[str] | None) -> List[Path] | None:
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


def load_policy_engine(
    policy_arg: Optional[str],
    *,
    default_policy_path: Path,
    fail_if_missing: bool = False,
    stage: str = "pipeline",
) -> PolicyEngine:
    """Load a policy engine with optional fail-closed semantics."""

    raw = (policy_arg or str(default_policy_path)).strip()
    normalized = raw.lower()
    if normalized in {"none", ""}:
        if fail_if_missing:
            raise PolicyViolationError(
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
            raise PolicyViolationError(message)
        print(f"⚠️ {message} (정책 미적용 상태로 진행)", flush=True)
        return PolicyEngine.empty()

    engine = _POLICY_CACHE.get(resolved)
    if engine is None:
        try:
            engine = PolicyEngine.from_file(resolved)
        except Exception as exc:
            message = f"[{stage}] 정책 파일을 불러오지 못했습니다 ({resolved}): {exc}"
            if fail_if_missing:
                raise PolicyViolationError(message) from exc
            print(f"⚠️ {message}", flush=True)
            return PolicyEngine.empty()
        _POLICY_CACHE[resolved] = engine
    return engine


def normalize_exts(exts: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if not exts:
        return None
    normalized: Set[str] = set()
    for ext in exts:
        value = (ext or "").strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        normalized.add(value)
    return normalized or None


__all__ = [
    "dir_size_bytes",
    "enforce_cache_limit",
    "load_policy_engine",
    "normalize_exts",
    "parse_roots",
    "warn_if_cache_limit_exceeded",
]
