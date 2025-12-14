"""Smart Folder + Policy aware scanner skeleton (Park David spec alignment).

이 모듈은 기존 FileFinder/PolicyEngine 흐름을 감싸서
allowed/denied 메타를 명시적으로 수집하는 베이스 스켈레톤이다.
현 시점에서는 infopilot의 scan 흐름과 병행 사용되며,
추후 통합 시 ScanResult를 파이프라인 입력으로 일관되게 전달하는 용도로 확장할 수 있다.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from core.data_pipeline.filefinder import FileFinder
from core.data_pipeline.policies.engine import PolicyEngine


@dataclass
class ScanConfig:
    roots: List[Path]
    exts: Optional[Iterable[str]] = None
    allow_hash: bool = False


@dataclass
class ScanResult:
    path: Path
    size: int
    mtime: float
    allowed: bool
    deny_reason: str = ""
    content_hash: str = ""

    def to_row(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "size": self.size,
            "mtime": self.mtime,
            "allowed": int(self.allowed),
            "deny_reason": self.deny_reason,
            "hash": self.content_hash,
        }


def _hash_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()


def run_scan(cfg: ScanConfig, policy_engine: Optional[PolicyEngine] = None) -> List[ScanResult]:
    finder = FileFinder(exts=cfg.exts or FileFinder.DEFAULT_EXTS, show_progress=False, scan_all_drives=False)
    records = finder.find(roots=cfg.roots, run_async=False)
    results: List[ScanResult] = []
    for rec in records:
        path = Path(rec["path"])
        allowed = True
        deny_reason = ""
        if policy_engine and policy_engine.has_policies:
            if not policy_engine.allows(path, agent="knowledge_search", include_manual=True):
                allowed = False
                deny_reason = "policy_denied"
        content_hash = _hash_file(path) if cfg.allow_hash and allowed else ""
        results.append(
            ScanResult(
                path=path,
                size=int(rec.get("size", 0) or 0),
                mtime=float(rec.get("mtime", 0.0) or 0.0),
                allowed=allowed,
                deny_reason=deny_reason,
                content_hash=content_hash,
            )
        )
    return results


def write_csv(results: List[ScanResult], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "mtime", "allowed", "deny_reason", "hash"])
        writer.writeheader()
        for row in results:
            writer.writerow(row.to_row())
