from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.data_pipeline.filefinder import FileFinder
from core.data_pipeline.policies.engine import PolicyEngine
from core.errors import PolicyViolationError

from .policy import normalize_exts, parse_roots


def _hash_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()


def write_scan_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "size", "mtime", "allowed", "deny_reason", "hash"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(payload)


def run_scan(
    out: Path,
    roots: List[Path] | None = None,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    exts: Optional[Iterable[str]] = None,
    agent: str,
    include_denied: bool = False,
    include_hash: bool = False,
) -> List[Dict[str, Any]]:
    scan_roots = roots
    if policy_engine and policy_engine.has_policies and not roots:
        candidate_roots = policy_engine.roots_for_agent(agent, include_manual=True)
        if candidate_roots:
            scan_roots = candidate_roots
            print("📁 정책 기반 스캔 루트:")
            for root in candidate_roots:
                print(f"   - {root}")

    normalized_exts = normalize_exts(exts)

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

    if not (policy_engine and policy_engine.has_policies):
        rows: List[Dict[str, Any]] = []
        for rec in files:
            path = Path(str(rec.get("path") or ""))
            payload = dict(rec)
            payload["allowed"] = 1
            payload["deny_reason"] = ""
            payload["hash"] = _hash_file(path) if include_hash and path.is_file() else ""
            rows.append(payload)
        if include_denied:
            write_scan_csv(rows, out)
        else:
            FileFinder.to_csv(files, out)
        print(f"📦 스캔 결과 저장: {out}")
        return rows if include_denied else files

    rows = []
    for rec in files:
        raw_path = rec.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        allowed, reason = policy_engine.check(path, agent=agent, include_manual=True)
        if not include_denied and not allowed:
            continue
        payload = dict(rec)
        payload["allowed"] = 1 if allowed else 0
        payload["deny_reason"] = "" if allowed else reason
        payload["hash"] = _hash_file(path) if include_hash and allowed and path.is_file() else ""
        rows.append(payload)
    write_scan_csv(rows, out)
    print(f"📦 스캔 결과 저장: {out}")
    return rows


def cmd_scan(
    args,
    *,
    default_policy_path: Path,
    agent: str,
) -> int:
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"

    # late import to avoid circular deps in CLI entrypoint
    from .policy import load_policy_engine

    policy_engine = load_policy_engine(
        policy_arg,
        default_policy_path=default_policy_path,
        fail_if_missing=policy_required,
        stage="scan",
    )
    roots = parse_roots(getattr(args, "roots", None))
    if not roots and policy_engine and policy_engine.has_policies:
        roots = policy_engine.roots_for_agent(agent, include_manual=True)
    if not roots:
        raise PolicyViolationError(
            "스마트 폴더 정책이나 스캔 루트가 없어 scan을 중단합니다. "
            "Park David Foundation 스펙에 따라 정책 기반 경계가 필수입니다."
        )
    rows = run_scan(
        Path(getattr(args, "out")),
        roots,
        policy_engine=policy_engine,
        exts=getattr(args, "exts", None),
        agent=agent,
        include_denied=bool(getattr(args, "include_denied", False)),
        include_hash=bool(getattr(args, "include_hash", False)),
    )
    if rows and isinstance(rows[0], dict) and "allowed" in rows[0]:
        return sum(1 for row in rows if str(row.get("allowed", "")).strip() not in {"0", "false", "False"})
    return len(rows)


__all__ = ["run_scan", "cmd_scan"]
