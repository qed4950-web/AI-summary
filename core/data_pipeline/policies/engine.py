"""Policy engine for smart folder configurations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from infopilot_core.data_pipeline.policies.loader import load_policy_file
from infopilot_core.utils import get_logger, resolve_repo_root

LOGGER = get_logger("policy.engine")


def _normalize_path(path: Path) -> Path:
    path = path.expanduser()
    try:
        return path.resolve(strict=False)
    except TypeError:  # Python <3.9 strict arg not supported
        try:
            return path.resolve()
        except OSError:
            return path
    except OSError:
        return path


@dataclass(frozen=True)
class SmartFolderPolicy:
    path: Path
    agents: frozenset[str]
    security: Dict[str, object]
    indexing: Dict[str, object]
    retention: Dict[str, object]

    @classmethod
    def from_dict(cls, data: Dict[str, object], *, base: Path) -> "SmartFolderPolicy":
        if "path" not in data or not data.get("path"):
            raise ValueError("Smart folder policy requires a 'path' key")
        raw_path = Path(str(data.get("path")))
        if not raw_path.is_absolute():
            raw_path = base / raw_path
        normalized_path = _normalize_path(raw_path)
        agents = frozenset(str(item) for item in data.get("agents", []) or [])
        security = dict(data.get("security", {}) or {})
        indexing = dict(data.get("indexing", {}) or {})
        retention = dict(data.get("retention", {}) or {})
        return cls(
            path=normalized_path,
            agents=agents,
            security=security,
            indexing=indexing,
            retention=retention,
        )

    @property
    def indexing_mode(self) -> str:
        mode = str(self.indexing.get("mode", "realtime") or "realtime").lower()
        if mode not in {"realtime", "scheduled", "manual"}:
            return "realtime"
        return mode

    def allows_agent(self, agent: str) -> bool:
        if not self.agents:
            return True
        return agent in self.agents


class PolicyEngine:
    def __init__(self, policies: Sequence[SmartFolderPolicy], *, source: Optional[Path] = None) -> None:
        self._policies = sorted(policies, key=lambda p: len(p.path.parts), reverse=True)
        self.source = source

    @classmethod
    def empty(cls) -> "PolicyEngine":
        return cls((), source=None)

    @classmethod
    def from_file(cls, path: Path) -> "PolicyEngine":
        repo_root = resolve_repo_root()
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if not path.exists():
            LOGGER.info("Policy file not found at %s; continuing without policies", path)
            return cls.empty()
        raw_policies = load_policy_file(path)
        policies = [SmartFolderPolicy.from_dict(entry, base=path.parent) for entry in raw_policies]
        LOGGER.info("Loaded %d smart folder policies from %s", len(policies), path)
        return cls(policies, source=path)

    def __len__(self) -> int:
        return len(self._policies)

    @property
    def has_policies(self) -> bool:
        return bool(self._policies)

    def roots_for_agent(self, agent: str, *, include_manual: bool = True) -> List[Path]:
        if not self._policies:
            return []
        roots: List[Path] = []
        for policy in self._policies:
            if not policy.allows_agent(agent):
                continue
            if not include_manual and policy.indexing_mode == "manual":
                continue
            roots.append(policy.path)
        # remove duplicates while preserving order
        seen = set()
        unique: List[Path] = []
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            unique.append(root)
        return unique

    def iter_policies(self) -> Sequence[SmartFolderPolicy]:
        return tuple(self._policies)

    def policy_for_path(self, path: Path) -> Optional[SmartFolderPolicy]:
        if not self._policies:
            return None
        normalized = _normalize_path(path)
        for policy in self._policies:
            try:
                normalized.relative_to(policy.path)
                return policy
            except ValueError:
                continue
        return None

    def allows(self, path: Path, *, agent: str, include_manual: bool = True) -> bool:
        if not self._policies:
            return True
        policy = self.policy_for_path(path)
        if policy is None:
            return False
        if not policy.allows_agent(agent):
            return False
        if not include_manual and policy.indexing_mode == "manual":
            return False
        return True

    def filter_records(
        self,
        records: Iterable[Dict[str, object]],
        *,
        agent: str,
        include_manual: bool = True,
    ) -> List[Dict[str, object]]:
        if not self._policies:
            return list(records)
        filtered: List[Dict[str, object]] = []
        for record in records:
            path_str = record.get("path") if isinstance(record, dict) else None
            if not path_str:
                continue
            if self.allows(Path(str(path_str)), agent=agent, include_manual=include_manual):
                filtered.append(record)
        return filtered
