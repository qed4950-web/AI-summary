"""Policy engine for smart folder configurations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from core.data_pipeline.policies.loader import load_policy_file
from core.utils import get_logger, resolve_repo_root

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


def _normalize_ext(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if not raw.startswith("."):
        raw = f".{raw}"
    return raw


def _normalize_exts(values: object) -> frozenset[str]:
    if not values:
        return frozenset()
    if not isinstance(values, list):
        values = [values]
    normalized = []
    for item in values:
        if item is None:
            continue
        ext = _normalize_ext(str(item))
        if ext:
            normalized.append(ext)
    return frozenset(normalized)


@dataclass(frozen=True)
class AgentRule:
    allow_types: frozenset[str] = frozenset()
    deny_types: frozenset[str] = frozenset()
    max_file_size_mb: int | None = None
    masking: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SmartFolderPolicy:
    path: Path
    agents: frozenset[str] = frozenset()
    sensitive_paths: frozenset[Path] = frozenset()
    allow_types: frozenset[str] = frozenset()
    deny_types: frozenset[str] = frozenset()
    max_file_size_mb: int | None = None
    agent_rules: Dict[str, AgentRule] = field(default_factory=dict)
    security: Dict[str, object] = field(default_factory=dict)
    indexing: Dict[str, object] = field(default_factory=dict)
    retention: Dict[str, object] = field(default_factory=dict)
    cache: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, object], *, base: Path) -> "SmartFolderPolicy":
        if "path" not in data:
            raise ValueError("Smart folder policy requires a 'path' key")
        raw_path_value = data.get("path")
        raw_path_str = str(raw_path_value or "").strip()
        policy_type = str(data.get("type") or "").lower()

        if not raw_path_str:
            if policy_type == "global":
                raw_path = resolve_repo_root()
            else:
                raise ValueError("Smart folder policy requires a non-empty 'path' for non-global entries")
        else:
            raw_path = Path(raw_path_str).expanduser()
            if not raw_path.is_absolute():
                raw_path = base / raw_path
        normalized_path = _normalize_path(raw_path)
        agents = frozenset(str(item) for item in data.get("agents", []) or [])
        allow_types = _normalize_exts(data.get("allow_types"))
        deny_types = _normalize_exts(data.get("deny_types"))
        max_file_size_mb_raw = data.get("max_file_size_mb")
        max_file_size_mb: int | None
        if max_file_size_mb_raw is None or max_file_size_mb_raw == "":
            max_file_size_mb = None
        else:
            try:
                max_file_size_mb = int(max_file_size_mb_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                max_file_size_mb = None

        agent_rules: Dict[str, AgentRule] = {}
        raw_agent_rules = data.get("agent_rules") or {}
        if isinstance(raw_agent_rules, dict):
            for agent_name, raw_rule in raw_agent_rules.items():
                if not agent_name:
                    continue
                if not isinstance(raw_rule, dict):
                    continue
                rule_allow = _normalize_exts(raw_rule.get("allow_types"))
                rule_deny = _normalize_exts(raw_rule.get("deny_types"))
                rule_max_raw = raw_rule.get("max_file_size_mb")
                rule_max: int | None
                if rule_max_raw is None or rule_max_raw == "":
                    rule_max = None
                else:
                    try:
                        rule_max = int(rule_max_raw)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        rule_max = None
                masking_raw = raw_rule.get("masking") or {}
                masking: Dict[str, bool] = {}
                if isinstance(masking_raw, dict):
                    for key, value in masking_raw.items():
                        if not key:
                            continue
                        masking[str(key)] = bool(value)
                agent_rules[str(agent_name)] = AgentRule(
                    allow_types=rule_allow,
                    deny_types=rule_deny,
                    max_file_size_mb=rule_max,
                    masking=masking,
                )

        sensitive_paths_raw = data.get("sensitive_paths") or []
        sensitive_paths: List[Path] = []
        for entry in sensitive_paths_raw:
            if entry is None:
                continue
            entry_str = str(entry).strip()
            if not entry_str:
                continue
            entry_path = Path(entry_str).expanduser()
            if not entry_path.is_absolute():
                entry_path = base / entry_path
            sensitive_paths.append(_normalize_path(entry_path))
        security = dict(data.get("security", {}) or {})
        indexing = dict(data.get("indexing", {}) or {})
        retention = dict(data.get("retention", {}) or {})
        cache = dict(data.get("cache", {}) or {})
        return cls(
            path=normalized_path,
            agents=agents,
            sensitive_paths=frozenset(sensitive_paths),
            allow_types=allow_types,
            deny_types=deny_types,
            max_file_size_mb=max_file_size_mb,
            agent_rules=agent_rules,
            security=security,
            indexing=indexing,
            retention=retention,
            cache=cache,
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

    def is_sensitive(self, path: Path) -> bool:
        if not self.sensitive_paths:
            return False
        normalized = _normalize_path(path)
        for sensitive_root in self.sensitive_paths:
            try:
                normalized.relative_to(sensitive_root)
                return True
            except ValueError:
                continue
        return False


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

    def check(self, path: Path, *, agent: str, include_manual: bool = True) -> tuple[bool, str]:
        if not self._policies:
            return True, "no_policies"
        policy = self.policy_for_path(path)
        if policy is None:
            return False, "out_of_scope"
        if policy.is_sensitive(path):
            return False, "sensitive_path"
        if not policy.allows_agent(agent):
            return False, "agent_denied"
        if not include_manual and policy.indexing_mode == "manual":
            return False, "manual_policy"

        effective_allow = policy.allow_types
        effective_deny = policy.deny_types
        max_size_mb = policy.max_file_size_mb
        rule = policy.agent_rules.get(agent)
        if rule:
            if rule.allow_types:
                effective_allow = rule.allow_types
            if rule.deny_types:
                effective_deny = frozenset(set(effective_deny) | set(rule.deny_types))
            if rule.max_file_size_mb is not None:
                max_size_mb = rule.max_file_size_mb

        if path.is_file():
            ext = _normalize_ext(path.suffix)
            if effective_deny and ext in effective_deny:
                return False, "type_denied"
            if effective_allow and ext not in effective_allow:
                return False, "type_not_allowed"
            if max_size_mb is not None:
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    return False, "stat_failed"
                if size_bytes > int(max_size_mb) * 1024 * 1024:
                    return False, "file_too_large"
        return True, "ok"

    def allows(self, path: Path, *, agent: str, include_manual: bool = True) -> bool:
        allowed, _ = self.check(path, agent=agent, include_manual=include_manual)
        return allowed

    def masking_rules_for_path(self, path: Path, *, agent: str) -> Dict[str, bool]:
        """Return policy masking rules for a given path/agent."""
        policy = self.policy_for_path(path)
        if policy is None:
            return {}
        rule = policy.agent_rules.get(agent)
        if rule and rule.masking:
            return dict(rule.masking)
        return {}

    def pii_mask_enabled_for_path(self, path: Path, *, agent: str) -> bool:
        """Best-effort toggle to enable meeting-style PII masking."""
        policy = self.policy_for_path(path)
        if policy is None:
            return False
        rules = self.masking_rules_for_path(path, agent=agent)
        if any(bool(value) for value in rules.values()):
            return True
        security = policy.security or {}
        return bool(security.get("pii_filter", False))

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
