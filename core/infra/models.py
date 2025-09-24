"""Model selection utilities for hybrid deployments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelSelector:
    registry: Dict[str, str]
    default_model: str

    def pick(self, policy_tag: Optional[str] = None, prefer_gpu: bool = False) -> str:
        key = f"{policy_tag or 'default'}:{'gpu' if prefer_gpu else 'cpu'}"
        return self.registry.get(key, self.registry.get(policy_tag or "default", self.default_model))
