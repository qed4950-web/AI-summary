from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class TaskStage:
    name: str
    handler: Callable[["TaskContext"], None]
    dependencies: Sequence[str] = field(default_factory=tuple)


@dataclass
class TaskContext:
    pipeline: Any
    job: Any
    data: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def record_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def stage_status(self) -> List[Dict[str, Any]]:
        return list(self.events)


class TaskGraph:
    def __init__(self, name: str = "taskgraph") -> None:
        self.name = name
        self._stages: List[TaskStage] = []
        self._stage_lookup: Dict[str, TaskStage] = {}

    def add_stage(self, name: str, handler: Callable[[TaskContext], None], *, dependencies: Optional[Iterable[str]] = None) -> None:
        stage = TaskStage(name=name, handler=handler, dependencies=tuple(dependencies or ()))
        self._stages.append(stage)
        self._stage_lookup[name] = stage

    def run(self, context: TaskContext) -> None:
        completed: Dict[str, bool] = {}
        for stage in self._stages:
            if stage.dependencies and not all(completed.get(dep) for dep in stage.dependencies):
                missing = [dep for dep in stage.dependencies if not completed.get(dep)]
                raise RuntimeError(f"TaskGraph dependency not satisfied for stage '{stage.name}': {missing}")
            event = {
                "stage": stage.name,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
            }
            context.record_event(event)
            try:
                stage.handler(context)
            except Exception as exc:
                event["status"] = "failed"
                event["finished_at"] = datetime.utcnow().isoformat()
                event["error"] = str(exc)
                raise
            else:
                event["status"] = "completed"
                event["finished_at"] = datetime.utcnow().isoformat()
                completed[stage.name] = True
