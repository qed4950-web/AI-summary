"""Pipeline orchestrator for photo tagging/deduplication."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.agents.taskgraph import TaskCancelled, TaskContext, TaskGraph

from core.utils import get_logger

from .models import PhotoAsset, PhotoJobConfig, PhotoRecommendation

LOGGER = get_logger("photo.pipeline")


class PhotoPipeline:
    """Placeholder pipeline for photo agent MVP."""

    def __init__(self, *, embedding_backend: str = "placeholder", tag_backend: str = "vision-api") -> None:
        self.embedding_backend = embedding_backend
        self.tag_backend = tag_backend
        self._cancel_event: Optional[Any] = None
        self._last_events: List[Dict[str, Any]] = []

    def run(
        self,
        job: PhotoJobConfig,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> PhotoRecommendation:
        LOGGER.info(
            "photo pipeline start: roots=%s embed=%s tag=%s",
            ",".join(str(r) for r in job.roots),
            self.embedding_backend,
            self.tag_backend,
        )
        context = TaskContext(pipeline=self, job=job)
        if progress_callback:
            context.extras["progress_callback"] = progress_callback
        if cancel_event:
            context.extras["cancel_event"] = cancel_event
        self._cancel_event = cancel_event
        graph = TaskGraph("photo_pipeline")
        graph.add_stage("scan", self._stage_scan)
        graph.add_stage("analyse", self._stage_analyse, dependencies=("scan",))
        graph.add_stage("persist", self._stage_persist, dependencies=("analyse",))

        try:
            graph.run(context)
        finally:
            self._cancel_event = None

        events = context.stage_status()
        self._last_events = events
        for event in events:
            LOGGER.info(
                "photo pipeline stage: %s status=%s",
                event.get("stage"),
                event.get("status"),
            )
        recommendation: Optional[PhotoRecommendation] = context.get("recommendation")
        if recommendation is None:
            raise RuntimeError("photo pipeline did not produce a recommendation")
        LOGGER.info("photo pipeline finished: report=%s", recommendation.report_path)
        return recommendation

    def last_events(self) -> List[Dict[str, Any]]:
        """Return TaskGraph events captured during the last run."""
        return list(self._last_events)

    # TaskGraph stages
    def _stage_scan(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        job: PhotoJobConfig = context.job
        photos = self._scan(job.roots)
        context.set("photos", photos)

    def _stage_analyse(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        photos: List[PhotoAsset] = context.get("photos") or []
        tagged = self._tag(photos)
        dedup_groups = self._deduplicate(tagged)
        best = self._pick_best(tagged)
        job: PhotoJobConfig = context.job
        recommendation = PhotoRecommendation(
            best_shots=best,
            duplicates=dedup_groups,
            similar_groups=dedup_groups,
            report_path=job.output_dir / "photo_report.json",
        )
        context.set("recommendation", recommendation)

    def _stage_persist(self, context: TaskContext) -> None:
        self._ensure_not_cancelled()
        job: PhotoJobConfig = context.job
        recommendation: PhotoRecommendation = context.get("recommendation")
        if recommendation is None:
            raise RuntimeError("photo pipeline persistence stage requires recommendation")
        self._persist(job, recommendation)

    def _scan(self, roots: Iterable[Path]) -> List[PhotoAsset]:
        assets: List[PhotoAsset] = []
        for root in roots:
            if not root.exists():
                LOGGER.warning("photo root missing: %s", root)
                continue
            for path in root.rglob("*"):
                self._ensure_not_cancelled()
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
                    continue
                assets.append(PhotoAsset(path=path, tags=[]))
        LOGGER.info("photos detected: %d", len(assets))
        return assets

    def _tag(self, photos: List[PhotoAsset]) -> List[PhotoAsset]:
        for asset in photos:
            asset.tags = ["tag-placeholder", asset.path.stem]
            asset.embedding = [0.0, 0.0, 0.0]
        return photos

    def _deduplicate(self, photos: List[PhotoAsset]) -> List[List[PhotoAsset]]:
        duplicates: List[List[PhotoAsset]] = []
        seen = {}
        for asset in photos:
            key = asset.path.stat().st_size if asset.path.exists() else None
            if key is None:
                continue
            bucket = seen.setdefault(key, [])
            bucket.append(asset)
        for bucket in seen.values():
            if len(bucket) > 1:
                duplicates.append(bucket)
        return duplicates

    def _pick_best(self, photos: List[PhotoAsset]) -> List[PhotoAsset]:
        sorted_photos = sorted(photos, key=lambda a: a.path.stat().st_mtime if a.path.exists() else 0, reverse=True)
        return sorted_photos[: min(20, len(sorted_photos))]

    def _persist(self, job: PhotoJobConfig, recommendation: PhotoRecommendation) -> None:
        import json

        job.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_shots": [str(asset.path) for asset in recommendation.best_shots],
            "duplicates": [[str(a.path) for a in group] for group in recommendation.duplicates],
            "similar_groups": [[str(a.path) for a in group] for group in recommendation.similar_groups],
            "policy_tag": job.policy_tag,
        }
        recommendation.report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_not_cancelled(self) -> None:
        if (
            self._cancel_event
            and hasattr(self._cancel_event, "is_set")
            and callable(getattr(self._cancel_event, "is_set"))
            and self._cancel_event.is_set()
        ):
            LOGGER.info("photo pipeline cancelled by user request")
            raise TaskCancelled("photo pipeline cancelled")
