from __future__ import annotations

from app.vision.types import (
    DistanceBand,
    HorizontalPosition,
    PathAnalysis,
    PathObstacle,
    PathRelevance,
    SceneObject,
)

OBSTRUCTION_TYPES = {
    "person",
    "chair",
    "couch",
    "sofa",
    "bench",
    "table",
    "dining table",
    "bed",
    "toilet",
    "tv",
    "potted plant",
    "refrigerator",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "dog",
    "cat",
    "suitcase",
    "backpack",
    "stairs",
    "curb",
    "pothole",
}


class PathAnalyzer:
    def analyze(self, objects: list[SceneObject]) -> PathAnalysis:
        obstacles: list[PathObstacle] = []
        for obj in objects:
            if obj.path_relevance in {PathRelevance.NONE, PathRelevance.LOW}:
                continue
            if obj.type.lower() not in OBSTRUCTION_TYPES and obj.path_relevance != PathRelevance.HIGH:
                continue
            if obj.distance_band == DistanceBand.FAR:
                continue
            obstacles.append(
                PathObstacle(
                    object_id=obj.id,
                    type=obj.type,
                    distance_range=obj.distance_range,
                    position=obj.position,
                    confidence=obj.confidence,
                )
            )
        if not objects:
            return PathAnalysis(path_status="uncertain", notes="No objects were detected with enough confidence.")
        high = [item for item in obstacles if any(o.id == item.object_id and o.path_relevance == PathRelevance.HIGH for o in objects)]
        if high:
            status = "blocked" if any(
                o.distance_band == DistanceBand.VERY_NEAR and o.position == HorizontalPosition.CENTER
                for o in objects
                if o.id in {h.object_id for h in high}
            ) else "partially_blocked"
        elif obstacles:
            status = "partially_blocked"
        else:
            status = "clear"
        return PathAnalysis(path_status=status, obstacles=obstacles)
