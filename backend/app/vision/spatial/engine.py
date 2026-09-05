from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Settings
from app.vision.types import DistanceBand, HorizontalPosition, PathRelevance, SceneObject

# Spoken ranges are coarse on purpose. Monocular depth is not metric.
DISTANCE_SPEECH = {
    DistanceBand.VERY_NEAR: "less than one meter",
    DistanceBand.NEAR: "one to two meters",
    DistanceBand.MID: "two to three meters",
    DistanceBand.FAR: "more than three meters",
    DistanceBand.UNKNOWN: "an unknown distance",
}


class SpatialReasoningEngine:
    def __init__(self, settings: Settings):
        self._higher_means_farther = settings.depth_higher_means_farther
        self._near_t = settings.depth_near_threshold
        self._mid_t = settings.depth_mid_threshold

    def enrich(
        self,
        objects: list[SceneObject],
        depth_map: Optional[np.ndarray],
        image_shape: tuple[int, ...],
    ) -> list[SceneObject]:
        enriched: list[SceneObject] = []
        for obj in objects:
            position = self.classify_position(obj.bbox.x + obj.bbox.width / 2)
            band, estimate, spoken = self.classify_distance(obj, depth_map)
            relevance = self.classify_path_relevance(obj, position, band)
            enriched.append(
                obj.model_copy(
                    update={
                        "position": position,
                        "distance_band": band,
                        "distance_estimate": estimate,
                        "distance_range": spoken,
                        "path_relevance": relevance,
                    }
                )
            )

        # Compute object-to-object spatial relationships
        relationships_by_id = self.compute_relationships(enriched)
        final_objects: list[SceneObject] = []
        for obj in enriched:
            rels = relationships_by_id.get(obj.id, [])
            final_objects.append(obj.model_copy(update={"relationships": rels}))
        return final_objects

    def compute_relationships(self, objects: list[SceneObject]) -> dict[int, list[str]]:
        """Compute grounded spatial relationships between detected objects."""
        relationships: dict[int, list[str]] = {obj.id: [] for obj in objects}
        surface_types = {"dining table", "table", "desk", "bench", "chair", "bed", "couch"}

        for i, obj_a in enumerate(objects):
            cx_a = obj_a.bbox.x + obj_a.bbox.width / 2
            cy_a = obj_a.bbox.y + obj_a.bbox.height / 2
            bot_a = obj_a.bbox.y + obj_a.bbox.height

            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                type_b = obj_b.type.lower()
                cx_b = obj_b.bbox.x + obj_b.bbox.width / 2
                cy_b = obj_b.bbox.y + obj_b.bbox.height / 2

                # 1. Surface support check (e.g. phone on table, cup on desk)
                if type_b in surface_types and obj_a.type.lower() not in surface_types:
                    x_overlap = (
                        obj_b.bbox.x <= cx_a <= (obj_b.bbox.x + obj_b.bbox.width)
                    )
                    y_support = (
                        obj_b.bbox.y - 0.05 <= bot_a <= (obj_b.bbox.y + obj_b.bbox.height * 0.85)
                    )
                    if x_overlap and y_support:
                        rel = f"on the {obj_b.type}"
                        if rel not in relationships[obj_a.id]:
                            relationships[obj_a.id].append(rel)

                # 2. Horizontal proximity check (beside / left / right)
                # Objects close horizontally within same depth band
                if obj_a.distance_band == obj_b.distance_band and obj_a.distance_band != DistanceBand.UNKNOWN:
                    dx = cx_a - cx_b
                    dy = abs(cy_a - cy_b)
                    if 0.05 < abs(dx) < 0.28 and dy < 0.30:
                        if dx < 0:
                            rel = f"to the left of {obj_b.type}"
                        else:
                            rel = f"to the right of {obj_b.type}"
                        if rel not in relationships[obj_a.id] and len(relationships[obj_a.id]) < 2:
                            relationships[obj_a.id].append(rel)

        return relationships

    def classify_position(self, cx_norm: float) -> HorizontalPosition:
        if cx_norm < 0.33:
            return HorizontalPosition.LEFT
        if cx_norm > 0.67:
            return HorizontalPosition.RIGHT
        return HorizontalPosition.CENTER

    def classify_distance(
        self,
        obj: SceneObject,
        depth_map: Optional[np.ndarray],
    ) -> tuple[DistanceBand, Optional[float], str]:
        if depth_map is None or depth_map.size == 0:
            return DistanceBand.UNKNOWN, None, DISTANCE_SPEECH[DistanceBand.UNKNOWN]
        h, w = depth_map.shape[:2]
        x1 = int(np.clip(obj.bbox.x * w, 0, w - 1))
        y1 = int(np.clip(obj.bbox.y * h, 0, h - 1))
        x2 = int(np.clip((obj.bbox.x + obj.bbox.width) * w, 0, w))
        y2 = int(np.clip((obj.bbox.y + obj.bbox.height) * h, 0, h))
        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            return DistanceBand.UNKNOWN, None, DISTANCE_SPEECH[DistanceBand.UNKNOWN]
        value = float(np.median(patch))
        finite = depth_map[np.isfinite(depth_map)]
        if finite.size == 0:
            return DistanceBand.UNKNOWN, None, DISTANCE_SPEECH[DistanceBand.UNKNOWN]
        lo, hi = np.percentile(finite, 5), np.percentile(finite, 95)
        if hi <= lo:
            return DistanceBand.UNKNOWN, None, DISTANCE_SPEECH[DistanceBand.UNKNOWN]
        norm = float(np.clip((value - lo) / (hi - lo), 0, 1))
        farness = norm if self._higher_means_farther else 1.0 - norm
        # Map relative farness to coarse indoor-ish bands. These numbers are
        # communication aids, not calibrated meters.
        if farness < self._near_t * 0.6:
            band = DistanceBand.VERY_NEAR
            estimate = 0.7
        elif farness < self._near_t:
            band = DistanceBand.NEAR
            estimate = 1.5
        elif farness < self._mid_t:
            band = DistanceBand.MID
            estimate = 2.5
        else:
            band = DistanceBand.FAR
            estimate = 4.0
        return band, estimate, DISTANCE_SPEECH[band]

    def classify_path_relevance(
        self,
        obj: SceneObject,
        position: HorizontalPosition,
        band: DistanceBand,
    ) -> PathRelevance:
        cx = obj.bbox.x + obj.bbox.width / 2
        in_corridor = 0.28 <= cx <= 0.72
        if not in_corridor:
            return PathRelevance.LOW if band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR} else PathRelevance.NONE
        if band == DistanceBand.VERY_NEAR:
            return PathRelevance.HIGH
        if band == DistanceBand.NEAR:
            return PathRelevance.HIGH if position == HorizontalPosition.CENTER else PathRelevance.MEDIUM
        if band == DistanceBand.MID:
            return PathRelevance.MEDIUM if position == HorizontalPosition.CENTER else PathRelevance.LOW
        return PathRelevance.LOW
