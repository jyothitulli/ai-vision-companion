from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import numpy as np

from app.vision.interfaces import ObjectTracker, RawDetection
from app.vision.types import BoundingBox, Movement, SceneObject


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ByteTrackTracker(ObjectTracker):
    """ByteTrack-style association with Ultralytics IDs when present.

    Ultralytics `model.track(tracker='bytetrack.yaml')` is used by the pipeline
    when available. This class also keeps a lightweight IoU fallback so unit
    tests and CPU environments remain deterministic.
    """

    def __init__(self, max_history: int = 12, iou_threshold: float = 0.4):
        self._max_history = max_history
        self._iou_threshold = iou_threshold
        self._next_id = 1
        self._history: dict[int, deque[tuple[float, tuple[float, float, float, float], str]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._class_by_id: dict[int, str] = {}

    def update(self, detections: list[RawDetection], timestamp_s: float) -> list[SceneObject]:
        assigned: list[SceneObject] = []
        used_tracks: set[int] = set()
        for det in detections:
            track_id = det.track_id
            if track_id is None:
                track_id = self._match_or_create(det, used_tracks)
            det.track_id = track_id
            used_tracks.add(track_id)
            self._history[track_id].append((timestamp_s, det.bbox_xyxy, det.class_name))
            self._class_by_id[track_id] = det.class_name
            assigned.append(
                SceneObject(
                    id=track_id,
                    type=det.class_name,
                    confidence=det.confidence,
                    bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    track_id=track_id,
                )
            )
        return assigned

    def _match_or_create(self, det: RawDetection, used: set[int]) -> int:
        best_id = None
        best_iou = self._iou_threshold
        for track_id, history in self._history.items():
            if track_id in used or not history:
                continue
            if self._class_by_id.get(track_id) != det.class_name:
                continue
            iou = _iou(history[-1][1], det.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_id = track_id
        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
        return best_id

    def movement_for(self, track_id: int) -> tuple[Movement, Optional[str]]:
        history = self._history.get(track_id)
        if not history or len(history) < 2:
            return Movement.UNKNOWN, None
        first = history[0]
        last = history[-1]
        dt = last[0] - first[0]
        if dt <= 0:
            return Movement.UNKNOWN, None
        # Area increase is a proxy for approaching the camera; this is not calibrated speed.
        def _area(box: tuple[float, float, float, float]) -> float:
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

        area_delta = _area(last[1]) - _area(first[1])
        cx_first = (first[1][0] + first[1][2]) / 2
        cx_last = (last[1][0] + last[1][2]) / 2
        dx = cx_last - cx_first
        rel_area = area_delta / max(_area(first[1]), 1.0)
        if abs(rel_area) < 0.08 and abs(dx) < 8:
            return Movement.STATIONARY, None
        direction = "toward_user" if rel_area > 0.12 else "away_from_user" if rel_area < -0.12 else None
        if rel_area > 0.12:
            return Movement.APPROACHING, direction
        if rel_area < -0.12:
            return Movement.RECEDING, direction
        if dx > 12:
            return Movement.MOVING, "right"
        if dx < -12:
            return Movement.MOVING, "left"
        return Movement.MOVING, direction


def normalize_boxes(objects: list[SceneObject], image: np.ndarray) -> list[SceneObject]:
    h, w = image.shape[:2]
    normalized: list[SceneObject] = []
    for obj in objects:
        # Incoming bbox may still be in pixels if produced by tracker fallback.
        x = obj.bbox.x
        y = obj.bbox.y
        width = obj.bbox.width
        height = obj.bbox.height
        if x > 1.5 or y > 1.5 or width > 1.5 or height > 1.5:
            x, y, width, height = x / w, y / h, width / w, height / h
        normalized.append(
            obj.model_copy(
                update={
                    "bbox": BoundingBox(
                        x=float(np.clip(x, 0, 1)),
                        y=float(np.clip(y, 0, 1)),
                        width=float(np.clip(width, 0, 1)),
                        height=float(np.clip(height, 0, 1)),
                    )
                }
            )
        )
    return normalized
