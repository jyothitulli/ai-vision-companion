from __future__ import annotations

import numpy as np

from app.vision.interfaces import ObjectDetector, RawDetection


class StubDetector(ObjectDetector):
    """Deterministic detector for tests. Does not load neural weights."""

    def __init__(self, detections: list[RawDetection] | None = None):
        self._detections = detections or []

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        _ = image
        return list(self._detections)
