from __future__ import annotations

import logging

import numpy as np

from app.vision.interfaces import RawDetection, SpecializedDetector

logger = logging.getLogger(__name__)

# These classes are architected here so they can be filled by a fine-tuned
# detector later. The COCO pretrained YOLO is NOT treated as an accessibility
# detector even if a name collides (e.g. there is no reliable "stairs" class).
TARGET_CLASSES = [
    "stairs",
    "ramp",
    "curb",
    "pothole",
    "uneven_surface",
    "blocked_pathway",
    "door",
    "elevator",
    "pedestrian_crossing",
    "traffic_signal",
    "tactile_paving",
]


class HeuristicSpecializedDetector(SpecializedDetector):
    """Conservative starter: maps a few COCO proxies with low confidence.

    `door` is not in COCO. `traffic light` and `stop sign` are weak proxies for
    crossing context only. Stairs/ramps/curbs remain unsupported until a
    fine-tuned model is evaluated.
    """

    PROXY = {
        "traffic light": ("traffic_signal", 0.4),
        "stop sign": ("pedestrian_crossing", 0.35),
    }

    def supported_classes(self) -> list[str]:
        return ["traffic_signal", "pedestrian_crossing"]

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        return []

    def from_general_detections(self, detections: list[RawDetection]) -> list[RawDetection]:
        specialized: list[RawDetection] = []
        for det in detections:
            proxy = self.PROXY.get(det.class_name.lower())
            if not proxy:
                continue
            name, cap = proxy
            specialized.append(
                RawDetection(
                    class_name=name,
                    confidence=min(det.confidence, cap),
                    bbox_xyxy=det.bbox_xyxy,
                    track_id=det.track_id,
                )
            )
        return specialized


class FineTunedSpecializedDetector(SpecializedDetector):
    """Loaded only when a trained accessibility weight file exists."""

    def __init__(self, weights_path: str, confidence: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(weights_path)
        self._confidence = confidence

    def supported_classes(self) -> list[str]:
        return TARGET_CLASSES

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(image, conf=self._confidence, verbose=False)
        detections: list[RawDetection] = []
        if not results or results[0].boxes is None:
            return detections
        names = results[0].names
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            detections.append(
                RawDetection(
                    class_name=str(names.get(cls_id, cls_id)),
                    confidence=float(box.conf[0].item()),
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )
        return detections
