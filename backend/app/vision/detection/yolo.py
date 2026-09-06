from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.config import Settings
from app.vision.interfaces import ObjectDetector, RawDetection

logger = logging.getLogger(__name__)


# Class-calibrated minimum confidence thresholds.
# Small personal items (phones, bottles, cups) have lower feature activation
# in full-scene captures, so a rigid 0.35 threshold drops them.
CLASS_CONFIDENCE_OVERRIDES: dict[str, float] = {
    "cell phone": 0.20,
    "phone": 0.20,
    "bottle": 0.22,
    "cup": 0.22,
    "mouse": 0.20,
    "remote": 0.22,
    "book": 0.24,
    "laptop": 0.25,
    "backpack": 0.28,
    "handbag": 0.28,
    "suitcase": 0.28,
}


class YOLODetector(ObjectDetector):
    """Ultralytics YOLO closed-set detector with class-calibrated confidence."""

    def __init__(
        self,
        settings: Settings,
        confidence: float | None = None,
        enable_overrides: bool = True,
    ):
        from ultralytics import YOLO

        weights = Path(settings.yolo_model)
        if not weights.is_file():
            weights = settings.weights_dir / Path(settings.yolo_model).name
        self._model = YOLO(str(weights) if weights.is_file() else settings.yolo_model)
        self._confidence = confidence if confidence is not None else settings.yolo_confidence
        self._device = settings.device
        self._enable_overrides = enable_overrides
        logger.info("yolo_loaded", extra={"model": settings.yolo_model, "device": settings.device})

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        # Predict with lower baseline threshold to capture small candidate objects if overrides enabled
        scan_conf = min(0.18, self._confidence) if self._enable_overrides else self._confidence
        results = self._model.predict(
            image,
            conf=scan_conf,
            device=self._device,
            verbose=False,
            imgsz=640,
        )
        detections: list[RawDetection] = []
        if not results or results[0].boxes is None:
            return detections
        result = results[0]
        names = result.names
        for box in result.boxes:
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, cls_id))
            target_conf = (
                CLASS_CONFIDENCE_OVERRIDES.get(class_name.lower(), self._confidence)
                if self._enable_overrides
                else self._confidence
            )
            if conf < target_conf:
                continue

            xyxy = box.xyxy[0].tolist()
            detections.append(
                RawDetection(
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    track_id=int(box.id[0].item()) if box.id is not None else None,
                )
            )
        return detections

    def track(self, image: np.ndarray) -> list[RawDetection]:
        scan_conf = min(0.18, self._confidence)
        results = self._model.track(
            image,
            conf=scan_conf,
            device=self._device,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
        )
        detections: list[RawDetection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, cls_id))
            conf = float(box.conf[0].item())

            required_conf = CLASS_CONFIDENCE_OVERRIDES.get(class_name.lower(), self._confidence)
            if conf < required_conf:
                continue

            xyxy = box.xyxy[0].tolist()
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())
            detections.append(
                RawDetection(
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    track_id=track_id,
                )
            )
        return detections


# Alias for explicit multi-model detector architecture
COCOObjectDetector = YOLODetector


class YOLOWorldFinder(ObjectDetector):
    """Open-vocabulary detector used for FIND OBJECT queries."""

    def __init__(self, settings: Settings):
        from ultralytics import YOLO

        self._model = YOLO(settings.yolo_world_model)
        self._confidence = max(0.15, settings.yolo_confidence - 0.1)
        self._device = settings.device
        self._classes: list[str] = []

    def set_classes(self, classes: list[str]) -> None:
        self._classes = classes
        self._model.set_classes(classes)

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        if not self._classes:
            return []
        results = self._model.predict(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
            imgsz=640,
        )
        detections: list[RawDetection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        if result.boxes is None:
            return detections
        for box in result.boxes:
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
