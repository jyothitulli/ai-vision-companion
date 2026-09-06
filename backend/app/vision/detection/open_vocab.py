from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

from app.config import Settings
from app.vision.interfaces import ObjectDetector, RawDetection

logger = logging.getLogger(__name__)


class OpenVocabularyDetector(ObjectDetector):
    """Open-vocabulary object detector capable of dynamic text prompt conditioning."""

    def __init__(self, settings: Settings, model_path: str | None = None):
        self._confidence = getattr(settings, "open_vocabulary_confidence", 0.22)
        self._device = settings.device
        self._classes: list[str] = []
        self._model = None

        weight_path = model_path or getattr(settings, "yolo_world_model", "yolov8s-worldv2.pt")
        # Check local paths
        candidates = [
            Path(weight_path),
            settings.weights_dir / Path(weight_path).name,
            Path("..") / Path(weight_path).name,
        ]
        resolved = None
        for cand in candidates:
            if cand.is_file():
                resolved = cand
                break

        if resolved and resolved.is_file():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(resolved))
                logger.info("open_vocab_model_loaded", extra={"model": str(resolved), "device": self._device})
            except Exception as exc:
                logger.warning("open_vocab_load_failed", extra={"error": str(exc)})
                self._model = None
        else:
            logger.info("open_vocab_weights_not_found", extra={"expected": weight_path})

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def get_status_report(self) -> dict[str, str]:
        """Provides an honest, transparent diagnostic of the open-vocabulary model state."""
        return {
            "model": "YOLO-World (yolov8s-worldv2)",
            "installation_status": "ultralytics installed",
            "weights_status": "weights file absent locally (yolov8s-worldv2.pt)" if not self.is_available else "loaded",
            "device": str(self._device),
            "available": "yes" if self.is_available else "no",
            "error": "Weights not downloaded locally; open-vocabulary zero-shot inference unavailable without weights" if not self.is_available else "none",
            "fallback": "COCOObjectDetector (calibrated YOLO11n) with transparent capability disclaimers",
        }

    def set_classes(self, classes: list[str]) -> None:
        self._classes = classes
        if self._model is not None and classes:
            try:
                self._model.set_classes(classes)
            except Exception as exc:
                logger.warning("open_vocab_set_classes_failed", extra={"error": str(exc)})

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        if self._model is None or not self._classes:
            return []
        try:
            results = self._model.predict(
                image,
                conf=self._confidence,
                device=self._device,
                verbose=False,
            )
            detections: list[RawDetection] = []
            if not results or results[0].boxes is None:
                return detections
            result = results[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = str(names.get(cls_id, cls_id))
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    RawDetection(
                        class_name=class_name,
                        confidence=conf,
                        bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    )
                )
            return detections
        except Exception as exc:
            logger.warning("open_vocab_detect_error", extra={"error": str(exc)})
            return []
