from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.vision.types import BoundingBox, Movement, SceneObject


@dataclass
class RawDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: Optional[int] = None


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[RawDetection]:
        raise NotImplementedError

    def track(self, image: np.ndarray) -> list[RawDetection]:
        """Optional multi-frame association. Default is per-frame detection."""
        return self.detect(image)


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Return a 2D relative-depth map aligned to the image."""
        raise NotImplementedError


class ObjectTracker(ABC):
    @abstractmethod
    def update(self, detections: list[RawDetection], timestamp_s: float) -> list[SceneObject]:
        raise NotImplementedError

    @abstractmethod
    def movement_for(self, track_id: int) -> tuple[Movement, Optional[str]]:
        raise NotImplementedError


class OCRProvider(ABC):
    @abstractmethod
    def read(self, image: np.ndarray) -> "OCRDocument":
        raise NotImplementedError


class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class SpecializedDetector(ABC):
    """Separate from COCO YOLO. Do not pretend general YOLO covers stairs/curbs."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[RawDetection]:
        raise NotImplementedError

    @abstractmethod
    def supported_classes(self) -> list[str]:
        raise NotImplementedError


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: BoundingBox


@dataclass
class OCRDocument:
    lines: list[OCRLine] = field(default_factory=list)
    full_text: str = ""
    structured: dict = field(default_factory=dict)
