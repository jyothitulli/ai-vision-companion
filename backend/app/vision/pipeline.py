from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import Settings
from app.ocr.provider import OCRProvider, build_ocr_provider
from app.reasoning.nlg import Intent, IntentParser, ResponseGenerator
from app.vision.depth.depth_anything import DepthAnythingEstimator, UnavailableDepthEstimator
from app.vision.detection.fusion import DetectionFusion
from app.vision.detection.open_vocab import OpenVocabularyDetector
from app.vision.detection.yolo import COCOObjectDetector, YOLODetector, YOLOWorldFinder
from app.vision.hazards.prioritizer import EventPrioritizer
from app.vision.interfaces import DepthEstimator, ObjectDetector, OCRDocument
from app.vision.path.analyzer import PathAnalyzer
from app.vision.scene.understander import SceneUnderstander
from app.vision.spatial.engine import SpatialReasoningEngine
from app.vision.specialized.detector import HeuristicSpecializedDetector
from app.vision.tracking.bytetrack import ByteTrackTracker, normalize_boxes
from app.vision.types import BoundingBox, PipelineResult, SceneObject
from app.vision.interfaces import RawDetection


MAX_SIDE = 960


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid_image")
    return image


def raw_to_objects(detections: list[RawDetection], image: np.ndarray) -> list[SceneObject]:
    h, w = image.shape[:2]
    objects: list[SceneObject] = []
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox_xyxy
        objects.append(
            SceneObject(
                id=det.track_id or index,
                type=det.class_name,
                confidence=det.confidence,
                bbox=BoundingBox(
                    x=x1 / w,
                    y=y1 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                ),
                track_id=det.track_id,
                specialized="_" in det.class_name or det.class_name in {"stairs", "ramp", "curb", "pothole", "door"},
            )
        )
    return objects


@dataclass
class VisionRuntime:
    detector: ObjectDetector
    depth: DepthEstimator
    tracker: ByteTrackTracker
    spatial: SpatialReasoningEngine
    path: PathAnalyzer
    scene: SceneUnderstander
    hazards: EventPrioritizer
    nlg: ResponseGenerator
    intents: IntentParser
    ocr: Optional[OCRProvider]
    finder: Optional[ObjectDetector]
    specialized: HeuristicSpecializedDetector
    settings: Settings
    fusion: Optional[DetectionFusion] = None

    def __post_init__(self):
        if self.fusion is None:
            self.fusion = DetectionFusion(iou_threshold=0.45)


def build_runtime(settings: Settings) -> VisionRuntime:
    detector = YOLODetector(settings)
    if settings.enable_depth:
        try:
            depth: DepthEstimator = DepthAnythingEstimator(settings)
        except Exception:
            depth = UnavailableDepthEstimator()
    else:
        depth = UnavailableDepthEstimator()
    finder = None
    if settings.enable_yolo_world:
        try:
            finder = OpenVocabularyDetector(settings)
            if not finder.is_available:
                finder = YOLOWorldFinder(settings)
        except Exception:
            finder = None
    ocr = None
    if settings.enable_ocr:
        try:
            ocr = build_ocr_provider()
        except Exception:
            ocr = None
    return VisionRuntime(
        detector=detector,
        depth=depth,
        tracker=ByteTrackTracker(),
        spatial=SpatialReasoningEngine(settings),
        path=PathAnalyzer(),
        scene=SceneUnderstander(),
        hazards=EventPrioritizer(),
        nlg=ResponseGenerator(),
        intents=IntentParser(),
        ocr=ocr,
        finder=finder,
        specialized=HeuristicSpecializedDetector(),
        fusion=DetectionFusion(iou_threshold=0.45),
        settings=settings,
    )


class VisionPipeline:
    def __init__(self, runtime: VisionRuntime):
        self.runtime = runtime

    def analyze(
        self,
        image_bgr: np.ndarray,
        intent: Intent,
        timestamp_s: Optional[float] = None,
        persist_tracks: bool = False,
    ) -> PipelineResult:
        latencies: dict[str, float] = {}
        warnings: list[str] = []
        image = preprocess_bgr(image_bgr)
        t0 = time.perf_counter()
        if persist_tracks and self.runtime.settings.enable_tracking:
            try:
                detections = self.runtime.detector.track(image)
            except Exception:
                detections = self.runtime.detector.detect(image)
        else:
            detections = self.runtime.detector.detect(image)
        latencies["detection_ms"] = (time.perf_counter() - t0) * 1000

        specialized = self.runtime.specialized.from_general_detections(detections)
        detector_outputs = [detections, specialized]

        if intent.mode == "find" and intent.target and self.runtime.finder is not None:
            t_find = time.perf_counter()
            try:
                if hasattr(self.runtime.finder, "set_classes"):
                    self.runtime.finder.set_classes([intent.target, *self._aliases(intent.target)])
                extra = self.runtime.finder.detect(image)
                detector_outputs.append(extra)
            except Exception:
                warnings.append("open_vocab_unavailable")
            latencies["open_vocab_ms"] = (time.perf_counter() - t_find) * 1000

        # Multi-model detection fusion with duplicate suppression
        detections = self.runtime.fusion.fuse(detector_outputs)

        t1 = time.perf_counter()
        depth_map = None
        if self.runtime.settings.enable_depth:
            depth_map = self.runtime.depth.estimate(image)
            if float(np.max(depth_map)) == 0.0:
                warnings.append("depth_unavailable")
        latencies["depth_ms"] = (time.perf_counter() - t1) * 1000

        ts = timestamp_s or time.time()
        self.runtime.tracker.update(detections, ts)
        objects = raw_to_objects(detections, image)
        moved: list[SceneObject] = []
        for obj in objects:
            if obj.track_id is None:
                moved.append(obj)
                continue
            movement, direction = self.runtime.tracker.movement_for(obj.track_id)
            moved.append(obj.model_copy(update={"movement": movement, "movement_direction": direction}))
        objects = moved
        objects = normalize_boxes(objects, image)
        objects = self.runtime.spatial.enrich(objects, depth_map, image.shape)

        path = self.runtime.path.analyze(objects)
        scene = self.runtime.scene.understand(objects)
        events = self.runtime.hazards.prioritize(objects, path)

        ocr_doc: OCRDocument | None = None
        if intent.mode == "read":
            if self.runtime.ocr is None:
                warnings.append("ocr_unavailable")
                answer = "I couldn't read the text. The OCR engine is unavailable."
            else:
                t_ocr = time.perf_counter()
                ocr_doc = self.runtime.ocr.read(image)
                latencies["ocr_ms"] = (time.perf_counter() - t_ocr) * 1000
                answer = self.runtime.nlg.read(
                    intent.question, ocr_doc.structured, ocr_doc.full_text, intent.ocr_focus
                )
        elif intent.mode == "find":
            answer = self.runtime.nlg.find(intent.target, objects)
        elif intent.mode == "ask":
            answer = self.runtime.nlg.ask(intent.question, objects, path, scene.description)
        else:
            answer = self.runtime.nlg.look(objects, path, scene.description)

        confidences = [obj.confidence for obj in objects] or [0.0]
        overall = float(np.mean(confidences[:5])) if objects else 0.2
        if scene.uncertain:
            overall = min(overall, scene.confidence)
        return PipelineResult(
            objects=objects,
            path=path,
            scene=scene,
            events=events,
            answer=answer,
            confidence=round(overall, 3),
            latencies_ms={key: round(value, 1) for key, value in latencies.items()},
            warnings=warnings,
            ocr_text=ocr_doc.full_text if ocr_doc else None,
        )

    def _aliases(self, target: str) -> list[str]:
        from app.reasoning.nlg import FIND_SYNONYMS

        return FIND_SYNONYMS.get(target, [target])
