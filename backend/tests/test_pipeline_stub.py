from __future__ import annotations

import numpy as np

from app.config import Settings
from app.ocr.provider import structure_ocr
from app.reasoning.nlg import Intent, IntentParser, ResponseGenerator
from app.vision.depth.depth_anything import UnavailableDepthEstimator
from app.vision.detection.stub import StubDetector
from app.vision.hazards.prioritizer import EventPrioritizer
from app.vision.interfaces import RawDetection
from app.vision.path.analyzer import PathAnalyzer
from app.vision.pipeline import VisionPipeline, VisionRuntime
from app.vision.scene.understander import SceneUnderstander
from app.vision.spatial.engine import SpatialReasoningEngine
from app.vision.specialized.detector import HeuristicSpecializedDetector
from app.vision.tracking.bytetrack import ByteTrackTracker


def _runtime(detections: list[RawDetection], depth: np.ndarray | None = None) -> VisionRuntime:
    settings = Settings(enable_depth=True, enable_ocr=False, enable_yolo_world=False, enable_model_warmup=False)

    class FixedDepth(UnavailableDepthEstimator):
        def estimate(self, image: np.ndarray) -> np.ndarray:
            if depth is not None:
                return depth
            h, w = image.shape[:2]
            ramp = np.linspace(0.2, 1.0, w, dtype=np.float32)
            return np.tile(ramp, (h, 1))

    return VisionRuntime(
        detector=StubDetector(detections),
        depth=FixedDepth(),
        tracker=ByteTrackTracker(),
        spatial=SpatialReasoningEngine(settings),
        path=PathAnalyzer(),
        scene=SceneUnderstander(),
        hazards=EventPrioritizer(),
        nlg=ResponseGenerator(),
        intents=IntentParser(),
        ocr=None,
        finder=None,
        specialized=HeuristicSpecializedDetector(),
        settings=settings,
    )


def test_pipeline_person_and_chair_spatial_language() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        RawDetection("person", 0.91, (20, 80, 140, 400)),
        RawDetection("chair", 0.88, (250, 180, 400, 420)),
    ]
    depth = np.ones((480, 640), dtype=np.float32) * 8
    depth[:, :180] = 2
    depth[:, 240:420] = 3
    pipeline = VisionPipeline(_runtime(detections, depth))
    result = pipeline.analyze(image, Intent(mode="look", question="Look."))
    text = result.answer.lower()
    assert "person" in text
    assert "chair" in text
    assert "2.37" not in result.answer
    assert "cross now" not in text
    assert result.objects[0].position.value in {"left", "center", "right"}
    assert result.objects[0].distance_range != "unknown" or "depth" in result.warnings


def test_pipeline_find_missing_target() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    pipeline = VisionPipeline(_runtime([]))
    result = pipeline.analyze(image, Intent(mode="find", question="Find my keys", target="keys"))
    assert "don't currently see" in result.answer.lower()


def test_ocr_structure_extracts_total() -> None:
    from app.vision.interfaces import OCRLine
    from app.vision.types import BoundingBox

    lines = [
        OCRLine(text="Coffee 3.50", confidence=0.9, bbox=BoundingBox(x=0.1, y=0.1, width=0.5, height=0.1)),
        OCRLine(text="Total 3.50", confidence=0.92, bbox=BoundingBox(x=0.1, y=0.3, width=0.5, height=0.1)),
    ]
    structured = structure_ocr("Coffee 3.50\nTotal 3.50", lines)
    assert structured["totals"]
    assert "3.50" in structured["price_like"]
