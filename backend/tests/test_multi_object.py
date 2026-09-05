import pytest
from app.reasoning.nlg import IntentParser, ResponseGenerator
from app.vision.detection.fusion import DetectionFusion, calculate_iou
from app.vision.interfaces import RawDetection
from app.vision.spatial.engine import SpatialReasoningEngine
from app.config import Settings
from app.vision.types import (
    BoundingBox,
    DistanceBand,
    HorizontalPosition,
    Movement,
    PathAnalysis,
    PathObstacle,
    PathRelevance,
    SceneObject,
)


def test_calculate_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert calculate_iou(box1, box2) == 1.0

    box3 = (10.0, 10.0, 20.0, 20.0)
    assert calculate_iou(box1, box3) == 0.0

    box4 = (5.0, 0.0, 15.0, 10.0)
    assert 0.3 < calculate_iou(box1, box4) < 0.4


def test_detection_fusion_duplicate_suppression():
    fusion = DetectionFusion(iou_threshold=0.45)
    # Simulate YOLO detection and Open-Vocab detection both detecting the same chair
    yolo_det = RawDetection(class_name="chair", confidence=0.88, bbox_xyxy=(100, 100, 200, 300))
    open_det = RawDetection(class_name="chair", confidence=0.72, bbox_xyxy=(105, 98, 202, 305))
    table_det = RawDetection(class_name="table", confidence=0.91, bbox_xyxy=(300, 150, 500, 400))

    fused = fusion.fuse([[yolo_det, table_det], [open_det]])
    assert len(fused) == 2
    # Chair should retain higher confidence (0.88)
    chair = [d for d in fused if d.class_name == "chair"][0]
    assert chair.confidence == 0.88


def test_spatial_relationships_surface_support():
    spatial = SpatialReasoningEngine(Settings(depth_higher_means_farther=True, depth_near_threshold=0.35, depth_mid_threshold=0.65))
    table = SceneObject(
        id=1,
        type="dining table",
        confidence=0.92,
        bbox=BoundingBox(x=0.2, y=0.4, width=0.6, height=0.5),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )
    laptop = SceneObject(
        id=2,
        type="laptop",
        confidence=0.88,
        bbox=BoundingBox(x=0.3, y=0.35, width=0.2, height=0.2),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )
    phone = SceneObject(
        id=3,
        type="cell phone",
        confidence=0.80,
        bbox=BoundingBox(x=0.6, y=0.42, width=0.08, height=0.12),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )

    enriched = spatial.enrich([table, laptop, phone], depth_map=None, image_shape=(480, 640, 3))
    laptop_enriched = [o for o in enriched if o.type == "laptop"][0]
    phone_enriched = [o for o in enriched if o.type == "cell phone"][0]

    assert any("on the dining table" in r for r in laptop_enriched.relationships)
    assert any("on the dining table" in r for r in phone_enriched.relationships)


def test_multi_object_grouped_voice_summary():
    gen = ResponseGenerator()
    table = SceneObject(
        id=1,
        type="table",
        confidence=0.92,
        bbox=BoundingBox(x=0.2, y=0.4, width=0.6, height=0.5),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )
    laptop = SceneObject(
        id=2,
        type="laptop",
        confidence=0.88,
        bbox=BoundingBox(x=0.3, y=0.35, width=0.2, height=0.2),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        relationships=["on the table"],
    )
    phone = SceneObject(
        id=3,
        type="phone",
        confidence=0.80,
        bbox=BoundingBox(x=0.6, y=0.42, width=0.08, height=0.12),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        relationships=["on the table"],
    )
    chair = SceneObject(
        id=4,
        type="chair",
        confidence=0.85,
        bbox=BoundingBox(x=0.05, y=0.3, width=0.2, height=0.4),
        position=HorizontalPosition.LEFT,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )

    path = PathAnalysis(path_status="clear")
    text = gen.look([chair, table, laptop, phone], path, "room.")
    assert "chair" in text.lower()
    assert "on the table" in text.lower()
    assert "laptop" in text.lower()
    assert "phone" in text.lower()


def test_ask_objects_on_surface():
    gen = ResponseGenerator()
    laptop = SceneObject(
        id=2,
        type="laptop",
        confidence=0.88,
        bbox=BoundingBox(x=0.3, y=0.35, width=0.2, height=0.2),
        relationships=["on the table"],
    )
    phone = SceneObject(
        id=3,
        type="phone",
        confidence=0.80,
        bbox=BoundingBox(x=0.6, y=0.42, width=0.08, height=0.12),
        relationships=["on the table"],
    )

    text = gen.ask("what is on the table?", [laptop, phone], PathAnalysis(path_status="clear"), "room")
    assert "on the table" in text.lower()
    assert "laptop" in text.lower()
    assert "phone" in text.lower()


def test_ask_specific_object_location():
    gen = ResponseGenerator()
    bottle = SceneObject(
        id=1,
        type="bottle",
        confidence=0.86,
        bbox=BoundingBox(x=0.7, y=0.4, width=0.1, height=0.2),
        position=HorizontalPosition.RIGHT,
        distance_range="one to two meters",
        distance_band=DistanceBand.NEAR,
    )
    text = gen.ask("where is the bottle?", [bottle], PathAnalysis(path_status="clear"), "room")
    assert "bottle" in text.lower()
    assert "right" in text.lower()
    assert "one to two meters" in text.lower()


def test_ask_directional_queries():
    gen = ResponseGenerator()
    left_person = SceneObject(
        id=1,
        type="person",
        confidence=0.90,
        bbox=BoundingBox(x=0.1, y=0.2, width=0.2, height=0.6),
        position=HorizontalPosition.LEFT,
        distance_range="two to three meters",
        distance_band=DistanceBand.MID,
    )
    text = gen.ask("what is to my left?", [left_person], PathAnalysis(path_status="clear"), "room")
    assert "left" in text.lower()
    assert "person" in text.lower()


def test_find_mode_single_and_multiple_targets():
    gen = ResponseGenerator()
    b1 = SceneObject(
        id=1,
        type="bottle",
        confidence=0.89,
        bbox=BoundingBox(x=0.2, y=0.4, width=0.1, height=0.2),
        position=HorizontalPosition.LEFT,
        distance_range="one meter",
    )
    b2 = SceneObject(
        id=2,
        type="bottle",
        confidence=0.82,
        bbox=BoundingBox(x=0.8, y=0.3, width=0.1, height=0.2),
        position=HorizontalPosition.RIGHT,
        distance_range="three meters",
    )

    # Multiple targets
    text_multi = gen.find("bottle", [b1, b2])
    assert "2 bottles" in text_multi or "two bottles" in text_multi.lower()
    assert "left" in text_multi.lower()
    assert "right" in text_multi.lower()

    # Single target
    text_single = gen.find("bottle", [b1])
    assert "see your bottle" in text_single.lower()
    assert "left" in text_single.lower()


def test_unsupported_micro_features_honest_reporting():
    gen = ResponseGenerator()
    text_handle = gen.ask("where is the door handle?", [], PathAnalysis(path_status="clear"), "room")
    assert "cannot be reliably recognized" in text_handle.lower() or "not reliably" in text_handle.lower()

    text_button = gen.ask("where is the elevator button?", [], PathAnalysis(path_status="clear"), "room")
    assert "cannot be reliably recognized" in text_button.lower() or "not reliably" in text_button.lower()
