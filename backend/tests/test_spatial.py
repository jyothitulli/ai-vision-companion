import numpy as np
import pytest

from app.config import Settings
from app.vision.spatial.engine import SpatialReasoningEngine
from app.vision.types import BoundingBox, DistanceBand, PathRelevance, SceneObject


@pytest.fixture
def spatial() -> SpatialReasoningEngine:
    return SpatialReasoningEngine(Settings(depth_higher_means_farther=True, depth_near_threshold=0.35, depth_mid_threshold=0.65))


def test_position_left_center_right(spatial: SpatialReasoningEngine) -> None:
    assert spatial.classify_position(0.1).value == "left"
    assert spatial.classify_position(0.5).value == "center"
    assert spatial.classify_position(0.9).value == "right"


def test_distance_bands_from_relative_depth(spatial: SpatialReasoningEngine) -> None:
    depth = np.ones((100, 100), dtype=np.float32) * 10
    depth[:, :30] = 1  # closer if higher means farther? 1 is nearer
    obj = SceneObject(
        id=1,
        type="chair",
        confidence=0.9,
        bbox=BoundingBox(x=0.05, y=0.2, width=0.15, height=0.4),
    )
    band, estimate, spoken = spatial.classify_distance(obj, depth)
    assert band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR, DistanceBand.MID}
    assert estimate is not None
    assert "meter" in spoken
    assert "2.37" not in spoken


def test_unknown_distance_without_depth(spatial: SpatialReasoningEngine) -> None:
    obj = SceneObject(id=1, type="person", confidence=0.9, bbox=BoundingBox(x=0.4, y=0.2, width=0.2, height=0.5))
    band, estimate, spoken = spatial.classify_distance(obj, None)
    assert band == DistanceBand.UNKNOWN
    assert estimate is None
    assert spoken == "an unknown distance"


def test_path_relevance_center_near(spatial: SpatialReasoningEngine) -> None:
    obj = SceneObject(id=1, type="chair", confidence=0.9, bbox=BoundingBox(x=0.4, y=0.3, width=0.2, height=0.4))
    relevance = spatial.classify_path_relevance(obj, spatial.classify_position(0.5), DistanceBand.NEAR)
    assert relevance in {PathRelevance.HIGH, PathRelevance.MEDIUM}
