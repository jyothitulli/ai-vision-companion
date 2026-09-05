from app.vision.path.analyzer import PathAnalyzer
from app.vision.types import BoundingBox, DistanceBand, HorizontalPosition, PathRelevance, SceneObject


def _obj(**kwargs) -> SceneObject:
    defaults = dict(
        id=1,
        type="chair",
        confidence=0.9,
        bbox=BoundingBox(x=0.4, y=0.3, width=0.2, height=0.4),
        position=HorizontalPosition.CENTER,
        distance_range="one to two meters",
        distance_band=DistanceBand.NEAR,
        path_relevance=PathRelevance.HIGH,
    )
    defaults.update(kwargs)
    return SceneObject(**defaults)


def test_chair_is_path_obstruction() -> None:
    analysis = PathAnalyzer().analyze([_obj()])
    assert analysis.path_status in {"partially_blocked", "blocked"}
    assert analysis.obstacles[0].type == "chair"


def test_far_bottle_is_not_treated_as_blockage() -> None:
    analysis = PathAnalyzer().analyze(
        [
            _obj(
                type="bottle",
                path_relevance=PathRelevance.LOW,
                distance_band=DistanceBand.FAR,
                distance_range="more than three meters",
            )
        ]
    )
    assert analysis.path_status == "clear"
    assert analysis.obstacles == []


def test_empty_objects_uncertain() -> None:
    analysis = PathAnalyzer().analyze([])
    assert analysis.path_status == "uncertain"
