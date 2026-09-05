from app.reasoning.nlg import IntentParser, ResponseGenerator
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


def test_look_does_not_list_everything() -> None:
    objects = [
        SceneObject(
            id=1,
            type="person",
            confidence=0.91,
            bbox=BoundingBox(x=0.1, y=0.2, width=0.2, height=0.5),
            position=HorizontalPosition.LEFT,
            distance_range="two to three meters",
            distance_band=DistanceBand.MID,
            movement=Movement.APPROACHING,
            path_relevance=PathRelevance.LOW,
        ),
        SceneObject(
            id=2,
            type="chair",
            confidence=0.84,
            bbox=BoundingBox(x=0.4, y=0.4, width=0.2, height=0.3),
            position=HorizontalPosition.CENTER,
            distance_range="one to two meters",
            distance_band=DistanceBand.NEAR,
            path_relevance=PathRelevance.HIGH,
        ),
        SceneObject(
            id=3,
            type="bottle",
            confidence=0.4,
            bbox=BoundingBox(x=0.8, y=0.7, width=0.05, height=0.1),
            position=HorizontalPosition.RIGHT,
            distance_range="more than three meters",
            distance_band=DistanceBand.FAR,
            path_relevance=PathRelevance.NONE,
        ),
    ]
    path = PathAnalysis(
        path_status="partially_blocked",
        obstacles=[
            PathObstacle(
                object_id=2,
                type="chair",
                distance_range="one to two meters",
                position=HorizontalPosition.CENTER,
                confidence=0.84,
            )
        ],
    )
    text = ResponseGenerator().look(objects, path, "room")
    assert "chair" in text.lower()
    assert "2.37" not in text
    assert text.lower().count("bottle") == 0


def test_confidence_hedging() -> None:
    gen = ResponseGenerator()
    low = SceneObject(
        id=1,
        type="stairs",
        confidence=0.32,
        bbox=BoundingBox(x=0.4, y=0.3, width=0.3, height=0.4),
        position=HorizontalPosition.CENTER,
        distance_range="two to three meters",
        distance_band=DistanceBand.MID,
    )
    text = gen.describe_object(low)
    assert "may be detecting" in text


def test_approaching_person_from_left() -> None:
    obj = SceneObject(
        id=1,
        type="person",
        confidence=0.91,
        bbox=BoundingBox(x=0.1, y=0.2, width=0.2, height=0.5),
        position=HorizontalPosition.LEFT,
        distance_range="two to three meters",
        distance_band=DistanceBand.MID,
        movement=Movement.APPROACHING,
        path_relevance=PathRelevance.LOW,
    )
    text = ResponseGenerator().describe_object(obj)
    assert "approaching from your left" in text.lower()


def test_never_says_safe_to_cross() -> None:
    parser = IntentParser()
    intent = parser.parse("is it safe to cross")
    assert intent.mode == "ask"
    objects = [
        SceneObject(
            id=1,
            type="traffic light",
            confidence=0.8,
            bbox=BoundingBox(x=0.5, y=0.2, width=0.1, height=0.2),
            position=HorizontalPosition.CENTER,
            distance_range="two to three meters",
            distance_band=DistanceBand.MID,
        )
    ]
    text = ResponseGenerator().ask("is the path clear", objects, PathAnalysis(path_status="clear"), "road")
    assert "safe to walk" in text or "cannot guarantee" in text
    assert "cross now" not in text.lower()


def test_find_missing_object() -> None:
    text = ResponseGenerator().find("keys", [])
    assert "don't currently see" in text.lower()
    assert "right" in text.lower()
