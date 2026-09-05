import time

from app.vision.hazards.prioritizer import EventPrioritizer
from app.vision.types import (
    BoundingBox,
    DistanceBand,
    HorizontalPosition,
    Movement,
    PathAnalysis,
    PathObstacle,
    PathRelevance,
    Priority,
    SceneObject,
)


def test_approaching_vehicle_is_p0() -> None:
    obj = SceneObject(
        id=1,
        type="car",
        confidence=0.88,
        bbox=BoundingBox(x=0.4, y=0.3, width=0.3, height=0.3),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.MID,
        distance_range="two to three meters",
        movement=Movement.APPROACHING,
        path_relevance=PathRelevance.MEDIUM,
        track_id=9,
    )
    events = EventPrioritizer(cooldown_s=8).prioritize( [obj], PathAnalysis(path_status="clear"), now=time.monotonic())
    assert any(event.priority == Priority.P0 for event in events)


def test_cooldown_suppresses_duplicate_announcement() -> None:
    prioritizer = EventPrioritizer(cooldown_s=8)
    obj = SceneObject(
        id=3,
        type="chair",
        confidence=0.9,
        bbox=BoundingBox(x=0.4, y=0.4, width=0.2, height=0.3),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.HIGH,
        track_id=3,
    )
    path = PathAnalysis(
        path_status="partially_blocked",
        obstacles=[PathObstacle(object_id=3, type="chair", distance_range="one to two meters", position=HorizontalPosition.CENTER, confidence=0.9)],
    )
    first = prioritizer.prioritize([obj], path, now=100.0)
    second = prioritizer.prioritize([obj], path, now=101.0)
    assert any(event.should_announce for event in first)
    assert all(not event.should_announce or event.cooldown_key not in {e.cooldown_key for e in first if e.should_announce} or not event.should_announce for event in second)
    assert not any(event.should_announce and event.cooldown_key == "path_obstruction:chair" for event in second)
