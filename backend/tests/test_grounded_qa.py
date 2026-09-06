"""Comprehensive Grounded Question-Answering and FIND Mode Benchmark Test Suite."""

from __future__ import annotations

import pytest

from app.reasoning.nlg import ResponseGenerator
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


@pytest.fixture
def sample_scene() -> list[SceneObject]:
    """A realistic desk & office scene with multiple interacting objects."""
    table = SceneObject(
        id=1,
        type="table",
        confidence=0.91,
        bbox=BoundingBox(x=0.25, y=0.35, width=0.55, height=0.55),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.MEDIUM,
    )
    laptop = SceneObject(
        id=2,
        type="laptop",
        confidence=0.88,
        bbox=BoundingBox(x=0.30, y=0.38, width=0.18, height=0.20),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
        relationships=["on the table"],
    )
    phone = SceneObject(
        id=3,
        type="cell phone",
        confidence=0.78,
        bbox=BoundingBox(x=0.52, y=0.42, width=0.08, height=0.12),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
        relationships=["on the table", "to the right of laptop"],
    )
    bottle = SceneObject(
        id=4,
        type="bottle",
        confidence=0.82,
        bbox=BoundingBox(x=0.62, y=0.36, width=0.07, height=0.18),
        position=HorizontalPosition.RIGHT,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
        relationships=["on the table"],
    )
    chair = SceneObject(
        id=5,
        type="chair",
        confidence=0.85,
        bbox=BoundingBox(x=0.05, y=0.25, width=0.18, height=0.60),
        position=HorizontalPosition.FAR_LEFT,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
    )
    person = SceneObject(
        id=6,
        type="person",
        confidence=0.94,
        bbox=BoundingBox(x=0.02, y=0.15, width=0.20, height=0.75),
        position=HorizontalPosition.LEFT,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
    )
    return [table, laptop, phone, bottle, chair, person]


@pytest.fixture
def path_clear() -> PathAnalysis:
    return PathAnalysis(path_status="clear", obstacles=[])


@pytest.fixture
def path_blocked() -> PathAnalysis:
    return PathAnalysis(
        path_status="partially_blocked",
        obstacles=[
            PathObstacle(
                object_id=1,
                type="table",
                distance_range="one to two meters",
                position=HorizontalPosition.CENTER,
                confidence=0.91,
            )
        ],
    )


# 1. GENERAL Q&A
def test_qa_general(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res1 = nlg.ask("What do you see?", sample_scene, path_clear, "office room")
    assert "table" in res1.lower()
    assert ("laptop" in res1.lower() or "chair" in res1.lower())

    res2 = nlg.ask("What is in front of me?", sample_scene, path_clear, "office room")
    assert len(res2) > 10


# 2. LOCATION Q&A
def test_qa_location(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_phone = nlg.ask("Where is the phone?", sample_scene, path_clear, "office room")
    assert "phone" in res_phone.lower() or "cell phone" in res_phone.lower()
    assert "on the table" in res_phone.lower() or "ahead" in res_phone.lower()

    res_bottle = nlg.ask("Where is the bottle?", sample_scene, path_clear, "office room")
    assert "bottle" in res_bottle.lower()
    assert "right" in res_bottle.lower() or "ahead" in res_bottle.lower()

    res_missing = nlg.ask("Where is the door?", sample_scene, path_clear, "office room")
    assert "don't currently see a door" in res_missing.lower()


# 3. EXISTENCE Q&A
def test_qa_existence(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_person = nlg.ask("Is anyone near me?", sample_scene, path_clear, "office room")
    assert "yes" in res_person.lower()
    assert "person" in res_person.lower()

    res_phone = nlg.ask("Is there a phone?", sample_scene, path_clear, "office room")
    assert "yes" in res_phone.lower()

    res_banana = nlg.ask("Is there a banana?", sample_scene, path_clear, "office room")
    assert "no" in res_banana.lower() or "don't see" in res_banana.lower()


# 4. COUNT Q&A
def test_qa_count(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_bottles = nlg.ask("How many bottles are there?", sample_scene, path_clear, "office room")
    assert "1 bottle" in res_bottles.lower() or "see 1" in res_bottles.lower()

    res_people = nlg.ask("How many people are there?", sample_scene, path_clear, "office room")
    assert "1" in res_people.lower()

    res_objects = nlg.ask("How many objects are there?", sample_scene, path_clear, "office room")
    assert "6 notable" in res_objects.lower()


# 5. REGION Q&A
def test_qa_region(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_left = nlg.ask("What is to my left?", sample_scene, path_clear, "office room")
    assert "on your left" in res_left.lower()

    res_right = nlg.ask("What is to my right?", sample_scene, path_clear, "office room")
    assert "on your right" in res_right.lower()
    assert "bottle" in res_right.lower()


# 6. RELATIONSHIPS Q&A
def test_qa_relationships(sample_scene: list[SceneObject], path_clear: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_table = nlg.ask("What is on the table?", sample_scene, path_clear, "office room")
    assert "on the table" in res_table.lower()
    assert "laptop" in res_table.lower() or "phone" in res_table.lower()

    res_laptop = nlg.ask("What is beside the laptop?", sample_scene, path_clear, "office room")
    assert "laptop" in res_laptop.lower() or "near the laptop" in res_laptop.lower()


# 7. ACCESSIBILITY Q&A
def test_qa_accessibility(sample_scene: list[SceneObject], path_blocked: PathAnalysis) -> None:
    nlg = ResponseGenerator()
    res_stairs = nlg.ask("Where are the stairs?", sample_scene, path_blocked, "hallway")
    assert "don't currently see stairs" in res_stairs.lower()
    assert "line of sight" in res_stairs.lower()

    res_path = nlg.ask("Is the path blocked?", sample_scene, path_blocked, "hallway")
    assert "table" in res_path.lower()
    assert "obstruct your path" in res_path.lower()

    res_button = nlg.ask("Where is the elevator button?", sample_scene, path_blocked, "hallway")
    assert "cannot be reliably recognized" in res_button.lower()


# 8. FIND MODE
def test_find_mode(sample_scene: list[SceneObject]) -> None:
    nlg = ResponseGenerator()
    # Single target
    res_phone = nlg.find("phone", sample_scene)
    assert "i see your phone" in res_phone.lower()
    assert "on the table" in res_phone.lower() or "on a table" in res_phone.lower()

    # Missing target with bounded camera advice
    res_missing = nlg.find("keys", sample_scene)
    assert "don't currently see your keys" in res_missing.lower()
    assert "moving the camera slowly" in res_missing.lower()
    assert "outside the camera frame" in res_missing.lower()
