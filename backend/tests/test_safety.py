"""Automated safety validation tests.

Verifies that the NLG and response generation layer NEVER output unsafe,
overconfident, or ungrounded navigational instructions.
"""

from __future__ import annotations

import pytest

from app.reasoning.nlg import ResponseGenerator
from app.vision.types import (
    BoundingBox,
    DistanceBand,
    HazardEvent,
    HorizontalPosition,
    Movement,
    PathAnalysis,
    PathObstacle,
    PathRelevance,
    Priority,
    SceneObject,
)

PROHIBITED_UNSAFE_STRINGS = [
    "the road is safe",
    "cross now",
    "walk forward safely",
    "you can proceed",
    "it is safe to cross",
    "safe to walk",
    "all clear to go",
    "guarantee it is safe",
]


@pytest.fixture
def nlg() -> ResponseGenerator:
    return ResponseGenerator()


def assert_safe_text(text: str) -> None:
    lowered = text.lower()
    for phrase in PROHIBITED_UNSAFE_STRINGS:
        assert phrase not in lowered, f"Prohibited unsafe phrase '{phrase}' found in response: '{text}'"


def test_safety_clear_path_includes_caution(nlg: ResponseGenerator) -> None:
    path = PathAnalysis(path_status="clear", obstacles=[])
    dummy = SceneObject(
        id=1,
        type="chair",
        confidence=0.88,
        bbox=BoundingBox(x=0.8, y=0.5, width=0.15, height=0.3),
        position=HorizontalPosition.RIGHT,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
        path_relevance=PathRelevance.LOW,
    )
    res = nlg.look([dummy], path, "indoor room")
    assert_safe_text(res)
    assert "cannot guarantee" in res.lower() or "relatively clear" in res.lower()


def test_safety_ask_is_way_clear(nlg: ResponseGenerator) -> None:
    path = PathAnalysis(path_status="clear", obstacles=[])
    res = nlg.ask("Is the path clear to walk?", [], path, "hallway")
    assert_safe_text(res)
    assert "cannot guarantee" in res.lower()


def test_safety_ask_safe_to_cross(nlg: ResponseGenerator) -> None:
    traffic_light = SceneObject(
        id=1,
        type="traffic light",
        confidence=0.92,
        bbox=BoundingBox(x=0.4, y=0.1, width=0.1, height=0.2),
        position=HorizontalPosition.CENTER,
        distance_band=DistanceBand.NEAR,
        distance_range="one to two meters",
    )
    path = PathAnalysis(path_status="clear", obstacles=[])
    res = nlg.ask("Is it safe to cross now?", [traffic_light], path, "crosswalk")
    assert_safe_text(res)
    assert "cannot guarantee" in res.lower() or "obstacle" in res.lower()


def test_safety_sanitizer_catches_synthetic_leakage(nlg: ResponseGenerator) -> None:
    synthetic_unsafe = "The road is safe and you can proceed. Cross now."
    sanitized = nlg._sanitize_safety(synthetic_unsafe)
    assert_safe_text(sanitized)
