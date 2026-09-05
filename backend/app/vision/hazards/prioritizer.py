from __future__ import annotations

import time
from collections import defaultdict
from uuid import uuid4

from app.vision.types import (
    DistanceBand,
    HazardEvent,
    Movement,
    PathAnalysis,
    Priority,
    SceneObject,
)

P0_TYPES = {"car", "truck", "bus", "motorcycle", "train"}
P1_TYPES = {"stairs", "curb", "ramp", "door", "doorway", "elevator", "pothole", "chair", "table", "dining table"}
CROSSING_TYPES = {"traffic light", "stop sign"}


class EventPrioritizer:
    def __init__(self, cooldown_s: float = 8.0):
        self._cooldown_s = cooldown_s
        self._last_announced: dict[str, float] = {}
        self._seen_counts: dict[str, int] = defaultdict(int)

    def prioritize(
        self,
        objects: list[SceneObject],
        path: PathAnalysis,
        now: float | None = None,
    ) -> list[HazardEvent]:
        now = now if now is not None else time.monotonic()
        events: list[HazardEvent] = []
        for obj in objects:
            events.extend(self._events_for_object(obj))
        if path.path_status == "blocked":
            events.append(
                HazardEvent(
                    event_id=str(uuid4()),
                    type="path_blocked",
                    priority=Priority.P0,
                    confidence=0.7,
                    message="An obstacle is very close ahead and may obstruct your path.",
                    cooldown_key="path_blocked",
                )
            )
        elif path.path_status == "partially_blocked" and path.obstacles:
            first = path.obstacles[0]
            events.append(
                HazardEvent(
                    event_id=str(uuid4()),
                    type="path_obstruction",
                    object_id=first.object_id,
                    priority=Priority.P1,
                    confidence=first.confidence,
                    message=(
                        f"A {first.type} is approximately {first.distance_range} ahead "
                        f"and may obstruct your path."
                    ),
                    cooldown_key=f"path_obstruction:{first.type}",
                )
            )
        return self.apply_cooldown(events, now)

    def _events_for_object(self, obj: SceneObject) -> list[HazardEvent]:
        events: list[HazardEvent] = []
        name = obj.type.lower()
        if name in P0_TYPES and obj.movement == Movement.APPROACHING:
            events.append(
                self._event(
                    obj,
                    "approaching_vehicle",
                    Priority.P0,
                    f"A {obj.type} appears to be approaching. I cannot determine whether a path is safe.",
                )
            )
        if obj.distance_band == DistanceBand.VERY_NEAR and obj.path_relevance.value in {"high", "medium"}:
            events.append(
                self._event(
                    obj,
                    "close_obstacle",
                    Priority.P0,
                    f"A {obj.type} is very close ahead, approximately {obj.distance_range}.",
                )
            )
        if name in P1_TYPES:
            events.append(
                self._event(
                    obj,
                    f"nav_{name}",
                    Priority.P1,
                    self._nav_message(obj),
                )
            )
        if name in CROSSING_TYPES:
            events.append(
                self._event(
                    obj,
                    "crossing_context",
                    Priority.P1,
                    "A pedestrian crossing or traffic signal is detected ahead. "
                    "I cannot reliably determine whether it is safe to cross.",
                )
            )
        if name == "person" and obj.distance_band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR, DistanceBand.MID}:
            movement = ""
            if obj.movement == Movement.APPROACHING:
                movement = " approaching"
            events.append(
                self._event(
                    obj,
                    "nearby_person",
                    Priority.P2,
                    f"A person is{movement} approximately {obj.distance_range} {self._pos(obj)}.",
                )
            )
        return events

    def _nav_message(self, obj: SceneObject) -> str:
        if obj.confidence < 0.45:
            return f"I may be detecting {obj.type} ahead, but I'm not confident."
        if obj.confidence < 0.65:
            return f"There may be {obj.type} approximately {obj.distance_range} {self._pos(obj)}."
        return f"{obj.type.capitalize()} {'are' if obj.type.endswith('s') else 'is'} detected approximately {obj.distance_range} {self._pos(obj)}."

    def _pos(self, obj: SceneObject) -> str:
        if obj.position.value == "center":
            return "ahead"
        return f"ahead on your {obj.position.value}"

    def _event(self, obj: SceneObject, etype: str, priority: Priority, message: str) -> HazardEvent:
        return HazardEvent(
            event_id=str(uuid4()),
            type=etype,
            object_id=obj.id,
            priority=priority,
            confidence=obj.confidence,
            message=message,
            cooldown_key=f"{etype}:{obj.track_id or obj.type}",
        )

    def apply_cooldown(self, events: list[HazardEvent], now: float) -> list[HazardEvent]:
        filtered: list[HazardEvent] = []
        for event in events:
            last = self._last_announced.get(event.cooldown_key)
            duplicate = last is not None and (now - last) < self._cooldown_s
            event.should_announce = not duplicate
            if event.should_announce:
                self._last_announced[event.cooldown_key] = now
                self._seen_counts[event.cooldown_key] += 1
            filtered.append(event)
        filtered.sort(key=lambda item: item.priority.value)
        return filtered
