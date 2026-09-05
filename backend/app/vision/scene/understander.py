from __future__ import annotations

from collections import Counter

from app.vision.types import SceneUnderstanding, SceneObject

INDOOR_HINTS = {"chair", "couch", "sofa", "bed", "dining table", "table", "tv", "toilet", "refrigerator", "oven", "sink"}
HALLWAY_HINTS = {"door", "doorway"}
ROAD_HINTS = {"car", "truck", "bus", "motorcycle", "traffic light", "stop sign", "bicycle"}


class SceneUnderstander:
    def understand(self, objects: list[SceneObject]) -> SceneUnderstanding:
        if not objects:
            return SceneUnderstanding(
                scene_type="unknown",
                description="I cannot confidently determine the scene.",
                confidence=0.2,
                uncertain=True,
            )
        types = [obj.type.lower() for obj in objects]
        counts = Counter(types)
        indoor = sum(counts[name] for name in INDOOR_HINTS)
        road = sum(counts[name] for name in ROAD_HINTS)
        doors = sum(counts[name] for name in HALLWAY_HINTS)
        people = counts.get("person", 0)

        if road >= 2 or (road >= 1 and "traffic light" in counts):
            scene_type = "road"
            description = "You appear to be near a road. Vehicles are visible ahead."
            confidence = min(0.55 + 0.1 * road, 0.8)
        elif doors and indoor == 0:
            scene_type = "hallway"
            description = "You appear to be in a hallway."
            confidence = 0.55
        elif indoor >= 2:
            scene_type = "room"
            description = "You appear to be in a room."
            confidence = min(0.5 + 0.08 * indoor, 0.78)
        elif people and indoor == 0 and road == 0:
            scene_type = "indoor_or_outdoor"
            description = "I cannot confidently determine the scene."
            confidence = 0.35
            return SceneUnderstanding(scene_type=scene_type, description=description, confidence=confidence, uncertain=True)
        else:
            return SceneUnderstanding(
                scene_type="unknown",
                description="I cannot confidently determine the scene.",
                confidence=0.3,
                uncertain=True,
            )

        extras = []
        for obj in objects:
            if obj.type.lower() in {"door", "dining table", "table", "chair", "stairs"}:
                extras.append(
                    f"A {obj.type} is approximately {obj.distance_range} "
                    f"{'ahead' if obj.position.value == 'center' else 'on your ' + obj.position.value}."
                )
        if extras:
            description = f"{description} {extras[0]}"
        return SceneUnderstanding(scene_type=scene_type, description=description, confidence=confidence, uncertain=confidence < 0.5)
