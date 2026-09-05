from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.types import DistanceBand, Movement, PathAnalysis, PipelineResult, Priority, SceneObject

FIND_SYNONYMS = {
    "keys": ["key", "keys"],
    "phone": ["cell phone", "phone", "mobile"],
    "bottle": ["bottle"],
    "glasses": ["glasses"],
    "backpack": ["backpack", "handbag", "suitcase"],
    "bag": ["handbag", "backpack", "suitcase"],
    "chair": ["chair"],
    "person": ["person"],
    "door": ["door", "doorway"],
    "book": ["book"],
    "remote": ["remote"],
    "laptop": ["laptop"],
}


@dataclass
class Intent:
    mode: str
    question: str
    target: str | None = None
    ocr_focus: str | None = None


class IntentParser:
    def parse(self, text: str, explicit_mode: str | None = None) -> Intent:
        lowered = (text or "").strip().lower()
        clean = lowered.rstrip(".?! ")
        if explicit_mode:
            return Intent(mode=explicit_mode, question=text or "", target=self._find_target(lowered), ocr_focus=self._ocr_focus(lowered))
        if clean in {"look", "what's around", "what is around me"}:
            return Intent(mode="look", question=text)
        if clean.startswith("find") or "where is my" in clean or "where's my" in clean:
            return Intent(mode="find", question=text, target=self._find_target(lowered))
        if "read" in clean or "what does this sign" in clean or "total" in clean or "prices" in clean:
            return Intent(mode="read", question=text, ocr_focus=self._ocr_focus(lowered))
        if "start assistance" in clean:
            return Intent(mode="assistance_start", question=text)
        if "stop assistance" in clean:
            return Intent(mode="assistance_stop", question=text)
        return Intent(mode="ask", question=text, target=self._find_target(lowered))

    def _find_target(self, text: str) -> str | None:
        for key, aliases in FIND_SYNONYMS.items():
            for alias in aliases + [key]:
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return key
        match = re.search(r"find (?:my |the )?([a-z ]+)", text)
        if match:
            return match.group(1).strip()
        return None

    def _ocr_focus(self, text: str) -> str | None:
        if "total" in text:
            return "total"
        if "price" in text:
            return "prices"
        if "summar" in text:
            return "summary"
        if "sign" in text:
            return "sign"
        return "read"


class ResponseGenerator:
    """Template NLG. Spatial facts come only from SceneObject fields."""

    def look(self, result_objects: list[SceneObject], path: PathAnalysis, scene_text: str) -> str:
        prioritized = self._select_objects(result_objects)
        if not prioritized:
            return "I don't clearly see notable objects right now. Try moving the camera slowly."
        sentences = [self.describe_object(obj, mention_path=True) for obj in prioritized[:3]]
        if scene_text.endswith(".") and "cannot confidently" not in scene_text.lower():
            sentences.insert(0, scene_text)
        if path.path_status == "clear" and not any(obj.path_relevance.value == "high" for obj in prioritized):
            sentences.append("Your path appears relatively clear in this frame, but I cannot guarantee it is safe to walk.")
        return " ".join(sentence for sentence in sentences if sentence)

    def ask(self, question: str, objects: list[SceneObject], path: PathAnalysis, scene_text: str) -> str:
        q = question.lower()
        if "path clear" in q or "is the way" in q:
            return self._path_answer(path)
        if "anyone" in q or "person" in q or "people" in q:
            people = [obj for obj in objects if obj.type.lower() == "person"]
            if not people:
                return "I don't currently see a person in this view."
            return " ".join(self.describe_object(obj) for obj in people[:2])
        if "front" in q or "ahead" in q or "looking at" in q:
            ahead = [obj for obj in objects if obj.position.value == "center"]
            if ahead:
                return " ".join(self.describe_object(obj, mention_path=True) for obj in ahead[:3])
            return self.look(objects, path, scene_text)
        if "door" in q:
            doors = [obj for obj in objects if "door" in obj.type.lower()]
            if not doors:
                return "I don't currently see a door in this view. Doorways are not always detected by the general object model."
            return " ".join(self.describe_object(obj) for obj in doors[:2])
        if "stair" in q:
            stairs = [obj for obj in objects if "stair" in obj.type.lower()]
            if not stairs:
                return "I don't currently see stairs. The general detector is not reliable for stairs yet."
            return " ".join(self.describe_object(obj) for obj in stairs)
        if "table" in q:
            tables = [obj for obj in objects if "table" in obj.type.lower()]
            if not tables:
                return "I don't currently see a table in this view."
            nearby = [obj for obj in objects if obj.type.lower() != "dining table"]
            return " ".join(self.describe_object(obj) for obj in tables[:1] + nearby[:2])
        if "chair" in q:
            chairs = [obj for obj in objects if obj.type.lower() == "chair"]
            if not chairs:
                return "I don't currently see a chair in this view."
            return " ".join(self.describe_object(obj, mention_path=True) for obj in chairs[:2])
        if "scene" in q or "where am i" in q:
            return scene_text
        return self.look(objects, path, scene_text)

    def find(self, target: str | None, objects: list[SceneObject]) -> str:
        if not target:
            return "Tell me which object to find, for example: find my bottle."
        matches = [obj for obj in objects if self._is_target(obj.type, target)]
        if not matches:
            return (
                f"I don't currently see your {target}. Try moving the camera slowly to the right, "
                "then to the left."
            )
        best = sorted(matches, key=lambda obj: obj.confidence, reverse=True)[0]
        location = "ahead" if best.position.value == "center" else f"slightly to your {best.position.value}"
        extra = ""
        if any(obj.type.lower() in {"dining table", "table"} for obj in objects) and best.bbox.y > 0.35:
            extra = " They appear to be on a table."
        return f"Your {target} {'are' if target.endswith('s') else 'is'} {location}.{extra}".strip()

    def read(self, question: str, structured: dict, full_text: str, focus: str | None) -> str:
        if not full_text.strip():
            return "I couldn't read any text. Hold the camera steady and fill the frame with the document."
        if focus == "total" and structured.get("totals"):
            return f"The total appears to be {structured['totals'][0]}."
        if focus == "prices" and structured.get("price_like"):
            prices = ", ".join(structured["price_like"][:8])
            return f"I found these price-like numbers: {prices}."
        if focus == "summary":
            compact = " ".join(full_text.split())
            return f"The text says: {compact[:400]}"
        compact = " ".join(full_text.split())
        return f"The visible text says: {compact[:500]}"

    def describe_object(self, obj: SceneObject, mention_path: bool = False) -> str:
        hedge = self._hedge(obj.confidence)
        place = "ahead" if obj.position.value == "center" else f"ahead on your {obj.position.value}"
        distance = obj.distance_range if obj.distance_band != DistanceBand.UNKNOWN else "ahead"
        obstruction = ""
        if mention_path and obj.path_relevance.value in {"high", "medium"}:
            obstruction = " and may obstruct your path"
        noun = obj.type
        if obj.movement == Movement.APPROACHING:
            origin = {
                "left": "from your left",
                "right": "from your right",
                "center": "from ahead",
            }[obj.position.value]
            return f"{hedge}{noun} is approaching {origin}.".replace("  ", " ")
        if obj.movement == Movement.RECEDING:
            return f"{hedge}{noun} is moving away, approximately {distance} {place}.".replace("  ", " ")
        return f"{hedge}{noun} is approximately {distance} {place}{obstruction}.".replace("  ", " ")

    def assistance_announcement(self, result: PipelineResult) -> str | None:
        announceable = [event for event in result.events if event.should_announce]
        if not announceable:
            return None
        top = announceable[0]
        if top.priority == Priority.P3:
            return None
        return top.message

    def _path_answer(self, path: PathAnalysis) -> str:
        if path.path_status == "clear":
            return "I don't see a clear obstruction in the center of this frame, but I cannot guarantee the path is safe."
        if path.path_status == "uncertain":
            return "I cannot confidently tell whether the path is clear."
        first = path.obstacles[0] if path.obstacles else None
        if first:
            return (
                f"A {first.type} is approximately {first.distance_range} ahead "
                f"and may obstruct your path."
            )
        return "The path may be partially blocked."

    def _select_objects(self, objects: list[SceneObject]) -> list[SceneObject]:
        ranked = sorted(
            objects,
            key=lambda obj: (
                {"high": 0, "medium": 1, "low": 2, "none": 3}[obj.path_relevance.value],
                {"very_near": 0, "near": 1, "mid": 2, "far": 3, "unknown": 4}[obj.distance_band.value],
                -obj.confidence,
            ),
        )
        # Avoid listing every COCO class; keep path-relevant and nearby items.
        selected: list[SceneObject] = []
        for obj in ranked:
            if obj.confidence < 0.35:
                continue
            if obj.path_relevance.value in {"high", "medium"} or obj.distance_band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR, DistanceBand.MID}:
                selected.append(obj)
            if len(selected) >= 4:
                break
        return selected

    def _hedge(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "A "
        if confidence >= 0.5:
            return "There may be a "
        return "I may be detecting a "

    def _is_target(self, class_name: str, target: str) -> bool:
        aliases = FIND_SYNONYMS.get(target, [target])
        lowered = class_name.lower()
        return any(alias in lowered or lowered in alias for alias in aliases)
