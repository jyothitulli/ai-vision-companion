from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.types import DistanceBand, Movement, PathAnalysis, PipelineResult, Priority, SceneObject

FIND_SYNONYMS: dict[str, list[str]] = {
    "keys": ["key", "keys", "keyring", "keychain"],
    "phone": ["cell phone", "phone", "mobile", "smartphone", "telephone"],
    "bottle": ["bottle", "water bottle", "flask"],
    "cup": ["cup", "mug", "coffee cup", "glass"],
    "glasses": ["glasses", "eyeglasses", "sunglasses", "spectacles"],
    "backpack": ["backpack", "bag", "bookbag"],
    "bag": ["handbag", "backpack", "suitcase", "tote", "purse", "bag"],
    "chair": ["chair", "armchair", "seat"],
    "person": ["person", "someone", "somebody", "human", "people"],
    "door": ["door", "doorway", "entrance"],
    "book": ["book", "notebook", "textbook"],
    "remote": ["remote", "controller", "remote control"],
    "laptop": ["laptop", "computer", "notebook"],
    "mouse": ["mouse", "computer mouse"],
    "keyboard": ["keyboard"],
    "table": ["dining table", "table", "desk"],
    "desk": ["dining table", "table", "desk"],
    "stairs": ["stairs", "staircase", "steps", "stairs up", "stairs down"],
    "curb": ["curb", "sidewalk edge"],
    "ramp": ["ramp", "wheelchair ramp"],
    "cane": ["cane", "white cane", "walking stick"],
    "wallet": ["wallet", "billfold"],
    "car": ["car", "automobile", "vehicle"],
    "bus": ["bus"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "clock": ["clock"],
    "trash can": ["trash can", "garbage can", "wastebasket", "bin"],
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
            return Intent(
                mode=explicit_mode,
                question=text or "",
                target=self._find_target(lowered),
                ocr_focus=self._ocr_focus(lowered),
            )
        if clean in {"look", "what's around", "what is around me", "what do you see"}:
            return Intent(mode="look", question=text)
        if (
            clean.startswith("find")
            or "where is my" in clean
            or "where's my" in clean
            or "locate my" in clean
            or "can you find" in clean
        ):
            return Intent(mode="find", question=text, target=self._find_target(lowered))
        if "read" in clean or "what does this sign" in clean or "total" in clean or "prices" in clean:
            return Intent(mode="read", question=text, ocr_focus=self._ocr_focus(lowered))
        if "start assistance" in clean:
            return Intent(mode="assistance_start", question=text)
        if "stop assistance" in clean:
            return Intent(mode="assistance_stop", question=text)
        return Intent(mode="ask", question=text, target=self._find_target(lowered))

    def _find_target(self, text: str) -> str | None:
        # Check specific multi-word micro-features first
        for micro in ["door handle", "elevator button", "cord", "wire"]:
            if micro in text:
                return micro

        for key, aliases in FIND_SYNONYMS.items():
            for alias in aliases + [key]:
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return key
        patterns = [
            r"find (?:my |the |a |an )?([a-z ]+)",
            r"where is (?:my |the |a |an )?([a-z ]+)",
            r"where are (?:my |the |any )?([a-z ]+)",
            r"locate (?:my |the )?([a-z ]+)",
            r"is there (?:a |an |any )?([a-z ]+)",
            r"do you see (?:a |an |my |the )?([a-z ]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                target = match.group(1).strip().rstrip(".?! ")
                # Remove common trailing noise words
                for noise in [" nearby", " ahead", " in front of me", " on the table", " here"]:
                    if target.endswith(noise):
                        target = target[: -len(noise)].strip()
                if target:
                    return target
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
    """Template NLG. Grounded strictly in SceneObject fields and geometric relationships."""

    def look(self, result_objects: list[SceneObject], path: PathAnalysis, scene_text: str) -> str:
        prioritized = self._select_objects(result_objects)
        if not prioritized:
            return "I don't clearly see notable objects right now. Try moving the camera slowly."

        # Intelligently group objects if surface relationships exist (e.g. laptop & phone on table)
        surface_groups: dict[str, list[str]] = {}
        standalone: list[SceneObject] = []

        for obj in prioritized:
            has_surface = False
            for rel in getattr(obj, "relationships", []):
                if rel.startswith("on the "):
                    surf_name = rel[len("on the ") :]
                    surface_groups.setdefault(surf_name, []).append(obj.type)
                    has_surface = True
                    break
            if not has_surface:
                standalone.append(obj)

        sentences: list[str] = []
        for obj in standalone[:4]:
            sentences.append(self.describe_object(obj, mention_path=True))

        for surf_name, items in surface_groups.items():
            if len(items) == 1:
                sentences.append(f"There is a {items[0]} on the {surf_name}.")
            elif len(items) == 2:
                sentences.append(f"On the {surf_name}, there is a {items[0]} and a {items[1]}.")
            else:
                formatted = ", ".join(items[:-1]) + f", and a {items[-1]}"
                sentences.append(f"On the {surf_name}, there is a {formatted}.")

        if scene_text.endswith(".") and "cannot confidently" not in scene_text.lower():
            sentences.insert(0, scene_text)
        if path.path_status == "clear" and not any(obj.path_relevance.value == "high" for obj in prioritized):
            sentences.append("Your path appears relatively clear in this frame, but I cannot guarantee it is safe to walk.")
        return " ".join(sentence for sentence in sentences if sentence)

    def ask(self, question: str, objects: list[SceneObject], path: PathAnalysis, scene_text: str) -> str:
        q = question.lower().strip()

        # 1. Path clearance questions
        if "path clear" in q or "is the way" in q or "path blocked" in q or "safe to walk" in q or "safe to cross" in q:
            return self._path_answer(path)

        # 2. Surface queries (e.g. "what is on the table?", "what is on the desk?")
        if "on the table" in q or "on the desk" in q or "on table" in q:
            table_items = [
                obj
                for obj in objects
                if any("on the" in r for r in getattr(obj, "relationships", []))
                or (obj.bbox.y < 0.75 and any(s.type.lower() in {"table", "dining table", "desk"} for s in objects if s.id != obj.id and s.bbox.x <= (obj.bbox.x + obj.bbox.width/2) <= (s.bbox.x + s.bbox.width)))
            ]
            # Exclude the table itself
            surface_items = [obj for obj in table_items if obj.type.lower() not in {"table", "dining table", "desk"}]
            if surface_items:
                item_names = [obj.type for obj in surface_items[:4]]
                if len(item_names) == 1:
                    return f"On the table, I see a {item_names[0]}."
                elif len(item_names) == 2:
                    return f"On the table, I see a {item_names[0]} and a {item_names[1]}."
                return f"On the table, I see a {', '.join(item_names[:-1])}, and a {item_names[-1]}."
            if any(obj.type.lower() in {"table", "dining table", "desk"} for obj in objects):
                return "I see a table ahead, but I don't clearly detect items resting on it right now."
            return "I don't currently see a table in this view."

        # 3. Directional / Region queries (e.g. "what is to my left?", "what is on my right?", "what is ahead?")
        if "to my left" in q or "on my left" in q or "on the left" in q:
            left_objs = [obj for obj in objects if obj.position.value == "left"]
            if left_objs:
                return "On your left, " + " ".join(self.describe_object(obj) for obj in left_objs[:2])
            return "I don't see notable objects on your left in this view."

        if "to my right" in q or "on my right" in q or "on the right" in q:
            right_objs = [obj for obj in objects if obj.position.value == "right"]
            if right_objs:
                return "On your right, " + " ".join(self.describe_object(obj) for obj in right_objs[:2])
            return "I don't see notable objects on your right in this view."

        if "directly ahead" in q or "in front" in q or "ahead" in q:
            ahead_objs = [obj for obj in objects if obj.position.value == "center"]
            if ahead_objs:
                return "Directly ahead, " + " ".join(self.describe_object(obj, mention_path=True) for obj in ahead_objs[:3])
            return self.look(objects, path, scene_text)

        # 4. Specific object location queries ("where is the bottle?", "where is my phone?", "is there a X?")
        target = IntentParser()._find_target(q)
        if target:
            # Check for unsupported micro-features
            if target in {"door_handle", "door handle", "elevator_button", "elevator button", "cord", "wire"}:
                return f"Small features like {target}s cannot be reliably recognized by this model at a distance."

            matches = [obj for obj in objects if self._is_target(obj.type, target)]
            if matches:
                if len(matches) == 1:
                    return self.describe_object(matches[0], mention_path=True)
                # Multiple instances
                first = matches[0]
                second = matches[1]
                return (
                    f"I see {len(matches)} {target}s. One is approximately {first.distance_range} on your {first.position.value}, "
                    f"and another is approximately {second.distance_range} on your {second.position.value}."
                )
            # Not found
            if target in {"stair", "stairs"}:
                return "I don't currently see stairs. Stair detection requires clear line of sight to the floor."
            if target in {"door", "doorway"}:
                return "I don't currently see a door in this view."
            return f"I don't currently see a {target} in this view."

        # 5. Scene context
        if "scene" in q or "where am i" in q:
            return scene_text

        # Default fallback to general look
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

        if len(matches) > 1:
            first = matches[0]
            second = matches[1]
            return (
                f"I see {len(matches)} {target}s. One is approximately {first.distance_range} on your {first.position.value}, "
                f"and another is approximately {second.distance_range} on your {second.position.value}."
            )

        best = sorted(matches, key=lambda obj: obj.confidence, reverse=True)[0]
        location = "ahead" if best.position.value == "center" else f"slightly to your {best.position.value}"
        dist = f"approximately {best.distance_range} " if best.distance_range != "an unknown distance" else ""
        
        # Check for surface relationships
        extra = ""
        for rel in getattr(best, "relationships", []):
            if "on the" in rel:
                extra = f" It appears to be {rel}."
                break
        if not extra and any(obj.type.lower() in {"dining table", "table", "desk"} for obj in objects) and best.bbox.y > 0.35:
            extra = " It appears to be on a table."

        return f"I see your {target} {dist}{location}.{extra}".strip().replace("  ", " ")

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
        
        # Mention surface relationship if present
        surface_rel = ""
        for rel in getattr(obj, "relationships", []):
            if "on the" in rel:
                surface_rel = f" {rel}"
                break

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
        return f"{hedge}{noun} is approximately {distance} {place}{surface_rel}{obstruction}.".replace("  ", " ")

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
        selected: list[SceneObject] = []
        for obj in ranked:
            if obj.confidence < 0.20:
                continue
            # Keep path-relevant hazards, nearby/mid objects, or surface-resting objects
            if (
                obj.path_relevance.value in {"high", "medium"}
                or obj.distance_band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR, DistanceBand.MID}
                or any("on the" in r for r in getattr(obj, "relationships", []))
            ):
                selected.append(obj)
            if len(selected) >= 6:
                break
        
        # If no objects qualified under path/distance heuristics, but objects exist, take the most confident
        if not selected and objects:
            for obj in ranked:
                if obj.confidence >= 0.22:
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

