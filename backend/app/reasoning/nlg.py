from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.types import DistanceBand, Movement, PathAnalysis, PipelineResult, Priority, SceneObject

FIND_SYNONYMS: dict[str, list[str]] = {
    "keys": ["key", "keys", "keyring", "keychain"],
    "phone": ["cell phone", "cell phones", "phone", "phones", "mobile", "smartphone", "telephone"],
    "bottle": ["bottle", "bottles", "water bottle", "water bottles", "flask"],
    "cup": ["cup", "cups", "mug", "mugs", "coffee cup", "glass"],
    "glasses": ["glasses", "eyeglasses", "sunglasses", "spectacles"],
    "backpack": ["backpack", "backpacks", "bag", "bags", "bookbag"],
    "bag": ["handbag", "backpack", "suitcase", "tote", "purse", "bag", "bags"],
    "chair": ["chair", "chairs", "armchair", "armchairs", "seat", "seats"],
    "person": ["person", "someone", "somebody", "human", "people", "persons"],
    "door": ["door", "doors", "doorway", "entrance"],
    "book": ["book", "books", "notebook", "textbook"],
    "remote": ["remote", "controller", "remote control"],
    "laptop": ["laptop", "laptops", "computer", "computers", "notebook"],
    "mouse": ["mouse", "computer mouse", "mice"],
    "keyboard": ["keyboard", "keyboards"],
    "table": ["dining table", "table", "tables", "desk", "desks"],
    "desk": ["dining table", "table", "desk", "desks"],
    "stairs": ["stairs", "staircase", "steps", "stairs up", "stairs down"],
    "curb": ["curb", "curbs", "sidewalk edge"],
    "ramp": ["ramp", "ramps", "wheelchair ramp"],
    "cane": ["cane", "white cane", "walking stick"],
    "wallet": ["wallet", "billfold"],
    "car": ["car", "cars", "automobile", "vehicle"],
    "bus": ["bus", "buses"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cycle"],
    "clock": ["clock", "clocks"],
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
            r"how many ([a-z ]+?)(?: are there| do you see|\?|$)",
            r"count of ([a-z ]+)",
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
                if target.endswith("s") and not target.endswith("ss") and len(target) > 3:
                    singular = target[:-1]
                    if singular in FIND_SYNONYMS:
                        return singular
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
            sentences.append("Your path appears relatively clear in this frame, but I cannot guarantee the path is clear.")
        result_text = " ".join(sentence for sentence in sentences if sentence)
        return self._sanitize_safety(result_text)

    def ask(self, question: str, objects: list[SceneObject], path: PathAnalysis, scene_text: str) -> str:
        q = question.lower().strip()

        # 1. Path clearance questions
        if "path clear" in q or "is the way" in q or "path blocked" in q or "safe to walk" in q or "safe to cross" in q:
            return self._sanitize_safety(self._path_answer(path))

        # 2. Count queries ("how many bottles are there?", "how many people are there?", "how many objects?")
        if q.startswith("how many") or "count of" in q or "number of" in q:
            if "objects" in q or "items" in q or "things" in q:
                count = len(objects)
                if count == 0:
                    return "I do not detect notable objects in this view."
                types = list(dict.fromkeys(obj.type for obj in objects))
                sample = ", ".join(types[:4])
                return f"I detect {count} notable object{'s' if count != 1 else ''} in this view, including {sample}."
            target = IntentParser()._find_target(q)
            if target:
                matches = [obj for obj in objects if self._is_target(obj.type, target)]
                if not matches:
                    return f"I do not see any {target}s in this view."
                if len(matches) == 1:
                    m = matches[0]
                    return f"I see 1 {target} in this view, approximately {m.distance_range} on your {m.position.value}."
                details = []
                for m in matches[:3]:
                    details.append(f"one is approximately {m.distance_range} on your {m.position.value}")
                return f"I see {len(matches)} {target}s in this view: {', and '.join(details)}."

        # 3. Existence queries ("is anyone near me?", "is there a bottle?", "is there a phone?")
        if q.startswith("is anyone") or "anyone near" in q or "someone near" in q or "anybody near" in q:
            nearby_people = [
                obj
                for obj in objects
                if obj.type.lower() == "person"
                and obj.distance_band in {DistanceBand.VERY_NEAR, DistanceBand.NEAR}
            ]
            if nearby_people:
                p = nearby_people[0]
                return f"Yes, there is a person approximately {p.distance_range} {self._pos_phrase(p)}."
            other_people = [obj for obj in objects if obj.type.lower() == "person"]
            if other_people:
                p = other_people[0]
                return f"I see a person, but they are farther away, approximately {p.distance_range} {self._pos_phrase(p)}."
            return "I do not detect anyone near you in this view."

        if (
            q.startswith("is there a ")
            or q.startswith("is there an ")
            or q.startswith("is there any ")
            or q.startswith("do you see a ")
            or q.startswith("do you see an ")
        ):
            target = IntentParser()._find_target(q)
            if target:
                if target in {"door_handle", "door handle", "elevator_button", "elevator button", "cord", "wire"}:
                    return f"Small features like {target}s cannot be reliably recognized by this model at a distance."
                matches = [obj for obj in objects if self._is_target(obj.type, target)]
                if matches:
                    m = matches[0]
                    rel_str = f" {m.relationships[0]}" if getattr(m, "relationships", []) else ""
                    return f"Yes, I see a {m.type} approximately {m.distance_range} {self._pos_phrase(m)}{rel_str}."
                if target in {"stair", "stairs"}:
                    return "I don't currently see stairs. Stair detection requires clear line of sight to the floor."
                if target in {"door", "doorway"}:
                    return "I don't currently see a door in this view."
                return f"No, I don't see a {target} in this view."

        # 4. Surface queries (e.g. "what is on the table?", "what is on the desk?")
        if "on the table" in q or "on the desk" in q or "on table" in q:
            table_items = [
                obj
                for obj in objects
                if any("on the" in r for r in getattr(obj, "relationships", []))
                or (
                    obj.bbox.y < 0.75
                    and any(
                        s.type.lower() in {"table", "dining table", "desk"}
                        for s in objects
                        if s.id != obj.id
                        and s.bbox.x <= (obj.bbox.x + obj.bbox.width / 2) <= (s.bbox.x + s.bbox.width)
                    )
                )
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

        # 5. Proximity / beside queries ("what is near the chair?", "what is beside the laptop?")
        if "beside" in q or "next to" in q or ("near the " in q and "me" not in q):
            for ref_key in ["chair", "table", "desk", "laptop", "bottle", "person", "phone"]:
                if ref_key in q:
                    ref_objs = [obj for obj in objects if self._is_target(obj.type, ref_key)]
                    if ref_objs:
                        ref = ref_objs[0]
                        nearby = [
                            other
                            for other in objects
                            if other.id != ref.id
                            and (
                                any(ref.type.lower() in r for r in getattr(other, "relationships", []))
                                or (
                                    other.distance_band == ref.distance_band
                                    and abs(other.bbox.x - ref.bbox.x) < 0.35
                                )
                            )
                        ]
                        if nearby:
                            names = [o.type for o in nearby[:3]]
                            return f"Near the {ref.type}, I see a {', a '.join(names)}."
                        return f"I see the {ref.type}, but no other notable objects immediately beside it."
                    return f"I don't currently see a {ref_key} in this view."

        # 6. Directional / Region queries
        if "far left" in q:
            objs = [obj for obj in objects if obj.position.value in {"far left", "left"} and obj.bbox.x < 0.25]
            if objs:
                return "On your far left, " + " ".join(self.describe_object(obj) for obj in objs[:2])
            return "I don't see notable objects on your far left in this view."

        if "far right" in q:
            objs = [obj for obj in objects if obj.position.value in {"far right", "right"} and (obj.bbox.x + obj.bbox.width) > 0.75]
            if objs:
                return "On your far right, " + " ".join(self.describe_object(obj) for obj in objs[:2])
            return "I don't see notable objects on your far right in this view."

        if "to my left" in q or "on my left" in q or "on the left" in q:
            left_objs = [obj for obj in objects if obj.position.value in {"left", "far left"}]
            if left_objs:
                return "On your left, " + " ".join(self.describe_object(obj) for obj in left_objs[:2])
            return "I don't see notable objects on your left in this view."

        if "to my right" in q or "on my right" in q or "on the right" in q:
            right_objs = [obj for obj in objects if obj.position.value in {"right", "far right"}]
            if right_objs:
                return "On your right, " + " ".join(self.describe_object(obj) for obj in right_objs[:2])
            return "I don't see notable objects on your right in this view."

        if "directly ahead" in q or "in front" in q or "ahead" in q:
            ahead_objs = [obj for obj in objects if obj.position.value == "center"]
            if ahead_objs:
                return "Directly ahead, " + " ".join(self.describe_object(obj, mention_path=True) for obj in ahead_objs[:3])
            return self.look(objects, path, scene_text)

        # 7. Specific object location queries ("where is the bottle?", "where is my phone?")
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

        # 8. Scene context
        if "scene" in q or "where am i" in q:
            return self._sanitize_safety(scene_text)

        # Default fallback to general look
        return self.look(objects, path, scene_text)

    def find(self, target: str | None, objects: list[SceneObject]) -> str:
        if not target:
            return "Tell me which object to find, for example: find my bottle."
        matches = [obj for obj in objects if self._is_target(obj.type, target)]
        if not matches:
            return (
                f"I don't currently see your {target}. Try moving the camera slowly to the right, "
                "then to the left. I cannot check areas outside the camera frame."
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

        answer = f"I see your {target} {dist}{location}.{extra}".strip().replace("  ", " ")
        return self._sanitize_safety(answer)

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

    def _pos_phrase(self, obj: SceneObject) -> str:
        if obj.position.value == "center":
            return "ahead"
        return f"ahead on your {obj.position.value}"

    def _sanitize_safety(self, text: str) -> str:
        """Reject and sanitize ungrounded or unsafe navigational claims."""
        sanitized = text
        unsafe_rules = [
            (re.compile(r"\bthe road is safe\b", re.I), "a roadway is detected ahead, but proceed with caution"),
            (re.compile(r"\bcross now\b", re.I), "a crossing may be present, but proceed with caution"),
            (re.compile(r"\bwalk forward safely\b", re.I), "there may be an obstacle ahead; proceed with caution"),
            (re.compile(r"\byou can proceed\b", re.I), "proceed with caution"),
            (re.compile(r"\bit is safe to cross\b", re.I), "I cannot confirm whether the crossing is clear"),
            (re.compile(r"\bit is safe to walk\b", re.I), "I cannot confirm whether the path is clear"),
            (re.compile(r"\ball clear to go\b", re.I), "proceed with caution"),
        ]
        for pattern, replacement in unsafe_rules:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

