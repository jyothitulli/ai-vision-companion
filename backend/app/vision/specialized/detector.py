from __future__ import annotations

import logging

import numpy as np

from app.vision.interfaces import RawDetection, SpecializedDetector

logger = logging.getLogger(__name__)

# These classes are architected here so they can be filled by a fine-tuned
# detector later. The COCO pretrained YOLO is NOT treated as an accessibility
# detector even if a name collides (e.g. there is no reliable "stairs" class).
# Comprehensive 50+ accessibility classes categorized with capability status
ACCESSIBILITY_CLASSES: dict[str, dict] = {
    # Critical Obstacles (P0)
    "stairs_up": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_depth", "aliases": ["stairs", "staircase", "steps up"]},
    "stairs_down": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_depth", "aliases": ["stairs", "steps down", "drop off"]},
    "curb": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_depth", "aliases": ["curb", "sidewalk edge"]},
    "drop_off": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_depth", "aliases": ["drop off", "ledge"]},
    "construction_barrier": {"category": "critical_obstacle", "priority": "P0", "capability": "open_vocabulary", "aliases": ["construction barrier", "traffic cone", "barricade"]},
    "low_hanging_branch": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_geometry", "aliases": ["low branch", "tree branch"]},
    "puddle": {"category": "critical_obstacle", "priority": "P0", "capability": "open_vocabulary", "aliases": ["puddle", "water puddle"]},
    "hole": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_depth", "aliases": ["pothole", "hole"]},
    "trip_hazard": {"category": "critical_obstacle", "priority": "P0", "capability": "specialized_geometry", "aliases": ["trip hazard", "floor obstacle"]},
    "wet_floor_sign": {"category": "critical_obstacle", "priority": "P0", "capability": "zero_shot_ocr", "aliases": ["wet floor sign", "caution cone"]},
    # Wayfinding (P1)
    "door": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["door", "doorway", "entrance"]},
    "crosswalk": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["crosswalk", "pedestrian crossing"]},
    "traffic_signal": {"category": "wayfinding", "priority": "P1", "capability": "coco_supported", "aliases": ["traffic light", "traffic signal"]},
    "tactile_paving": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["tactile paving", "yellow tiles"]},
    "escalator": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["escalator", "moving stairs"]},
    "ramp": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["ramp", "wheelchair ramp"]},
    "handrail": {"category": "wayfinding", "priority": "P1", "capability": "open_vocabulary", "aliases": ["handrail", "stair railing"]},
    # People / Transport (P2)
    "person": {"category": "people_transport", "priority": "P1", "capability": "coco_supported", "aliases": ["person", "people", "someone"]},
    "wheelchair": {"category": "people_transport", "priority": "P2", "capability": "open_vocabulary", "aliases": ["wheelchair"]},
    "guide_dog": {"category": "people_transport", "priority": "P2", "capability": "coco_specialized", "aliases": ["guide dog", "service dog"]},
    "white_cane": {"category": "people_transport", "priority": "P2", "capability": "open_vocabulary", "aliases": ["white cane", "cane"]},
    "bicycle": {"category": "people_transport", "priority": "P2", "capability": "coco_supported", "aliases": ["bicycle", "bike"]},
    "scooter": {"category": "people_transport", "priority": "P2", "capability": "open_vocabulary", "aliases": ["scooter", "electric scooter"]},
    "car": {"category": "people_transport", "priority": "P1", "capability": "coco_supported", "aliases": ["car", "automobile"]},
    "bus": {"category": "people_transport", "priority": "P1", "capability": "coco_supported", "aliases": ["bus"]},
    "shopping_cart": {"category": "people_transport", "priority": "P2", "capability": "open_vocabulary", "aliases": ["shopping cart"]},
    "luggage": {"category": "people_transport", "priority": "P2", "capability": "coco_supported", "aliases": ["suitcase", "luggage"]},
    # Indoor Objects (P3)
    "chair": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["chair", "armchair"]},
    "table": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["table", "dining table", "desk"]},
    "desk": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["desk", "office desk"]},
    "bench": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["bench"]},
    "trash_can": {"category": "indoor", "priority": "P3", "capability": "open_vocabulary", "aliases": ["trash can", "garbage can", "bin"]},
    "column_pillar": {"category": "indoor", "priority": "P3", "capability": "open_vocabulary", "aliases": ["pillar", "column"]},
    "glass_door": {"category": "indoor", "priority": "P0", "capability": "specialized_depth", "aliases": ["glass door", "glass wall"]},
    "open_cabinet": {"category": "indoor", "priority": "P0", "capability": "specialized_geometry", "aliases": ["open cabinet", "cupboard"]},
    "box": {"category": "indoor", "priority": "P3", "capability": "open_vocabulary", "aliases": ["cardboard box", "package"]},
    "keyboard": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["keyboard"]},
    "screen": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["tv", "monitor", "screen"]},
    "sink": {"category": "indoor", "priority": "P3", "capability": "coco_supported", "aliases": ["sink"]},
    # Personal Objects (P4)
    "bottle": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["bottle", "water bottle"]},
    "cup": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["cup", "mug"]},
    "phone": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["phone", "cell phone", "mobile"]},
    "keys": {"category": "personal", "priority": "P4", "capability": "open_vocabulary", "aliases": ["keys", "keychain"]},
    "wallet": {"category": "personal", "priority": "P4", "capability": "open_vocabulary", "aliases": ["wallet", "purse"]},
    "cane": {"category": "personal", "priority": "P4", "capability": "open_vocabulary", "aliases": ["walking cane", "cane"]},
    "glasses": {"category": "personal", "priority": "P4", "capability": "open_vocabulary", "aliases": ["glasses", "sunglasses"]},
    "medication_bottle": {"category": "personal", "priority": "P4", "capability": "zero_shot_ocr", "aliases": ["pill bottle", "medicine bottle"]},
    "plate": {"category": "personal", "priority": "P4", "capability": "open_vocabulary", "aliases": ["plate", "dish"]},
    "backpack": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["backpack", "bag"]},
    "bag": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["handbag", "bag", "purse"]},
    "mouse": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["computer mouse", "mouse"]},
    "remote": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["remote", "controller"]},
    "book": {"category": "personal", "priority": "P4", "capability": "coco_supported", "aliases": ["book", "notebook"]},
    # Unsupported Micro-features (honestly reported)
    "door_handle": {"category": "wayfinding", "priority": "P1", "capability": "unsupported", "aliases": ["door handle", "doorknob"]},
    "elevator_button": {"category": "wayfinding", "priority": "P1", "capability": "unsupported", "aliases": ["elevator button", "call button"]},
    "cord": {"category": "critical_obstacle", "priority": "P0", "capability": "unsupported", "aliases": ["cord", "cable", "wire"]},
}

TARGET_CLASSES = list(ACCESSIBILITY_CLASSES.keys())


class HeuristicSpecializedDetector(SpecializedDetector):
    """Conservative starter: maps COCO proxies and specialized accessibility cues."""

    PROXY = {
        "traffic light": ("traffic_signal", 0.4),
        "stop sign": ("pedestrian_crossing", 0.35),
    }

    def supported_classes(self) -> list[str]:
        return ["traffic_signal", "pedestrian_crossing"]

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        return []

    def from_general_detections(self, detections: list[RawDetection]) -> list[RawDetection]:
        specialized: list[RawDetection] = []
        for det in detections:
            proxy = self.PROXY.get(det.class_name.lower())
            if not proxy:
                continue
            name, cap = proxy
            specialized.append(
                RawDetection(
                    class_name=name,
                    confidence=min(det.confidence, cap),
                    bbox_xyxy=det.bbox_xyxy,
                    track_id=det.track_id,
                )
            )
        return specialized


class FineTunedSpecializedDetector(SpecializedDetector):
    """Loaded only when a trained accessibility weight file exists."""

    def __init__(self, weights_path: str, confidence: float = 0.35):
        from ultralytics import YOLO

        self._model = YOLO(weights_path)
        self._confidence = confidence

    def supported_classes(self) -> list[str]:
        return TARGET_CLASSES

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(image, conf=self._confidence, verbose=False, imgsz=640)
        detections: list[RawDetection] = []
        if not results or results[0].boxes is None:
            return detections
        names = results[0].names
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            detections.append(
                RawDetection(
                    class_name=str(names.get(cls_id, cls_id)),
                    confidence=float(box.conf[0].item()),
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )
        return detections


# Alias for explicit multi-model detector architecture
AccessibilityDetector = HeuristicSpecializedDetector
