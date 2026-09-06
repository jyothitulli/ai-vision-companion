# Vision Companion: Perception Limitations & Safety Disclaimers

This document provides a transparent, evidence-based breakdown of the Vision Companion perception system's operational boundaries, model capabilities, and physical constraints.

---

## 1. Object Detection Capability Tiers

The system vocabulary contains 50+ classes, but they are **not** all equally detectable. The perception pipeline explicitly separates targets into four distinct capability tiers:

### Tier A: COCO-Supported Classes (Active & Verified)
* **Classes**: `person`, `chair`, `dining table` / `desk`, `laptop`, `cell phone`, `bottle`, `cup`, `mouse`, `keyboard`, `backpack`, `handbag` / `bag`, `suitcase`, `bicycle`, `car`, `bus`, `traffic light`, `bench`, `book`.
* **Model**: YOLO11n (`yolo11n.pt`) with class-calibrated confidence thresholds (e.g., 0.20 for small objects like `cell phone` and `mouse`, 0.30 for `person` and `chair`).
* **Operational Status**: Fully supported, evaluated on photographic scenes, and integrated into LOOK, ASK, and FIND modes.

### Tier B: Open-Vocabulary Classes (Experimental / Zero-Shot)
* **Classes**: `white_cane`, `crosswalk`, `pedestrian_light`, `elevator_door`, `tactile_paving`, `handrail`, `grab_bar`, `trash_can`, `fire_extinguisher`, `water_fountain`, `charging_station`, `bus_stop_sign`, `vending_machine`, `pillar`, `bollard`, `scooter`.
* **Model**: `OpenVocabularyDetector` architecture (designed for YOLO-World / `yolov8s-worldv2.pt`).
* **Operational Status**: When open-vocabulary weights are absent locally, the pipeline executes a transparent fallback to the calibrated base detector and informs the user if a queried class cannot be detected, rather than hallucinating.

### Tier C: Specialized Accessibility Hazards (Geometry & Line-of-Sight)
* **Classes**: `stairs_up`, `stairs_down`, `drop_off`, `curb`, `ramp`, `low_ceiling`, `overhanging_obstacle`, `uneven_surface`.
* **Model**: Depth gradient heuristic analyzer and `FineTunedSpecializedDetector` interface (`training/train_accessibility.py`).
* **Operational Status**: Requires clear line of sight to the floor surface and adequate lighting. If camera pitch does not capture the ground plane, the system explicitly responds: *"I don't currently see stairs. Stair detection requires clear line of sight to the floor."*

### Tier D: Unsupported Micro-Features & Transparent Surfaces
* **Classes**: `door_handle`, `elevator_button`, `keyhole`, `puddle`, `slippery_floor`, `braille_signage`, `card_reader`, `cord` / `wire`, `glass_door`.
* **Operational Status**: **NOT RELIABLY SUPPORTED.** At typical room distances (1–4 meters), camera resolution (640×640) and standard RGB optics cannot reliably resolve millimeter-scale features or optical transparency.
* **Guarantee**: The language layer will **never hallucinate** these items and will honestly instruct the user: *"Small features like elevator buttons cannot be reliably recognized by this model at a distance."*

---

## 2. Depth Perception: Relative vs. Metric Distance

* **Model**: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`).
* **Fundamental Constraint**: Monocular depth estimation infers an **affine-invariant relative depth map**, not calibrated metric distances.
* **Camera Calibration**: Without known focal length, sensor height, and pitch angle, reporting exact values (e.g. *"2.37 meters"*) is scientifically ungrounded and misleading.
* **System Policy**: Depth values are mapped into coarse, qualitative communicative bands:
  - `VERY_NEAR`: *"less than one meter"*
  - `NEAR`: *"one to two meters"*
  - `MID`: *"two to three meters"*
  - `FAR`: *"more than three meters"*
* Spoken output always includes hedging terms like *"approximately"* to prevent false reliance.

---

## 3. Spatial Relationships & Grounding

* Spatial relationships (`"on the table"`, `"to the left of the laptop"`) are computed using geometric containment, vertical edge support, and depth band agreement.
* Each relationship is assigned a computed confidence score. Only relationships with confidence $\ge 0.50$ are verbalized.
* Spatial relationships are never hallucinated when bounding-box geometry does not support physical contact or elevation agreement.

---

## 4. Safety & Navigation Disclaimers

1. **Advisory Tool Only**: Vision Companion is an assistive perception tool designed to describe scene context. It is **not** an autonomous navigation system or white cane replacement.
2. **Prohibited Claims**: The NLG layer contains hardcoded safety filters that reject statements such as *"The road is safe"*, *"Cross now"*, or *"You can proceed"*.
3. **Street Crossing**: Traffic lights and crosswalks are detected as informational visual context. The system explicitly reminds the user that it cannot determine whether crossing is safe.
