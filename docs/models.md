# Vision Companion: Models & Detection Capability Matrix

This document provides technical documentation of the models running in Vision Companion, their weight dependencies, execution devices, and honest capability statuses.

---

## 1. Running Models Summary

| Subsystem | Model | Weights / Version | Device | Operational Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Detection** | Ultralytics YOLO11n | `yolo11n.pt` | CPU / CUDA | **Active & Evaluated** | 80 COCO classes with class-calibrated confidence overrides. |
| **Monocular Depth** | Depth Anything V2 Small | `Depth-Anything-V2-Small-hf` | CPU / CUDA | **Active & Evaluated** | Relative affine-invariant depth; mapped to coarse distance bands. |
| **Object Tracking** | ByteTrack + IoU Fallback | Ultralytics tracker | CPU | **Active & Evaluated** | Multi-frame track persistence and approaching/receding proxy analysis. |
| **Document / Sign OCR** | RapidOCR ONNX | `ch_PP-OCRv4` ONNX | CPU | **Active & Evaluated** | Offline text detection and recognition for signs, labels, prices. |
| **Speech-to-Text (STT)** | Faster-Whisper | `whisper-tiny` | CPU / CUDA | **Active & Evaluated** | Edge speech transcription with microphone fallback. |
| **Text-to-Speech (TTS)** | Platform TTS (Expo / Edge) | System native | Platform | **Active & Evaluated** | Natural synthesized voice feedback. |
| **Open-Vocabulary** | YOLO-World v2 | `yolov8s-worldv2.pt` | CPU / CUDA | **Experimental / Fallback** | Architecture implemented; gracefully falls back if weights not found locally. |
| **Specialized Accessibility**| Geometry & Depth Heuristics | Rule-based + YOLO finetune | CPU | **Active Interface** | Detects dropoffs, curbs, stairs with line-of-sight disclaimers. |

---

## 2. Capability Matrix: 50+ Vocabulary Classes

Every target class in the Vision Companion vocabulary has an explicit, tested capability tier:

### A. COCO-Supported Classes (Active & Evaluated)
*Genuinely detected by YOLO11n with calibrated thresholds:*
- `person` (0.30)
- `chair` (0.30)
- `table` / `dining table` / `desk` (0.30)
- `laptop` (0.25)
- `cell phone` (0.20)
- `bottle` (0.22)
- `cup` (0.22)
- `mouse` (0.20)
- `keyboard` (0.30)
- `backpack` (0.28)
- `handbag` / `bag` (0.28)
- `suitcase` / `luggage` (0.28)
- `bicycle` (0.30)
- `car` (0.30)
- `bus` (0.30)
- `traffic light` (0.30)
- `bench` (0.30)
- `book` (0.24)

### B. Open-Vocabulary Classes (Zero-Shot / Experimental)
*Supported when open-vocabulary weights are present; transparent fallback with honest disclaimers when absent:*
- `white_cane`
- `crosswalk` / `pedestrian_crossing`
- `pedestrian_light`
- `elevator_door`
- `tactile_paving`
- `handrail`
- `grab_bar`
- `trash_can`
- `fire_extinguisher`
- `water_fountain`
- `charging_station`
- `bus_stop_sign`
- `vending_machine`
- `pillar`
- `bollard`
- `scooter`

### C. Specialized Accessibility Hazards (Geometry & Line-of-Sight)
*Requires clear line of sight to the floor surface; returns disclaimers if ground plane is occluded:*
- `stairs_up`
- `stairs_down`
- `drop_off`
- `curb`
- `ramp`
- `low_ceiling`
- `overhanging_obstacle`
- `uneven_surface`

### D. Unsupported Classes (Micro-Features & Transparent Surfaces)
*Cannot be reliably resolved at camera distance; system explicitly informs the user rather than guessing:*
- `door_handle` / `doorknob`
- `elevator_button`
- `keyhole`
- `puddle`
- `slippery_floor`
- `braille_signage`
- `card_reader`
- `cord` / `wire`
- `glass_door`

---

## 3. Class-Calibrated Confidence Thresholds

Standard object detectors drop small personal objects at full-room scale under rigid global thresholds ($\ge 0.35$). Vision Companion uses class-calibrated activation thresholds:

| Object Class | Global Default | Calibrated Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| `cell phone` | 0.35 | **0.20** | Small feature surface area on desk / hand |
| `mouse` | 0.35 | **0.20** | Low profile, peripheral desk placement |
| `bottle` | 0.35 | **0.22** | Transparent / reflective cylindrical geometry |
| `cup` | 0.35 | **0.22** | Top-down foreshortening |
| `book` | 0.35 | **0.24** | Planar geometry, variable cover art |
| `laptop` | 0.35 | **0.25** | Open / closed aspect ratio variation |
| `person` | 0.35 | **0.30** | High feature distinctiveness |
| `chair` | 0.35 | **0.30** | High training frequency in COCO |
