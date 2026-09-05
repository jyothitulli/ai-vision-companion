"""
Vision Companion: Multi-Object Detection & Pipeline Comparison Tool
Usage:
    python tools/compare_detections.py <image_path> [--mode look|ask|find|read] [--target <object_to_find>] [--question "<question>"]
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure backend package is in python path
backend_path = (Path(__file__).resolve().parent.parent / "backend").resolve()
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings
from app.reasoning.nlg import Intent
from app.vision.pipeline import VisionPipeline, build_runtime
from app.vision.specialized.detector import ACCESSIBILITY_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Companion Detection Comparison Tool")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--mode", type=str, default="look", choices=["look", "ask", "find", "read"], help="Interaction mode")
    parser.add_argument("--question", type=str, default="What do you see?", help="Question for ask mode")
    parser.add_argument("--target", type=str, default=None, help="Target for find mode")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"Error: Could not read image at {img_path}")
        sys.exit(1)

    h, w = image_bgr.shape[:2]
    settings = Settings()
    runtime = build_runtime(settings)
    pipeline = VisionPipeline(runtime)

    intent = Intent(mode=args.mode, question=args.question, target=args.target)

    t_start = time.perf_counter()
    result = pipeline.analyze(image_bgr, intent)
    total_ms = (time.perf_counter() - t_start) * 1000

    print("\n" + "=" * 60)
    print("VISION COMPANION DETECTION REPORT")
    print("=" * 60)
    print(f"Image: {img_path.name} ({w}x{h})")
    print(f"Mode: {args.mode.upper()}" + (f" (Target: {args.target})" if args.target else ""))
    print(f"Overall Confidence: {result.confidence:.2f}")

    print("\nDetected Objects:")
    print("-" * 60)
    if not result.objects:
        print("  None detected.")
    for i, obj in enumerate(result.objects, 1):
        rel_str = f" | Relations: {', '.join(obj.relationships)}" if getattr(obj, "relationships", []) else ""
        print(f"{i}. {obj.type.upper()}")
        print(f"   Confidence: {obj.confidence:.2f}")
        print(f"   Position: {obj.position.value}")
        print(f"   Depth: approximately {obj.distance_range} ({obj.distance_band.value})")
        print(f"   Path Relevance: {obj.path_relevance.value}{rel_str}")

    print("\nPotentially Missed Requested / Accessibility Classes:")
    print("-" * 60)
    detected_types = {obj.type.lower() for obj in result.objects}
    # Check accessibility vocabulary
    notable_to_check = ["stairs_up", "stairs_down", "door", "curb", "ramp", "crosswalk", "person", "chair", "table", "phone", "bottle"]
    for cls_name in notable_to_check:
        meta = ACCESSIBILITY_CLASSES.get(cls_name, {})
        aliases = meta.get("aliases", [cls_name])
        found = any(alias in detected_types or any(alias in d for d in detected_types) for alias in aliases)
        if not found:
            cap = meta.get("capability", "unsupported")
            print(f"  - {cls_name:<20} [Status: {cap}] (not observed in this view)")

    print("\nSpoken Response:")
    print("-" * 60)
    print(f"\"{result.answer}\"")

    print("\nPipeline Timing:")
    print("-" * 60)
    for stage, ms in result.latencies_ms.items():
        print(f"  {stage:<18}: {ms:6.1f} ms")
    print(f"  {'Total':<18}: {total_ms:6.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
