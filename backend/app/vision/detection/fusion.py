from __future__ import annotations

import logging
from app.vision.interfaces import RawDetection

logger = logging.getLogger(__name__)


def calculate_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class DetectionFusion:
    """Fuses multi-model detection outputs with IoU duplicate suppression and confidence calibration."""

    def __init__(self, iou_threshold: float = 0.45):
        self.iou_threshold = iou_threshold

    def fuse(self, detector_outputs: list[list[RawDetection]]) -> list[RawDetection]:
        # Flatten all detection candidates
        all_candidates: list[RawDetection] = []
        for output in detector_outputs:
            if output:
                all_candidates.extend(output)

        if not all_candidates:
            return []

        # Sort descending by confidence
        sorted_candidates = sorted(all_candidates, key=lambda d: d.confidence, reverse=True)
        fused: list[RawDetection] = []
        suppressed_indices: set[int] = set()

        for i, lead in enumerate(sorted_candidates):
            if i in suppressed_indices:
                continue
            fused.append(lead)

            for j in range(i + 1, len(sorted_candidates)):
                if j in suppressed_indices:
                    continue
                comp = sorted_candidates[j]

                # Check if classes are identical or compatible synonyms
                same_class = (
                    lead.class_name.lower() == comp.class_name.lower()
                    or lead.class_name.lower() in comp.class_name.lower()
                    or comp.class_name.lower() in lead.class_name.lower()
                )

                iou = calculate_iou(lead.bbox_xyxy, comp.bbox_xyxy)
                # Suppress duplicates with high spatial overlap
                if same_class and iou >= self.iou_threshold:
                    suppressed_indices.add(j)
                # Also suppress almost identical bounding boxes even if class names slightly vary
                elif iou >= 0.75:
                    suppressed_indices.add(j)

        return fused
