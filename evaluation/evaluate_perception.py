"""Perception System Evaluation Framework.

Computes precision, recall, F1, small-object recall, and latency breakdown
on real photographic test scenes comparing baseline vs. upgraded pipelines.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings
from app.reasoning.nlg import Intent
from app.vision.detection.yolo import YOLODetector
from app.vision.pipeline import VisionPipeline, build_runtime


def run_evaluation() -> dict:
    gt_path = ROOT / "tests" / "assets" / "scenes" / "ground_truth.json"
    if not gt_path.exists():
        print(f"Error: {gt_path} not found.")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    scenes_dir = gt_path.parent

    settings = Settings()
    runtime = build_runtime(settings)
    pipeline = VisionPipeline(runtime)

    # Baseline detector (pure 0.35 global threshold, no overrides)
    baseline_yolo = YOLODetector(settings, confidence=0.35, enable_overrides=False)

    results = {
        "baseline": {"tp": 0, "fp": 0, "fn": 0, "small_tp": 0, "small_total": 0, "latencies": []},
        "upgraded": {"tp": 0, "fp": 0, "fn": 0, "small_tp": 0, "small_total": 0, "latencies": []},
    }

    print("=" * 70)
    print("VISION COMPANION: REAL MULTI-OBJECT ACCURACY BENCHMARK")
    print("=" * 70)

    for sc in scenes:
        img_file = scenes_dir / sc["file"]
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        gt_objs = [o.lower() for o in sc.get("ground_truth_objects", [])]
        small_objs = [s.lower() for s in sc.get("small_objects", [])]
        results["baseline"]["small_total"] += len(small_objs)
        results["upgraded"]["small_total"] += len(small_objs)

        print(f"\nEvaluating Scene: {sc['file']}")
        print(f"  Ground Truth: {', '.join(gt_objs)}")

        # 1. Run Baseline
        t0 = time.perf_counter()
        raw_base = baseline_yolo.detect(img)
        t_base = (time.perf_counter() - t0) * 1000
        results["baseline"]["latencies"].append(t_base)
        base_det_names = [d.class_name.lower() for d in raw_base]
        print(f"  Baseline Detections (conf>=0.35): {base_det_names}")

        for gt in gt_objs:
            if any(gt in d or d in gt for d in base_det_names):
                results["baseline"]["tp"] += 1
            else:
                results["baseline"]["fn"] += 1

        for d in base_det_names:
            if not any(gt in d or d in gt for gt in gt_objs):
                results["baseline"]["fp"] += 1

        for sm in small_objs:
            if any(sm in d or d in sm for d in base_det_names):
                results["baseline"]["small_tp"] += 1

        # 2. Run Upgraded Pipeline
        t0 = time.perf_counter()
        res_upgraded = pipeline.analyze(img, Intent(mode="look", question="what do you see"))
        t_up = (time.perf_counter() - t0) * 1000
        results["upgraded"]["latencies"].append(t_up)
        up_det_names = [o.type.lower() for o in res_upgraded.objects]
        print(f"  Upgraded Detections (calibrated):  {up_det_names}")

        for gt in gt_objs:
            if any(gt in d or d in gt for d in up_det_names):
                results["upgraded"]["tp"] += 1
            else:
                results["upgraded"]["fn"] += 1

        for d in up_det_names:
            if not any(gt in d or d in gt for gt in gt_objs):
                results["upgraded"]["fp"] += 1

        for sm in small_objs:
            if any(sm in d or d in sm for d in up_det_names):
                results["upgraded"]["small_tp"] += 1

    # Calculate metrics
    def calc_metrics(bucket: dict) -> dict:
        tp = bucket["tp"]
        fp = bucket["fp"]
        fn = bucket["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        sm_rec = bucket["small_tp"] / bucket["small_total"] if bucket["small_total"] > 0 else 0.0
        avg_lat = float(np.mean(bucket["latencies"])) if bucket["latencies"] else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "small_recall": sm_rec,
            "avg_latency_ms": avg_lat,
        }

    m_base = calc_metrics(results["baseline"])
    m_up = calc_metrics(results["upgraded"])

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Baseline (0.35)':<18} | {'Upgraded (Calibrated)':<22}")
    print("-" * 70)
    print(f"{'True Positives (TP)':<25} | {m_base['tp']:<18} | {m_up['tp']:<22}")
    print(f"{'False Positives (FP)':<25} | {m_base['fp']:<18} | {m_up['fp']:<22}")
    print(f"{'False Negatives (FN)':<25} | {m_base['fn']:<18} | {m_up['fn']:<22}")
    print(f"{'Overall Precision':<25} | {m_base['precision']*100:6.1f}%{' '*11} | {m_up['precision']*100:6.1f}%")
    print(f"{'Overall Recall':<25} | {m_base['recall']*100:6.1f}%{' '*11} | {m_up['recall']*100:6.1f}%")
    print(f"{'F1 Score':<25} | {m_base['f1']:6.3f}{' '*12} | {m_up['f1']:6.3f}")
    print(f"{'Small-Object Recall':<25} | {m_base['small_recall']*100:6.1f}%{' '*11} | {m_up['small_recall']*100:6.1f}%")
    print(f"{'Average Latency':<25} | {m_base['avg_latency_ms']:6.1f} ms{' '*10} | {m_up['avg_latency_ms']:6.1f} ms")
    print("=" * 70 + "\n")

    return {"baseline": m_base, "upgraded": m_up}


if __name__ == "__main__":
    run_evaluation()
