from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def detection_metrics(preds: list[dict], gts: list[dict], iou_threshold: float = 0.5) -> dict:
    """Minimal precision/recall/F1 for a single image set. Not a COCO evaluator."""
    tp = fp = fn = 0
    used = set()
    for pred in preds:
        match = None
        for index, gt in enumerate(gts):
            if index in used or pred["type"] != gt["type"]:
                continue
            if _iou(pred["bbox"], gt["bbox"]) >= iou_threshold:
                match = index
                break
        if match is None:
            fp += 1
        else:
            tp += 1
            used.add(match)
    fn = len(gts) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "map_proxy": precision * recall, "tp": tp, "fp": fp, "fn": fn}


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def latency_report(samples_ms: list[float]) -> dict:
    arr = np.array(samples_ms, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else None,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else None,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else None,
    }


def write_report(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path = output.with_suffix(".html")
    results = payload.get("results") or {}
    rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in {
            **(payload.get("targets") or {}),
            **{k: results[k] for k in results},
        }.items()
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Vision Companion evaluation</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:24px}"
        "td,th{padding:8px;border:1px solid #444;text-align:left}</style></head><body>"
        f"<h1>Vision Companion evaluation</h1><p>{payload.get('note')}</p>"
        f"<p>Generated {payload.get('generated_at')}</p><table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
        "<p>This report does not invent accuracy. Empty results mean measurements were not supplied.</p>"
        "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an evaluation report from prediction JSON.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/latest.json"))
    args = parser.parse_args()
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "This script never fabricates model accuracy. Provide predictions JSON to populate metrics.",
        "targets": {
            "object_detection_latency_ms": "measure on device",
            "depth_latency_ms": "measure on device",
            "end_to_end_latency_ms": "measure on device",
            "false_alert_rate": "requires labeled continuous-assistance logs",
            "missed_event_rate": "requires labeled continuous-assistance logs",
        },
        "results": None,
    }
    if args.predictions and args.predictions.exists():
        data = json.loads(args.predictions.read_text(encoding="utf-8"))
        payload["results"] = detection_metrics(data.get("predictions", []), data.get("ground_truth", []))
    write_report(payload, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
