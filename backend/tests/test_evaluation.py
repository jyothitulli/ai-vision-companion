import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "evaluation"))

from evaluate import detection_metrics, latency_report  # noqa: E402


def test_detection_metrics_perfect_match() -> None:
    bbox = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.4}
    metrics = detection_metrics(
        [{"type": "chair", "bbox": bbox}],
        [{"type": "chair", "bbox": bbox}],
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_latency_report_empty() -> None:
    report = latency_report([])
    assert report["count"] == 0
    assert report["mean_ms"] is None
