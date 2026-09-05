import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION") != "1",
    reason="Set RUN_MODEL_INTEGRATION=1 after weights are downloaded.",
)


def test_detector_returns_structured_objects() -> None:
    import numpy as np

    from app.config import Settings
    from app.reasoning.nlg import Intent
    from app.vision.pipeline import VisionPipeline, build_runtime

    settings = Settings(enable_depth=False, enable_ocr=False, enable_yolo_world=False, enable_model_warmup=False)
    pipeline = VisionPipeline(build_runtime(settings))
    image = np.zeros((480, 640, 3), dtype="uint8")
    result = pipeline.analyze(image, Intent(mode="look", question="Look."))
    assert result.answer
    assert result.confidence >= 0
    assert "2.37" not in result.answer
