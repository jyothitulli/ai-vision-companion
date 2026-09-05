from __future__ import annotations

import logging

import numpy as np

from app.config import Settings
from app.vision.interfaces import DepthEstimator

logger = logging.getLogger(__name__)


class DepthAnythingEstimator(DepthEstimator):
    """Depth Anything V2 Small via Hugging Face transformers.

    Output is relative depth, not metric distance. Callers must treat values as
    ordinal (nearer/farther in this frame) and convert to spoken ranges with
    an explicit uncalibrated heuristic.
    """

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = torch.device(settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(settings.depth_model)
        self._model = AutoModelForDepthEstimation.from_pretrained(settings.depth_model)
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch
        logger.info("depth_model_loaded", extra={"model": settings.depth_model, "device": str(self._device)})

    def estimate(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            predicted = outputs.predicted_depth
        depth = self._torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        return depth.astype(np.float32)


class UnavailableDepthEstimator(DepthEstimator):
    def estimate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.float32)
