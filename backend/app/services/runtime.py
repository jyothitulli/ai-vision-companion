from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings, get_settings
from app.speech.whisper import WhisperRecognizer
from app.vision.pipeline import VisionPipeline, VisionRuntime, build_runtime

logger = logging.getLogger(__name__)

_runtime: Optional[VisionRuntime] = None
_pipeline: Optional[VisionPipeline] = None
_speech: Optional[WhisperRecognizer] = None


def get_runtime(settings: Settings | None = None) -> VisionRuntime:
    global _runtime
    if _runtime is None:
        _runtime = build_runtime(settings or get_settings())
    return _runtime


def get_pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline(get_runtime())
    return _pipeline


def get_speech() -> WhisperRecognizer:
    global _speech
    if _speech is None:
        _speech = WhisperRecognizer(get_settings())
    return _speech


def warmup() -> None:
    try:
        get_pipeline()
        logger.info("vision_pipeline_ready")
    except Exception:
        logger.exception("vision_pipeline_warmup_failed")
    try:
        get_speech()
        logger.info("speech_ready")
    except Exception:
        logger.exception("speech_warmup_failed")
