from __future__ import annotations

import logging

from app.config import Settings
from app.vision.interfaces import SpeechRecognizer

logger = logging.getLogger(__name__)


class WhisperRecognizer(SpeechRecognizer):
    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(settings.whisper_model, device=settings.device, compute_type="int8")
        logger.info("whisper_loaded", extra={"model": settings.whisper_model})

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
