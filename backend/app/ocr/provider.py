from __future__ import annotations

import logging
import re

import numpy as np

from app.vision.interfaces import OCRDocument, OCRLine, OCRProvider
from app.vision.types import BoundingBox

logger = logging.getLogger(__name__)


class RapidOCRProvider(OCRProvider):
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> OCRDocument:
        result, _ = self._engine(image)
        lines: list[OCRLine] = []
        if not result:
            return OCRDocument(lines=[], full_text="", structured={"paragraphs": []})
        h, w = image.shape[:2]
        texts: list[str] = []
        for item in result:
            box, text, score = item[0], item[1], float(item[2])
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            lines.append(
                OCRLine(
                    text=text,
                    confidence=score,
                    bbox=BoundingBox(
                        x=x1 / w,
                        y=y1 / h,
                        width=(x2 - x1) / w,
                        height=(y2 - y1) / h,
                    ),
                )
            )
            texts.append(text)
        full = "\n".join(texts)
        return OCRDocument(lines=lines, full_text=full, structured=structure_ocr(full, lines))


class PaddleOCRProvider(OCRProvider):
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def read(self, image: np.ndarray) -> OCRDocument:
        raw = self._engine.ocr(image, cls=True)
        lines: list[OCRLine] = []
        h, w = image.shape[:2]
        texts: list[str] = []
        pages = raw or []
        for page in pages:
            if not page:
                continue
            for item in page:
                box, (text, score) = item
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                lines.append(
                    OCRLine(
                        text=str(text),
                        confidence=float(score),
                        bbox=BoundingBox(
                            x=x1 / w,
                            y=y1 / h,
                            width=(x2 - x1) / w,
                            height=(y2 - y1) / h,
                        ),
                    )
                )
                texts.append(str(text))
        full = "\n".join(texts)
        return OCRDocument(lines=lines, full_text=full, structured=structure_ocr(full, lines))


def structure_ocr(full_text: str, lines: list[OCRLine]) -> dict:
    amounts = re.findall(r"(?:[$€£]|USD|INR|Rs\.?)\s?\d+(?:[.,]\d{2})?", full_text, flags=re.I)
    totals = re.findall(r"(?:total|amount due|grand total)\s*[:\-]?\s*([^\n]+)", full_text, flags=re.I)
    prices = re.findall(r"\d+[.,]\d{2}", full_text)
    return {
        "paragraphs": [line.text for line in lines],
        "currency_mentions": amounts,
        "totals": [item.strip() for item in totals],
        "price_like": prices[:20],
        "line_count": len(lines),
        "mean_confidence": round(sum(line.confidence for line in lines) / max(len(lines), 1), 3),
    }


def build_ocr_provider() -> OCRProvider:
    try:
        provider = PaddleOCRProvider()
        logger.info("ocr_provider", extra={"impl": "paddleocr"})
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("paddleocr_unavailable", extra={"error": str(exc)})
        provider = RapidOCRProvider()
        logger.info("ocr_provider", extra={"impl": "rapidocr"})
        return provider
