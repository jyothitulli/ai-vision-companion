
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AnalyzeResponse, HealthResponse, TranscribeResponse
from app.config import Settings, get_settings
from app.db.models import DetectionEvent, Interaction
from app.db.session import get_db
from app.logging import request_id_ctx
from app.rate_limit import limiter
from app.reasoning.nlg import Intent
from app.services.assistance import assistance_manager
from app.services import runtime as runtime_mod
from app.vision.pipeline import decode_image

get_pipeline = runtime_mod.get_pipeline
get_speech = runtime_mod.get_speech

logger = logging.getLogger(__name__)
router = APIRouter()

SAFE_USER_ERRORS = {
    "invalid_image": "I couldn't process the image. Please try again.",
    "image_too_large": "The photo is too large. Please capture again.",
    "unsupported_type": "That image type isn't supported. Use JPEG or PNG.",
    "missing_image": "I didn't receive a photo.",
    "model_unavailable": "The vision model isn't ready. Please try again in a moment.",
    "stt_failure": "I couldn't hear that clearly. Please try again.",
}


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.app_env == "development":
        if settings.api_key in {"change-me-in-production", "dev-local-key", "", "none"}:
            return
        if not x_api_key or x_api_key == settings.api_key:
            return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="I couldn't authorize this request.")


def _validate_upload(file: UploadFile, settings: Settings) -> None:
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type and content_type not in settings.allowed_image_type_set:
        raise HTTPException(status_code=400, detail=SAFE_USER_ERRORS["unsupported_type"])


async def _read_image(file: UploadFile, settings: Settings) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=SAFE_USER_ERRORS["missing_image"])
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=SAFE_USER_ERRORS["image_too_large"])
    return data


def _to_response(mode: str, result, processing_ms: float, session_id: Optional[str]) -> AnalyzeResponse:
    return AnalyzeResponse(
        success=True,
        mode=mode,
        answer=result.answer,
        objects=[obj.model_dump() for obj in result.objects],
        events=[event.model_dump() for event in result.events],
        path=result.path.model_dump(),
        scene=result.scene.model_dump(),
        confidence=result.confidence,
        processing_time_ms=round(processing_ms, 1),
        latencies_ms=result.latencies_ms,
        warnings=result.warnings,
        request_id=request_id_ctx.get(),
        session_id=session_id,
    )


async def _persist(
    db: AsyncSession,
    mode: str,
    question: Optional[str],
    result,
    processing_ms: float,
    session_id: Optional[str],
) -> None:
    db.add(
        Interaction(
            session_id=session_id,
            mode=mode,
            question=question,
            response=result.answer,
            latency_ms=int(processing_ms),
            confidence=result.confidence,
        )
    )
    for event in result.events:
        if event.should_announce:
            db.add(
                DetectionEvent(
                    session_id=session_id,
                    type=event.type,
                    confidence=event.confidence,
                    priority=event.priority.value,
                    announced=True,
                )
            )
    await db.commit()


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    models = {
        "detector": settings.yolo_model,
        "depth": settings.depth_model if settings.enable_depth else "disabled",
        "ocr": "configured",
        "stt": settings.whisper_model,
        "warmup": str(settings.enable_model_warmup),
    }
    db_kind = "sqlite" if "sqlite" in settings.resolved_database_url() else "postgresql"
    return HealthResponse(status="ok", device=settings.device, models=models, database=db_kind)


@router.post("/vision/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_analyze)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    return await _run_mode("look", file, None, session_id, db, settings)


@router.post("/vision/ask", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_analyze)
async def ask(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    return await _run_mode("ask", file, question, session_id, db, settings)


@router.post("/vision/read", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_analyze)
async def read(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(default="Read this."),
    session_id: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    return await _run_mode("read", file, question, session_id, db, settings)


@router.post("/vision/find", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_analyze)
async def find(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(...),
    session_id: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    return await _run_mode("find", file, f"Find my {target}", session_id, db, settings, target=target)


@router.post("/assistance/start", dependencies=[Depends(require_api_key)])
async def assistance_start(session_id: str = Form(...)) -> dict:
    assistance_manager.start(session_id)
    return {
        "success": True,
        "mode": "assistance",
        "answer": "Continuous assistance is on. I will only announce important changes.",
        "session_id": session_id,
        "request_id": request_id_ctx.get(),
    }


@router.post("/assistance/stop", dependencies=[Depends(require_api_key)])
async def assistance_stop(session_id: str = Form(...)) -> dict:
    assistance_manager.stop(session_id)
    return {
        "success": True,
        "mode": "assistance",
        "answer": "Continuous assistance is off.",
        "session_id": session_id,
        "request_id": request_id_ctx.get(),
    }


@router.post("/assistance/frame", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_analyze)
async def assistance_frame(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    if not assistance_manager.is_active(session_id):
        assistance_manager.start(session_id)
    response = await _run_mode("assistance", file, "continuous", session_id, db, settings, persist_tracks=True)
    return response


@router.post("/speech/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(get_settings().rate_limit_speech)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> TranscribeResponse:
    import tempfile
    from pathlib import Path

    started = time.perf_counter()
    suffix = Path(file.filename or "speech.wav").suffix or ".wav"
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=SAFE_USER_ERRORS["stt_failure"])
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            text = runtime_mod.get_speech().transcribe(tmp.name)
    except Exception:
        logger.exception("stt_failed")
        raise HTTPException(status_code=500, detail=SAFE_USER_ERRORS["stt_failure"]) from None
    return TranscribeResponse(
        success=True,
        text=text,
        processing_time_ms=round((time.perf_counter() - started) * 1000, 1),
        request_id=request_id_ctx.get(),
    )


async def _run_mode(
    mode: str,
    file: UploadFile,
    question: Optional[str],
    session_id: Optional[str],
    db: AsyncSession,
    settings: Settings,
    target: Optional[str] = None,
    persist_tracks: bool = False,
) -> AnalyzeResponse:
    started = time.perf_counter()
    _validate_upload(file, settings)
    data = await _read_image(file, settings)
    try:
        image = decode_image(data)
    except ValueError:
        raise HTTPException(status_code=400, detail=SAFE_USER_ERRORS["invalid_image"]) from None
    # Image bytes are local to this request and are not written to disk.
    try:
        pipeline = runtime_mod.get_pipeline()
        parser = pipeline.runtime.intents
        intent = parser.parse(question or "Look.", explicit_mode=mode if mode != "assistance" else "look")
        if mode == "find":
            intent = Intent(mode="find", question=question or "", target=target)
        if mode == "assistance":
            intent = Intent(mode="look", question="continuous")
        result = pipeline.analyze(image, intent, persist_tracks=persist_tracks or mode == "assistance")
        if mode == "assistance":
            spoken = assistance_manager.ingest(session_id or "default", result, intent)
            result.answer = spoken or ""
        processing_ms = (time.perf_counter() - started) * 1000
        try:
            await _persist(db, mode, question, result, processing_ms, session_id)
        except Exception:
            logger.exception("persist_failed")
        logger.info(
            "vision_request",
            extra={
                "mode": mode,
                "processing_time_ms": round(processing_ms, 1),
                "confidence": result.confidence,
                "object_count": len(result.objects),
            },
        )
        return _to_response(mode, result, processing_ms, session_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("pipeline_failed")
        raise HTTPException(status_code=500, detail=SAFE_USER_ERRORS["model_unavailable"]) from None
