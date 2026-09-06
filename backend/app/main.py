from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.config import get_settings
from app.db.session import init_db
from app.logging import configure_logging, new_request_id, request_id_ctx
from app.rate_limit import limiter
from app.services.runtime import warmup

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(settings.app_debug)
    await init_db()
    if settings.enable_model_warmup:
        warmup()
    yield


app = FastAPI(
    title="Around You API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
)
app.include_router(router, prefix="/api")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Something went wrong. Please try again.",
                "request_id": request_id,
            },
        )
    response.headers["X-Request-Id"] = request_id
    return response


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    index_path = STATIC_DIR / "index.html"
    if "text/html" in accept and index_path.is_file():
        return FileResponse(index_path)
    return {"name": "Around You", "tagline": "Understand the world around you.", "docs": "/docs", "health": "/api/health", "app": "/app"}


@app.get("/app")
async def app_view():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"error": "Web UI not found"})
