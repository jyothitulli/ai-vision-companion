from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    success: bool = True
    mode: str
    answer: str
    objects: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    path: dict[str, Any] | None = None
    scene: dict[str, Any] | None = None
    confidence: float
    processing_time_ms: float
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    request_id: str
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict[str, str]
    database: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    request_id: str


class TranscribeResponse(BaseModel):
    success: bool = True
    text: str
    processing_time_ms: float
    request_id: str
