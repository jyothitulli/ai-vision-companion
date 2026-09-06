from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.reasoning.nlg import Intent
from app.services import runtime as runtime_mod
from app.vision.pipeline import VisionPipeline
from app.vision.types import PathAnalysis, PipelineResult, SceneUnderstanding


class FakePipeline(VisionPipeline):
    def __init__(self) -> None:  # noqa: D401
        self.runtime = MagicMock()
        self.runtime.intents.parse.side_effect = lambda text, explicit_mode=None: Intent(
            mode=explicit_mode or "look", question=text or "", target="bottle" if "bottle" in (text or "") else None
        )

    def analyze(self, image, intent, timestamp_s=None, persist_tracks=False):  # noqa: ANN001
        assert image is not None
        return PipelineResult(
            objects=[],
            path=PathAnalysis(path_status="uncertain"),
            scene=SceneUnderstanding(scene_type="unknown", description="I cannot confidently determine the scene.", confidence=0.2, uncertain=True),
            events=[],
            answer="A chair is approximately one to two meters ahead and may obstruct your path.",
            confidence=0.8,
            latencies_ms={"detection_ms": 12.0},
        )


def _jpeg() -> bytes:
    import cv2

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_upload(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: FakePipeline())
    client = TestClient(app)
    response = client.post("/api/vision/analyze", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "chair" in body["answer"]


def test_ask_and_find_and_read(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: FakePipeline())
    client = TestClient(app)
    jpeg = _jpeg()
    ask = client.post(
        "/api/vision/ask",
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        data={"question": "What is in front of me?"},
    )
    assert ask.status_code == 200
    find = client.post(
        "/api/vision/find",
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        data={"target": "bottle"},
    )
    assert find.status_code == 200
    read = client.post(
        "/api/vision/read",
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        data={"question": "Read this."},
    )
    assert read.status_code == 200


def test_assistance_start_stop_and_frame(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: FakePipeline())
    client = TestClient(app)
    start = client.post("/api/assistance/start", data={"session_id": "s1"})
    assert start.status_code == 200
    frame = client.post(
        "/api/assistance/frame",
        files={"file": ("frame.jpg", _jpeg(), "image/jpeg")},
        data={"session_id": "s1"},
    )
    assert frame.status_code == 200
    stop = client.post("/api/assistance/stop", data={"session_id": "s1"})
    assert stop.status_code == 200


def test_rejects_oversized_payload(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: FakePipeline())
    from app.config import get_settings

    settings = get_settings()
    client = TestClient(app)
    huge = b"x" * (settings.max_upload_bytes + 10)
    response = client.post("/api/vision/analyze", files={"file": ("frame.jpg", huge, "image/jpeg")})
    assert response.status_code == 413


def test_rejects_non_image(monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: FakePipeline())
    client = TestClient(app)
    response = client.post(
        "/api/vision/ask",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"question": "what is ahead"},
    )
    assert response.status_code == 400


import pytest


@pytest.mark.anyio
async def test_concurrency_non_blocking_event_loop(monkeypatch) -> None:
    import asyncio
    import time
    from httpx import ASGITransport, AsyncClient

    class SlowBlockingPipeline(FakePipeline):
        def analyze(self, image, intent, timestamp_s=None, persist_tracks=False):
            time.sleep(0.3)  # Synchronous blocking call
            return super().analyze(image, intent, timestamp_s, persist_tracks)

    monkeypatch.setattr(runtime_mod, "get_pipeline", lambda: SlowBlockingPipeline())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        t0 = time.perf_counter()

        async def send_vision():
            return await ac.post(
                "/api/vision/analyze",
                files={"file": ("frame.jpg", _jpeg(), "image/jpeg")},
            )

        async def send_health():
            # Wait a tiny fraction so vision request starts first
            await asyncio.sleep(0.05)
            h_start = time.perf_counter()
            resp = await ac.get("/api/health")
            h_duration = time.perf_counter() - h_start
            return resp, h_duration

        vision_task = asyncio.create_task(send_vision())
        health_task = asyncio.create_task(send_health())

        vision_res, (health_res, health_duration) = await asyncio.gather(vision_task, health_task)

        assert vision_res.status_code == 200
        assert health_res.status_code == 200
        # Health check took far less than the 0.3s blocking sleep of vision analyze
        assert health_duration < 0.25

