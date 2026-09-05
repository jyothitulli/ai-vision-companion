import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    "path",
    [
        "/api/vision/analyze",
        "/api/vision/ask",
        "/api/vision/read",
        "/api/vision/find",
        "/api/assistance/start",
        "/api/assistance/stop",
    ],
)
def test_endpoints_exist(path: str) -> None:
    client = TestClient(app)
    response = client.post(path)
    assert response.status_code != 404
