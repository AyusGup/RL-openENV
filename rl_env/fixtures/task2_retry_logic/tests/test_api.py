"""API tests for the retry incident fixture."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import UPSTREAM_RESPONSES, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_upstream_responses() -> None:
    UPSTREAM_RESPONSES[:] = [503, 503, 503, 200]


def test_upstream_health_eventually_recovers() -> None:
    response = client.get("/api/upstream/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "upstream_status": 200}
