"""Tests for the retry handler."""

from __future__ import annotations

from app.retry_handler import Response, retry_request


def test_retry_request_succeeds_on_fourth_attempt() -> None:
    attempts = {"count": 0}

    def flaky_request() -> Response:
        attempts["count"] += 1
        if attempts["count"] < 4:
            return Response(status_code=503, payload={"ok": False})
        return Response(status_code=200, payload={"ok": True})

    response = retry_request(flaky_request, max_retries=3, sleep_fn=lambda _: None)

    assert response.status_code == 200
    assert attempts["count"] == 4
