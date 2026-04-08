"""API surface that uses the retry helper."""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from app.retry_handler import MaxRetriesExceeded, Response, retry_request

app = FastAPI()
UPSTREAM_RESPONSES: list[int] = [503, 503, 503, 200]


def make_upstream_request() -> Response:
    """Return the next canned upstream result."""

    status_code = UPSTREAM_RESPONSES.pop(0) if UPSTREAM_RESPONSES else 200
    return Response(status_code=status_code, payload={"status": status_code})


@app.get("/api/upstream/health")
def upstream_health() -> dict:
    """Call the flaky upstream with retries."""
    
    try:
        result = retry_request(make_upstream_request, max_retries=3, sleep_fn=lambda _: None)
    except MaxRetriesExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "upstream_status": result.status_code}
