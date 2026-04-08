"""Retry helper used by the upstream health check."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


class MaxRetriesExceeded(RuntimeError):
    """Raised when the upstream request never succeeds."""


@dataclass
class Response:
    """Small response object used in tests and the demo API."""

    status_code: int
    payload: dict


def retry_request(
    request_fn: Callable[[], Response],
    max_retries: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Response:
    """Call an upstream request with exponential backoff."""
    
    for attempt in range(max_retries):
        response = request_fn()
        if response.status_code == 200:
            return response
        sleep_fn(2**attempt)
    raise MaxRetriesExceeded(f"Failed after {max_retries} retries")
