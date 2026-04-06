"""Tests for Service B latency."""

from __future__ import annotations

import time

from service_b.main import process_payload


def test_service_b_processing_stays_under_budget() -> None:
    started = time.perf_counter()
    payload = process_payload({"item_id": 42})
    elapsed = time.perf_counter() - started

    assert payload["item_id"] == 42
    assert payload["enrichment"]["item_id"] == 42
    assert payload["elapsed_ms"] < 200
    assert elapsed < 0.20
