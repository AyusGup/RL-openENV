"""Cross-service regression tests."""

from __future__ import annotations

import time
from pathlib import Path

from service_b.main import process_payload
from tests.test_service_a import extract_http_timeout_seconds


def test_rca_template_is_available() -> None:
    template = (Path(__file__).resolve().parents[1] / "RCA_template.md").read_text(
        encoding="utf-8"
    )

    assert "## Root Cause" in template
    assert "## Prevention" in template


def test_service_a_timeout_exceeds_service_b_processing_time() -> None:
    timeout_seconds = extract_http_timeout_seconds()

    started = time.perf_counter()
    payload = process_payload({"item_id": 42})
    elapsed = time.perf_counter() - started

    assert payload["elapsed_ms"] > 0
    assert elapsed < timeout_seconds
