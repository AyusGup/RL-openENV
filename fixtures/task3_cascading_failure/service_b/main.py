"""Service B processing entry point."""

from __future__ import annotations

import time

from service_b.database import get_enrichment_data


def process_payload(payload: dict) -> dict:
    """Enrich a request payload."""
    
    started = time.perf_counter()
    enrichment = get_enrichment_data(int(payload["item_id"]))
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "item_id": payload["item_id"],
        "enrichment": enrichment,
        "elapsed_ms": elapsed_ms,
    }
