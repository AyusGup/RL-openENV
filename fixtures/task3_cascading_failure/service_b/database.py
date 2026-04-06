"""Database access layer for Service B."""

from __future__ import annotations

import time

QUERY_LATENCY_SECONDS = 0.25


def get_enrichment_data(item_id: int) -> dict:
    """Simulate a slow enrichment query."""
    
    time.sleep(QUERY_LATENCY_SECONDS)
    return {
        "item_id": item_id,
        "metadata": {"tier": "gold"},
        "analytics": {"score": 98},
    }
