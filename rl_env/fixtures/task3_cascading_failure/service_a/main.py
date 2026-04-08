"""Service A caller for the downstream processing path."""

from __future__ import annotations

import httpx

from service_a.config import SERVICE_B_URL


async def call_service_b(payload: dict, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Forward work to Service B."""
    
    async with httpx.AsyncClient(timeout=0.1, transport=transport) as client:
        response = await client.post(SERVICE_B_URL, json=payload)
        response.raise_for_status()
        return response.json()
