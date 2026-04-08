"""Async typed client for the SRE OpenEnv HTTP API."""

from __future__ import annotations

from typing import List, Optional

import httpx

from .models import SREAction, SREObservation, SREState, SREStepResult, TaskSummary
from .utils.port_resolver import resolve_base_url


class SREEnv:
    """Async client wrapper around the SRE OpenEnv endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0):
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            follow_redirects=True,
            timeout=self.timeout,
        )

    async def __aenter__(self) -> "SREEnv":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def reset(self, task_id: Optional[str] = None) -> SREObservation:
        """Reset the environment and return the initial observation."""
        response = await self._client.post("/reset", json={"task_id": task_id})
        response.raise_for_status()
        return SREObservation.model_validate(response.json())

    async def step(self, action: SREAction) -> SREStepResult:
        """Execute one action in the environment."""
        response = await self._client.post("/step", json=action.model_dump())
        response.raise_for_status()
        return SREStepResult.model_validate(response.json())

    async def state(self) -> Optional[SREState]:
        """Return the current environment state, if any."""
        response = await self._client.get("/state")
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            return None
        return SREState.model_validate(payload)

    async def tasks(self) -> List[TaskSummary]:
        """Return available task summaries."""
        response = await self._client.get("/tasks")
        response.raise_for_status()
        payload = response.json()
        return [TaskSummary.model_validate(item) for item in payload]
