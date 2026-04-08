"""Async typed client for the SRE OpenEnv HTTP API."""

from __future__ import annotations

import asyncio
from typing import List, Optional

import httpx

from .models import SREAction, SREObservation, SREState, SREStepResult, TaskSummary

_DOCKER_STARTUP_TIMEOUT = 60  # seconds to wait for the container to become healthy
_DOCKER_POLL_INTERVAL = 1.5   # seconds between health-check polls
_DOCKER_HOST_PORT = 7861      # fixed host port mapped to the container


class SREEnv:
    """Async client wrapper around the SRE OpenEnv endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:7861", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            follow_redirects=True,
            timeout=self.timeout,
        )
        self._container_proc: Optional[asyncio.subprocess.Process] = None

    # ------------------------------------------------------------------
    # Docker factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str],
        timeout: float = 60.0,
        task_id: Optional[str] = None,
    ) -> "SREEnv":
        """Start *image_name* as a Docker container and return a connected client.

        Raises ``RuntimeError`` if *image_name* is ``None`` or empty — set the
        ``IMAGE_NAME`` environment variable before calling this factory.
        """
        if not image_name:
            raise RuntimeError(
                "IMAGE_NAME is required. Set the IMAGE_NAME environment variable to the Docker image to use."
            )

        base_url = f"http://127.0.0.1:{_DOCKER_HOST_PORT}"

        # Build the docker run command.
        docker_cmd = [
            "docker", "run", "--rm",
            "-p", f"{_DOCKER_HOST_PORT}:7861",
            "--env", "PORT=7861",
        ]
        if task_id:
            docker_cmd += ["--env", f"SRE_TASK_NAME={task_id}"]
        docker_cmd.append(image_name)

        print(f"[ENV] Starting container: {' '.join(docker_cmd)}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        instance = cls(base_url=base_url, timeout=timeout)
        instance._container_proc = proc

        # Poll /health until the server is ready.
        health_url = f"{base_url}/health"
        deadline = asyncio.get_event_loop().time() + _DOCKER_STARTUP_TIMEOUT
        last_exc: Exception = RuntimeError("timeout")
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await probe.get(health_url)
                    if resp.status_code < 500:
                        print(f"[ENV] Container healthy at {base_url}", flush=True)
                        return instance
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                await asyncio.sleep(_DOCKER_POLL_INTERVAL)

        # Startup timed out — kill the container and raise.
        proc.terminate()
        raise RuntimeError(
            f"SRE env container ({image_name}) did not become healthy within "
            f"{_DOCKER_STARTUP_TIMEOUT}s — last error: {last_exc}"
        )

    async def __aenter__(self) -> "SREEnv":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client and stop any managed Docker container."""
        await self._client.aclose()
        if self._container_proc is not None:
            try:
                self._container_proc.terminate()
                await asyncio.wait_for(self._container_proc.wait(), timeout=10.0)
            except Exception:  # noqa: BLE001
                self._container_proc.kill()

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
