"""Typed OpenEnv client for the SRE environment."""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.containers.runtime.providers import (
    ContainerProvider,
    LocalDockerProvider,
)

from .models import SREAction, SREObservation, SREState, TaskSummary


class SREEnv(EnvClient[SREAction, SREObservation, SREState]):
    """Client for the SRE environment using OpenEnv WebSocket sessions."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        provider: Optional[ContainerProvider] = None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            provider=provider,
            message_timeout_s=timeout,
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _step_payload(self, action: SREAction) -> Dict:
        return {
            "tool": action.tool,
            "command": action.command,
            "file_path": action.file_path,
            "file_content": action.file_content,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SREObservation]:
        obs_data = payload.get("observation", {})
        observation = SREObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            file_tree=obs_data.get("file_tree", []),
            alert_message=obs_data.get("alert_message", ""),
            score=obs_data.get("score"),
            message=obs_data.get("message", ""),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SREState:
        return SREState.model_validate(payload)

    async def tasks(self) -> List[TaskSummary]:
        """Return available task summaries from the custom `/tasks` endpoint."""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get("/tasks")
            response.raise_for_status()
        return [TaskSummary.model_validate(item) for item in response.json()]

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str],
        timeout: float = 60.0,
        task_id: Optional[str] = None,
        host_port: int = 8000,
        provider: Optional[ContainerProvider] = None,
    ) -> "SREEnv":
        """Start the image in Docker and return a connected client."""
        if not image_name:
            raise RuntimeError(
                "IMAGE_NAME is required. Set the IMAGE_NAME environment variable to the Docker image to use."
            )

        if provider is None:
            provider = LocalDockerProvider()

        env_vars = {"PORT": "8000"}
        if task_id:
            env_vars["SRE_TASK_NAME"] = task_id

        base_url = provider.start_container(
            image_name,
            port=host_port,
            env_vars=env_vars,
        )
        provider.wait_for_ready(base_url, timeout_s=timeout)

        client = cls(base_url=base_url, timeout=timeout, provider=provider)
        await client.connect()
        return client
