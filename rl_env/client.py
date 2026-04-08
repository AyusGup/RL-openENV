"""SRE Incident Response Environment Clients."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    SREAction,
    SREObservation,
    SREState,
    SREStepResult,
)


class SREEnv(
    EnvClient[SREAction, SREObservation, SREState]
):
    """
    Client for the SRE Incident Response Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SREEnv(base_url="http://localhost:7861") as client:
        ...     obs = client.reset()
        ...     result = client.step(SREAction(tool="terminal", command="ls /workspace"))
        ...     print(result.observation.stdout)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SREEnv.from_docker_image("sre_env:latest", task_id="task_1")
        >>> try:
        ...     obs = client.reset()
        ...     result = client.step(SREAction(tool="submit"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SREAction) -> Dict:
        """
        Convert SREAction to JSON payload for step message.

        Args:
            action: SREAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[SREObservation]:
        """
        Parse server response into StepResult[SREObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SREObservation
        """
        step_result = SREStepResult.model_validate(payload)
        return StepResult(
            observation=step_result.observation,
            reward=step_result.reward.value,
            done=step_result.done,
        )

    def _parse_state(self, payload: Dict) -> Optional[SREState]:
        """
        Parse server response into SREState object.

        Args:
            payload: JSON response from state request

        Returns:
            SREState object, or None if the payload is empty
        """
        if payload is None:
            return None
        return SREState.model_validate(payload)

