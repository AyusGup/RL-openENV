"""Typed models for the SRE incident response environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class SREAction(Action):
    """An action the agent can take."""

    tool: Literal["terminal", "editor", "replay", "submit"]
    command: str = ""
    file_path: str = ""
    file_content: str = ""


class SREObservation(Observation):
    """What the agent sees after each action."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    file_tree: list[str] = Field(default_factory=list)
    alert_message: str = ""
    score: float | None = None
    message: str = ""
    last_action_error: str | None = None


class SREState(State):
    """Episode state metadata."""

    task_id: str = ""
    task_name: str = ""
    max_steps: int = 50
    cumulative_reward: float = 0.0
    done: bool = False
    workspace_root: str = "/workspace"


class TaskSummary(BaseModel):
    """Public task metadata exposed by the HTTP API."""

    id: str
    name: str
    difficulty: str
