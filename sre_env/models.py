"""Typed models for the SRE incident response environment."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SREAction(BaseModel):
    """An action the agent can take."""

    tool: Literal["terminal", "editor", "replay", "submit"]
    command: str = ""
    file_path: str = ""
    file_content: str = ""


class SREObservation(BaseModel):
    """What the agent sees after each action."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    file_tree: list[str] = Field(default_factory=list)
    alert_message: str = ""


class SREReward(BaseModel):
    """Typed reward payload for a single environment step."""

    value: float = 0.0


class SREStepInfo(BaseModel):
    """Extra typed metadata returned from step()."""

    score: Optional[float] = None
    message: str = ""
    last_action_error: Optional[str] = None


class SREStepResult(BaseModel):
    """Typed return payload from step(action)."""

    observation: SREObservation
    reward: SREReward
    done: bool = False
    info: SREStepInfo = Field(default_factory=SREStepInfo)


class SREState(BaseModel):
    """Episode state metadata."""

    episode_id: str = ""
    task_id: str = ""
    task_name: str = ""
    step_count: int = 0
    max_steps: int = 50
    cumulative_reward: float = 0.0
    done: bool = False
    workspace_root: str = "/workspace"


class TaskSummary(BaseModel):
    """Public task metadata exposed by the HTTP API."""

    id: str
    name: str
    difficulty: str
