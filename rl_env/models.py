"""
Data models SRE Incident Response Environments.

The sre_env environment is an SRE incident response environment where an agent
diagnoses and remediates infrastructure incidents.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class SREAction(Action):
    """Action the agent can take in the SRE incident response environment."""

    tool: Literal["terminal", "editor", "replay", "submit"] = Field(
        ..., description="Tool to invoke (terminal, editor, replay, or submit)"
    )
    command: str = Field(default="", description="Shell command to run (terminal tool)")
    file_path: str = Field(default="", description="Target file path (editor tool)")
    file_content: str = Field(
        default="", description="New file content to write (editor tool)"
    )


class SREObservation(Observation):
    """Observation the agent receives after each action in the SRE environment."""

    stdout: str = Field(default="", description="Standard output from the last command")
    stderr: str = Field(default="", description="Standard error from the last command")
    exit_code: int = Field(default=0, description="Exit code of the last command")
    file_tree: list[str] = Field(
        default_factory=list,
        description="Snapshot of the repository file tree",
    )
    alert_message: str = Field(
        default="", description="Alert or incident description shown to the agent"
    )


class SREReward(BaseModel):
    """Typed reward payload for a single SRE environment step."""

    value: float = Field(default=0.0, description="Reward value for the current step")


class SREStepInfo(BaseModel):
    """Extra typed metadata returned alongside each SRE environment step."""

    score: Optional[float] = Field(
        default=None, description="Grader score after submission (0.0–1.0)"
    )
    message: str = Field(default="", description="Human-readable status or error message")
    last_action_error: Optional[str] = Field(
        default=None, description="Error message from the last action, if any"
    )
    grading_breakdown: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-component grading scores for completed episodes",
    )


class SREStepResult(BaseModel):
    """Typed return payload from SREEnv.step(action)."""

    observation: SREObservation = Field(..., description="Current environment observation")
    reward: SREReward = Field(..., description="Reward for the current step")
    done: bool = Field(default=False, description="Whether the episode has ended")
    info: SREStepInfo = Field(
        default_factory=SREStepInfo,
        description="Additional step metadata",
    )


class SREState(BaseModel):
    """Episode state metadata tracked internally by the SRE environment."""

    episode_id: str = Field(default="", description="Unique identifier for this episode")
    task_id: str = Field(default="", description="Identifier of the active task")
    task_name: str = Field(default="", description="Human-readable name of the active task")
    step_count: int = Field(default=0, description="Number of steps taken so far")
    max_steps: int = Field(default=50, description="Maximum steps allowed per episode")
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated this episode"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
    workspace_root: str = Field(
        default="/workspace", description="Absolute path to the agent workspace"
    )


class TaskSummary(BaseModel):
    """Public task metadata exposed by the SRE environment HTTP API."""

    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    difficulty: str = Field(..., description="Difficulty level (e.g. easy, medium, hard)")
