"""Typed models for the SRE incident response environment."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Any


@dataclass
class SREAction:
    """An action the agent can take.

    Tools:
        terminal: Run a shell command (cat, grep, ls, pytest, python, etc.)
        editor: Write content to a file (create or overwrite)
        submit: Signal episode complete, trigger grading
    """

    tool: Literal["terminal", "editor", "submit"]
    command: str = ""              # For terminal: the shell command
    file_path: str = ""            # For editor: relative path in workspace
    file_content: str = ""         # For editor: full file content to write


@dataclass
class SREObservation:
    """What the agent sees after each action."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    file_tree: list[str] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    alert_message: str = ""        # Populated on reset()
    score: Optional[float] = None  # Final score on submit


@dataclass
class SREState:
    """Episode state metadata."""

    episode_id: str = ""
    task_id: str = ""
    task_name: str = ""
    step_count: int = 0
    max_steps: int = 50
    cumulative_reward: float = 0.0
    done: bool = False
    workspace_root: str = "/workspace" # Root of the episode's sandbox
