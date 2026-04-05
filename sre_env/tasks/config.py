"""Task configuration models for fixture-backed SRE tasks."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..models import TaskSummary


class RegexCheck(BaseModel):
    """A regex-based correctness check for a task."""

    file: str
    pattern: str
    message: str = ""


class TaskConfig(BaseModel):
    """Configuration loaded from a task fixture directory."""

    id: str
    name: str
    difficulty: str
    alert_message: str = ""
    alert_source: str = "prometheus-alertmanager"
    severity: str = "MEDIUM"
    expected_fix_files: list[str] = Field(default_factory=list)
    grading_weights: dict[str, float] = Field(default_factory=dict)
    regex_checks: list[RegexCheck] = Field(default_factory=list)

    @classmethod
    def from_json(cls, task_path: Path) -> "TaskConfig":
        """Load a task configuration from a fixture directory."""
        config_path = task_path / "task_config.json"
        with open(config_path, "r", encoding="utf-8") as file_handle:
            raw_config = json.load(file_handle)
        return cls.model_validate(raw_config)

    def to_summary(self) -> TaskSummary:
        """Return the public task metadata shown to clients."""
        return TaskSummary(id=self.id, name=self.name, difficulty=self.difficulty)
