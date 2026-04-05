"""Discover and manage all available SRE tasks."""

from pathlib import Path
from typing import Dict, List, Optional

from ..models import TaskSummary
from .config import TaskConfig


class TaskRegistry:
    """Auto-discovers and indexes all tasks from the /fixtures directory."""

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self.tasks: Dict[str, TaskConfig] = {}
        self.load_tasks()

    def load_tasks(self) -> None:
        """Scan fixtures_dir for task configurations."""
        if not self.fixtures_dir.exists():
            return

        # Find all subdirectories that have a task_config.json
        for task_path in sorted(self.fixtures_dir.iterdir()):
            if task_path.is_dir() and (task_path / "task_config.json").exists():
                try:
                    config = TaskConfig.from_json(task_path)
                    self.tasks[config.id] = config
                except Exception as e:
                    # Log and skip invalid tasks
                    print(f"Error loading task from {task_path}: {e}")

    def get_task(self, task_id: str) -> Optional[TaskConfig]:
        """Fetch a specific task's config."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[TaskConfig]:
        """Return all discovered tasks."""
        return [self.tasks[task_id] for task_id in sorted(self.tasks)]

    def list_summaries(self) -> List[TaskSummary]:
        """Return public task summaries for the API."""
        return [task.to_summary() for task in self.list_tasks()]

    def default_task_id(self) -> Optional[str]:
        """Return the default task id for empty reset requests."""
        tasks = self.list_tasks()
        if not tasks:
            return None
        return tasks[0].id
