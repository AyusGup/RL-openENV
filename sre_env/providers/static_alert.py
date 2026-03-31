"""Provides alert/incident data for the current episode task."""

import json
from pathlib import Path
from typing import Any, Dict

from .base import AlertProvider


class StaticAlertProvider(AlertProvider):
    """Reads alert data from a JSON file in the task's fixture folder."""

    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir

    async def get_alert(self, task_id: str) -> Dict[str, Any]:
        """Fetch alert content for a specific task.

        Args:
            task_id: The ID of the task to get an alert for.

        Returns:
            dict: Task alert details (message, severity, source).
        """
        # Look for the task's directory and its config file
        task_dir = self.fixture_dir / task_id
        config_path = task_dir / "task_config.json"

        if not config_path.exists():
            return {
                "message": f"Alert: Task {task_id} scenario started.",
                "severity": "UNKNOWN",
                "source": "monitoring-gateway",
            }

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return {
                    "message": config.get("alert_message", "Unexpected issues detected."),
                    "severity": config.get("severity", "MEDIUM"),
                    "source": config.get("alert_source", "prometheus-alertmanager"),
                }
        except (json.JSONDecodeError, IOError):
            return {
                "message": "Error reading alert data.",
                "severity": "CRITICAL",
                "source": "environment-engine",
            }
