"""Provides log file access from the episode's workspace."""

from pathlib import Path
from typing import List

from .base import LogProvider


class StaticLogProvider(LogProvider):
    """Reads logs from static files in the /logs directory of a task."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.log_dir = workspace_root / "logs"

    async def get_log(self, log_name: str) -> str:
        """Read a specific log file.

        Args:
            log_name: Name of the log (e.g., error.log).

        Returns:
            str: Log contents or error if not found.
        """
        log_path = self.log_dir / log_name
        if not log_path.exists() or not log_path.is_file():
            return f"Error: Log file '{log_name}' not found."

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                # We only return the last 100 lines to mimic a real log tail
                # and prevent overwhelming the agent's context window.
                lines = f.readlines()
                return "".join(lines[-100:])
        except Exception as e:
            return f"Error reading log: {str(e)}"

    async def list_logs(self) -> List[str]:
        """List all .log files in the logs/ directory.

        Returns:
            list[str]: Log names.
        """
        if not self.log_dir.exists():
            return []

        return [f.name for f in self.log_dir.glob("*.log")]
