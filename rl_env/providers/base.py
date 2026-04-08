"""Abstract protocols for swappable data sources.

Competition: uses static file implementations.
Production: swap in live API implementations.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class LogProvider(Protocol):
    """Provides log data to the environment."""

    async def get_log(self, log_name: str) -> str:
        """Fetch log content.

        Args:
            log_name: Name of the log file (e.g. error.log)

        Returns:
            str: Log contents.
        """
        ...

    async def list_logs(self) -> list[str]:
        """List available logs for the current task.

        Returns:
            list[str]: Log names.
        """
        ...


@runtime_checkable
class AlertProvider(Protocol):
    """Provides alert/incident data."""

    async def get_alert(self, task_id: str) -> dict[str, Any]:
        """Fetch alert content for a specific task.

        Args:
            task_id: ID of the task.

        Returns:
            dict: Alert details (message, severity, source).
        """
        ...


@runtime_checkable
class MetricsProvider(Protocol):
    """Provides system metrics."""

    async def get_metrics(self, metric_name: str) -> dict[str, Any]:
        """Fetch performance metrics.

        Args:
            metric_name: Name of the metric (e.g. latency, error_rate).

        Returns:
            dict: Metric data.
        """
        ...


@runtime_checkable
class CommandExecutor(Protocol):
    """Executes shell commands in a controlled environment."""

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout: int = 10,
    ) -> tuple[str, str, int]:
        """Run a command.

        Args:
            command: Shell command.
            cwd: Working directory.
            timeout: Command timeout.

        Returns:
            tuple: (stdout, stderr, exit_code).
        """
        ...
