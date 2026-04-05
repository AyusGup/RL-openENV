"""Executes shell commands in a controlled sandbox with timeout security."""

import asyncio
import os
import shlex
from pathlib import Path
from typing import Tuple

from .base import CommandExecutor


class SandboxExecutor(CommandExecutor):
    """Executes shell commands locally with resource constraints.

    In a production environment, this would run commands inside a Docker container
    or a specialized sandbox. For the competition, we use subprocess with timeouts.
    """

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout: int = 10,
    ) -> Tuple[str, str, int]:
        """Run a command asynchronously.

        Args:
            command: The shell command to run.
            cwd: Working directory (the episode's workspace).
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple[str, str, int]: (stdout, stderr, exit_code)
        """
        # Ensure the working directory exists
        if not cwd.exists():
            return ("", f"Error: Workspace directory {cwd} does not exist.", 1)

        try:
            normalized_command = self._normalize_command(command)
            # We use asyncio to prevent blocking the main FastAPI event loop
            process = await asyncio.create_subprocess_shell(
                normalized_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                # Set dynamic environment variables if needed
                env=os.environ.copy(),
            )

            try:
                # Wait for the process to complete with a timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return (
                    stdout.decode(errors="replace"),
                    stderr.decode(errors="replace"),
                    process.returncode or 0,
                )

            except asyncio.TimeoutError:
                # Kill the process if it exceeds the timeout
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
                return (
                    "",
                    f"Command timed out after {timeout} seconds.",
                    124,  # Standard exit code for timeout
                )

        except Exception as e:
            return ("", f"Execution Error: {str(e)}", 1)

    def _normalize_command(self, command: str) -> str:
        """Translate a small subset of Unix-style commands for Windows dev shells."""
        if os.name != "nt":
            return command

        if command.startswith("cat "):
            target = command[4:].replace("/", "\\")
            return f"type {target}"

        if command.startswith("python "):
            return f"py -3 {command[7:]}"

        if command == "python":
            return "py -3"

        return command
