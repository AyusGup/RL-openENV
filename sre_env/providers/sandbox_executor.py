"""Execute workspace-scoped commands with a conservative allowlist."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Sequence, Tuple

from .base import CommandExecutor


class SandboxExecutor(CommandExecutor):
    """Run a small allowlisted command set inside the containerized app.

    The environment itself already runs inside Docker when deployed. This
    executor provides an additional in-process policy layer by enforcing
    timeouts, restricting commands to a small safe subset, and resolving file
    arguments against the active workspace.
    """

    _ALLOWED_COMMANDS = {"cat", "ls", "python", "pytest", "find", "pwd"}

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout: int = 10,
    ) -> Tuple[str, str, int]:
        """Run a command asynchronously."""
        if not cwd.exists():
            return ("", f"Error: Workspace directory {cwd} does not exist.", 1)

        try:
            argv = self._build_argv(command, cwd)
            completed = await asyncio.to_thread(
                subprocess.run,
                argv,
                cwd=str(cwd),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (completed.stdout, completed.stderr, completed.returncode)
        except subprocess.TimeoutExpired:
            return ("", f"Command timed out after {timeout} seconds.", 124)
        except ValueError as exc:
            return ("", f"Error: {exc}", 1)
        except Exception as exc:
            return ("", f"Execution Error: {str(exc)}", 1)

    def _build_argv(self, command: str, cwd: Path) -> Sequence[str]:
        """Parse and validate a command into subprocess argv."""
        try:
            parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise ValueError(f"Invalid command syntax: {exc}") from exc

        if not parts:
            raise ValueError("Empty command.")

        cmd = parts[0].lower()
        if cmd not in self._ALLOWED_COMMANDS:
            raise ValueError(
                f"Command '{parts[0]}' is not allowed. Allowed commands: {', '.join(sorted(self._ALLOWED_COMMANDS))}."
            )

        if any(token in command for token in ("&&", "||", ";", "|", ">", "<")):
            raise ValueError("Shell operators and redirection are not allowed.")

        if cmd == "cat":
            if len(parts) != 2:
                raise ValueError("cat expects exactly one file path.")
            return self._cat_command(parts[1], cwd)

        if cmd == "ls":
            target = parts[1] if len(parts) > 1 else "."
            self._resolve_workspace_path(target, cwd)
            return self._ls_command(target)

        if cmd == "pwd":
            return self._pwd_command()

        if cmd == "find":
            return self._find_command(parts, cwd)

        if cmd in {"python", "pytest"}:
            return self._python_like_command(parts, cwd)

        raise ValueError(f"Unsupported command '{parts[0]}'.")

    def _cat_command(self, target: str, cwd: Path) -> Sequence[str]:
        resolved = self._resolve_workspace_path(target, cwd)
        if os.name == "nt":
            return ["cmd.exe", "/c", "type", str(resolved)]
        return ["cat", str(resolved)]

    def _ls_command(self, target: str) -> Sequence[str]:
        normalized = "." if target == "." else target.replace("/", "\\") if os.name == "nt" else target
        if os.name == "nt":
            return ["cmd.exe", "/c", "dir", normalized]
        return ["ls", normalized]

    def _pwd_command(self) -> Sequence[str]:
        if os.name == "nt":
            return ["cmd.exe", "/c", "cd"]
        return ["pwd"]

    def _find_command(self, parts: Sequence[str], cwd: Path) -> Sequence[str]:
        if len(parts) not in {3, 4}:
            raise ValueError("find supports 'find <path> -name <pattern>' only.")

        search_root = parts[1]
        if parts[2] != "-name" or len(parts) != 4:
            raise ValueError("find supports 'find <path> -name <pattern>' only.")

        resolved_root = self._resolve_workspace_path(search_root, cwd)
        pattern = parts[3]
        if os.name == "nt":
            return [
                "powershell.exe",
                "-Command",
                f"Get-ChildItem -Path '{resolved_root}' -Recurse -Filter '{pattern}' | ForEach-Object {{ $_.FullName }}",
            ]
        return ["find", str(resolved_root), "-name", pattern]

    def _python_like_command(self, parts: Sequence[str], cwd: Path) -> Sequence[str]:
        if parts[0].lower() == "pytest":
            pytest_args = list(parts[1:])
            self._validate_pytest_targets(pytest_args, cwd)
            return [self._python_executable(), "-m", "pytest", *pytest_args]

        python_args = list(parts[1:])
        if python_args[:2] == ["-m", "pytest"]:
            self._validate_pytest_targets(python_args[2:], cwd)
        else:
            raise ValueError("Only python -m pytest is allowed.")
        return [self._python_executable(), *python_args]

    def _validate_pytest_targets(self, args: Sequence[str], cwd: Path) -> None:
        """Validate pytest target paths and forbid dangerous options."""
        disallowed_prefixes = ("--rootdir", "--basetemp", "--confcutdir", "--import-mode")
        for arg in args:
            if arg.startswith("-"):
                if arg.startswith(disallowed_prefixes):
                    raise ValueError(f"pytest option '{arg}' is not allowed.")
                continue
            self._resolve_workspace_path(arg, cwd)

    def _resolve_workspace_path(self, value: str, cwd: Path) -> Path:
        """Resolve a relative workspace path and reject path escapes."""
        normalized = value.replace("\\", "/")
        pure_path = PurePosixPath(normalized)
        if pure_path.is_absolute():
            raise ValueError("Absolute paths are not allowed.")
        if ".." in pure_path.parts:
            raise ValueError("Path traversal is not allowed.")

        resolved = (cwd / Path(normalized)).resolve()
        workspace_root = cwd.resolve()
        if workspace_root != resolved and workspace_root not in resolved.parents:
            raise ValueError("Path must stay inside the workspace.")
        return resolved

    def _python_executable(self) -> str:
        """Return the preferred Python executable for the current OS."""
        return "py" if os.name == "nt" else "python"
