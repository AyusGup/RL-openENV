"""Tests for the workspace command executor."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sre_env.providers.sandbox_executor import SandboxExecutor


def test_sandbox_executor_reads_workspace_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello\n", encoding="utf-8")

    stdout, stderr, exit_code = asyncio.run(
        SandboxExecutor().execute("cat note.txt", workspace)
    )

    assert exit_code == 0
    assert stdout.strip() == "hello"
    assert stderr == ""


def test_sandbox_executor_blocks_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("nope\n", encoding="utf-8")

    stdout, stderr, exit_code = asyncio.run(
        SandboxExecutor().execute("cat ../secret.txt", workspace)
    )

    assert stdout == ""
    assert exit_code == 1
    assert "Path traversal is not allowed" in stderr


def test_sandbox_executor_allows_pytest_module_invocation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    test_dir = workspace / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_smoke.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    stdout, stderr, exit_code = asyncio.run(
        SandboxExecutor().execute("python -m pytest -q tests/test_smoke.py", workspace, timeout=20)
    )

    assert exit_code == 0
    assert "1 passed" in stdout
    assert stderr == ""
