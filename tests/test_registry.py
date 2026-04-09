"""Task registry and grader regression tests."""

import asyncio
import re
from pathlib import Path

from rl_env.server.grader import SREGrader
from rl_env.server.sre_environment import SREEnvironment
from rl_env.tasks import TaskRegistry


def test_grader_scores_fixed_workspace_in_range(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "rl_env" / "fixtures"
    workspace_root = tmp_path / "workspace"
    env = SREEnvironment(fixtures_dir, workspace_root)
    registry = TaskRegistry(fixtures_dir)
    task = registry.get_task("task1_wrong_status")

    assert task is not None
    env.reset(task.id)

    target_file = workspace_root / "app" / "main.py"
    original = target_file.read_text(encoding="utf-8")
    target_file.write_text(re.sub(r"status_code\s*=\s*200", "status_code=201", original), encoding="utf-8")

    score = asyncio.run(SREGrader().grade_episode(task, fixtures_dir / task.id, workspace_root))
    assert 0.0 <= score <= 1.0


def test_registry_discovers_single_task() -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "rl_env" / "fixtures"
    registry = TaskRegistry(fixtures_dir)

    assert registry.default_task_id() == "task1_wrong_status"
    assert [task.id for task in registry.list_tasks()] == [
        "task1_wrong_status",
        "task2_retry_logic",
        "task3_cascading_failure",
    ]


def test_registry_exposes_new_task_metadata() -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "rl_env" / "fixtures"
    registry = TaskRegistry(fixtures_dir)

    task2 = registry.get_task("task2_retry_logic")
    task3 = registry.get_task("task3_cascading_failure")

    assert task2 is not None
    assert task2.difficulty == "medium"
    assert task2.max_steps == 16
    assert "app/retry_handler.py" in task2.expected_fix_files
    assert task3 is not None
    assert task3.difficulty == "hard"
    assert task3.max_steps == 24
    assert "service_a/main.py" in task3.expected_fix_files


def test_grader_counts_expected_new_file_creation(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "rl_env" / "fixtures"
    workspace_root = tmp_path / "workspace"
    env = SREEnvironment(fixtures_dir, workspace_root)
    registry = TaskRegistry(fixtures_dir)
    task = registry.get_task("task2_retry_logic")

    assert task is not None
    env.reset(task.id)

    retry_handler = workspace_root / "app" / "retry_handler.py"
    retry_handler.write_text(
        retry_handler.read_text(encoding="utf-8").replace(
            "range(max_retries)", "range(max_retries + 1)"
        ),
        encoding="utf-8",
    )
    (workspace_root / "RCA.md").write_text(
        (
            "# Incident RCA Report\n\n"
            "## Root Cause\nretry loop boundary mismatch\n\n"
            "## Fix Applied\nupdated loop to `range(max_retries + 1)` in retry_handler.\n"
        ),
        encoding="utf-8",
    )

    score = SREGrader()._check_file_changes(
        task.expected_fix_files,
        fixtures_dir / task.id,
        workspace_root,
    )
    assert score == 1.0


def test_grader_runs_hidden_fixture_tests_for_task2(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "rl_env" / "fixtures"
    workspace_root = tmp_path / "workspace"
    env = SREEnvironment(fixtures_dir, workspace_root)
    registry = TaskRegistry(fixtures_dir)
    task = registry.get_task("task2_retry_logic")

    assert task is not None
    env.reset(task.id)
    assert not (workspace_root / "tests").exists()

    retry_handler = workspace_root / "app" / "retry_handler.py"
    retry_handler.write_text(
        retry_handler.read_text(encoding="utf-8").replace(
            "range(max_retries)", "range(max_retries + 1)"
        ),
        encoding="utf-8",
    )
    (workspace_root / "RCA.md").write_text(
        (
            "# Incident RCA Report\n\n"
            "## Root Cause\nretry loop boundary mismatch\n\n"
            "## Fix Applied\nupdated loop to `range(max_retries + 1)` in retry_handler.\n"
        ),
        encoding="utf-8",
    )

    test_score = asyncio.run(
        SREGrader()._check_tests(task.id, fixtures_dir / task.id, workspace_root)
    )
    assert test_score == 1.0

