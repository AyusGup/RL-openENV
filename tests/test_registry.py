"""Task registry and grader regression tests."""

import asyncio
import re
from pathlib import Path

from sre_env.server.grader import SREGrader
from sre_env.server.sre_environment import SREEnvironment
from sre_env.tasks import TaskRegistry


def test_grader_scores_fixed_workspace_in_range(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    workspace_root = tmp_path / "workspace"
    env = SREEnvironment(fixtures_dir, workspace_root)
    registry = TaskRegistry(fixtures_dir)
    task = registry.get_task("task1_wrong_status")

    assert task is not None
    asyncio.run(env.reset(task.id))

    target_file = workspace_root / "app" / "main.py"
    original = target_file.read_text(encoding="utf-8")
    target_file.write_text(re.sub(r"status_code\s*=\s*200", "status_code=201", original), encoding="utf-8")

    score = asyncio.run(SREGrader().grade_episode(task, fixtures_dir / task.id, workspace_root))
    assert 0.0 <= score <= 1.0


def test_registry_discovers_single_task() -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    registry = TaskRegistry(fixtures_dir)

    assert registry.default_task_id() == "task1_wrong_status"
    assert [task.id for task in registry.list_tasks()] == ["task1_wrong_status"]
