"""API-level tests for the SRE environment server."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

import sre_env.server.app as app_module
from sre_env.server.sre_environment import SREEnvironment


def build_client(tmp_path: Path) -> TestClient:
    """Create a test client backed by an isolated workspace."""
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    workspace_root = tmp_path / "workspace"
    app_module.env = SREEnvironment(fixtures_dir, workspace_root)
    return TestClient(app_module.app)


def test_reset_defaults_to_single_task(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/reset", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["alert_message"]
    assert "app/main.py" in body["file_tree"]
    assert not any(path.startswith("tests/") for path in body["file_tree"])
    state = client.get("/state").json()
    assert state["max_steps"] == 8


def test_tasks_endpoint_lists_all_tasks(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "task1_wrong_status",
            "name": "Task 1: FastAPI Status Code Mismatch",
            "difficulty": "easy",
        },
        {
            "id": "task2_retry_logic",
            "name": "Task 2: Off-by-One Retry Bug",
            "difficulty": "medium",
        },
        {
            "id": "task3_cascading_failure",
            "name": "Task 3: Cascading Timeout Failure",
            "difficulty": "hard",
        },
    ]


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_step_and_state_round_trip(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.post("/reset", json={"task_id": "task1_wrong_status"})

    step_response = client.post("/step", json={"tool": "terminal", "command": "cat app/main.py"})
    state_response = client.get("/state")

    assert step_response.status_code == 200
    body = step_response.json()
    assert "status_code=200" in body["observation"]["stdout"]
    assert body["reward"]["value"] >= 0.0
    assert body["done"] is False
    assert state_response.status_code == 200
    assert state_response.json()["task_id"] == "task1_wrong_status"
    assert state_response.json()["step_count"] == 1


def test_task1_replay_runs_against_workspace_app(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.post("/reset", json={"task_id": "task1_wrong_status"})

    response = client.post("/step", json={"tool": "replay", "command": "create_item_contract"})

    assert response.status_code == 200
    body = response.json()
    assert "replay=create_item_contract" in body["observation"]["stdout"]
    assert "observed_status=200" in body["observation"]["stdout"]
    assert body["reward"]["value"] > 0.0
    assert body["done"] is False


def test_replay_is_rejected_for_non_task1(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.post("/reset", json={"task_id": "task2_retry_logic"})

    response = client.post("/step", json={"tool": "replay", "command": "create_item_contract"})

    assert response.status_code == 400
    assert "Replay is not supported for task 'task2_retry_logic'" in response.json()["detail"]


def test_submit_returns_normalized_score(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.post("/reset", json={})
    source = client.post("/step", json={"tool": "terminal", "command": "cat app/main.py"}).json()["observation"]["stdout"]
    fixed_source = re.sub(r"status_code\s*=\s*200", "status_code=201", source)

    client.post(
        "/step",
        json={
            "tool": "editor",
            "file_path": "app/main.py",
            "file_content": fixed_source,
        },
    )
    response = client.post("/step", json={"tool": "submit"})

    assert response.status_code == 200
    assert 0.0 <= response.json()["info"]["score"] <= 1.0
    assert 0.0 <= response.json()["reward"]["value"] <= 1.0


def test_step_limit_auto_grades_workspace(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    client.post("/reset", json={"task_id": "task1_wrong_status"})
    app_module.env.state.max_steps = 1

    response = client.post("/step", json={"tool": "terminal", "command": "cat app/main.py"})

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["info"]["score"] is not None
    assert "auto-graded" in body["info"]["message"]


def test_reset_can_target_task2(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/reset", json={"task_id": "task2_retry_logic"})

    assert response.status_code == 200
    body = response.json()
    assert "app/retry_handler.py" in body["file_tree"]
    assert "RCA.md" not in body["file_tree"]
    state = client.get("/state").json()
    assert state["max_steps"] == 16


def test_reset_can_target_task3(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/reset", json={"task_id": "task3_cascading_failure"})

    assert response.status_code == 200
    body = response.json()
    assert "service_a/main.py" in body["file_tree"]
    assert "service_b/database.py" in body["file_tree"]
    assert "RCA_template.md" in body["file_tree"]
    state = client.get("/state").json()
    assert state["max_steps"] == 24
