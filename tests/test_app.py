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


def test_tasks_endpoint_lists_single_task(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "task1_wrong_status",
            "name": "Task 1: FastAPI Status Code Mismatch",
            "difficulty": "easy",
        }
    ]


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
