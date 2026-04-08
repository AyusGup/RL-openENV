"""API-level tests for the SRE environment server."""

import json
import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

import sre_env.server.app as app_module


def build_client(tmp_path: Path, fixtures_dir: Path | None = None) -> TestClient:
    """Create a test client backed by an isolated workspace."""
    resolved_fixtures = fixtures_dir or (Path(__file__).resolve().parents[1] / "sre_env" / "fixtures")
    workspace_root = tmp_path / "workspace"
    app = app_module.create_sre_app(resolved_fixtures, workspace_root)
    return TestClient(app)


def ws_reset(ws, task_id: str | None = None) -> dict:
    payload = {"type": "reset", "data": {}}
    if task_id:
        payload["data"]["task_id"] = task_id
    ws.send_json(payload)
    response = ws.receive_json()
    assert response["type"] == "observation"
    return response["data"]


def ws_step(ws, action: dict) -> dict:
    ws.send_json({"type": "step", "data": action})
    response = ws.receive_json()
    assert response["type"] == "observation"
    return response["data"]


def ws_state(ws) -> dict:
    ws.send_json({"type": "state"})
    response = ws.receive_json()
    assert response["type"] == "state"
    return response["data"]


def test_template_endpoints_exist(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    assert client.get("/health").status_code == 200
    assert client.get("/schema").status_code == 200
    assert client.get("/metadata").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_reset_endpoint_accepts_task_id(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/reset", json={"task_id": "task2_retry_logic"})

    assert response.status_code == 200
    body = response.json()
    assert "app/retry_handler.py" in body["observation"]["file_tree"]


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


def test_step_and_state_round_trip_via_websocket(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task1_wrong_status")
        step_result = ws_step(ws, {"tool": "terminal", "command": "cat app/main.py"})
        state = ws_state(ws)

    assert "status_code=200" in step_result["observation"]["stdout"]
    assert step_result["reward"] >= 0.0
    assert step_result["done"] is False
    assert state["task_id"] == "task1_wrong_status"
    assert state["step_count"] == 1


def test_task1_replay_runs_against_workspace_app(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task1_wrong_status")
        step_result = ws_step(ws, {"tool": "replay", "command": "create_item_contract"})

    body = step_result["observation"]
    assert "replay=create_item_contract" in body["stdout"]
    assert "observed_status=200" in body["stdout"]
    assert "contract_ok=false" in body["stdout"]
    assert step_result["done"] is False


def test_task2_replay_runs_against_workspace_app(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task2_retry_logic")
        step_result = ws_step(ws, {"tool": "replay", "command": "retry_health_contract"})

    body = step_result["observation"]
    assert "replay=retry_health_contract" in body["stdout"]
    assert "observed_status=503" in body["stdout"]
    assert "contract_ok=false" in body["stdout"]
    assert step_result["done"] is False


def test_task3_replay_checks_timeout_budget(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task3_cascading_failure")
        step_result = ws_step(ws, {"tool": "replay", "command": "cascading_timeout_budget"})

    body = step_result["observation"]
    assert "replay=cascading_timeout_budget" in body["stdout"]
    assert "contract_ok=false" in body["stdout"]
    assert step_result["done"] is False


def test_replay_invalid_name_surfaces_error(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task2_retry_logic")
        step_result = ws_step(ws, {"tool": "replay", "command": "create_item_contract"})

    assert "Unknown replay target for task2_retry_logic" in step_result["observation"]["stderr"]


def test_submit_returns_normalized_score(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task1_wrong_status")
        read_result = ws_step(ws, {"tool": "terminal", "command": "cat app/main.py"})
        source = read_result["observation"]["stdout"]
        fixed_source = re.sub(r"status_code\s*=\s*200", "status_code=201", source)

        ws_step(
            ws,
            {
                "tool": "editor",
                "file_path": "app/main.py",
                "file_content": fixed_source,
            },
        )
        submit_result = ws_step(ws, {"tool": "submit"})

    assert submit_result["done"] is True
    assert 0.0 <= submit_result["reward"] <= 1.0
    assert 0.0 <= submit_result["observation"]["score"] <= 1.0


def test_step_limit_auto_grades_workspace(tmp_path: Path) -> None:
    source_fixtures = Path(__file__).resolve().parents[1] / "sre_env" / "fixtures"
    patched_fixtures = tmp_path / "fixtures"
    shutil.copytree(source_fixtures, patched_fixtures)

    config_path = patched_fixtures / "task1_wrong_status" / "task_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_steps"] = 1
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    client = build_client(tmp_path, fixtures_dir=patched_fixtures)

    with client.websocket_connect("/ws") as ws:
        ws_reset(ws, task_id="task1_wrong_status")
        step_result = ws_step(ws, {"tool": "terminal", "command": "cat app/main.py"})

    assert step_result["done"] is True
    assert step_result["observation"]["score"] is not None
    assert "auto-graded" in step_result["observation"]["message"]


def test_reset_can_target_task2(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        reset_result = ws_reset(ws, task_id="task2_retry_logic")
        state = ws_state(ws)

    assert "app/retry_handler.py" in reset_result["observation"]["file_tree"]
    assert not any(path.startswith("tests/") for path in reset_result["observation"]["file_tree"])
    assert state["max_steps"] == 16


def test_reset_can_target_task3(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with client.websocket_connect("/ws") as ws:
        reset_result = ws_reset(ws, task_id="task3_cascading_failure")
        state = ws_state(ws)

    file_tree = reset_result["observation"]["file_tree"]
    assert "service_a/main.py" in file_tree
    assert "service_b/database.py" in file_tree
    assert not any(path.startswith("tests/") for path in file_tree)
    assert state["max_steps"] == 24
