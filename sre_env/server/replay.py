"""Task-scoped replay probes for post-fix verification."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


@dataclass
class ReplayResult:
    """Structured result from a replay probe."""

    replay_name: str
    success: bool
    status_code: int
    response_body: Any
    evidence_log: str


class ReplayExecutor:
    """Run deterministic, task-aware replay probes against workspace code."""

    def run(self, task_id: str, replay_name: str, workspace_root: Path) -> ReplayResult:
        normalized_replay = replay_name.strip()
        if task_id == "task1_wrong_status":
            if normalized_replay != "create_item_contract":
                raise ValueError(
                    "Unknown replay target for task1_wrong_status. Expected 'create_item_contract'."
                )
            return self._run_task1_create_item_contract(workspace_root)

        raise ValueError(f"Replay is not supported for task '{task_id}'.")

    def _run_task1_create_item_contract(self, workspace_root: Path) -> ReplayResult:
        app_module = self._load_workspace_module(
            module_name="task1_workspace_app_main",
            module_path=workspace_root / "app" / "main.py",
        )
        app = getattr(app_module, "app", None)
        db = getattr(app_module, "db", None)
        if app is None:
            raise ValueError("Workspace app/main.py does not expose a FastAPI 'app'.")
        if isinstance(db, list):
            db.clear()

        payload = {"name": "replay-item"}
        with TestClient(app) as client:
            response = client.post("/api/items", json=payload)

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        success = (
            response.status_code == 201
            and isinstance(response_body, dict)
            and "id" in response_body
            and isinstance(response_body.get("item"), dict)
            and response_body.get("item", {}).get("name") == payload["name"]
        )
        evidence = [
            "replay=create_item_contract",
            "request=POST /api/items",
            f"expected_status=201",
            f"observed_status={response.status_code}",
            f"contract_ok={str(success).lower()}",
            f"response_body={json.dumps(response_body, separators=(',', ':'))}",
        ]
        return ReplayResult(
            replay_name="create_item_contract",
            success=success,
            status_code=response.status_code,
            response_body=response_body,
            evidence_log="\n".join(evidence) + "\n",
        )

    def _load_workspace_module(self, module_name: str, module_path: Path):
        if not module_path.exists():
            raise ValueError(f"Replay target file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load replay module from {module_path}")

        workspace_root = str(module_path.parents[1])
        original_sys_path = list(sys.path)
        previous_module = sys.modules.pop(module_name, None)
        try:
            sys.path.insert(0, workspace_root)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        finally:
            sys.path[:] = original_sys_path
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
