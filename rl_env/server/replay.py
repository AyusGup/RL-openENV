"""Task-scoped replay probes for post-fix verification."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import time
import types
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

        if task_id == "task2_retry_logic":
            if normalized_replay != "retry_health_contract":
                raise ValueError(
                    "Unknown replay target for task2_retry_logic. Expected 'retry_health_contract'."
                )
            return self._run_task2_retry_health_contract(workspace_root)

        if task_id == "task3_cascading_failure":
            if normalized_replay != "cascading_timeout_budget":
                raise ValueError(
                    "Unknown replay target for task3_cascading_failure. Expected 'cascading_timeout_budget'."
                )
            return self._run_task3_cascading_timeout_budget(workspace_root)

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
            "expected_status=201",
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

    def _run_task2_retry_health_contract(self, workspace_root: Path) -> ReplayResult:
        app_module = self._load_workspace_module(
            module_name="task2_workspace_app_main",
            module_path=workspace_root / "app" / "main.py",
        )
        app = getattr(app_module, "app", None)
        upstream_responses = getattr(app_module, "UPSTREAM_RESPONSES", None)
        if app is None:
            raise ValueError("Workspace app/main.py does not expose a FastAPI 'app'.")
        if isinstance(upstream_responses, list):
            upstream_responses[:] = [503, 503, 503, 200]

        with TestClient(app) as client:
            response = client.get("/api/upstream/health")

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        success = response.status_code == 200 and response_body == {
            "ok": True,
            "upstream_status": 200,
        }
        evidence = [
            "replay=retry_health_contract",
            "request=GET /api/upstream/health",
            "expected_status=200",
            f"observed_status={response.status_code}",
            "expected_body={\"ok\":true,\"upstream_status\":200}",
            f"contract_ok={str(success).lower()}",
            f"response_body={json.dumps(response_body, separators=(',', ':'))}",
        ]
        return ReplayResult(
            replay_name="retry_health_contract",
            success=success,
            status_code=response.status_code,
            response_body=response_body,
            evidence_log="\n".join(evidence) + "\n",
        )

    def _run_task3_cascading_timeout_budget(self, workspace_root: Path) -> ReplayResult:
        timeout_seconds = self._extract_service_a_timeout_seconds(workspace_root / "service_a" / "main.py")
        service_b_main = self._load_workspace_module(
            module_name="task3_workspace_service_b_main",
            module_path=workspace_root / "service_b" / "main.py",
        )
        process_payload = getattr(service_b_main, "process_payload", None)
        if process_payload is None:
            raise ValueError("Workspace service_b/main.py does not expose process_payload.")

        started = time.perf_counter()
        payload = process_payload({"item_id": 42})
        elapsed_seconds = time.perf_counter() - started
        elapsed_ms = int(elapsed_seconds * 1000)

        within_timeout_range = 0.3 <= timeout_seconds <= 1.0
        service_b_within_budget = elapsed_seconds < 0.20
        timeout_exceeds_elapsed = elapsed_seconds < timeout_seconds
        payload_ok = isinstance(payload, dict) and payload.get("item_id") == 42
        success = (
            within_timeout_range
            and service_b_within_budget
            and timeout_exceeds_elapsed
            and payload_ok
        )

        evidence = [
            "replay=cascading_timeout_budget",
            "request=service_a timeout budget vs service_b processing time",
            f"timeout_seconds={timeout_seconds:.3f}",
            f"observed_elapsed_seconds={elapsed_seconds:.3f}",
            f"observed_elapsed_ms={elapsed_ms}",
            f"within_timeout_range={str(within_timeout_range).lower()}",
            f"service_b_within_budget={str(service_b_within_budget).lower()}",
            f"timeout_exceeds_elapsed={str(timeout_exceeds_elapsed).lower()}",
            f"contract_ok={str(success).lower()}",
            f"payload={json.dumps(payload, separators=(',', ':'))}",
        ]
        return ReplayResult(
            replay_name="cascading_timeout_budget",
            success=success,
            status_code=200,
            response_body={
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "payload": payload,
            },
            evidence_log="\n".join(evidence) + "\n",
        )

    def _extract_service_a_timeout_seconds(self, source_path: Path) -> float:
        if not source_path.exists():
            raise ValueError(f"Replay target file not found: {source_path}")
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "AsyncClient":
                continue
            for keyword in node.keywords:
                if keyword.arg == "timeout" and isinstance(keyword.value, ast.Constant):
                    return float(keyword.value.value)
        raise ValueError("Could not find httpx.AsyncClient(timeout=...) in service_a/main.py.")

    def _load_workspace_module(self, module_name: str, module_path: Path):
        if not module_path.exists():
            raise ValueError(f"Replay target file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load replay module from {module_path}")

        workspace_root = str(module_path.parent.parent.resolve())
        package_name = module_path.parent.name
        original_sys_path = list(sys.path)
        previous_module = sys.modules.pop(module_name, None)
        previous_package_modules = {
            key: value
            for key, value in list(sys.modules.items())
            if key == package_name or key.startswith(f"{package_name}.")
        }
        for key in list(previous_package_modules.keys()):
            sys.modules.pop(key, None)
        try:
            sys.path.insert(0, workspace_root)
            namespace_pkg = types.ModuleType(package_name)
            namespace_pkg.__path__ = [str(module_path.parent.resolve())]  # type: ignore[attr-defined]
            namespace_pkg.__package__ = package_name
            sys.modules[package_name] = namespace_pkg
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        finally:
            sys.path[:] = original_sys_path
            sys.modules.pop(module_name, None)
            for key in list(sys.modules.keys()):
                if key == package_name or key.startswith(f"{package_name}."):
                    sys.modules.pop(key, None)
            sys.modules.update(previous_package_modules)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
