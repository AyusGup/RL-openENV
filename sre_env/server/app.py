"""FastAPI server for the SRE Incident Response environment."""

import os
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app

from sre_env.models import SREAction, SREObservation, TaskSummary
from sre_env.server.sre_environment import SREEnvironment
from sre_env.tasks.registry import TaskRegistry

PACKAGE_ROOT = Path(os.getenv("OPENENV_REPO_ROOT", Path(__file__).resolve().parents[1]))
FIXTURES_DIR = Path(os.getenv("OPENENV_FIXTURES_DIR", PACKAGE_ROOT / "fixtures"))
WORKSPACE_ROOT = Path(os.getenv("OPENENV_WORKSPACE_ROOT", PACKAGE_ROOT / "workspace"))
DEFAULT_TASK_ID = os.getenv("SRE_TASK_NAME")
PORT = int(os.getenv("PORT", 8000))


def create_sre_app(
    fixtures_dir: Path,
    workspace_root: Path,
    default_task_id: str | None = None,
) -> FastAPI:
    """Create a template-style OpenEnv app and attach `/tasks` extension route."""

    def env_factory() -> SREEnvironment:
        return SREEnvironment(
            fixtures_dir=fixtures_dir,
            workspace_root=workspace_root,
            default_task_id=default_task_id,
        )

    app = create_app(
        env_factory,
        SREAction,
        SREObservation,
        env_name="sre_env",
        max_concurrent_envs=1,
    )

    registry = TaskRegistry(fixtures_dir)

    @app.get("/tasks", response_model=List[TaskSummary], tags=["Environment Info"])
    async def list_tasks() -> List[TaskSummary]:
        return registry.list_summaries()

    return app


app = create_sre_app(FIXTURES_DIR, WORKSPACE_ROOT, DEFAULT_TASK_ID)


def main(host: str = "0.0.0.0", port: int = PORT) -> None:
    """Run the SRE server with Uvicorn."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
