"""FastAPI server for the SRE Incident Response environment."""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from sre_env.models import SREAction, SREObservation, SREState, SREStepResult, TaskSummary
from sre_env.server.sre_environment import SREEnvironment

# Configuration
PACKAGE_ROOT = Path(os.getenv("OPENENV_REPO_ROOT", Path(__file__).resolve().parents[1]))
FIXTURES_DIR = Path(os.getenv("OPENENV_FIXTURES_DIR", PACKAGE_ROOT / "fixtures"))
WORKSPACE_ROOT = Path(os.getenv("OPENENV_WORKSPACE_ROOT", PACKAGE_ROOT / "workspace"))

app = FastAPI(title="SRE Incident Response OpenEnv")

# Initialize environment
env = SREEnvironment(FIXTURES_DIR, WORKSPACE_ROOT)

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "SRE Incident Response Environment is running."}

@app.get("/health")
async def health():
    """Container health probe endpoint."""
    return {"status": "healthy"}

@app.get("/tasks", response_model=List[TaskSummary])
async def list_tasks():
    """List all available incident scenarios."""
    return env.registry.list_summaries()

@app.post("/reset", response_model=SREObservation)
async def reset(request: ResetRequest):
    """Start a new incident response episode."""
    observation = await env.reset(request.task_id)
    if observation.stderr.startswith("Error:"):
        raise HTTPException(status_code=400, detail=observation.stderr)
    return observation

@app.post("/step", response_model=SREStepResult)
async def step(action: SREAction):
    """Execute an agent action."""
    result = await env.step(action)
    if result.observation.stderr.startswith("Error:") and not result.done:
        raise HTTPException(status_code=400, detail=result.observation.stderr)
    return result

@app.get("/state", response_model=Optional[SREState])
async def get_state():
    """Get the current episode state."""
    return env.state

if __name__ == "__main__":
   """Run the FastAPI app with Uvicorn."""

   uvicorn.run(app, host="0.0.0.0", port=7860)
