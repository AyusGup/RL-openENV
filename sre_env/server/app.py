"""FastAPI server for the SRE Incident Response environment."""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..models import SREAction, SREObservation, SREState
from .sre_environment import SREEnvironment

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
FIXTURES_DIR = BASE_DIR / "fixtures"
WORKSPACE_ROOT = BASE_DIR / "workspace"

app = FastAPI(title="SRE Incident Response OpenEnv")

# Initialize environment
env = SREEnvironment(FIXTURES_DIR, WORKSPACE_ROOT)

class ResetRequest(BaseModel):
    task_id: str

@app.get("/")
async def root():
    return {"message": "SRE Incident Response Environment is running."}

@app.get("/tasks")
async def list_tasks():
    """List all available incident scenarios."""
    return env.registry.list_tasks()

@app.post("/reset", response_model=SREObservation)
async def reset(request: ResetRequest):
    """Start a new incident response episode."""
    observation = await env.reset(request.task_id)
    if "Error" in observation.stderr:
        raise HTTPException(status_code=400, detail=observation.stderr)
    return observation

@app.post("/step", response_model=SREObservation)
async def step(action: SREAction):
    """Execute an agent action."""
    return await env.step(action)

@app.get("/state", response_model=Optional[SREState])
async def get_state():
    """Get the current episode state."""
    return env.state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
