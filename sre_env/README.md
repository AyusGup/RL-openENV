---
title: SRE Incident Response OpenEnv
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SRE Incident Response - OpenEnv

An OpenEnv-style environment for evaluating agents on realistic SRE incident-response tasks: investigating alerts, reading logs, fixing broken services, validating the fix, and submitting for grading.

## Features
- **Three-task benchmark**: Easy, medium, and hard incident scenarios with distinct failure modes.
- **Action Space**: Simple `terminal`, `editor`, `replay`, and `submit` tools.
- **Provider Pattern**: Swappable data sources for logs, metrics, and execution.
- **Deterministic Grading**: Using `difflib`, `pytest` exit codes, and regex-based RCA scoring.

## Motivation
This environment models a real operational workflow humans perform during incident response: inspect alerts, read logs, inspect source, apply a fix, run verification, and submit a resolution. The tasks progress from a simple API contract bug to retry logic drift and finally a multi-service timeout incident with an RCA requirement.

## Interface
Action space:
- `terminal`: run one workspace-scoped shell command
- `editor`: replace one file with full contents
- `replay`: run a deterministic task-specific validation probe
- `submit`: finish the episode and trigger grading

Observation model:
- `stdout`
- `stderr`
- `exit_code`
- `file_tree`
- `alert_message`

Step result model:
- `observation`
- `reward`
- `done`
- `info`

State model:
- `episode_id`
- `task_id`
- `task_name`
- `step_count`
- `max_steps`
- `cumulative_reward`
- `done`
- `workspace_root`

## Repository Layout
- `openenv.yaml`: OpenEnv manifest used by `openenv validate` and `openenv push`.
- `fixtures/`: task fixtures, hidden tests, and replay assets.
- `pyproject.toml`, `uv.lock`, `Dockerfile`: self-contained deployment dependencies/build config.

Python client (typed async):
```python
from sre_env import SREAction, SREEnv

async with SREEnv() as env:
    observation = await env.reset(task_id="task2_retry_logic")
    result = await env.step(SREAction(tool="terminal", command="cat app/retry_handler.py"))
    state = await env.state()
```
`SREEnv()` resolves base URL in this order: explicit arg, `OPENENV_BASE_URL`, `.openenv_port`, then `http://127.0.0.1:7860`.

## Linux Setup
From this directory (`sre_env`), create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Validate the OpenEnv manifest locally:
```bash
openenv validate .
```

Start the environment server in one terminal:
```bash
source .venv/bin/activate
cd ..
python -m sre_env.server.app
```
Server startup prefers `PORT` when set. If `PORT` is unset, it tries `7860` and falls back to an OS-selected free port when `7860` is busy. The selected port is written to `.openenv_port`.

## Hugging Face Secrets
For Hugging Face Spaces, add these in your Space settings:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Put `HF_TOKEN` in Space Secrets, not plain Variables.

Deploy to Hugging Face Spaces with:
```bash
openenv push . --repo-id Jha-ayush/rl-openenv
```

Runtime workspace is created under `workspace/` by default.

To create a token:
1. Sign in to Hugging Face.
2. Open `https://huggingface.co/settings/tokens`.
3. Create a fine-grained token.
4. Grant permission to make Inference Providers calls.

## Tasks

### Task 1: FastAPI Status Code Mismatch
Difficulty: easy

The item-creation flow violates its API contract. The agent must inspect logs and source, identify the bug, fix it, run tests, and submit the workspace for deterministic grading.

### Task 2: Off-by-One Retry Bug
Difficulty: medium

The upstream retry handler gives up one attempt too early. The agent must inspect logs, patch the retry loop, verify the fix, and write an `RCA.md`.

### Task 3: Cascading Timeout Failure
Difficulty: hard

Service A times out before Service B can complete a slower enrichment path. The agent must inspect logs across both services, adjust the caller timeout, improve Service B latency, verify the tests, and write an `RCA.md`.

## License
MIT
