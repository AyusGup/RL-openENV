---
title: SRE Incident Response OpenEnv
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# SRE Incident Response - OpenEnv

An OpenEnv-style environment for evaluating agents on realistic SRE incident-response tasks: investigating alerts, reading logs, fixing broken services, validating the fix, and submitting for grading.

## Features
- **Three-task benchmark**: Easy, medium, and hard incident scenarios with distinct failure modes.
- **Action Space**: Simple `terminal`, `editor`, `replay`, and `submit` tools.
- **Provider Pattern**: Swappable data sources for logs, metrics, and execution.
- **Deterministic Grading**: Weighted `file_change`, continuous `tests_pass`, and `regex_match` scoring.
- **Structured Inference Loop**: Built-in guards to reduce replay/cat spam and auto-submit once criteria are met.

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

Runtime behavior notes:
- `done=true` is returned when the agent submits or when server step budget is exhausted (auto-grade path).
- For RCA-required tasks, inference policy prefers: code edit -> RCA -> replay -> submit.
- Inference caps replay spam to at most 2 consecutive replay actions without an intervening code edit.
- Inference will force a final submit if the loop exits without `done=true`.
- If LLM credits are exhausted mid-episode (provider `402`), inference degrades gracefully:
  - after at least one code edit: fallback to replay-first flow,
  - before any code edit: fallback to submit (no synthetic/hardcoded file content is injected).

Scoring notes:
- Final task score is computed by the server grader on submit/auto-grade.
- Grader also returns per-component scores in step metadata: `file_change`, `tests_pass`, `regex_match`.
- `tests_pass` is continuous (`passed / total`) when pytest summaries are parseable.
- Grader test subprocess timeout is `120s` (to reduce false failures on slower filesystems).
- Server grading remains raw in `[0, 1]`; inference normalizes emitted task scores into strict `(0, 1)` for evaluator compatibility.
- Step-level reward shaping is conservative (`base_step_penalty=-0.005`) and discourages no-op loops (duplicate cat/replay spam).

## Repository Layout
- `openenv.yaml`: OpenEnv manifest used by `openenv validate` and `openenv push`.
- `fixtures/`: task fixtures, hidden tests, and replay assets.
- `pyproject.toml`, `uv.lock`, `Dockerfile`: self-contained deployment dependencies/build config.

Python client (typed async):
```python
from rl_env import SREAction, SREEnv

async with SREEnv("http://127.0.0.1:8000") as env:
    observation = await env.reset(task_id="task2_retry_logic")
    result = await env.step(SREAction(tool="terminal", command="cat app/retry_handler.py"))
    state = await env.state()
```

## Linux Setup
From this directory (`rl_env`), create a virtual environment and install dependencies:
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
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Inference Runtime Knobs
Key environment variables used by `inference.py`:
- `MAX_STEPS` (default `20`)
- `SUCCESS_SCORE_THRESHOLD` (default `0.5`)
- `HTTP_TIMEOUT_SECONDS` (default `180`)
- `SCORE_EPSILON` (default `1e-6`)
- `ENABLE_GRADE_BREAKDOWN_LOGS` (default `0`; set `1` to enable per-component `[GRADE]` logs)

## Hugging Face Secrets
For Hugging Face Spaces, add these in your Space settings:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Port configuration:
- Space metadata `app_port` is set to `8000` in this README front matter.
- Runtime uses `PORT` when provided by the platform; otherwise the app defaults to `8000`.

Put `HF_TOKEN` in Space Secrets, not plain Variables.

Deploy to Hugging Face Spaces with:
```bash
openenv push . --repo-id Jha-ayush/rl-openenv
```

## GitHub Actions Auto Deploy
This repository includes an auto-deploy workflow at `.github/workflows/deploy-hf-space.yml`.

Trigger behavior:
- Runs on every push to `main` (including merge commits).
- Can also be started manually with `workflow_dispatch`.

Required GitHub repository settings:
- Secret: `HF_TOKEN`
- Variable: `HF_SPACE_REPO_ID` (example: `Jha-ayush/rl-openenv`)

Once these are set, merging to `main` will automatically run:
```bash
openenv push . --repo-id "$HF_SPACE_REPO_ID"
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
RCA grading includes a semantic alignment check in `Fix Applied` for the actual loop-boundary fix (e.g. `max_retries + 1`).

### Task 3: Cascading Timeout Failure
Difficulty: hard

Service A times out before Service B can complete a slower enrichment path. The agent must inspect logs across both services, adjust the caller timeout, improve Service B latency, verify the tests, and write an `RCA.md`.
RCA grading expects `Fix Applied` to cover both sides of the remediation:
- Service A timeout/deadline adjustment.
- Service B/database latency-side improvement.

## License
MIT
