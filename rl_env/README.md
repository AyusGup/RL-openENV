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

A high-fidelity **SRE (Site Reliability Engineering)** evaluation environment built on the OpenEnv framework. It is designed to challenge AI agents with realistic production-grade incident response scenarios, requiring them to perform end-to-end remediation: from initial alert triage to root cause analysis (RCA).

## 🚀 Environment Overview
Unlike generic coding benchmarks, the SRE OpenEnv focuses on **observability-driven debugging**. Agents are not just given a bug description; they are given a production alert and a "broken" environment. To succeed, they must:
- **Triage Alerts**: Interpret high-level monitoring signals (Prometheus/Alertmanager style).
- **Navigate Microservices**: Explore complex service hierarchies and dependencies.
- **Analyze Observability Data**: Parse application logs and service-to-service communication traces.
- **Implement Fixes**: Patch code or configuration in a persistent workspace.
- **Verify Remediation**: Use task-specific "replay" probes to confirm the fix works in vivo.
- **Document the Fix**: Write an Incident RCA Report (`RCA.md`) that accurately describes the problem and the solution.

## ✨ Key Features
- **Progressive Difficulty**: Three curated tasks ranging from single-file logic bugs to multi-service cascading failures.
- **Persistent Workspace**: A realistic filesystem where actions have consequences and state is maintained.
- **Rich Action Space**: Full access to a shell (`terminal`), a file manager (`editor`), and specialized validation tools (`replay`).
- **Hardened Grading**: A deterministic scoring engine that evaluates code changes, test pass rates, and the accuracy of the RCA documentation.
- **Observability Stack**: Simulated logs and monitoring metadata tailored to each incident.

## 🎯 Motivation
The SRE role is unique: it requires a blend of software engineering and systems operations. This environment evaluates whether an AI agent can bridge that gap. We measure an agent's ability to maintain a mental model of a system under pressure, avoid unproductive "cat/grep" loops, and provide high-quality documentation that matches its technical actions.

## 🛠️ Incident Response Workflow
The environment enforces a realistic lifecycle for every incident:
1.  **Trigger**: An alert message is surfaced in the initial observation.
2.  **Investigation**: The agent uses `terminal` to inspect logs (`logs/app.log`) and explore the `file_tree`.
3.  **Remediation**: The agent applies a fix using the `editor`.
4.  **Verification**: The agent runs the `replay` command to verify the system's health.
5.  **Documentation**: For complex tasks, the agent must write an `RCA.md`.
6.  **Resolution**: The agent calls `submit` to close the incident and receive a final grade.

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
