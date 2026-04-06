# SRE Incident Response - OpenEnv

An OpenEnv-style environment for evaluating agents on realistic SRE incident-response tasks: investigating alerts, reading logs, fixing broken services, validating the fix, and submitting for grading.

## Features
- **Three-task benchmark**: Easy, medium, and hard incident scenarios with distinct failure modes.
- **Action Space**: Simple `terminal`, `editor`, and `submit` tools.
- **Provider Pattern**: Swappable data sources for logs, metrics, and execution.
- **Deterministic Grading**: Using `difflib`, `pytest` exit codes, and regex-based RCA scoring.

## Motivation
This environment models a real operational workflow humans perform during incident response: inspect alerts, read logs, inspect source, apply a fix, run verification, and submit a resolution. The tasks progress from a simple API contract bug to retry logic drift and finally a multi-service timeout incident with an RCA requirement.

## Interface
Action space:
- `terminal`: run one workspace-scoped shell command
- `editor`: replace one file with full contents
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

## Getting Started
To install dependencies:
```bash
pip install -e .
```

To run the environment server:
```bash
python -m sre_env.server.app
```

To run the baseline inference script:
```bash
python inference.py
```

The inference client uses the OpenAI-compatible model endpoint to choose the next environment action step by step and emits the required `[START]`, `[STEP]`, and `[END]` logs.

To validate the OpenEnv manifest locally:
```bash
openenv validate
```

## WSL Setup
From WSL, create a Linux virtual environment in the repo root:
```bash
cd /mnt/c/Users/ag835/My_projects/rl-openenv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Start the environment server in one terminal:
```bash
source .venv/bin/activate
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

In a second terminal, set the required inference variables and run:
```bash
source .venv/bin/activate
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENAI_API_KEY="hf_your_token_here"
python inference.py
```

You can copy the template in `.env.example` into your shell manually or load it with your preferred dotenv workflow. Do not commit real secrets.

## Hugging Face Secrets
For Hugging Face Spaces, add these in your Space settings:
- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY`

Put `OPENAI_API_KEY` in Space Secrets, not plain Variables.

To create a token:
1. Sign in to Hugging Face.
2. Open `https://huggingface.co/settings/tokens`.
3. Create a fine-grained token.
4. Grant permission to make Inference Providers calls.

The default OpenAI-compatible endpoint used by this repo is:
```text
https://router.huggingface.co/v1
```

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
