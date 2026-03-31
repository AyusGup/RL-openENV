# SRE Incident Response - OpenEnv

An OpenEnv-compliant environment where an AI agent diagnoses and fixes production incidents.

## Features
- **3 SRE Scenarios**: From simple status code mismatches to complex cascading timeout failures.
- **Action Space**: Simple `terminal`, `editor`, and `submit` tools.
- **Provider Pattern**: Swappable data sources for logs, metrics, and execution.
- **Deterministic Grading**: Using `difflib`, `pytest` exit codes, and regex-based RCA scoring.

## Getting Started
To install dependencies:
```bash
pip install -e .
```

To run the environment server:
```bash
python -m sre_env.server.app
```

## Task 1: FastAPI Status Code Mismatch
A POST endpoint returns 200 instead of 201. The agent must find the bug and fix it.

## Task 2: Retry Logic Off-by-One
Transient failures aren't being retried enough due to an off-by-one error in a loop.

## Task 3: Cascading Timeout Failure
Service B slowed down, and Service A's hardcoded timeout is too low.

## License
MIT
