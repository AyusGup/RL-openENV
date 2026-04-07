"""Run inference baseline across all SRE tasks and print a summary table."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

TASKS = [
    "task1_wrong_status",
    "task2_retry_logic",
    "task3_cascading_failure",
]

END_PATTERN = re.compile(
    r"^\[END\]\s+success=(?P<success>true|false)\s+steps=(?P<steps>\d+)\s+score=(?P<score>\d+(?:\.\d+)?)\s+rewards=(?P<rewards>.*)$"
)


@dataclass
class TaskResult:
    """Structured baseline result for one task run."""

    task: str
    success: bool
    steps: int
    score: float
    rewards: str


def parse_end_line(task: str, output: str) -> TaskResult:
    """Extract the final [END] summary line from inference output."""
    for line in reversed(output.splitlines()):
        match = END_PATTERN.match(line.strip())
        if match:
            return TaskResult(
                task=task,
                success=match.group("success") == "true",
                steps=int(match.group("steps")),
                score=float(match.group("score")),
                rewards=match.group("rewards"),
            )
    raise RuntimeError(f"Could not parse [END] line for {task}.")


def run_task(task: str, repo_root: Path) -> TaskResult:
    """Run the baseline inference script once for a single task."""
    env = os.environ.copy()
    env["SRE_TASK_NAME"] = task
    command = [sys.executable, "inference.py"]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"inference.py exited with code {completed.returncode} for {task}.\n"
            f"Output:\n{combined_output}"
        )
    return parse_end_line(task, combined_output)


def print_summary(results: List[TaskResult]) -> None:
    """Print a deterministic, compact summary table."""
    print("\nBaseline Summary")
    print("task\tsuccess\tsteps\tscore")
    total_score = 0.0
    for result in results:
        total_score += result.score
        print(
            f"{result.task}\t{str(result.success).lower()}\t{result.steps}\t{result.score:.3f}"
        )
    average_score = total_score / len(results) if results else 0.0
    print(f"average_score\t-\t-\t{average_score:.3f}")


def main() -> None:
    """Entry point for baseline batch execution."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
        raise RuntimeError(
            "Missing API credentials. Set OPENAI_API_KEY (or HF_TOKEN/API_KEY) before running."
        )

    repo_root = Path(__file__).resolve().parent
    results: List[TaskResult] = []
    for task in TASKS:
        print(f"\n=== Running {task} ===")
        results.append(run_task(task, repo_root))
    print_summary(results)


if __name__ == "__main__":
    main()
