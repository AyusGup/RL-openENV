"""Pure-function tests for inference-side guardrails."""

from __future__ import annotations

import pytest

from inference import (
    build_rca_content,
    choose_forced_action,
    history_has_successful_replay,
    normalize_action,
    task_requires_rca,
)


def test_build_rca_content_for_task2_mentions_retry_root_cause() -> None:
    content = build_rca_content(
        task_name="task2_retry_logic",
        alert_message="retry issue",
        template="# Incident RCA Report\n\n## Root Cause\n",
    )

    assert "## Root Cause" in content
    assert "range(max_retries + 1)" in content
    assert "## Fix Applied" in content


def test_normalize_action_accepts_replay() -> None:
    action = normalize_action({"tool": "replay", "command": "retry_health_contract"})

    assert action["tool"] == "replay"
    assert action["command"] == "retry_health_contract"


def test_normalize_action_rejects_replay_without_command() -> None:
    with pytest.raises(RuntimeError):
        normalize_action({"tool": "replay"})


def test_history_has_successful_replay_detects_contract_success() -> None:
    assert history_has_successful_replay(
        [
            {
                "action": "replay(retry_health_contract)",
                "stdout": "replay=retry_health_contract\ncontract_ok=true\n",
            }
        ]
    )


def test_choose_forced_action_creates_rca_after_replay_success() -> None:
    action = choose_forced_action(
        task_name="task2_retry_logic",
        file_tree=["app/retry_handler.py"],
        known_files={},
        history=[
            {"action": "write(app/retry_handler.py)", "reward": "0.02", "stdout": ""},
            {
                "action": "replay(retry_health_contract)",
                "reward": "0.05",
                "stdout": "contract_ok=true",
            },
        ],
        latest_step_result={
            "info": {"last_action_error": "cat: RCA.md: No such file or directory"},
        },
        alert_message="retry issue",
        step_number=13,
        max_steps=16,
    )

    assert action is not None
    assert action["tool"] == "editor"
    assert action["file_path"] == "RCA.md"
    assert "## Prevention" in action["file_content"]


def test_choose_forced_action_waits_for_verification_before_rca() -> None:
    action = choose_forced_action(
        task_name="task2_retry_logic",
        file_tree=["app/retry_handler.py"],
        known_files={},
        history=[{"action": "write(app/retry_handler.py)", "reward": "0.02", "stdout": ""}],
        latest_step_result={"info": {"last_action_error": None}},
        alert_message="retry issue",
        step_number=5,
        max_steps=16,
    )

    assert action is None


def test_choose_forced_action_creates_rca_near_budget_without_replay() -> None:
    action = choose_forced_action(
        task_name="task3_cascading_failure",
        file_tree=["service_a/main.py", "service_b/database.py"],
        known_files={},
        history=[{"action": "write(service_a/main.py)", "reward": "0.02", "stdout": ""}],
        latest_step_result={"info": {"last_action_error": None}},
        alert_message="timeout issue",
        step_number=24,
        max_steps=24,
    )

    assert action is not None
    assert action["tool"] == "editor"
    assert action["file_path"] == "RCA.md"


def test_task_requires_rca_reads_task_config() -> None:
    assert task_requires_rca("task1_wrong_status") is True
    assert task_requires_rca("task2_retry_logic") is True
    assert task_requires_rca("task3_cascading_failure") is True
