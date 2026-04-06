"""Pure-function tests for inference-side guardrails."""

from __future__ import annotations

from inference import build_rca_content, choose_forced_action


def test_build_rca_content_for_task2_mentions_retry_root_cause() -> None:
    content = build_rca_content(
        task_name="task2_retry_logic",
        alert_message="retry issue",
        template="# Incident RCA Report\n\n## Root Cause\n",
    )

    assert "## Root Cause" in content
    assert "range(max_retries + 1)" in content
    assert "## Fix Applied" in content


def test_choose_forced_action_creates_rca_after_progress() -> None:
    action = choose_forced_action(
        task_name="task2_retry_logic",
        file_tree=["app/retry_handler.py", "RCA_template.md"],
        known_files={"RCA_template.md": "# Incident RCA Report\n"},
        history=[
            {"action": "write(app/retry_handler.py)", "reward": "0.02", "stdout": ""},
            {
                "action": "python -m pytest -q tests/test_retry.py",
                "reward": "0.05",
                "stdout": "1 passed in 0.11s",
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
        file_tree=["app/retry_handler.py", "RCA_template.md"],
        known_files={"RCA_template.md": "# Incident RCA Report\n"},
        history=[{"action": "write(app/retry_handler.py)", "reward": "0.02", "stdout": ""}],
        latest_step_result={"info": {"last_action_error": None}},
        alert_message="retry issue",
        step_number=5,
        max_steps=16,
    )

    assert action is None
