"""Tests for reward shaping behavior."""

from __future__ import annotations

from sre_env.models import SREAction, SREObservation
from sre_env.server.reward import SREStepRewarder


def test_rewarder_does_not_reward_fake_source_file_created_after_reset() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py", "logs/error.log"])

    reward = rewarder.calculate_reward(
        SREAction(tool="terminal", command="cat service_a/fake.py"),
        SREObservation(stdout="print('fake')", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    assert reward == -0.01


def test_rewarder_gives_small_replay_reward_and_bonus_on_success() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py", "logs/error.log"])

    reward = rewarder.calculate_reward(
        SREAction(tool="replay", command="create_item_contract"),
        SREObservation(stdout="contract_ok=true\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    assert reward == 0.02


def test_rewarder_rewards_rca_when_expected_fix_file() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py", "RCA_template.md"])

    reward = rewarder.calculate_reward(
        SREAction(
            tool="editor",
            file_path="RCA.md",
            file_content=(
                "# Incident RCA Report\n\n"
                "## Root Cause\nA status-code mismatch broke the API contract.\n\n"
                "## Affected Services\nitem creation endpoint and dependent clients.\n\n"
                "## Fix Applied\nSet create_item to return HTTP 201.\n\n"
                "## Prevention\nKeep replay checks and API contract tests.\n"
            ),
        ),
        SREObservation(stdout="", exit_code=0),
        expected_fix_files=["app/main.py", "RCA.md"],
    )

    assert reward > 0.0


def test_rewarder_penalizes_duplicate_cat_only_when_output_unchanged() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])

    first = rewarder.calculate_reward(
        SREAction(tool="terminal", command="cat app/main.py"),
        SREObservation(stdout="print('v1')\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    second = rewarder.calculate_reward(
        SREAction(tool="terminal", command="cat app/main.py"),
        SREObservation(stdout="print('v1')\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    assert first == 0.04
    assert second == -0.03


def test_rewarder_allows_post_edit_cat_without_duplicate_penalty() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])

    rewarder.calculate_reward(
        SREAction(tool="terminal", command="cat app/main.py"),
        SREObservation(stdout="print('v1')\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    rewarder.calculate_reward(
        SREAction(tool="editor", file_path="app/main.py", file_content="print('v2')\n"),
        SREObservation(stdout="", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    after_edit = rewarder.calculate_reward(
        SREAction(tool="terminal", command="cat app/main.py"),
        SREObservation(stdout="print('v2')\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    assert after_edit == -0.01
