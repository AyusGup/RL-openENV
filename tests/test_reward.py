"""Tests for simplified reward policy behavior."""

from __future__ import annotations

from rl_env.models import SREAction, SREObservation
from rl_env.server.reward import SREStepRewarder


def test_compile_transition_invalid_to_valid_yields_positive_component() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])
    rewarder.compile_validity_by_file["app/main.py"] = False

    reward = rewarder.calculate_reward(
        SREAction(
            tool="editor",
            file_path="app/main.py",
            file_content="def ok():\n    return 1\n",
        ),
        SREObservation(stdout="", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    breakdown = rewarder.get_last_breakdown()
    assert breakdown["compile_component"] == 0.2
    assert reward == (
        breakdown["base_penalty"]
        + breakdown["compile_component"]
        + breakdown["replay_test_component"]
        - breakdown["complexity_component"]
        + breakdown["heuristic_component"]
    )


def test_compile_transition_valid_to_invalid_yields_negative_component() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])
    rewarder.compile_validity_by_file["app/main.py"] = True

    reward = rewarder.calculate_reward(
        SREAction(
            tool="editor",
            file_path="app/main.py",
            file_content="def broken(:\n    pass\n",
        ),
        SREObservation(stdout="", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    breakdown = rewarder.get_last_breakdown()
    assert breakdown["compile_component"] == -0.1
    assert breakdown["compile_error_type"] == "SyntaxError"
    assert isinstance(reward, float)


def test_syntax_error_never_raises_and_returns_numeric_reward() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])
    rewarder.compile_validity_by_file["app/main.py"] = False

    reward = rewarder.calculate_reward(
        SREAction(
            tool="editor",
            file_path="app/main.py",
            file_content="def still_broken(:\n",
        ),
        SREObservation(stdout="", exit_code=0),
        expected_fix_files=["app/main.py"],
    )

    breakdown = rewarder.get_last_breakdown()
    assert isinstance(reward, float)
    assert breakdown["compile_valid_now"] is False
    assert breakdown["compile_error_type"] == "SyntaxError"


def test_unexpected_parse_error_is_caught_and_marked_parseerror() -> None:
    rewarder = SREStepRewarder()
    is_valid, error_type = rewarder._safe_parse_python(123)  # type: ignore[arg-type]

    assert is_valid is False
    assert error_type == "ParseError"


def test_replay_delta_reward_is_positive_only_once() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])

    first = rewarder.calculate_reward(
        SREAction(tool="replay", command="retry_health_contract"),
        SREObservation(stdout="contract_ok=true\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    first_breakdown = rewarder.get_last_breakdown()

    second = rewarder.calculate_reward(
        SREAction(tool="replay", command="retry_health_contract"),
        SREObservation(stdout="contract_ok=true\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    second_breakdown = rewarder.get_last_breakdown()

    assert first_breakdown["replay_test_component"] == 1.0
    assert second_breakdown["replay_test_component"] == 0.0
    assert first > second


def test_pytest_terminal_command_does_not_add_test_component() -> None:
    rewarder = SREStepRewarder()
    rewarder.seed_initial_files(["app/main.py"])

    rewarder.calculate_reward(
        SREAction(tool="terminal", command="python -m pytest -q"),
        SREObservation(stdout="1 passed\n", exit_code=0),
        expected_fix_files=["app/main.py"],
    )
    breakdown = rewarder.get_last_breakdown()

    assert breakdown["replay_test_component"] == 0.0
