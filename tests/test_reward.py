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
