"""Advanced reward shaping for SRE incident response."""

from typing import Set

from ..models import SREAction


class SREStepRewarder:
    """Calculates granular rewards at every step to guide the agent."""

    def __init__(self):
        self.seen_logs: Set[str] = set()
        self.seen_source: Set[str] = set()

    def calculate_reward(self, action: SREAction) -> float:
        """Assign rewards for diagnostic actions.

        This helps the agent learn that reading logs and source code
        is part of a successful SRE workflow.
        """
        reward = -0.01  # Base step cost (prevents rambling)

        if action.tool == "terminal":
            cmd = action.command.lower()
            
            # 1. Reward for reading logs (first time only)
            if "cat " in cmd and ".log" in cmd:
                log_name = cmd.split("cat ")[-1].strip()
                if log_name not in self.seen_logs:
                    reward += 0.05
                    self.seen_logs.add(log_name)

            # 2. Reward for inspecting source code (first time)
            if "cat " in cmd and ".py" in cmd:
                py_name = cmd.split("cat ")[-1].strip()
                if py_name not in self.seen_source:
                    reward += 0.05
                    self.seen_source.add(py_name)

        return reward
